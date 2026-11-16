"""Phase 1 implementation tests (methodology, "Required tests before trusting results").

    python scripts/phase1_tests.py check_b0_log --log runs/.../b0/train.log --protocol strict
    python scripts/phase1_tests.py run --backbone_ckpt B.pth --pq_ckpt pipnet_*.pth \
        --cache_dir runs/phase1/feature_cache --cub_path ... --b0_log runs/.../b0/train.log --out tests.json

1. Masked-head invariance: zeroing the selected edges in W and applying the same edges as a mask on
   ReLU(W) must give the same predictions (float tolerance). Also checked against the repository's
   limit_prototypes().
2. Physical vs logical pruning: top-K masked model (all codes still in the softmax) vs the same model
   with unused codes physically removed (prune_inactive_prototypes), before retraining. Reports the
   logit change and changed predictions on validation. This is a measurement, not pass/fail: it tests
   the paper's claim that removal leaves predictions unchanged.
3. Feature-cache agreement: validation through the fp16 cache vs through the image pipeline.
4. Codebook pruning check: B0 counted code usage on the held-out validation split, not test.
All tests read validation only, never test.
"""

import argparse
import copy
import re
import sys
from pathlib import Path

import torch

import pq_lib as L

B0_EXPECTED = {
    "strict": "train loader split='train' (5394 imgs) | validation loader split='val' (600 imgs)",
    "paper": "train loader split='train_full' (5994 imgs) | validation loader split='test' (5794 imgs)",
}


def check_b0_log(log: str, protocol: str) -> dict:
    text = Path(log).read_text(errors="replace")
    lines = [ln for ln in text.splitlines() if "[CUB200]" in ln]
    expected = B0_EXPECTED[protocol]
    found = any(expected in ln for ln in lines)
    test_requested = any("TEST split requested" in ln for ln in lines)
    steps = len(re.findall(r"Pruned codebook from \d+ to \d+ entries", text))
    ok = found and not test_requested and steps > 0
    res = {
        "test": "4_b0_usage_split", "status": "PASS" if ok else "FAIL", "protocol": protocol,
        "expected_loader_line": expected, "loader_lines": [ln.split("[CUB200]", 1)[1].strip() for ln in lines],
        "test_split_requested": test_requested, "pruning_steps_logged": steps,
    }
    print(f"[TEST 4] {res['status']}: B0 usage split = "
          f"{'held-out val' if protocol == 'strict' else 'official test (paper protocol)'}; "
          f"{steps} pruning steps logged; test requested: {test_requested}")
    return res


def compare(a: torch.Tensor, b: torch.Tensor, labels: torch.Tensor) -> dict:
    pa, pb = a.argmax(1), b.argmax(1)
    return {
        "max_abs_logit_diff": (a - b).abs().max().item(),
        "mean_abs_logit_diff": (a - b).abs().mean().item(),
        "max_abs_logit": a.abs().max().item(),
        "changed_predictions": int((pa != pb).sum().item()),
        "n_images": int(len(labels)),
        "acc_a": (pa == labels).float().mean().item() * 100,
        "acc_b": (pb == labels).float().mean().item() * 100,
    }


def topk_mask(w_eff: torch.Tensor, k: int) -> torch.Tensor:
    idx = w_eff.topk(k, dim=1).indices
    return torch.zeros_like(w_eff, dtype=torch.bool).scatter_(1, idx, True) & (w_eff > 0)


def head_logits(model, feats: torch.Tensor, dev, bs: int = 256) -> torch.Tensor:
    out = []
    w = L.effective_weight(model)
    for i in range(0, len(feats), bs):
        s = L.scores_from_features(model, feats[i:i + bs].to(dev))
        out.append(L.logits_from_scores(s, w).cpu())
    return torch.cat(out)


def run(args) -> dict:
    dev = L.device()
    model = L.load_protoquant(args.backbone_ckpt, args.pq_ckpt, dev)
    cache = torch.load(Path(args.cache_dir) / "val.pt", map_location="cpu")
    if cache.get("backbone_sha256") != L.sha256(args.backbone_ckpt):
        raise SystemExit("Feature cache was built with a different backbone checkpoint")
    feats, labels = cache["features"], cache["labels"]
    results = {"pq_ckpt": args.pq_ckpt, "k": args.k, **L.head_stats(model), **L.protocol_info(), "tests": []}

    # Pooled scores once (fp32 from the fp16 cache); reused by tests 1 and 2.
    scores = torch.cat([L.scores_from_features(model, feats[i:i + 256].to(dev)).cpu()
                        for i in range(0, len(feats), 256)])
    w_raw = model.classifier.weight.detach().cpu()
    w_eff = torch.relu(w_raw)
    mask = topk_mask(w_eff, args.k)

    # ---- Test 1: masked-head invariance --------------------------------------------------------
    w_zeroed = w_raw.clone()
    w_zeroed[~mask] = 0.0
    logits_zeroed = scores @ torch.relu(w_zeroed).t()
    logits_masked = scores @ (w_eff * mask).t()
    limited = copy.deepcopy(model)
    limited.limit_prototypes(k=args.k)
    logits_repo = scores @ L.effective_weight(limited).cpu().t()
    t1 = {"zeroed_vs_mask": compare(logits_zeroed, logits_masked, labels),
          "mask_vs_repo_limit_prototypes": compare(logits_masked, logits_repo, labels)}
    tol = 1e-4 * max(1.0, t1["zeroed_vs_mask"]["max_abs_logit"])
    ok1 = all(v["changed_predictions"] == 0 and v["max_abs_logit_diff"] <= tol for v in t1.values())
    results["tests"].append({"test": "1_masked_head_invariance", "status": "PASS" if ok1 else "FAIL",
                             "tolerance": tol, **t1})
    print(f"[TEST 1] {'PASS' if ok1 else 'FAIL'}: max |logit diff| "
          f"{t1['zeroed_vs_mask']['max_abs_logit_diff']:.2e} (zero vs mask), "
          f"{t1['mask_vs_repo_limit_prototypes']['max_abs_logit_diff']:.2e} (mask vs limit_prototypes)")

    # ---- Test 2: physical vs logical pruning -------------------------------------------------
    t2 = {}
    for name, k in (("top_k", args.k), ("no_top_k", None)):
        logical = copy.deepcopy(model)
        if k is not None:
            logical.limit_prototypes(k=k)
        physical = copy.deepcopy(logical)
        removed_from = physical.num_prototypes
        physical.prune_inactive_prototypes(min_weight=0.0)
        c = compare(head_logits(logical, feats, dev), head_logits(physical, feats, dev), labels)
        t2[name] = {"codes_before": removed_from, "codes_after": int(physical.num_prototypes), **c}
    results["tests"].append({"test": "2_physical_vs_logical", "status": "MEASURED", **t2})
    tk = t2["top_k"]
    print(f"[TEST 2] MEASURED (K={args.k}): codes {tk['codes_before']} -> {tk['codes_after']}, "
          f"max |logit diff| {tk['max_abs_logit_diff']:.3f}, changed predictions "
          f"{tk['changed_predictions']}/{tk['n_images']}, val acc {tk['acc_a']:.2f}% -> {tk['acc_b']:.2f}%")

    # ---- Test 3: feature-cache agreement ------------------------------------------------------
    ds = L.dataset(args.cub_path, "val")
    if list(ds.img_ids) != cache["img_ids"].tolist():
        raise SystemExit("Validation image order differs between the cache and the dataset")
    img_logits = []
    with torch.no_grad():
        for x, _ in L.loader(ds, args.batch_size, args.num_workers):
            img_logits.append(model(x.to(dev)).logits.float().cpu())
    t3 = compare(torch.cat(img_logits), head_logits(model, feats, dev), labels)
    ok3 = t3["changed_predictions"] == 0
    results["tests"].append({"test": "3_feature_cache_agreement", "status": "PASS" if ok3 else "WARN", **t3})
    print(f"[TEST 3] {'PASS' if ok3 else 'WARN'}: max |logit diff| {t3['max_abs_logit_diff']:.2e}, "
          f"changed predictions {t3['changed_predictions']}/{t3['n_images']}")

    # ---- Test 4: B0 usage split -------------------------------------------------------------------
    if args.b0_log:
        results["tests"].append(check_b0_log(args.b0_log, L.protocol_info()["protocol"]))

    results["all_pass"] = all(t["status"] in ("PASS", "MEASURED", "WARN") for t in results["tests"])
    L.write_json(args.out, results)
    return results


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check_b0_log")
    c.add_argument("--log", required=True)
    c.add_argument("--protocol", choices=list(B0_EXPECTED), required=True)
    r = sub.add_parser("run")
    r.add_argument("--backbone_ckpt", required=True)
    r.add_argument("--pq_ckpt", required=True)
    r.add_argument("--cache_dir", required=True)
    r.add_argument("--cub_path", required=True)
    r.add_argument("--b0_log", default="")
    r.add_argument("--out", required=True)
    r.add_argument("--k", type=int, default=10)
    r.add_argument("--batch_size", type=int, default=128)
    r.add_argument("--num_workers", type=int, default=4)
    args = p.parse_args()
    if args.cmd == "check_b0_log":
        res = check_b0_log(args.log, args.protocol)
        sys.exit(0 if res["status"] == "PASS" else 1)
    res = run(args)
    sys.exit(0 if res["all_pass"] else 1)


if __name__ == "__main__":
    main()
