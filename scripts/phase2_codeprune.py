"""Code-level (structured) pruning, one strict seed (docs/phase0_protocol.md, section 12).

    python scripts/phase2_codeprune.py --backbone_ckpt B.pth --pq_ckpt runs/.../stage2/pipnet_*.pth \
        --cache_dir runs/phase1/feature_cache --masking_dir runs/phase2/strict_s40/masking --seed 40

Edge pruning (B1, P1) chooses which class-code edges to keep; the codes that survive are whatever those edges
happen to touch (about 1,300 at K = 10). Code-level pruning chooses the codes first:

  1. score every code m over all classes:  CB1 (control): sum_k w_km;  CP1: sum_k w_km * E[s_m | y = k]
     (w = ReLU(W), class means from the training split, as P1);
  2. keep the M best codes;
  3. keep the top K edges of each class among those codes, ranked by the matching edge score (B1: w_km,
     P1: w_km * E[s_m | y = k]). A class with fewer than K positive edges on the kept codes keeps fewer.

Mask keys are "<score>-M<M>_K<K>", e.g. CP1-M800_K10. Writes to masking_dir: codeprune_masks.pt (loadable by
phase2_retrain.py), codeprune.json (one-shot val + test top-1, codes kept, edges, active codes) and
codeprune_test_preds.npz (per-image test predictions). Refuses to run again on an existing codeprune.json.

Code-matched control (section 12, post hoc): with --match_active_to CP1-M1000_K10 --tag matched, rebuilds that
CP1 mask, counts its active codes A for this seed, and builds only the CB1 mask whose budget M gives the active
count closest to A (ties: smaller M). Key "CB1-eqCP1M1000_K10"; files codeprune_matched{.json,_masks.pt,
_test_preds.npz}, so the earlier one-shot results are not touched and test is read once for the new mask.
"""

import argparse
from pathlib import Path

import numpy as np
import torch

import phase2_lib as P
import pq_lib as L
from phase2_masking import acc, load_split, pooled_scores


def code_masks(w_eff: torch.Tensor, scores: dict, budgets: list[int], k: int) -> tuple[dict, dict]:
    masks, info = {}, {}
    pos = w_eff > 0
    for name, edge in (("CB1", scores["B1"]), ("CP1", scores["P1"])):
        code_score = edge.masked_fill(~pos, 0).sum(0)            # (M,) summed over classes
        for m in budgets:
            keep = torch.zeros(w_eff.shape[1], dtype=torch.bool)
            keep[code_score.topk(min(m, len(code_score))).indices] = True
            mask = P.topk_per_class(edge, w_eff * keep, k)         # only kept codes are eligible
            key = f"{name}-M{m}_K{k}"
            masks[key] = mask
            per_class = mask.sum(1)
            info[key] = {"code_budget": m, "classes_below_k": int((per_class < k).sum()),
                         "classes_with_no_edge": int((per_class == 0).sum())}
    return masks, info


def matched_cb1(w_eff: torch.Tensor, scores: dict, target_key: str) -> tuple[dict, dict, torch.Tensor]:
    """CB1 mask with as many active codes as the CP1 mask `target_key` (e.g. CP1-M1000_K10)."""
    name, k = target_key.rsplit("_K", 1)
    if not name.startswith("CP1-M"):
        raise SystemExit(f"--match_active_to expects a CP1-M<M>_K<K> key, got {target_key}")
    k, m_ref = int(k), int(name.split("-M")[1])
    ref_mask = code_masks(w_eff, scores, [m_ref], k)[0][target_key]
    target = int(ref_mask.any(0).sum())
    edge = scores["B1"]
    code_score = edge.masked_fill(~(w_eff > 0), 0).sum(0)
    best = None
    for m in range(target, w_eff.shape[1] + 1):        # active codes <= M, so start at the target
        keep = torch.zeros(w_eff.shape[1], dtype=torch.bool)
        keep[code_score.topk(m).indices] = True
        mask = P.topk_per_class(edge, w_eff * keep, k)
        gap = abs(int(mask.any(0).sum()) - target)
        if best is None or gap < best[0]:
            best = (gap, m, mask)
        if gap == 0:
            break
    _, m, mask = best
    key = f"CB1-eq{name.replace('-', '')}_K{k}"
    per_class = mask.sum(1)
    info = {key: {"code_budget": m, "target_key": target_key, "target_active_codes": target,
                  "classes_below_k": int((per_class < k).sum()), "classes_with_no_edge": int((per_class == 0).sum())}}
    return {key: mask}, info, ref_mask


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone_ckpt", required=True)
    ap.add_argument("--pq_ckpt", required=True, help="dense Stage 2 checkpoint (Phase 1, strict)")
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--masking_dir", required=True, help="the seed's Phase 2 masking folder")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--budgets", type=int, nargs="+", default=[1200, 1000, 800, 600, 400])
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--match_active_to", default=None,
                    help="build only the CB1 mask matched in active codes to this CP1 key (e.g. CP1-M1000_K10)")
    ap.add_argument("--tag", default="", help="output name suffix, e.g. 'matched' -> codeprune_matched.json")
    args = ap.parse_args()

    if L.protocol_info()["protocol"] != "strict":
        raise SystemExit("runs under PQ_CUB_PROTOCOL=strict")
    out = Path(args.masking_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = "codeprune" + (f"_{args.tag}" if args.tag else "")
    if args.match_active_to and not args.tag:
        raise SystemExit("--match_active_to needs --tag, so the earlier codeprune.json is not overwritten")
    if (out / f"{stem}.json").exists():
        raise SystemExit(f"{out / f'{stem}.json'} exists: test was already read for these masks")

    dev = L.device()
    model = L.load_protoquant(args.backbone_ckpt, args.pq_ckpt, dev)
    w_eff = L.effective_weight(model).cpu()
    bb_sha = L.sha256(args.backbone_ckpt)
    s, y, ids = {}, {}, {}
    for split in ("train", "val", "test"):
        d = load_split(Path(args.cache_dir), split, bb_sha)
        s[split], y[split], ids[split] = pooled_scores(model, d["features"], dev), d["labels"], d["img_ids"]
    mu_in, mu_out = P.class_means(s["train"], y["train"], w_eff.shape[0])
    scores = P.edge_scores(w_eff, mu_in, mu_out)
    extra = {}
    if args.match_active_to:
        masks, info, ref_mask = matched_cb1(w_eff, scores, args.match_active_to)
        saved = out / "codeprune_masks.pt"   # the CP1 mask must be the one that was retrained
        if saved.is_file():
            old = torch.load(saved, map_location="cpu")["masks"].get(args.match_active_to)
            same = old is not None and bool(torch.equal(old.bool(), ref_mask))
            extra["cp1_mask_matches_saved"] = same
            if not same:
                print(f"[CODE] WARNING: rebuilt {args.match_active_to} differs from {saved}", flush=True)
        for key, i in info.items():
            print(f"[CODE] {key}: target {i['target_active_codes']} active codes -> CB1 budget M = {i['code_budget']}",
                  flush=True)
    else:
        masks, info = code_masks(w_eff, scores, args.budgets, args.k)

    w = w_eff.double()
    rows, preds = [], {}
    for key, mask in masks.items():
        method, k = key.rsplit("_K", 1)
        wm = w * mask
        lt = s["test"] @ wm.t()
        row = {"method": method, "K": int(k), "val_top1": acc(s["val"] @ wm.t(), y["val"]),
               "test_top1": acc(lt, y["test"]), **P.mask_stats(mask), **info[key]}
        rows.append(row)
        preds[key] = lt.argmax(1).numpy()
        print(f"[CODE] {key:16s} val {row['val_top1']:6.2f} test {row['test_top1']:6.2f} "
              f"edges {row['nonzero_edges']:5d} active codes {row['active_codes']:5d} "
              f"classes < K {row['classes_below_k']}", flush=True)

    torch.save({"masks": masks, "pq_ckpt": args.pq_ckpt, "seed": args.seed}, out / f"{stem}_masks.pt")
    np.savez_compressed(out / f"{stem}_test_preds.npz", img_ids=ids["test"].numpy(), labels=y["test"].numpy(), **preds)
    L.write_json(out / f"{stem}.json", {"seed": args.seed, "pq_ckpt": args.pq_ckpt, **L.protocol_info(), **extra,
                                         "models": rows})


if __name__ == "__main__":
    main()
