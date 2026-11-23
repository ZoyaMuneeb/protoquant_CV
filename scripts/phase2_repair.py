"""Closed-form weight repair after pruning, no retraining, one strict seed (docs/phase0_protocol.md, section 13).

    python scripts/phase2_repair.py --backbone_ckpt B.pth --pq_ckpt runs_bb2/phase1/strict_s40/stage2/pipnet_*.pth \
        --cache_dir runs_bb2/phase1/feature_cache --masking_dir runs_bb2/phase2/strict_s40/masking --seed 40

A pruning mask keeps some class-code edges and zeroes the rest; one-shot evaluation leaves the surviving weights
unchanged. Repair re-fits only the surviving weights, on the training split's cached features, keeping the mask
(support) and w >= 0, in the spirit of OBS / layer-wise reconstruction (SparseGPT) for a prototype head:

  LS  (main) per class k, non-negative least squares against the DENSE model's logits:
          min_{v >= 0}  mean_i (s_i[S_k] . v - z_ik)^2 + ridge * |v|^2,   z = s_dense @ ReLU(W)^T
      200 independent small problems (|S_k| <= K), solved exactly (scipy nnls). Needs no labels.
  CE  (control) the surviving weights re-fitted to the training LABELS: full-batch Adam on cross-entropy over the
      cached pooled scores, projected onto w >= 0 and the mask after every step (a fixed number of steps).

Each repair is evaluated in two settings:
  logical   every code stays in the softmax (as in one-shot masking, phase2_masking.py);
  physical  codes no class uses are removed first (prune_inactive_prototypes, as the retraining recipe does), so
            the softmax renormalises over the kept codes; pooled scores are recomputed and repair fits on those.
Physical repair is the fair comparison with retraining; logical repair isolates the effect of the weights.

Model keys: "<method>+<variant>_K<K>", variant in {LS, CE, phys, phys+LS, phys+CE}, e.g. P1+phys+LS_K10 and
CP1-M1000+phys+LS_K10 ("+phys" alone = physical removal with the weights unchanged). Masks come from the seed's
masking folder (masks.pt, codeprune_masks.pt, codeprune_matched_masks.pt). Writes to masking_dir:
repair.json (val + test top-1 per model, edges, codes, fit statistics), repair_test_preds.npz (per-image test
predictions, for the paired bootstrap) and repair_weights.pt (the repaired weights). Test is read once: the
script refuses to run again on an existing repair.json.
"""

import argparse
import copy
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import nnls

import phase2_lib as P
import pq_lib as L
from phase2_masking import acc, load_split

MASK_FILES = ("masks.pt", "codeprune_masks.pt", "codeprune_matched_masks.pt")
DEFAULT_KEYS = ["B1_K10", "P1_K10", "random_K10", "CB1-M1000_K10", "CP1-M1000_K10", "CB1-eqCP1M1000_K10"]


@torch.no_grad()
def pooled_scores(model, feats: torch.Tensor, dev, codes: torch.Tensor | None = None, bs: int = 256) -> torch.Tensor:
    """Pooled code activations (N x M'), softmax over all codes or, if `codes` is given, over those codes only
    (= the model after physical removal of every other code)."""
    out = []
    for i in range(0, len(feats), bs):
        sim = model._detect_prototypes(feats[i:i + bs].to(dev).float())
        if codes is not None:
            sim = sim[:, codes.to(sim.device)]
        out.append(model.pool_layer(F.softmax(sim / model.temperature, dim=1)).double().cpu())
    return torch.cat(out)


def repair_ls(s: torch.Tensor, z: torch.Tensor, mask: torch.Tensor, ridge: float) -> tuple[torch.Tensor, dict]:
    """Per-class non-negative least squares of the dense logits z (N x C) on the kept codes' scores s (N x M)."""
    n = s.shape[0]
    a_all, z_np = s.numpy(), z.numpy()
    w = torch.zeros(mask.shape, dtype=torch.float64)
    resid = []
    for k in range(mask.shape[0]):
        idx = torch.nonzero(mask[k]).flatten().numpy()
        if len(idx) == 0:
            continue
        a, b = a_all[:, idx], z_np[:, k]
        if ridge > 0:   # mean-squared objective + ridge * |v|^2 as one augmented least-squares problem
            a = np.vstack([a, np.sqrt(ridge * n) * np.eye(len(idx))])
            b = np.concatenate([b, np.zeros(len(idx))])
        v, _ = nnls(a, b, maxiter=50 * len(idx) + 100)
        w[k, idx] = torch.from_numpy(v)
        resid.append(float(np.mean((a_all[:, idx] @ v - z_np[:, k]) ** 2)))
    return w, {"train_logit_mse": float(np.mean(resid)) if resid else None}


def repair_ce(s: torch.Tensor, y: torch.Tensor, w0: torch.Tensor, mask: torch.Tensor, steps: int, lr: float,
              dev) -> tuple[torch.Tensor, dict]:
    """Surviving weights re-fitted to the labels: projected full-batch Adam on cross-entropy, from w0 * mask."""
    sd, yd, m = s.float().to(dev), y.to(dev), mask.float().to(dev)
    w = (w0.float().to(dev) * m).clone().requires_grad_(True)
    opt = torch.optim.Adam([w], lr=lr)
    with torch.no_grad():
        before = F.cross_entropy(sd @ w.t(), yd).item()
    for _ in range(steps):
        opt.zero_grad()
        F.cross_entropy(sd @ w.t(), yd).backward()
        opt.step()
        with torch.no_grad():
            w.clamp_(min=0).mul_(m)
    with torch.no_grad():
        after = F.cross_entropy(sd @ w.t(), yd).item()
    return w.detach().double().cpu(), {"train_ce_before": before, "train_ce_after": after}


@torch.no_grad()
def check_physical(model, feats: torch.Tensor, mask: torch.Tensor, w_eff: torch.Tensor, dev) -> float:
    """Max |logit| difference between our physical pooled scores and the real compact model
    (prune_inactive_prototypes) on a few cached feature maps."""
    m = copy.deepcopy(model)
    m.classifier.weight.mul_(mask.to(m.classifier.weight.device, m.classifier.weight.dtype))
    m.prune_inactive_prototypes(min_weight=0.0)
    f = feats.to(dev).float()
    real = m.classifier(m.pool_layer(F.softmax(m._detect_prototypes(f) / m.temperature, dim=1))).double().cpu()
    codes = torch.nonzero(mask.any(0)).flatten()
    ours = pooled_scores(model, feats, dev, codes) @ (w_eff.double() * mask)[:, codes].t()
    return float((real - ours).abs().max())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone_ckpt", required=True)
    ap.add_argument("--pq_ckpt", required=True, help="dense Stage 2 checkpoint (Phase 1, strict)")
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--masking_dir", required=True, help="the seed's Phase 2 masking folder")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--keys", nargs="+", default=DEFAULT_KEYS, help="mask keys to repair (missing ones are skipped)")
    ap.add_argument("--ridge", type=float, default=1e-4)
    ap.add_argument("--ce_steps", type=int, default=300)
    ap.add_argument("--ce_lr", type=float, default=0.01)
    args = ap.parse_args()

    if L.protocol_info()["protocol"] != "strict":
        raise SystemExit("runs under PQ_CUB_PROTOCOL=strict")
    out = Path(args.masking_dir)
    if (out / "repair.json").exists():
        raise SystemExit(f"{out / 'repair.json'} exists: test was already read for these models")
    torch.manual_seed(args.seed)

    masks, source = {}, {}
    for name in MASK_FILES:
        p = out / name
        if not p.is_file():
            continue
        saved = torch.load(p, map_location="cpu")
        if Path(saved["pq_ckpt"]).name != Path(args.pq_ckpt).name:
            raise SystemExit(f"{p} was built from a different Stage 2 checkpoint")
        for key, mask in saved["masks"].items():
            masks.setdefault(key, mask.bool())
            source.setdefault(key, name)
    keys = [k for k in args.keys if k in masks]
    for k in args.keys:
        if k not in masks:
            print(f"[REPAIR] skip {k}: no such mask in {out}", flush=True)
    if not keys:
        raise SystemExit("no masks to repair")

    dev = L.device()
    model = L.load_protoquant(args.backbone_ckpt, args.pq_ckpt, dev)
    w_eff = L.effective_weight(model).cpu().double()
    bb_sha = L.sha256(args.backbone_ckpt)
    feats, y, ids, s = {}, {}, {}, {}
    for split in ("train", "val", "test"):
        d = load_split(Path(args.cache_dir), split, bb_sha)
        feats[split], y[split], ids[split] = d["features"], d["labels"], d["img_ids"]
        s[split] = pooled_scores(model, feats[split], dev)
    z_train = s["train"] @ w_eff.t()   # the dense model's training logits: the LS target
    dense_val = acc(s["val"] @ w_eff.t(), y["val"])
    print(f"[REPAIR] dense val {dense_val:.2f}; keys {keys}", flush=True)

    rows, preds, weights, checks = [], {}, {}, {}
    for key in keys:
        method, k = key.rsplit("_K", 1)
        mask = masks[key] & (w_eff > 0)
        codes = torch.nonzero(mask.any(0)).flatten()
        checks[key] = check_physical(model, feats["val"][:8], mask, w_eff, dev)
        if checks[key] > 1e-3:
            raise SystemExit(f"physical scores differ from the compact model for {key}: {checks[key]:.2e}")
        s_phys = {sp: pooled_scores(model, feats[sp], dev, codes) for sp in ("train", "val", "test")}
        base_val = acc(s["val"] @ (w_eff * mask).t(), y["val"])
        for variant in ("phys", "LS", "CE", "phys+LS", "phys+CE"):
            phys = variant.startswith("phys")
            sc = s_phys if phys else s
            m_cols = mask[:, codes] if phys else mask
            w0 = (w_eff * mask)[:, codes] if phys else w_eff * mask
            t0 = time.time()
            fit = {}
            if variant.endswith("LS"):
                w, fit = repair_ls(sc["train"], z_train, m_cols, args.ridge)
            elif variant.endswith("CE"):
                w, fit = repair_ce(sc["train"], y["train"], w0, m_cols, args.ce_steps, args.ce_lr, dev)
            else:
                w = w0
            secs = time.time() - t0
            lv, lt = sc["val"] @ w.t(), sc["test"] @ w.t()
            name = f"{method}+{variant}_K{k}"
            row = {"method": f"{method}+{variant}", "K": int(k), "base_key": key, "mask_file": source[key],
                   "variant": variant, "physical": phys, "codebook_rows": int(len(codes) if phys else mask.shape[1]),
                   "val_top1": acc(lv, y["val"]), "test_top1": acc(lt, y["test"]),
                   "val_ce": F.cross_entropy(lv, y["val"]).item(), "one_shot_val_top1": base_val,
                   "repair_seconds": secs, **P.mask_stats(w > 0), **fit}
            rows.append(row)
            preds[name] = lt.argmax(1).numpy()
            weights[name] = {"weight": w.float(), "codes": codes if phys else None}
            print(f"[REPAIR] {name:28s} val {row['val_top1']:6.2f} (one-shot {base_val:6.2f}) "
                  f"test {row['test_top1']:6.2f}  edges {row['nonzero_edges']:5d}  codes {row['codebook_rows']:5d}  "
                  f"{secs:6.2f} s", flush=True)

    torch.save({"weights": weights, "pq_ckpt": args.pq_ckpt, "seed": args.seed}, out / "repair_weights.pt")
    np.savez_compressed(out / "repair_test_preds.npz", img_ids=ids["test"].numpy(), labels=y["test"].numpy(), **preds)
    L.write_json(out / "repair.json", {
        "seed": args.seed, "pq_ckpt": args.pq_ckpt, **L.protocol_info(), "dense_val_top1": dense_val,
        "ridge": args.ridge, "ce_steps": args.ce_steps, "ce_lr": args.ce_lr,
        "physical_check_max_logit_error": checks, "models": rows})


if __name__ == "__main__":
    main()
