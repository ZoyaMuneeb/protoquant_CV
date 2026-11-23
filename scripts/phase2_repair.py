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

CE step sweep (section 13, post hoc): with --ce_sweep 300 1000 3000 10000 --tag cesweep, only the CE refit is run
(keys "<method>+CEsel_K<K>" and "<method>+phys+CEsel_K<K>"), evaluated on validation at each step count; per model
the step count with the best validation top-1 is kept (ties: fewer steps) and test is read once for it. Files
repair_cesweep{.json,_test_preds.npz,_weights.pt}; the validation curve is saved in the json.

Retraining recipe on cached features (section 13, post hoc): with --cached_retrain --tag cachedrt, the masks are
retrained with the retraining recipe of phase2_retrain.py (AdamW lr 0.05, weight decay 5e-4, warm-up + cosine,
label smoothing 0.1, batch 1,024, 145 epochs, mask held fixed, unused codes removed, best epoch on validation) but
on the cached features instead of augmented images (no MixUp / CutMix / image augmentation). Two variants:
"<method>+phys+rtH_K<K>" (head only, codes frozen) and "<method>+phys+rtCH_K<K>" (codes + head). Head only vs
codes + head isolates code adaptation; codes + head vs real retraining isolates augmentation and image training.
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


def repair_ce_sweep(s: torch.Tensor, y: torch.Tensor, s_val: torch.Tensor, y_val: torch.Tensor, w0: torch.Tensor,
                    mask: torch.Tensor, checkpoints: list[int], lr: float, dev) -> tuple[torch.Tensor, dict]:
    """repair_ce run up to max(checkpoints) steps, evaluated on validation at every checkpoint. Returns the weights
    of the checkpoint with the best validation top-1 (ties: fewer steps) and the whole curve. Test is not used."""
    sd, yd, m = s.float().to(dev), y.to(dev), mask.float().to(dev)
    svd, yvd = s_val.float().to(dev), y_val.to(dev)
    w = (w0.float().to(dev) * m).clone().requires_grad_(True)
    opt = torch.optim.Adam([w], lr=lr)
    with torch.no_grad():
        before = F.cross_entropy(sd @ w.t(), yd).item()
    curve, best, step = [], None, 0
    for target in sorted(checkpoints):
        while step < target:
            opt.zero_grad()
            F.cross_entropy(sd @ w.t(), yd).backward()
            opt.step()
            with torch.no_grad():
                w.clamp_(min=0).mul_(m)
            step += 1
        with torch.no_grad():
            lv = svd @ w.t()
            point = {"steps": step, "train_ce": F.cross_entropy(sd @ w.t(), yd).item(),
                     "val_top1": acc(lv.cpu(), y_val), "val_ce": F.cross_entropy(lv, yvd).item(),
                     "nonzero_edges": int((w > 0).sum())}
        curve.append(point)
        if best is None or point["val_top1"] > best[0]["val_top1"]:
            best = (point, w.detach().double().cpu().clone())
    return best[1], {"train_ce_before": before, "ce_steps_selected": best[0]["steps"],
                     "train_ce_after": best[0]["train_ce"], "ce_curve": curve}


def cached_logits(feats: torch.Tensor, codes: torch.Tensor, w_raw: torch.Tensor, temperature: float,
                  eps: float = 1e-6) -> torch.Tensor:
    """ProtoQuantNet.forward from cached backbone features (B, C, H, W), differentiable in codes and w_raw:
    cosine similarity per position, softmax over codes / T, max over positions, ReLU(W) head."""
    x = feats.float().flatten(2).transpose(1, 2)                        # (B, HW, C)
    x = x / (x.norm(dim=-1, keepdim=True) + eps)
    c = codes / (codes.norm(dim=-1, keepdim=True) + eps)
    s = F.softmax((x @ c.t()) / temperature, dim=-1).amax(dim=1)       # (B, M)
    return s @ torch.relu(w_raw).t()


def retrain_cached(feats: dict, y: dict, codes0: torch.Tensor, w0_raw: torch.Tensor, mask: torch.Tensor,
                   train_codes: bool, temperature: float, args, seed: int, dev) -> tuple[dict, dict]:
    """The retraining recipe (phase2_retrain.py) run on the cached training features instead of augmented images:
    AdamW (lr, weight decay), warm-up + cosine schedule stepped per epoch, label smoothing 0.1 (no MixUp/CutMix:
    they need images), batch 1,024, mask re-applied after every step, best epoch on validation. Head only
    (train_codes=False) or codes + head. Returns val/test logits of the best epoch and the fit record."""
    from src.utils import SchedulerArgs, create_schedulers
    codes = torch.nn.Parameter(codes0.float().clone().to(dev), requires_grad=train_codes)
    w = torch.nn.Parameter(w0_raw.float().clone().to(dev))
    m = mask.float().to(dev)
    opt = torch.optim.AdamW([w] + ([codes] if train_codes else []), lr=args.rt_lr, weight_decay=args.rt_wd,
                            betas=(0.9, 0.999), eps=1e-8)
    sched = create_schedulers([opt], SchedulerArgs(epochs=args.rt_epochs, warmup_epochs=args.rt_warmup,
                                                   lr=args.rt_lr))[0]
    ftr, ytr = feats["train"].to(dev), y["train"].to(dev)
    g = torch.Generator().manual_seed(seed)
    bs = args.rt_batch

    def predict(split: str) -> torch.Tensor:
        with torch.no_grad():
            f = feats[split]
            return torch.cat([cached_logits(f[i:i + 256].to(dev), codes, w, temperature).cpu()
                              for i in range(0, len(f), 256)]).double()

    best, curve, t0 = None, [], time.time()
    for epoch in range(args.rt_epochs):
        perm = torch.randperm(len(ytr), generator=g)
        for i in range(0, len(perm) - bs + 1, bs):          # drop_last, as in retraining
            idx = perm[i:i + bs].to(dev)
            opt.zero_grad()
            F.cross_entropy(cached_logits(ftr[idx], codes, w, temperature), ytr[idx], label_smoothing=0.1).backward()
            opt.step()
            with torch.no_grad():
                w.mul_(m)                                   # pruned edges stay pruned
        lv = predict("val")
        val = acc(lv, y["val"])
        sched.step(epoch)
        curve.append(val)
        if best is None or val > best[0]:
            best = (val, epoch, codes.detach().clone(), w.detach().clone())
    with torch.no_grad():
        codes.copy_(best[2])
        w.copy_(best[3])
    lv, lt = predict("val"), predict("test")
    fit = {"best_epoch": best[1], "epochs": args.rt_epochs, "train_codes": train_codes, "val_curve": curve,
           "train_seconds": time.time() - t0,
           "code_change_mean_cosine": float(F.cosine_similarity(best[2].cpu(), codes0.float(), dim=1).mean())}
    return {"val": lv, "test": lt, "w_eff": torch.relu(best[3]).double().cpu(), "codes": best[2].cpu()}, fit


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
    ap.add_argument("--ce_sweep", type=int, nargs="+", default=None,
                    help="CE step counts to try, e.g. 300 1000 3000 10000: runs only the CE refit (variants CEsel and "
                         "phys+CEsel), picks the step count by validation top-1 per model, reads test once for it")
    ap.add_argument("--tag", default="", help="output name suffix, e.g. 'cesweep' -> repair_cesweep.json")
    ap.add_argument("--cached_retrain", action="store_true",
                    help="run only the retraining recipe on cached features (variants phys+rtH: head only, "
                         "phys+rtCH: codes + head), best epoch on validation, test read once")
    ap.add_argument("--rt_epochs", type=int, default=145)
    ap.add_argument("--rt_warmup", type=int, default=5)
    ap.add_argument("--rt_lr", type=float, default=0.05)
    ap.add_argument("--rt_wd", type=float, default=5e-4)
    ap.add_argument("--rt_batch", type=int, default=1024)
    args = ap.parse_args()

    if L.protocol_info()["protocol"] != "strict":
        raise SystemExit("runs under PQ_CUB_PROTOCOL=strict")
    if (args.ce_sweep or args.cached_retrain) and not args.tag:
        raise SystemExit("--ce_sweep / --cached_retrain need --tag, so the earlier repair.json is not overwritten")
    if args.ce_sweep and args.cached_retrain:
        raise SystemExit("run --ce_sweep and --cached_retrain separately")
    stem = "repair" + (f"_{args.tag}" if args.tag else "")
    variants = (("CEsel", "phys+CEsel") if args.ce_sweep else ("phys+rtH", "phys+rtCH") if args.cached_retrain
                else ("phys", "LS", "CE", "phys+LS", "phys+CE"))
    out = Path(args.masking_dir)
    if (out / f"{stem}.json").exists():
        raise SystemExit(f"{out / f'{stem}.json'} exists: test was already read for these models")
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
        for variant in variants:
            phys = variant.startswith("phys")
            sc = s_phys if phys else s
            m_cols = mask[:, codes] if phys else mask
            w0 = (w_eff * mask)[:, codes] if phys else w_eff * mask
            t0 = time.time()
            fit, lv = {}, None
            if variant.startswith("phys+rt"):
                w0_raw = (model.classifier.weight.detach().cpu().double() * mask)[:, codes]
                res, fit = retrain_cached(feats, y, model.codes.detach().cpu()[codes], w0_raw, m_cols,
                                          variant == "phys+rtCH", model.temperature, args, args.seed, dev)
                w, lv, lt = res["w_eff"], res["val"], res["test"]
            elif variant.endswith("LS"):
                w, fit = repair_ls(sc["train"], z_train, m_cols, args.ridge)
            elif variant.endswith("CE"):
                w, fit = repair_ce(sc["train"], y["train"], w0, m_cols, args.ce_steps, args.ce_lr, dev)
            elif variant.endswith("CEsel"):
                w, fit = repair_ce_sweep(sc["train"], y["train"], sc["val"], y["val"], w0, m_cols, args.ce_sweep,
                                         args.ce_lr, dev)
            else:
                w = w0
            secs = time.time() - t0
            if lv is None:
                lv, lt = sc["val"] @ w.t(), sc["test"] @ w.t()
            name = f"{method}+{variant}_K{k}"
            row = {"method": f"{method}+{variant}", "K": int(k), "base_key": key, "mask_file": source[key],
                   "variant": variant, "physical": phys, "codebook_rows": int(len(codes) if phys else mask.shape[1]),
                   "val_top1": acc(lv, y["val"]), "test_top1": acc(lt, y["test"]),
                   "val_ce": F.cross_entropy(lv, y["val"]).item(), "one_shot_val_top1": base_val,
                   "repair_seconds": secs, **P.mask_stats(w > 0), **fit}
            rows.append(row)
            preds[name] = lt.argmax(1).numpy()
            weights[name] = {"weight": w.float(), "codes": codes if phys else None,
                             **({"code_vectors": res["codes"]} if variant == "phys+rtCH" else {})}
            for p in fit.get("ce_curve", []):
                print(f"[REPAIR]   {name}: {p['steps']:6d} steps  train CE {p['train_ce']:.4f}  "
                      f"val {p['val_top1']:6.2f}  val CE {p['val_ce']:.4f}  edges {p['nonzero_edges']}", flush=True)
            print(f"[REPAIR] {name:28s} val {row['val_top1']:6.2f} (one-shot {base_val:6.2f}) "
                  f"test {row['test_top1']:6.2f}  edges {row['nonzero_edges']:5d}  codes {row['codebook_rows']:5d}  "
                  f"{secs:6.2f} s" + (f"  steps {fit['ce_steps_selected']}" if "ce_steps_selected" in fit else "")
                  + (f"  best epoch {fit['best_epoch']}" if "best_epoch" in fit else ""),
                  flush=True)

    torch.save({"weights": weights, "pq_ckpt": args.pq_ckpt, "seed": args.seed}, out / f"{stem}_weights.pt")
    np.savez_compressed(out / f"{stem}_test_preds.npz", img_ids=ids["test"].numpy(), labels=y["test"].numpy(), **preds)
    L.write_json(out / f"{stem}.json", {
        "seed": args.seed, "pq_ckpt": args.pq_ckpt, **L.protocol_info(), "dense_val_top1": dense_val,
        "ridge": args.ridge, "ce_steps": args.ce_steps, "ce_lr": args.ce_lr, "ce_sweep": args.ce_sweep,
        "selection": ("CE step count chosen per model by validation top-1 (ties: fewer steps)" if args.ce_sweep else
                      "best epoch by validation top-1 (as retraining)" if args.cached_retrain else None),
        "cached_retrain": {k: getattr(args, k) for k in ("rt_epochs", "rt_warmup", "rt_lr", "rt_wd", "rt_batch")}
        if args.cached_retrain else None,
        "physical_check_max_logit_error": checks, "models": rows})


if __name__ == "__main__":
    main()
