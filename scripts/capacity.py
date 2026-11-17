"""Capacity metrics for every Phase 1/2 model (protocol section 5), one table per K.

    python scripts/capacity.py --backbone_ckpt B.pth --phase1_root ../runs/phase1 --phase2_root ../runs/phase2 \
        --out_json results/capacity.json --out_md results/capacity.md

Per seed, for the dense head, every one-shot mask and every retrained model:
  codebook entries, codebook memory (MB, fp32), nonzero edges of ReLU(W);
  latency (ms per batch): head only (backbone features in, logits out) and full model (images in), median
    of --iters timed batches after --warmup, fixed batch size, 224 x 224, fp32, synthetic inputs;
  peak GPU memory (GB) of the full-model forward at that batch size;
  explanation size: per test image, the smallest number of codes whose contributions ReLU(W)_ŷm · s_m
    reach 90 % of the predicted class's total; mean and median over the test set;
  retraining cost (GPU-minutes, sum of the training epochs in retrain.json).

One-shot masks are logical (zeroed edges, full codebook), so they compute exactly what the dense model
computes: their latency and memory are the dense model's, and are not measured again. Explanation size
uses the cached test features and the models' own predictions; labels are not used, so this is not a test
accuracy read.
"""

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
import torch

import pq_lib as L
from phase2_masking import load_split, pooled_scores

IMG = 224


def latency(fn, x: torch.Tensor, dev, warmup: int, iters: int) -> tuple[float, float | None]:
    """Median ms per call and peak GPU memory (GB) while running fn(x)."""
    cuda = dev.type == "cuda"
    with torch.no_grad():
        for _ in range(warmup):
            fn(x)
        if cuda:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        times = []
        for _ in range(iters):
            if cuda:
                a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                a.record(); fn(x); b.record()
                torch.cuda.synchronize()
                times.append(a.elapsed_time(b))
            else:
                t0 = time.perf_counter(); fn(x); times.append((time.perf_counter() - t0) * 1000)
    peak = torch.cuda.max_memory_allocated() / 2**30 if cuda else None
    return float(statistics.median(times)), peak


def measure_speed(model, dev, args) -> dict:
    w = L.effective_weight(model)
    head = lambda f: L.logits_from_scores(L.scores_from_features(model, f), w)  # noqa: E731
    full = lambda x: model(x).logits  # noqa: E731
    feats = torch.randn(args.batch_size, 768, IMG // 32, IMG // 32, device=dev)
    imgs = torch.randn(args.batch_size, 3, IMG, IMG, device=dev)
    head_ms, _ = latency(head, feats, dev, args.warmup, args.iters)
    full_ms, peak = latency(full, imgs, dev, args.warmup, args.iters)
    return {"head_ms": head_ms, "full_ms": full_ms, "peak_gpu_gb": peak}


@torch.no_grad()
def explanation_size(s: torch.Tensor, w: torch.Tensor, dev, frac: float = 0.9, bs: int = 1024) -> dict:
    """s: (N, M) pooled code activations; w: (C, M) = ReLU(W) (masked). Per image: codes needed to reach
    frac of the predicted class's evidence."""
    sizes = []
    w = w.to(dev).float()
    for i in range(0, len(s), bs):
        sb = s[i:i + bs].to(dev).float()
        pred = (sb @ w.t()).argmax(1)
        contrib = w[pred] * sb                                   # (B, M)
        total = contrib.sum(1, keepdim=True)
        c = contrib.sort(1, descending=True).values.cumsum(1)
        n = (c < frac * total).sum(1) + 1
        n[total.squeeze(1) <= 0] = 0
        sizes.append(n.cpu())
    sizes = torch.cat(sizes).double()
    return {"expl_size_mean": sizes.mean().item(), "expl_size_median": sizes.median().item()}


def codebook_stats(model, mask: torch.Tensor | None = None) -> dict:
    w = L.effective_weight(model)
    if mask is not None:
        w = w * mask.to(w.device)
    codes = model.codes.detach()
    return {"codebook_entries": int(codes.shape[0]), "codebook_mb": codes.numel() * 4 / 2**20,
            "nonzero_edges": int((w > 0).sum()), "active_codes": int((w > 0).any(0).sum())}


def newest(pattern: str, root: Path) -> Path | None:
    hits = sorted(root.glob(pattern), key=lambda p: p.stat().st_mtime)
    return hits[-1] if hits else None


def run_seed(seed: int, args, dev, bb_sha: str) -> list[dict]:
    p1 = Path(args.phase1_root) / f"strict_s{seed}"
    p2 = Path(args.phase2_root) / f"strict_s{seed}"
    pq = newest("stage2/pipnet_*.pth", p1)
    if pq is None:
        print(f"[CAP] seed {seed}: no Phase 1 Stage 2 checkpoint, skipped", flush=True)
        return []
    rows = []
    test = load_split(Path(args.phase1_root) / "feature_cache", "test", bb_sha)

    dense = L.load_protoquant(args.backbone_ckpt, str(pq), dev)
    s = pooled_scores(dense, test["features"], dev)
    w = L.effective_weight(dense).cpu()
    speed = measure_speed(dense, dev, args)
    rows.append({"seed": seed, "model": "dense", "K": None, "kind": "dense", **codebook_stats(dense),
                 **speed, **explanation_size(s, w, dev), "retrain_gpu_min": 0.0})
    print(f"[CAP] s{seed} dense: {rows[-1]}", flush=True)

    masks_pt = p2 / "masking" / "masks.pt"
    if masks_pt.is_file():
        masks = torch.load(masks_pt, map_location="cpu")["masks"]
        for key, mask in masks.items():
            method, k = key.rsplit("_K", 1)
            rows.append({"seed": seed, "model": method, "K": int(k), "kind": "one-shot",
                         **codebook_stats(dense, mask), **speed, "speed_as_dense": True,
                         **explanation_size(s, w * mask, dev), "retrain_gpu_min": 0.0})
    del dense, s

    for d in sorted((p2 / "retrain").glob("*/")):
        ck = d / "pipnet_retrained.pth"
        if not (d / "DONE").exists() or not ck.is_file():
            continue
        method, k = d.name.rsplit("_K", 1)
        model = L.load_protoquant(args.backbone_ckpt, str(ck), dev)
        s = pooled_scores(model, test["features"], dev)
        info = json.loads((d / "retrain.json").read_text()) if (d / "retrain.json").is_file() else {}
        cost = sum(h.get("seconds", 0) for h in info.get("history", [])) / 60
        rows.append({"seed": seed, "model": "B2" if method == "B1" else method, "K": int(k), "kind": "retrained",
                     **codebook_stats(model), **measure_speed(model, dev, args),
                     **explanation_size(s, L.effective_weight(model).cpu(), dev), "retrain_gpu_min": cost})
        print(f"[CAP] s{seed} retrained {d.name}: codes {rows[-1]['codebook_entries']} "
              f"head {rows[-1]['head_ms']:.2f} ms full {rows[-1]['full_ms']:.2f} ms", flush=True)
        del model, s
        if dev.type == "cuda":
            torch.cuda.empty_cache()
    return rows


def fmt(vals, nd=2):
    vals = [v for v in vals if v is not None]
    if not vals:
        return "-"
    if len(vals) == 1:
        return f"{vals[0]:,.{nd}f}"
    return f"{statistics.mean(vals):,.{nd}f} ± {statistics.stdev(vals):,.{nd}f}"


def write_md(rows: list[dict], meta: dict, path: Path) -> None:
    dense = [r for r in rows if r["kind"] == "dense"]
    head_ref = {r["seed"]: r["head_ms"] for r in dense}
    full_ref = {r["seed"]: r["full_ms"] for r in dense}
    out = ["# Capacity metrics", "",
           f"Device: {meta['device']}. Latency: median of {meta['iters']} batches of {meta['batch_size']} at "
           f"{IMG}x{IMG}, fp32, after {meta['warmup']} warm-up batches; head = backbone features in, logits out; "
           "full = images in. Peak GPU memory: full-model forward at that batch size. Explanation size: codes "
           "needed for 90 % of the predicted class's evidence, test set (mean / median). One-shot masks keep the "
           "full codebook, so their latency and memory equal the dense model's (marked =dense). "
           "Speed-up = dense latency / model latency, same seed. Mean ± sd over seeds.", ""]

    def table(sel, title):
        if not sel:
            return
        out.extend([f"## {title}", "",
                    "| Model | Kind | Codebook entries | Codebook MB | Edges | Head ms | Head speed-up | Full ms "
                    "| Full speed-up | Peak GPU GB | Expl. size mean | Expl. size median | Retrain GPU-min |",
                    "|---|---|---|---|---|---|---|---|---|---|---|---|---|"])
        groups: dict = {}
        for r in sel:
            groups.setdefault((r["kind"], r["model"]), []).append(r)
        for (kind, model), rs in groups.items():
            asd = kind == "one-shot"
            hs = [head_ref[r["seed"]] / r["head_ms"] for r in rs if r["seed"] in head_ref]
            fs = [full_ref[r["seed"]] / r["full_ms"] for r in rs if r["seed"] in full_ref]
            out.append(" | ".join([
                f"| {model}", kind, fmt([r["codebook_entries"] for r in rs], 0), fmt([r["codebook_mb"] for r in rs]),
                fmt([r["nonzero_edges"] for r in rs], 0),
                "=dense" if asd else fmt([r["head_ms"] for r in rs]), "=dense" if asd else fmt(hs),
                "=dense" if asd else fmt([r["full_ms"] for r in rs]), "=dense" if asd else fmt(fs),
                "=dense" if asd else fmt([r["peak_gpu_gb"] for r in rs]),
                fmt([r["expl_size_mean"] for r in rs], 1), fmt([r["expl_size_median"] for r in rs], 1),
                fmt([r["retrain_gpu_min"] for r in rs], 1)]) + " |")
        out.append("")

    table(dense, "Dense head (no pruning)")
    for k in sorted({r["K"] for r in rows if r["K"] is not None}):
        sel = [r for r in rows if r["K"] == k]
        sel.sort(key=lambda r: (r["kind"] != "one-shot", r["model"]))
        table(sel, f"K = {k} edges per class")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(out))
    print(f"wrote {path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone_ckpt", required=True)
    ap.add_argument("--phase1_root", required=True)
    ap.add_argument("--phase2_root", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[40, 41, 42])
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--out_md", required=True)
    args = ap.parse_args()

    dev = L.device()
    torch.backends.cudnn.benchmark = False
    bb_sha = L.sha256(args.backbone_ckpt)
    meta = {"device": torch.cuda.get_device_name(0) if dev.type == "cuda" else "cpu",
            "batch_size": args.batch_size, "warmup": args.warmup, "iters": args.iters, "image_size": IMG,
            "precision": "fp32", "torch": torch.__version__,
            "tf32_matmul": torch.backends.cuda.matmul.allow_tf32, "tf32_cudnn": torch.backends.cudnn.allow_tf32}
    rows = []
    for seed in args.seeds:
        rows += run_seed(seed, args, dev, bb_sha)
    if not rows:
        raise SystemExit("no models found")
    L.write_json(args.out_json, {**meta, **L.protocol_info(), "rows": rows})
    write_md(rows, meta, Path(args.out_md))


if __name__ == "__main__":
    main()
