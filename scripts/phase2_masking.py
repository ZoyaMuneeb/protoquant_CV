"""Phase 2, one-shot masking (no retraining) + deletion-effect control, for one strict seed.

    python scripts/phase2_masking.py --backbone_ckpt B.pth --pq_ckpt runs/phase1/strict_s40/stage2/pipnet_*.pth \
        --cache_dir runs/phase1/feature_cache --out_dir runs/phase2/strict_s40/masking --seed 40

Writes to out_dir:
  masks.pt           every mask (B1, random, P1, P1contrast, P3 at each K) + the P1 statistics
  results.json       per method and K: val and test top-1, edges, active codes (P3: classes at minimum)
  test_preds.npz     per-image test predictions of every masked model (Phase 3 paired bootstrap)
  deletion.json      deletion-effect control: 100 single-edge deletions vs each score (Spearman)

Edge masking is logical: masked edges are set to zero, every code stays in the softmax. Scores come
from the fp16 feature cache (Test 3 checked it against the image pipeline). Test is read once per
masked model; the script refuses to run again on an existing results.json.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr

import phase2_lib as P
import pq_lib as L


def pooled_scores(model, feats: torch.Tensor, dev, bs: int = 256) -> torch.Tensor:
    return torch.cat([L.scores_from_features(model, feats[i:i + bs].to(dev)).double().cpu()
                      for i in range(0, len(feats), bs)])


def load_split(cache_dir: Path, split: str, bb_sha: str) -> dict:
    d = torch.load(cache_dir / f"{split}.pt", map_location="cpu")
    if d.get("backbone_sha256") != bb_sha:
        raise SystemExit(f"{split}.pt in the feature cache was built with a different backbone")
    return d


def acc(logits: torch.Tensor, labels: torch.Tensor) -> float:
    return (logits.argmax(1) == labels).double().mean().item() * 100


def deletion_effect(s_val, y_val, w_eff, scores, n_edges: int, seed: int) -> dict:
    """Remove single edges from the dense head, one at a time, and measure the change in validation
    cross-entropy. Edges: a fixed sample spanning low, medium and high P1 scores (thirds)."""
    w = w_eff.double()
    logits = s_val @ w.t()
    base_loss = F.cross_entropy(logits, y_val).item()
    nz = torch.nonzero(w > 0)
    p1 = scores["P1"][nz[:, 0], nz[:, 1]]
    order = p1.argsort()
    thirds = torch.tensor_split(order, 3)
    g = torch.Generator().manual_seed(seed)
    per = [n_edges // 3 + (1 if i < n_edges % 3 else 0) for i in range(3)]
    picked = torch.cat([t[torch.randperm(len(t), generator=g)[:n]] for t, n in zip(thirds, per)])
    rows = []
    for idx in picked.tolist():
        k, m = nz[idx].tolist()
        new = logits.clone()
        new[:, k] -= w[k, m] * s_val[:, m]
        damage = F.cross_entropy(new, y_val).item() - base_loss
        rows.append({"class": k, "code": m, "damage": damage,
                     **{name: scores[name][k, m].item() for name in ("B1", "P1", "P1contrast")}})
    dmg = np.array([r["damage"] for r in rows])
    corr = {}
    for name in ("B1", "P1", "P1contrast"):
        rho, p = spearmanr(dmg, [r[name] for r in rows])
        corr[name] = {"spearman_rho": float(rho), "p_value": float(p)}
    return {"n_edges": len(rows), "base_val_loss": base_loss, "spearman": corr, "edges": rows}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone_ckpt", required=True)
    ap.add_argument("--pq_ckpt", required=True, help="dense Stage 2 checkpoint (Phase 1, strict)")
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--ks", type=int, nargs="+", default=[5, 10, 20])
    ap.add_argument("--deletion_edges", type=int, default=100)
    args = ap.parse_args()

    if L.protocol_info()["protocol"] != "strict":
        raise SystemExit("Phase 2 runs under PQ_CUB_PROTOCOL=strict")
    out = Path(args.out_dir)
    if (out / "results.json").exists():
        raise SystemExit(f"{out / 'results.json'} exists: test was already read for these masks")
    out.mkdir(parents=True, exist_ok=True)
    dev = L.device()
    model = L.load_protoquant(args.backbone_ckpt, args.pq_ckpt, dev)
    w_eff = L.effective_weight(model).cpu()
    cache = Path(args.cache_dir)
    bb_sha = L.sha256(args.backbone_ckpt)

    # Pooled code activations s (N x M) per split, from the cache.
    s, y, ids = {}, {}, {}
    for split in ("train", "val", "test"):
        d = load_split(cache, split, bb_sha)
        s[split], y[split], ids[split] = pooled_scores(model, d["features"], dev), d["labels"], d["img_ids"]
        print(f"[P2] {split}: scores {tuple(s[split].shape)}", flush=True)

    mu_in, mu_out = P.class_means(s["train"], y["train"], w_eff.shape[0])
    scores = P.edge_scores(w_eff, mu_in, mu_out)
    masks, p3_info = P.build_masks(w_eff, scores, args.ks, args.seed)

    w = w_eff.double()
    results = {"seed": args.seed, "pq_ckpt": args.pq_ckpt, "ks": args.ks, **L.protocol_info(), "models": []}
    dense = {"method": "dense", "K": None, "val_top1": acc(s["val"] @ w.t(), y["val"]),
             "test_top1": acc(s["test"] @ w.t(), y["test"]), **P.mask_stats(w_eff > 0)}
    results["models"].append(dense)
    preds = {"dense": (s["test"] @ w.t()).argmax(1).numpy()}
    print(f"[P2] dense: val {dense['val_top1']:.2f}  test {dense['test_top1']:.2f}  edges {dense['nonzero_edges']}")
    for key, mask in masks.items():
        method, k = key.rsplit("_K", 1)
        wm = w * mask
        lv, lt = s["val"] @ wm.t(), s["test"] @ wm.t()
        row = {"method": method, "K": int(k), "val_top1": acc(lv, y["val"]), "test_top1": acc(lt, y["test"]),
               **P.mask_stats(mask), **p3_info.get(key, {})}
        results["models"].append(row)
        preds[key] = lt.argmax(1).numpy()
        print(f"[P2] {key:15s} val {row['val_top1']:6.2f}  test {row['test_top1']:6.2f}  "
              f"edges {row['nonzero_edges']:5d}  active codes {row['active_codes']:5d}", flush=True)

    torch.save({"masks": masks, "mu_in": mu_in.float(), "mu_out": mu_out.float(),
                "pq_ckpt": args.pq_ckpt, "seed": args.seed}, out / "masks.pt")
    np.savez_compressed(out / "test_preds.npz", img_ids=ids["test"].numpy(), labels=y["test"].numpy(),
                        **{k: v for k, v in preds.items()})
    deletion = deletion_effect(s["val"], y["val"], w_eff, scores, args.deletion_edges, args.seed)
    L.write_json(out / "deletion.json", {"seed": args.seed, **deletion})
    for name, c in deletion["spearman"].items():
        print(f"[P2] deletion effect: Spearman(damage, {name}) = {c['spearman_rho']:.3f} (p={c['p_value']:.2g})")
    L.write_json(out / "results.json", results)


if __name__ == "__main__":
    main()
