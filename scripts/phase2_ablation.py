"""Phase 4 ablation: P3 without the one-edge-per-class minimum (one-shot masking), for one strict seed.

    python scripts/phase2_ablation.py --backbone_ckpt B.pth --pq_ckpt runs/phase1/strict_s40/stage2/pipnet_*.pth \
        --cache_dir runs/phase1/feature_cache --masking_dir runs/phase2/strict_s40/masking --seed 40

Builds P3 masks exactly as phase2_masking.py does (same P1 scores from the training split, same budget
N = K x 200), but without forcing every class to keep one edge: classes can end with no edge and are then
never predicted. Checks that, with the minimum, the rebuilt masks equal the saved Phase 2 P3 masks
(recorded as p3_mask_reproduced; a mismatch prints a warning).

Writes to masking_dir: ablation.json (val + test top-1, edges, active codes, classes with no edge) and
ablation_test_preds.npz (per-image test predictions, for the paired bootstrap). Refuses to run again on an
existing ablation.json (test is read once per masked model).
"""

import argparse
from pathlib import Path

import numpy as np
import torch

import phase2_lib as P
import pq_lib as L
from phase2_masking import acc, load_split, pooled_scores


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone_ckpt", required=True)
    ap.add_argument("--pq_ckpt", required=True, help="dense Stage 2 checkpoint (Phase 1, strict)")
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--masking_dir", required=True, help="the seed's Phase 2 masking folder (has masks.pt)")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--ks", type=int, nargs="+", default=[5, 10, 20])
    args = ap.parse_args()

    if L.protocol_info()["protocol"] != "strict":
        raise SystemExit("Phase 4 runs under PQ_CUB_PROTOCOL=strict")
    out = Path(args.masking_dir)
    if (out / "ablation.json").exists():
        raise SystemExit(f"{out / 'ablation.json'} exists: test was already read for these masks")
    saved = torch.load(out / "masks.pt", map_location="cpu")
    if Path(saved["pq_ckpt"]).name != Path(args.pq_ckpt).name:
        raise SystemExit("masks.pt was built from a different Stage 2 checkpoint")

    dev = L.device()
    model = L.load_protoquant(args.backbone_ckpt, args.pq_ckpt, dev)
    w_eff = L.effective_weight(model).cpu()
    bb_sha = L.sha256(args.backbone_ckpt)
    s, y, ids = {}, {}, {}
    for split in ("train", "val", "test"):
        d = load_split(Path(args.cache_dir), split, bb_sha)
        s[split], y[split], ids[split] = pooled_scores(model, d["features"], dev), d["labels"], d["img_ids"]
    mu_in, mu_out = P.class_means(s["train"], y["train"], w_eff.shape[0])
    p1 = P.edge_scores(w_eff, mu_in, mu_out)["P1"]

    w = w_eff.double()
    rows, preds = [], {}
    for k in args.ks:
        with_min, _ = P.global_budget(p1, w_eff, k, min_one=True)
        ref = saved["masks"].get(f"P3_K{k}")
        reproduced = ref is not None and torch.equal(with_min, ref)
        if not reproduced:
            print(f"[ABL] WARNING: rebuilt P3_K{k} differs from the saved Phase 2 mask "
                  f"({int((with_min ^ ref).sum()) if ref is not None else 'no saved mask'} edges)", flush=True)
        mask, info = P.global_budget(p1, w_eff, k, min_one=False)
        wm = w * mask
        lt = s["test"] @ wm.t()
        key = f"P3nomin_K{k}"
        row = {"method": "P3nomin", "K": k, "val_top1": acc(s["val"] @ wm.t(), y["val"]), "test_top1": acc(lt, y["test"]),
               **P.mask_stats(mask), **info, "p3_mask_reproduced": reproduced,
               "same_mask_as_P3": bool(ref is not None and torch.equal(mask, ref))}
        rows.append(row)
        preds[key] = lt.argmax(1).numpy()
        print(f"[ABL] {key}: val {row['val_top1']:.2f} test {row['test_top1']:.2f} "
              f"classes with no edge {info['classes_with_no_edge']} same as P3 {row['same_mask_as_P3']}", flush=True)

    np.savez_compressed(out / "ablation_test_preds.npz", img_ids=ids["test"].numpy(), labels=y["test"].numpy(), **preds)
    L.write_json(out / "ablation.json", {"seed": args.seed, "pq_ckpt": args.pq_ckpt, **L.protocol_info(), "models": rows})


if __name__ == "__main__":
    main()
