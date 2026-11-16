"""Cache deterministic (un-augmented) backbone features once, in fp16.

    python scripts/cache_features.py --backbone_ckpt B.pth --cub_path ... --out_dir runs/phase1/feature_cache

One file per split: <split>.pt = {features (N, 768, 7, 7) fp16, labels, img_ids, backbone sha256}.
About 0.9 GB for train + val + test. Used for importance scoring (P1, P3) and evaluation; anything that
trains uses the normal augmented image pipeline. Test 3 checks cache vs image agreement.
"""

import argparse
from pathlib import Path

import torch

import pq_lib as L


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--backbone_ckpt", required=True)
    p.add_argument("--cub_path", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=4)
    args = p.parse_args()

    dev = L.device()
    backbone = L.get_backbone(L.load_backbone_classifier(args.backbone_ckpt, dev)).eval()
    bb_sha = L.sha256(args.backbone_ckpt)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for split in args.splits:
        target = out_dir / f"{split}.pt"
        if target.exists() and torch.load(target, map_location="cpu").get("backbone_sha256") == bb_sha:
            print(f"[CACHE] {target} exists for this backbone; skipped")
            continue
        ds = L.dataset(args.cub_path, split)
        feats, labels = [], []
        with torch.no_grad():
            for x, y in L.loader(ds, args.batch_size, args.num_workers):
                feats.append(backbone(x.to(dev, non_blocking=True)).half().cpu())
                labels.append(y)
        feats = torch.cat(feats)
        torch.save({
            "features": feats, "labels": torch.cat(labels), "img_ids": torch.tensor(ds.img_ids),
            "split": split, "backbone_ckpt": args.backbone_ckpt, "backbone_sha256": bb_sha,
        }, target)
        print(f"[CACHE] {split}: {tuple(feats.shape)} fp16, {feats.numel() * 2 / 2**20:.0f} MB -> {target}")


if __name__ == "__main__":
    main()
