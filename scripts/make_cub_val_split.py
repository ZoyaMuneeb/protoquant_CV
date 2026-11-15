#!/usr/bin/env python
"""Phase 0: create the fixed, class-stratified validation split for CUB-200-2011.

Holds out 10 % of the official training images (stratified by class, fixed
seed). The official test images stay 'test'. The output has one line per image
(<img_id> <train|val|test>) and is shared by every run and every seed.

Usage:
    python scripts/make_cub_val_split.py --cub_root /path/to/CUB_200_2011 \
        --out splits/cub200_val10_seed0.txt
"""
import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cub_root", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--val_fraction", type=float, default=0.10)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    labels = pd.read_csv(args.cub_root / "image_class_labels.txt", sep=" ", names=["img_id", "label"])
    tt = pd.read_csv(args.cub_root / "train_test_split.txt", sep=" ", names=["img_id", "is_train"])
    df = labels.merge(tt, on="img_id", validate="1:1")

    train = df[df.is_train == 1]
    _, val_ids = train_test_split(
        train.img_id.values,
        test_size=args.val_fraction,
        stratify=train.label.values,
        random_state=args.seed,
        shuffle=True,
    )
    val_ids = set(int(i) for i in val_ids)
    df["split"] = ["test" if t == 0 else ("val" if i in val_ids else "train") for i, t in zip(df.img_id, df.is_train)]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        f.write(
            f"# CUB-200-2011 strict-protocol split | val_fraction={args.val_fraction} "
            f"stratified by class | sklearn train_test_split random_state={args.seed}\n"
        )
        for i, s in zip(df.img_id, df.split):
            f.write(f"{i} {s}\n")
    sha = hashlib.sha256(args.out.read_bytes()).hexdigest()

    per_class = df[df.split == "val"].groupby("label").size()
    summary = {
        "file": str(args.out),
        "sha256": sha,
        "seed": args.seed,
        "val_fraction": args.val_fraction,
        "counts": df.split.value_counts().to_dict(),
        "val_per_class_min": int(per_class.min()),
        "val_per_class_max": int(per_class.max()),
        "classes_in_val": int(per_class.size),
    }
    print(json.dumps(summary, indent=2))
    args.out.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
