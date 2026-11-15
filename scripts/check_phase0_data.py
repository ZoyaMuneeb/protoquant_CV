#!/usr/bin/env python
"""Phase 0 data checks: dataset counts, crops, and the strict/paper splits.

Usage (from the protoquant/ folder):
    python scripts/check_phase0_data.py --raw_root /path/CUB_200_2011 \
        --cropped_parent /path/cub200_cropped [--out results/phase0_data_check.json]
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "codebook_train"))
from src.datasets import cub as cubmod  # noqa: E402


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_root", required=True, type=Path)
    ap.add_argument("--cropped_parent", required=True, type=Path)
    ap.add_argument("--split_file", type=Path, default=cubmod.DEFAULT_SPLIT_FILE)
    ap.add_argument("--out", type=Path, default=REPO / "results" / "phase0_data_check.json")
    args = ap.parse_args()
    report, failures = {}, []

    def check(name, ok, detail=""):
        report.setdefault("checks", {})[name] = {"ok": bool(ok), "detail": detail}
        if not ok:
            failures.append(name)
        print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")

    root = args.cropped_parent / "CUB_200_2011"
    imgs = pd.read_csv(root / "images.txt", sep=" ", names=["img_id", "path"])
    bbox = pd.read_csv(root / "bounding_boxes.txt", sep=" ", names=["img_id", "x", "y", "w", "h"])
    tt = pd.read_csv(root / "train_test_split.txt", sep=" ", names=["img_id", "is_train"])
    lab = pd.read_csv(root / "image_class_labels.txt", sep=" ", names=["img_id", "label"])
    df = imgs.merge(bbox, on="img_id").merge(tt, on="img_id").merge(lab, on="img_id")

    check("official counts", (len(df), int(df.is_train.sum()), int((df.is_train == 0).sum())) == (11788, 5994, 5794),
          f"total={len(df)} train={int(df.is_train.sum())} test={int((df.is_train == 0).sum())}")
    check("200 classes", df.label.nunique() == 200 and df.label.min() == 1 and df.label.max() == 200)

    # Every cropped image exists and matches its (int-truncated) bounding box.
    bad, missing, small = [], 0, 0
    for r in df.itertuples():
        p = root / "images" / r.path
        if not p.is_file():
            missing += 1
            continue
        with Image.open(p) as im:
            w, h = im.size
            mode = im.mode
        if (w, h) != (int(r.w), int(r.h)) or mode != "RGB":
            bad.append((r.img_id, (w, h), (int(r.w), int(r.h)), mode))
        if min(w, h) < 32:
            small += 1
    check("all crops present", missing == 0, f"missing={missing}")
    check("crop size == bbox size, RGB", not bad, f"mismatches={len(bad)} examples={bad[:3]}")
    report["crops_smaller_than_32px"] = small
    sizes = df[["w", "h"]].astype(int)
    report["bbox_size_stats"] = {"w_median": int(sizes.w.median()), "h_median": int(sizes.h.median()),
                                 "w_min": int(sizes.w.min()), "h_min": int(sizes.h.min())}

    # Split file: deterministic, stratified, no test leakage.
    split = cubmod.load_split_file(args.split_file)
    m = df.merge(split, on="img_id", validate="1:1")
    report["split_file"] = str(args.split_file.relative_to(REPO)) if args.split_file.is_relative_to(REPO) else str(args.split_file)
    report["split_sha256"] = sha(args.split_file)
    check("split covers all images", len(m) == 11788)
    check("test == official test", ((m.split == "test") == (m.is_train == 0)).all())
    vc = m[m.split == "val"].groupby("label").size()
    check("val stratified (3 per class, 600 total)", len(vc) == 200 and vc.min() == 3 and vc.max() == 3 and vc.sum() == 600,
          f"val={vc.sum()} per-class min={vc.min()} max={vc.max()}")
    with tempfile.TemporaryDirectory() as td:
        regen = Path(td) / "s.txt"
        subprocess.run([sys.executable, str(REPO / "scripts" / "make_cub_val_split.py"), "--cub_root", str(args.raw_root),
                        "--out", str(regen)], check=True, capture_output=True)
        check("split regenerates bit-identically", sha(regen) == sha(args.split_file))

    # Loader behaviour under both protocols.
    for proto, expect in [("paper", (5994, 5794, "test")), ("strict", (5394, 600, "val"))]:
        os.environ["PQ_CUB_PROTOCOL"] = proto
        tr, va = cubmod.get_cub200(str(args.cropped_parent), None, None)
        ok = (len(tr), len(va), va.split) == expect and not set(tr.img_ids) & set(va.img_ids)
        check(f"loader protocol={proto}", ok, f"train={len(tr)} val_loader={len(va)} ({va.split})")
    os.environ["PQ_CUB_PROTOCOL"] = "strict"
    te = cubmod.get_cub200_test(str(args.cropped_parent), None)
    tr, va = cubmod.get_cub200(str(args.cropped_parent), None, None)
    check("strict: train/val/test disjoint and complete",
          not (set(tr.img_ids) & set(te.img_ids)) and not (set(va.img_ids) & set(te.img_ids))
          and len(set(tr.img_ids) | set(va.img_ids) | set(te.img_ids)) == 11788)
    x, y = va[0]
    check("strict val sample loads", isinstance(x, Image.Image) and 0 <= y < 200)

    report["passed"] = not failures
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\n{'ALL CHECKS PASSED' if not failures else 'FAILED: ' + ', '.join(failures)} -> {args.out}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
