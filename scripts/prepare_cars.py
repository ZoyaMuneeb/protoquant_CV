#!/usr/bin/env python
"""Convert Stanford Cars (196 classes) to the CUB layout used by the strict-protocol loader.

    python scripts/prepare_cars.py --src ../data/raw/stanford_cars --out ../data/cars196

Accepts any of the common distributions of the dataset (searched recursively under --src):
  * devkit: cars_train_annos.mat + cars_test_annos_withlabels.mat (+ cars_meta.mat for class names), with the
    images in folders named after their split (cars_train/, cars_test/; the two share file names);
  * all-in-one: cars_annos.mat (relative_im_path, class, test) + car_ims/;
  * Hugging Face Parquet (e.g. tanganke/stanford_cars): train-*.parquet + test-*.parquet with columns
    image ({bytes, path}) and label (0-based, class names in the file metadata). Needs pyarrow. Only these two
    splits are read; other Parquet files (robustness copies of the test set) are ignored.
Writes <out>/stanford_cars/{images/{train,test}/..., images.txt, image_class_labels.txt, train_test_split.txt,
classes.txt}: image ids 1..N (official training images first, then test, each in file-name / file order),
labels 1..196. Image bytes are copied unchanged. Full images are kept (no crop to the boxes), as in the
official Stanford Cars configuration. Checks: 196 classes in both splits, 8,144 train / 8,041 test images (a
warning if the counts differ), every file present, and a sample of images decodes.
"""

import argparse
import io
import json
import random
import shutil
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.io import loadmat

# A row: (split, output file name, label 1..196, source = image file Path or encoded image bytes).


def text(x) -> str:
    """A string from a .mat cell, however deeply loadmat nests it."""
    while isinstance(x, np.ndarray):
        x = x.flat[0]
    return str(x)


def find(src: Path, stem: str) -> Path | None:
    """<stem>.mat anywhere under src, also as re-downloaded copies like '<stem> (1).mat'."""
    hits = sorted(p for p in src.rglob(f"{stem}*.mat")
                  if p.stem == stem or p.stem.startswith(f"{stem} ("))
    return hits[0] if hits else None


def image_dir(src: Path, fnames: list[str], split: str) -> Path:
    """The folder holding every listed image of this split. Train and test images share file names
    (00001.jpg ...), so the folder must name the split and contain all the files."""
    cands = {p.parent for p in src.rglob(fnames[0]) if p.is_file()}
    cands = [d for d in cands if split in d.name.lower() or split in str(d.relative_to(src)).lower()]
    ok = [d for d in sorted(cands) if all((d / f).is_file() for f in fnames)]
    if len(ok) != 1:
        raise SystemExit(f"need exactly one folder under {src} named '*{split}*' with all {len(fnames)} {split} "
                         f"images, found {[str(d) for d in ok] or [str(d) for d in sorted(cands)]}")
    return ok[0]


def read_devkit(src: Path) -> tuple[list, list[str]]:
    tr, te = find(src, "cars_train_annos"), find(src, "cars_test_annos_withlabels")
    if tr is None or te is None:
        if tr is not None and find(src, "cars_test_annos") is not None:
            raise SystemExit("found cars_test_annos.mat, which has no test labels: add "
                             "cars_test_annos_withlabels.mat (e.g. from the Kaggle 'standford-cars-dataset-meta')")
        return [], []
    rows = []
    for path, split in ((tr, "train"), (te, "test")):
        ann = loadmat(path)["annotations"][0]
        fnames = [text(a["fname"]) for a in ann]
        folder = image_dir(src, fnames, split)
        rows += [(split, f, int(text(a["class"])), folder / f) for f, a in zip(fnames, ann)]
    meta = find(src, "cars_meta")
    names = [text(c) for c in loadmat(meta)["class_names"][0]] if meta else []
    return rows, names


def read_all_in_one(src: Path) -> tuple[list, list[str]]:
    path = find(src, "cars_annos")
    if path is None:
        return [], []
    m = loadmat(path)
    ann = m["annotations"][0]
    rows = []
    for a in ann:
        rel = text(a["relative_im_path"])
        f = path.parent / rel
        if not f.is_file():
            raise SystemExit(f"{rel} (listed in {path.name}) not found next to it; keep car_ims/ beside the file")
        rows.append(("test" if int(text(a["test"])) else "train", f.name, int(text(a["class"])), f))
    names = [text(c) for c in m["class_names"][0]] if "class_names" in m else []
    return rows, names


def read_parquet(src: Path) -> tuple[list, list[str]]:
    files = {s: sorted(src.rglob(f"{s}-*.parquet")) for s in ("train", "test")}
    if not files["train"] and not files["test"]:
        return [], []
    if not files["train"] or not files["test"]:
        raise SystemExit(f"found Parquet files for only one split under {src}: {files}")
    try:
        import pyarrow.parquet as pq
    except ImportError:
        raise SystemExit("reading the Hugging Face Parquet files needs pyarrow: pip install pyarrow") from None
    rows, names = [], []
    for split in ("train", "test"):
        k = 0
        for f in files[split]:
            table = pq.read_table(f, columns=["image", "label"])
            if not names:
                meta = json.loads((table.schema.metadata or {}).get(b"huggingface", b"{}"))
                labels = meta.get("info", {}).get("features", {}).get("label", {}).get("names") or []
                names = list(labels.values()) if isinstance(labels, dict) else list(labels)
            for img, label in zip(table.column("image").to_pylist(), table.column("label").to_pylist()):
                k += 1
                data = img["bytes"] if isinstance(img, dict) else img
                ext = Path((img.get("path") or "") if isinstance(img, dict) else "").suffix.lower()
                if ext not in (".jpg", ".jpeg", ".png"):
                    ext = {"JPEG": ".jpg", "PNG": ".png"}.get(Image.open(io.BytesIO(data)).format, ".jpg")
                rows.append((split, f"{k:05d}{ext}", int(label) + 1, data))
    return rows, names


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, type=Path, help="folder holding the downloaded Stanford Cars files")
    ap.add_argument("--out", required=True, type=Path, help="dataset root; images go to <out>/stanford_cars")
    ap.add_argument("--symlink", action="store_true", help="link image files instead of copying them")
    args = ap.parse_args()

    root = args.out / "stanford_cars"
    if (root / "images.txt").exists():
        raise SystemExit(f"{root} already prepared; delete it to rebuild")
    layout, rows, names = None, [], []
    for layout, reader in (("devkit", read_devkit), ("cars_annos.mat", read_all_in_one),
                           ("huggingface parquet", read_parquet)):
        rows, names = reader(args.src)
        if rows:
            break
    if not rows:
        raise SystemExit(f"found no devkit annotations, cars_annos.mat or train-/test-*.parquet under {args.src}")

    rows.sort(key=lambda r: (r[0] != "train", r[1]))
    missing = [str(s) for *_, s in rows if isinstance(s, Path) and not s.is_file()]
    if missing:
        raise SystemExit(f"{len(missing)} images listed in the annotations are missing, e.g. {missing[:3]}")

    (root / "images" / "train").mkdir(parents=True, exist_ok=True)
    (root / "images" / "test").mkdir(parents=True, exist_ok=True)
    with open(root / "images.txt", "w") as fi, open(root / "image_class_labels.txt", "w") as fl, \
            open(root / "train_test_split.txt", "w") as fs:
        for i, (split, name, label, source) in enumerate(rows, start=1):
            rel = f"{split}/{name}"
            dst = root / "images" / rel
            if isinstance(source, bytes):
                dst.write_bytes(source)
            elif args.symlink:
                dst.symlink_to(source.resolve())
            else:
                shutil.copy2(source, dst)
            fi.write(f"{i} {rel}\n")
            fl.write(f"{i} {label}\n")
            fs.write(f"{i} {1 if split == 'train' else 0}\n")
    labels = sorted({r[2] for r in rows})
    with open(root / "classes.txt", "w") as fc:
        for c in labels:
            fc.write(f"{c} {names[c - 1] if len(names) >= c else c}\n")

    n_train = sum(r[0] == "train" for r in rows)
    n_test = len(rows) - n_train
    classes_train = {r[2] for r in rows if r[0] == "train"}
    classes_test = {r[2] for r in rows if r[0] == "test"}
    checks = {
        "196 classes, labels 1..196": labels == list(range(1, 197)),
        "every class in train and in test": classes_train == classes_test == set(labels),
        "8,144 train / 8,041 test (official counts)": (n_train, n_test) == (8144, 8041),
    }
    bad = []
    for split, name, _, _ in random.Random(0).sample(rows, min(200, len(rows))):
        try:
            Image.open(root / "images" / split / name).convert("RGB").size
        except Exception as e:  # noqa: BLE001
            bad.append(f"{split}/{name}: {e}")
    checks["200 sampled images decode"] = not bad
    summary = {"layout": layout, "root": str(root), "train": n_train, "test": n_test, "classes": len(labels),
               "checks": {k: "PASS" if v else ("WARN" if k.startswith("8,144") else "FAIL") for k, v in checks.items()},
               "undecodable_sample": bad[:5]}
    print(json.dumps(summary, indent=2))
    (root / "prepare_check.json").write_text(json.dumps(summary, indent=2) + "\n")
    if any(v == "FAIL" for v in summary["checks"].values()):
        raise SystemExit("Stanford Cars preparation check FAILED")


if __name__ == "__main__":
    main()
