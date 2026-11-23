#!/usr/bin/env python
"""Convert Stanford Cars (196 classes) to the CUB layout used by the strict-protocol loader.

    python scripts/prepare_cars.py --src ../data/raw/stanford_cars --out ../data/cars196

Accepts either common distribution of the dataset (searched recursively under --src):
  * devkit: cars_train_annos.mat + cars_test_annos_withlabels.mat (+ cars_meta.mat for class names), with the
    images in folders that contain the listed file names (cars_train/, cars_test/);
  * all-in-one: cars_annos.mat (relative_im_path, class, test) + car_ims/.
Writes <out>/stanford_cars/{images/{train,test}/..., images.txt, image_class_labels.txt, train_test_split.txt,
classes.txt}: image ids 1..N (official training images first, then test, each sorted by file name), labels
1..196. Full images are kept (no crop to the boxes), as in the official Stanford Cars configuration. Checks:
196 classes in both splits, 8,144 train / 8,041 test images (a warning if the counts differ), every file
present, and a sample of images decodes.
"""

import argparse
import json
import random
import shutil
from pathlib import Path

from PIL import Image
from scipy.io import loadmat


def text(x) -> str:
    """A string from a .mat cell, however deeply loadmat nests it."""
    while hasattr(x, "flat") and not isinstance(x, str):
        x = x.flat[0]
    return str(x)


def find(src: Path, name: str) -> Path | None:
    hits = sorted(src.rglob(name))
    return hits[0] if hits else None


def image_dir(src: Path, fname: str) -> Path:
    hits = [p.parent for p in src.rglob(fname) if p.is_file()]
    if not hits:
        raise SystemExit(f"no image named {fname} under {src}")
    return hits[0]


def read_devkit(src: Path) -> tuple[list, list[str]]:
    tr, te = find(src, "cars_train_annos.mat"), find(src, "cars_test_annos_withlabels.mat")
    if tr is None or te is None:
        return [], []
    rows = []
    for path, split in ((tr, "train"), (te, "test")):
        ann = loadmat(path)["annotations"][0]
        folder = image_dir(src, str(ann[0]["fname"][0]))
        rows += [(split, folder / str(a["fname"][0]), int(a["class"][0][0])) for a in ann]
    meta = find(src, "cars_meta.mat")
    names = [text(c) for c in loadmat(meta)["class_names"][0]] if meta else []
    return rows, names


def read_all_in_one(src: Path) -> tuple[list, list[str]]:
    path = find(src, "cars_annos.mat")
    if path is None:
        return [], []
    m = loadmat(path)
    ann = m["annotations"][0]
    rows = []
    for a in ann:
        rel = str(a["relative_im_path"][0])
        f = path.parent / rel
        if not f.is_file():
            f = image_dir(src, Path(rel).name) / Path(rel).name
        rows.append(("test" if int(a["test"][0][0]) else "train", f, int(a["class"][0][0])))
    names = [text(c) for c in m["class_names"][0]] if "class_names" in m else []
    return rows, names


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, type=Path, help="folder holding the downloaded Stanford Cars files")
    ap.add_argument("--out", required=True, type=Path, help="dataset root; images go to <out>/stanford_cars")
    ap.add_argument("--symlink", action="store_true", help="link images instead of copying them")
    args = ap.parse_args()

    rows, names = read_devkit(args.src)
    layout = "devkit"
    if not rows:
        rows, names = read_all_in_one(args.src)
        layout = "cars_annos.mat"
    if not rows:
        raise SystemExit(f"found neither the devkit annotation files nor cars_annos.mat under {args.src}")

    root = args.out / "stanford_cars"
    if (root / "images.txt").exists():
        raise SystemExit(f"{root} already prepared; delete it to rebuild")
    rows.sort(key=lambda r: (r[0] != "train", r[1].name))
    missing = [str(f) for _, f, _ in rows if not f.is_file()]
    if missing:
        raise SystemExit(f"{len(missing)} images listed in the annotations are missing, e.g. {missing[:3]}")

    (root / "images" / "train").mkdir(parents=True, exist_ok=True)
    (root / "images" / "test").mkdir(parents=True, exist_ok=True)
    with open(root / "images.txt", "w") as fi, open(root / "image_class_labels.txt", "w") as fl, \
            open(root / "train_test_split.txt", "w") as fs:
        for i, (split, f, label) in enumerate(rows, start=1):
            rel = f"{split}/{f.name}"
            dst = root / "images" / rel
            if args.symlink:
                dst.symlink_to(f.resolve())
            else:
                shutil.copy2(f, dst)
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
    sample = random.Random(0).sample(rows, min(200, len(rows)))
    bad = []
    for _, f, _ in sample:
        try:
            Image.open(f).convert("RGB").size
        except Exception as e:  # noqa: BLE001
            bad.append(f"{f.name}: {e}")
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
