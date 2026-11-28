"""Inference on one test image with the report's models, plus the evidence behind each prediction.

    python scripts/predict.py --backbone_ckpt "$BB" --runs_root ../runs_bb2 --test_index 0
    python scripts/predict.py --backbone_ckpt "$BB" --runs_root ../runs_bb2 --random 3 --out results/demo

For every model the script prints the top-5 classes (softmax of the logits), the explanation size N90 (codes making
up 90 % of the predicted class's additive score, as in the report) and the three codes that contribute most, with
their share of that score and where they match in the 7 x 7 feature map. With --out it also saves one PNG per image:
the image with each model's top-3 matched patches boxed.

Models (one seed, K = 10), names as in the report:
  Dense    ProtoQuant before pruning (3,000 codes)
  B1       magnitude mask, one-shot (no recovery)
  P1       activation-aware mask, one-shot
  B1-CFR   B1 mask, codes + head trained on cached features (145 epochs)
  B1-IR    B1 mask, codes + head retrained on augmented images (145 epochs)
Models whose files are missing for the seed are skipped with a message.

Images come from the official test split, cropped to the bird's bounding box as in the classification runs, with the
same deterministic transform as evaluation. --image runs on any other image file instead (no ground-truth label).
Set PQ_DATASET=cars196 (and the Cars backbone and runs folder) for Stanford Cars.
"""

import argparse
import os
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import interpretability as I  # noqa: E402
import pq_lib as L  # noqa: E402

MODELS = {"Dense": "dense", "B1": "B1_K10|oneshot", "P1": "P1_K10|oneshot", "B1-CFR": "B1_K10|rtCH",
          "B1-IR": "B1_K10|images145"}
GRID, IMG, PATCH = 7, 224, 32
COLORS = {"Dense": "#7f7f7f", "B1": "#4779a4", "P1": "#d3923e", "B1-CFR": "#299282", "B1-IR": "#234f70"}


def class_names(root: Path, n: int) -> list[str]:
    f = root / "classes.txt"
    if not f.is_file():
        return [f"class {i + 1}" for i in range(n)]
    names = [line.split(" ", 1)[1].strip() for line in f.read_text().splitlines() if line.strip()]
    return [s.split(".", 1)[1].replace("_", " ") if s[:3].isdigit() and "." in s[:5] else s for s in names]


@torch.no_grad()
def explain(m: dict, feats: torch.Tensor, dev) -> dict:
    """Prediction, top-5, N90 and top-3 contributing codes of one model for one image (feats: 1, D, 7, 7)."""
    s, rows, cols = I.activations(m, feats, dev)
    s, rows, cols = s[0], rows[0], cols[0]
    logits = s @ m["w"].t()
    prob = F.softmax(logits, dim=0)
    pred = int(logits.argmax())
    contrib = s * m["w"][pred]                                   # v_m = W_{y m} s_m(x), the report's Eq. for N90
    srt = contrib.sort(descending=True).values
    n90 = int((srt.cumsum(0) < 0.9 * contrib.sum() - 1e-6).sum()) + 1
    top = contrib.argsort(descending=True)[:3].tolist()
    return {"pred": pred, "top5": prob.topk(5), "n90": n90, "edges": int((m["w"] > 0).sum()),
            "codes": int((m["w"] > 0).any(0).sum()),
            "top": [(int(m["ids"][j]), float(contrib[j] / contrib.sum()), int(rows[j]), int(cols[j])) for j in top]}


def denormalize(x: torch.Tensor, transform) -> torch.Tensor:
    mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    for t in getattr(transform, "transforms", []):
        if t.__class__.__name__ == "Normalize":
            mean, std = t.mean, t.std
    x = x * torch.tensor(std)[:, None, None] + torch.tensor(mean)[:, None, None]
    return x.clamp(0, 1).permute(1, 2, 0).numpy()


def save_figure(path: Path, image, results: dict, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    skip = round((IMG - PATCH) / (GRID - 1))
    fig, axes = plt.subplots(1, len(results), figsize=(2.1 * len(results), 2.6))
    for ax, (name, r) in zip([axes] if len(results) == 1 else axes, results.items()):
        ax.imshow(image)
        ax.set_xticks([]), ax.set_yticks([])
        for k, (_, _, row, col) in enumerate(r["top"]):
            ax.add_patch(mpatches.Rectangle((col * skip, row * skip), PATCH, PATCH, fill=False,
                                            ec=COLORS.get(name, "black"), lw=1.6))
            ax.text(col * skip + 2, row * skip + 10, str(k + 1), color="white", fontsize=7, weight="bold")
        ax.set_title(f"{name}: {r['name'][:22]}\n{100 * r['prob']:.0f} %, N90 = {r['n90']}", fontsize=7)
    fig.suptitle(title, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backbone_ckpt", required=True, help="fine-tuned backbone .pth of the dataset")
    ap.add_argument("--runs_root", required=True, help="holds phase1/ and phase2/ (CUB: ../runs_bb2, Cars: ../runs_cars)")
    ap.add_argument("--cub_path", default=os.environ.get("CUB_PATH"),
                    help="dataset root (default: $CUB_PATH, else ../data/cub200_cropped or ../data/cars196)")
    ap.add_argument("--seed", type=int, default=40)
    ap.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    ap.add_argument("--test_index", type=int, nargs="+", default=None, help="test image index(es)")
    ap.add_argument("--random", type=int, default=0, metavar="N", help="N random test images")
    ap.add_argument("--image", default=None, help="any image file instead of a test image")
    ap.add_argument("--out", default=None, help="folder for one PNG per image")
    args = ap.parse_args()

    if args.cub_path is None:
        args.cub_path = str(L.REPO_ROOT.parent / "data" / ("cars196" if L.DATASET == "cars196" else "cub200_cropped"))
    dev = L.device()
    t0 = time.time()
    keys = sorted({MODELS[n].split("|")[0] for n in args.models if n != "Dense"})
    found = I.collect_models(argparse.Namespace(runs_root=args.runs_root, backbone_ckpt=args.backbone_ckpt,
                                                keys=keys), args.seed)
    models = {n: found[MODELS[n]] for n in args.models if MODELS[n] in found}
    for n in args.models:
        if n not in models:
            print(f"[PREDICT] {n}: not available for seed {args.seed}, skipped")
    backbone = L.get_backbone(L.load_backbone_classifier(args.backbone_ckpt, dev)).eval()
    transform = L.eval_transform(args.cub_path)
    print(f"[PREDICT] device {dev}; loaded {', '.join(models)} (seed {args.seed}) in {time.time() - t0:.1f} s")

    if args.image:
        items = [(Path(args.image).name, Image.open(args.image).convert("RGB"), None)]
        names = class_names(Path(args.cub_path) / L.DATASET_CLASS.SUBDIR, L.NUM_CLASSES)
    else:
        ds = L.dataset(args.cub_path, "test")
        names = class_names(Path(ds.root), L.NUM_CLASSES)
        idx = args.test_index or (random.sample(range(len(ds)), args.random) if args.random else [0])
        items = []
        for i in idx:
            with Image.open(os.path.join(ds.root, "images", ds.image_paths[i])) as im:
                items.append((f"test image {i} ({ds.image_paths[i]})", im.convert("RGB"), ds.labels[i]))

    for title, im, label in items:
        x = transform(im)[None]
        t1 = time.time()
        with torch.no_grad():
            feats = backbone(x.to(dev)).float().cpu()
        results = {n: explain(m, feats, dev) for n, m in models.items()}
        ms = 1000 * (time.time() - t1)
        print(f"\n=== {title}" + (f"  |  true class: {names[label]}" if label is not None else ""))
        print(f"    backbone + {len(models)} heads: {ms:.0f} ms")
        for n, r in results.items():
            p, c = r["top5"]
            mark = "" if label is None else ("  correct" if r["pred"] == label else "  WRONG")
            r["name"], r["prob"] = names[r["pred"]], float(p[0])
            print(f"\n  {n:<7} {r['codes']:,} codes, {r['edges']:,} edges  ->  {names[r['pred']]} "
                  f"({100 * float(p[0]):.1f} %){mark}   N90 = {r['n90']}")
            print("          top-5: " + ", ".join(f"{names[int(k)]} {100 * float(v):.1f} %" for v, k in zip(p, c)))
            for k, (code, share, row, col) in enumerate(r["top"]):
                print(f"          code {code:>4} (dense id): {100 * share:4.1f} % of the class score, "
                      f"matched at row {row}, col {col}")
        if args.out:
            out = Path(args.out)
            out.mkdir(parents=True, exist_ok=True)
            f = out / (title.split(" (")[0].replace(" ", "_") + ".png")
            save_figure(f, denormalize(x[0], transform), results,
                        title + (f" - true: {names[label]}" if label is not None else ""))
            print(f"\n  saved {f}")


if __name__ == "__main__":
    main()
