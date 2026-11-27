"""Prototype figures for the report (CUB): what a code shows before and after recovery, and one explanation.

    # (a) needs only the output of scripts/interpretability.py (codes_s<seed>.npz, images_train.txt); no GPU
    python scripts/make_prototype_figures.py --save_dir ../runs_bb2/interpretability --seed 40 --out results/figures

    # (b) also explains one test image with the dense model and B1-IR (needs the checkpoints; CPU is enough)
    python scripts/make_prototype_figures.py --save_dir ../runs_bb2/interpretability --seed 40 --out results/figures \
        --explain --backbone_ckpt "$BB" --runs_root ../runs_bb2 --cub_raw ~/data/raw/CUB_200_2011

(a) prototypes.pdf: for a few codes kept by both recovery methods, the top-5 training patches of the same code in
    the dense model, after image recovery (B1 mask, 145 epochs = B1-IR) and after cached-feature recovery
    (B1 mask, codes + head = B1-CFR). Each block is labelled with the code's purity and best part. Codes are chosen
    by --codes, or else the codes with the highest dense purity among those all three models use (stated in the
    log, so the choice can be reported). Same patches, same image positions and purity as interpretability.py.
(b) explanation.pdf: one uncropped CUB test image (resized to 224 x 224 as in the purity protocol). The B1-IR model's
    three largest contributions to its predicted class, each with the patch it matched and the code's top-3
    training patches; the title gives how many codes make up 90 % of the score for B1-IR and for the dense model.
    The image is --test_index, or else the first test image both models classify correctly.
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

IMG, PATCH, GRID = 224, 32, 7
MODELS = [("dense", "Dense ProtoQuant"), ("B1_K10|images145", "B1-IR"), ("B1_K10|rtCH", "B1-CFR")]
INK, INK2 = "#0b0b0b", "#52514e"
# the report's colours: dense / reference, B1-IR, B1-CFR; boxes on the patches use the model's colour
COLOR = {"dense": "#7f7f7f", "B1_K10|images145": "#234f70", "B1_K10|rtCH": "#299282"}
BOX = COLOR["B1_K10|images145"]


def span(i: int) -> int:
    """Top-left pixel of patch row/column i on the 224 x 224 image (the authors' skip of 32, last patch clamped)."""
    skip = round((IMG - PATCH) / (GRID - 1))
    return min(i * skip, IMG - PATCH)


def load_image(path: str) -> Image.Image:
    with Image.open(path) as im:
        return im.convert("RGB").resize((IMG, IMG), Image.BILINEAR)


def crop(img: Image.Image, pos: int, context: int = 16) -> tuple[np.ndarray, tuple[int, int]]:
    """The 32 px patch at grid position pos with `context` px around it; returns the crop and the patch offset."""
    r, c = divmod(int(pos), GRID)
    y, x = span(r), span(c)
    size = PATCH + 2 * context                       # same size everywhere: the window shifts at the image edge
    y0, x0 = min(max(0, y - context), IMG - size), min(max(0, x - context), IMG - size)
    return np.asarray(img)[y0:y0 + size, x0:x0 + size], (x - x0, y - y0)


def show_patch(ax, img, pos, edge=BOX):
    a, (px, py) = crop(img, pos)
    ax.imshow(a)
    ax.add_patch(mpatches.Rectangle((px - 0.5, py - 0.5), PATCH, PATCH, fill=False, ec="white", lw=2.2))
    ax.add_patch(mpatches.Rectangle((px - 0.5, py - 0.5), PATCH, PATCH, fill=False, ec=edge, lw=1.2))
    ax.set_xticks([]), ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def per_code(npz, model: str) -> dict:
    """dense code id -> (top-10 image indices, positions, purity, part index) for one model."""
    ids = npz[f"{model}|codes"]
    pur = npz.get(f"{model}|train|purity")
    part = npz.get(f"{model}|train|part")
    return {int(c): (npz[f"{model}|train|img"][i], npz[f"{model}|train|pos"][i],
                     None if pur is None else float(pur[i]), None if part is None else int(part[i]))
            for i, c in enumerate(ids)}


def choose_codes(tables: list[dict], n: int) -> list[int]:
    common = set(tables[0]).intersection(*tables[1:])
    if not common:
        raise SystemExit("no code is used by all three models")
    return sorted(common, key=lambda c: (-(tables[0][c][2] or 0), c))[:n]


def fig_prototypes(npz, paths, parts, codes, n_patches=5):
    tables = [per_code(npz, m) for m, _ in MODELS]
    codes = codes or choose_codes(tables, 3)
    cache = {}

    def img(i):
        if i not in cache:
            cache[i] = load_image(paths[i])
        return cache[i]

    fig = plt.figure(figsize=(7.16, 0.95 * len(codes) + 0.45))
    outer = fig.add_gridspec(len(codes), len(MODELS), wspace=0.08, hspace=0.55)
    for r, code in enumerate(codes):
        dense_top = {(int(a), int(b)) for a, b in zip(*tables[0][code][:2])}
        for c, ((model, title), table) in enumerate(zip(MODELS, tables)):
            top_img, top_pos, pur, part = table[code]
            same = sum((int(a), int(b)) in dense_top for a, b in zip(top_img, top_pos))
            inner = outer[r, c].subgridspec(1, n_patches, wspace=0.05)
            for k in range(n_patches):
                ax = fig.add_subplot(inner[0, k])
                show_patch(ax, img(int(top_img[k])), int(top_pos[k]), COLOR[model])
            label = []
            if pur is not None:
                label.append(f"purity {pur:.2f}" + (f" ({parts[part]})" if parts and part is not None else ""))
            if c:
                label.append(f"retention {10 * same} %")
            ax0 = fig.add_subplot(outer[r, c], frameon=False)
            ax0.set_xticks([]), ax0.set_yticks([])
            if r == 0:
                ax0.text(0.5, 1.42, title, transform=ax0.transAxes, ha="center", fontsize=7.5, color=COLOR[model],
                         weight="bold")
            ax0.set_title(", ".join(label), fontsize=6.5, color=INK, pad=2)
            if c == 0:
                ax0.set_ylabel(f"code {code}", fontsize=6.5, color=INK2)
    print("[FIG] prototypes: codes", codes, flush=True)
    return fig


def contributions(m: dict, feats, dev):
    """Pooled scores, strongest positions and the predicted class's per-code contributions for one image."""
    from interpretability import activations
    s, rows, cols = activations(m, feats, dev)
    s, rows, cols = s[0], rows[0], cols[0]
    pred = int((s @ m["w"].t()).argmax())
    contrib = s * m["w"][pred]
    srt = contrib.sort(descending=True).values
    n90 = int((srt.cumsum(0) < 0.9 * contrib.sum() - 1e-6).sum()) + 1
    return pred, contrib, rows, cols, n90


def fig_explanation(args, npz, paths):
    import torch
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import interpretability as I
    import pq_lib as L

    dev = L.device()
    margs = argparse.Namespace(runs_root=args.runs_root, backbone_ckpt=args.backbone_ckpt, keys=["B1_K10"])
    models = I.collect_models(margs, args.seed)
    dense, b2 = models["dense"], models.get("B1_K10|images145")
    if b2 is None:
        raise SystemExit("no 145-epoch B1 checkpoint (budget/B1_K10_E145) for this seed")
    backbone = L.get_backbone(L.load_backbone_classifier(args.backbone_ckpt, dev)).eval()
    ds = I.RawCUB(Path(args.cub_raw), train=False)
    classes = [l.split(" ", 1)[1].split(".", 1)[1].replace("_", " ")
               for l in (Path(args.cub_raw) / "classes.txt").read_text().splitlines()]
    labels = {int(i): int(c) - 1 for i, c in (l.split() for l in
                                                (Path(args.cub_raw) / "image_class_labels.txt").read_text().splitlines())}

    def run(i):
        x = ds[i][0][None].to(dev)
        with torch.no_grad():
            f = backbone(x).float().cpu()
        return contributions(b2, f, dev), contributions(dense, f, dev)

    if args.test_index is not None:
        idx = args.test_index
        (pred, contrib, rows, cols, n90), dres = run(idx)
    else:
        for idx in range(len(ds)):
            (pred, contrib, rows, cols, n90), dres = run(idx)
            if pred == dres[0] == labels[ds.ids[idx]]:
                break
    b2_table = per_code(npz, "B1_K10|images145")
    image = load_image(ds.paths[idx])
    top = contrib.argsort(descending=True)[:3].tolist()
    share = (contrib[top] / contrib.sum()).tolist()
    fig = plt.figure(figsize=(7.16, 2.0))
    gs = fig.add_gridspec(1, 4, width_ratios=[1.25, 1, 1, 1], wspace=0.25)
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(image)
    ax.set_xticks([]), ax.set_yticks([])
    for k, j in enumerate(top):
        y, x = span(int(rows[j])), span(int(cols[j]))
        ax.add_patch(mpatches.Rectangle((x, y), PATCH, PATCH, fill=False, ec=BOX, lw=1.4))
        ax.text(x + 2, y + 9, str(k + 1), color="white", fontsize=7, weight="bold")
    ax.set_title(f"Predicted: {classes[pred]}", fontsize=7, color=INK)
    for k, j in enumerate(top):
        code = int(b2["ids"][j])
        inner = gs[0, k + 1].subgridspec(2, 3, hspace=0.3, wspace=0.05)
        a = fig.add_subplot(inner[0, 1])
        show_patch(a, image, int(rows[j]) * GRID + int(cols[j]))
        a.set_title(f"{k + 1}: code {code}, {100 * share[k]:.0f} % of score", fontsize=6, color=INK)
        if k == 0:
            a.text(-0.15, 0.5, "matched\npatch", transform=a.transAxes, ha="right", va="center", fontsize=5.5,
                   color=INK2)
        top_img, top_pos = b2_table[code][:2]
        for t in range(3):
            b = fig.add_subplot(inner[1, t])
            show_patch(b, load_image(paths[int(top_img[t])]), int(top_pos[t]), INK2)
            if k == 0 and t == 0:
                b.text(-0.15, 0.5, "code's top\ntraining\npatches", transform=b.transAxes, ha="right",
                       va="center", fontsize=5.5, color=INK2)
    fig.suptitle(f"N_90 (codes making up 90 % of the predicted class score): B1-IR {n90}, dense {dres[4]}", fontsize=7.5,
                 color=INK, y=1.02)
    print(f"[FIG] explanation: test image {idx} ({ds.paths[idx]}), class {classes[pred]}, B1-IR codes {top}, "
          f"n90 B1-IR {n90} / dense {dres[4]}", flush=True)
    return fig


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--save_dir", required=True, help="interpretability.py --save_dir (codes_s<seed>.npz)")
    ap.add_argument("--seed", type=int, default=40)
    ap.add_argument("--out", default="results/figures")
    ap.add_argument("--codes", type=int, nargs="*", default=None, help="dense code ids to show (default: auto)")
    ap.add_argument("--image_root", default=None, help="replace the folder of images_train.txt (another machine)")
    ap.add_argument("--explain", action="store_true")
    ap.add_argument("--backbone_ckpt")
    ap.add_argument("--runs_root")
    ap.add_argument("--cub_raw")
    ap.add_argument("--test_index", type=int, default=None)
    args = ap.parse_args()

    save, out = Path(args.save_dir), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    npz = dict(np.load(save / f"codes_s{args.seed}.npz"))
    paths = (save / "images_train.txt").read_text().split()
    if args.image_root:
        paths = [str(Path(args.image_root) / Path(p).parent.name / Path(p).name) for p in paths]
    parts_file = Path(args.cub_raw or "") / "parts" / "parts.txt"
    parts = None
    if args.cub_raw and parts_file.is_file():   # the merged-part order of interpretability.load_parts
        merged = sorted({l.split(" ", 1)[1].strip().replace("left", "right") for l in parts_file.read_text().splitlines()})
        parts = [p.replace("right ", "") for p in merged]
    plt.rcParams.update({"font.family": "serif", "pdf.fonttype": 42, "ps.fonttype": 42})

    figs = {"prototypes": fig_prototypes(npz, paths, parts, args.codes)}
    if args.explain:
        if not (args.backbone_ckpt and args.runs_root and args.cub_raw):
            raise SystemExit("--explain needs --backbone_ckpt, --runs_root and --cub_raw")
        figs["explanation"] = fig_explanation(args, npz, paths)
    for name, fig in figs.items():
        for ext in ("pdf", "png"):
            fig.savefig(out / f"{name}_s{args.seed}.{ext}", dpi=300, bbox_inches="tight")
        print(f"[FIG] wrote {out / name}_s{args.seed}.pdf/.png", flush=True)


if __name__ == "__main__":
    main()
