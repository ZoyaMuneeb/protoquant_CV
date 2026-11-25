"""Interpretability of the dense, pruned and retrained models (docs/phase0_protocol.md, section 15). No training.

    python scripts/interpretability.py --backbone_ckpt B.pth --runs_root ../runs_bb2 --cub_raw <raw CUB_200_2011> \
        --code_key CP1-M1000_K10 --seeds 40 41 42 --out_dir results/backbone2 --save_dir ../runs_bb2/interpretability

Follows the authors' code (codebook_train/other_benchmarks/purity_benchmark), vectorised over cached feature maps:

  purity (CUB only)    util/eval_cub_csv.py: for every code some class uses (max weight > 1e-5), the 10 images with
                       the highest pooled score among the official train (or test) images, uncropped, resized to
                       224 x 224; the 32 x 32 patch at the code's strongest position; per part (left / right merged)
                       the share of those images showing the part in which it lies inside the patch; the code's
                       purity is its best part's share. Mean and std over codes.
  explanation size     per class: edges with weight > 0.1 (ProtoQuantNet.calculate_local_size), and all nonzero
                       edges; per test image (pipnet/test.py): codes with pooled score x weight > 1e-3 for the
                       predicted class, pooled scores below 0.1 set to 0.

Deviations from the authors' code, on purpose: eval_pipnet subtracts 1e-3 from every head weight at every test batch
and main.py zeroes codes never found above 0.1 in the training set; both change the model being measured, so neither
is done here.

Added by us:
  codes for 90 %       per test image, how many codes make up 90 % of the predicted class's score. Unlike the
                       thresholds above it does not depend on the scale of the weights.
  meaning kept         same code in this model and in the dense model: share of its 10 top training patches that are
                       the same (image and position; also image only), cosine between the code vectors, and the
                       change in its purity.
  random patches       purity when each code's 10 patches are drawn at random: the chance level.

--check_authors N runs the authors' own functions on N training images and compares them with this script.
"""

import argparse
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import pq_lib as L
from phase2_masking import load_split

TOPK, IMG, PATCH, MIN_WEIGHT = 10, 224, 32, 1e-5
MASK_FILES = ("masks.pt", "codeprune_masks.pt", "codeprune_matched_masks.pt")
NAMES = {"B1": "Magnitude", "P1": "Activation-aware", "random": "Random"}
VARIANTS = {"oneshot": "pruned, no retraining", "phys": "pruned, unused codes removed",
            "rtH": "cached features, head only", "rtCH": "cached features, codes + head"}


# ---- models --------------------------------------------------------------------------------------------------
def entry(codes, w_eff, ids, temperature):
    """A model as tensors: code vectors (M, D), ReLU(W) (C, M), the dense index of every code (M,)."""
    codes, w_eff, ids = codes.detach().float().cpu(), w_eff.detach().float().cpu(), ids.detach().cpu().long()
    if not (codes.shape[0] == w_eff.shape[1] == len(ids)):
        raise SystemExit(f"inconsistent model: {codes.shape[0]} codes, {w_eff.shape[1]} columns, {len(ids)} ids")
    return {"codes": codes, "w": w_eff, "ids": ids, "T": float(temperature)}


def newest(folder: Path, pattern: str):
    files = sorted(folder.glob(pattern), key=lambda p: p.stat().st_mtime)
    return files[-1] if files else None


def label(key: str, variant: str) -> str:
    base = key.rsplit("_K", 1)[0]
    name = NAMES.get(base)
    if name is None:
        name = ("Code-level magnitude" if base.startswith("CB1") else
                "Code-level activation-aware" if base.startswith("CP1") else base)
    if variant.startswith("images"):
        return f"{name}: retrained on images, {variant[6:]} epochs"
    return f"{name}: {VARIANTS[variant]}"


def collect_models(args, seed: int) -> dict:
    """{name: model} for one seed: the dense model first, then every variant of every key that exists."""
    root = Path(args.runs_root)
    pq = newest(root / "phase1" / f"strict_s{seed}" / "stage2", "pipnet_*.pth")
    if pq is None:
        raise SystemExit(f"no dense Stage 2 checkpoint for seed {seed} in {root / 'phase1'}")
    model = L.load_protoquant(args.backbone_ckpt, str(pq), torch.device("cpu"))
    codes, w, T = model.codes.detach(), L.effective_weight(model), model.temperature
    models = {"dense": entry(codes, w, torch.arange(len(codes)), T)}

    base = root / "phase2" / f"strict_s{seed}"
    masks = {}
    for f in MASK_FILES:
        p = base / "masking" / f
        if p.is_file():
            saved = torch.load(p, map_location="cpu")
            if Path(saved["pq_ckpt"]).name != pq.name:
                raise SystemExit(f"{p} was built from a different Stage 2 checkpoint")
            for k, m in saved["masks"].items():
                masks.setdefault(k, m.bool())
    rt_file = base / "masking" / "repair_cachedrt_weights.pt"
    rt = {}
    if rt_file.is_file():
        saved = torch.load(rt_file, map_location="cpu")
        if Path(saved["pq_ckpt"]).name != pq.name:
            raise SystemExit(f"{rt_file} was built from a different Stage 2 checkpoint")
        rt = saved["weights"]

    for key in args.keys:
        if key not in masks:
            print(f"[INTERP] seed {seed}: skip {key} (no mask)", flush=True)
            continue
        mask = masks[key]
        method, k = key.rsplit("_K", 1)
        ids = torch.nonzero((mask & (w > 0)).any(0)).flatten()
        models[f"{key}|oneshot"] = entry(codes, w * mask, torch.arange(len(codes)), T)
        models[f"{key}|phys"] = entry(codes[ids], (w * mask)[:, ids], ids, T)
        for v in ("rtH", "rtCH"):
            r = rt.get(f"{method}+phys+{v}_K{k}")
            if r is None:
                print(f"[INTERP] seed {seed}: skip {key} {v} (not in {rt_file.name})", flush=True)
                continue
            models[f"{key}|{v}"] = entry(r["code_vectors"] if v == "rtCH" else codes[r["codes"]], r["weight"],
                                         r["codes"], T)
        # budget/ holds the 145-epoch runs, retrain/ the 30-epoch ones; the longest run of each key is used.
        runs = sorted((base / "budget").glob(f"{key}_E*/pipnet_retrained.pth"),
                      key=lambda p: int(p.parent.name.rsplit("_E", 1)[1]))
        path = runs[-1] if runs else base / "retrain" / key / "pipnet_retrained.pth"
        if not path.is_file():
            print(f"[INTERP] seed {seed}: skip {key} retrained on images (no checkpoint)", flush=True)
            continue
        ck = torch.load(path, map_location="cpu")
        epochs = path.parent.name.rsplit("_E", 1)[1] if runs else "30"
        rids = torch.nonzero(mask.any(0)).flatten()
        models[f"{key}|images{epochs}"] = entry(ck["model"]["codes"], torch.relu(ck["model"]["classifier.weight"]),
                                                rids, T)
    return models


# ---- feature maps --------------------------------------------------------------------------------------------
class RawCUB(torch.utils.data.Dataset):
    """Uncropped CUB images with the authors' projection transform, in ImageFolder order."""

    def __init__(self, root: Path, train: bool, limit: int = 0):
        from torchvision import transforms as T
        split = dict(l.split() for l in (root / "train_test_split.txt").read_text().splitlines())
        rows = [l.split() for l in (root / "images.txt").read_text().splitlines()]
        rows = sorted((p, i) for i, p in rows if (split[i] == "1") == train)
        rows = sorted(rows, key=lambda r: tuple(r[0].split("/")))
        if limit:
            rows = rows[:: max(1, len(rows) // limit)][:limit]
        self.paths, self.ids = [str(root / "images" / r[0]) for r in rows], [int(r[1]) for r in rows]
        self.tf = T.Compose([T.Resize((IMG, IMG)), T.ToTensor(),
                             T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        with Image.open(self.paths[i]) as im:
            return self.tf(im.convert("RGB")), im.size[0], im.size[1]   # x, width, height


@torch.no_grad()
def extract(backbone, ds, dev, bs: int, workers: int):
    feats, widths, heights = [], [], []
    for x, w, h in L.loader(ds, bs, workers):
        feats.append(backbone(x.to(dev, non_blocking=True)).float().cpu())
        widths.append(w)
        heights.append(h)
    return torch.cat(feats), torch.cat(widths).double(), torch.cat(heights).double()


@torch.no_grad()
def activations(m: dict, feats: torch.Tensor, dev, bs: int = 64):
    """Pooled scores (N, M) and the row / column of each code's strongest position (as get_topk_cub)."""
    c = m["codes"].to(dev)
    c = c / (c.norm(dim=-1, keepdim=True) + 1e-6)
    pooled, rows, cols = [], [], []
    for i in range(0, len(feats), bs):
        z = feats[i:i + bs].to(dev).float().permute(0, 2, 3, 1)         # B, H, W, D
        z = z / (z.norm(dim=-1, keepdim=True) + 1e-6)
        p = F.softmax((z @ c.t()) / m["T"], dim=-1)                      # B, H, W, M
        H = p.shape[1]
        best, idx = p.permute(0, 2, 1, 3).flatten(1, 2).max(1)           # column first, as the authors
        pooled.append(best.cpu())
        cols.append((idx // H).to(torch.uint8).cpu())
        rows.append((idx % H).to(torch.uint8).cpu())
    return torch.cat(pooled), torch.cat(rows), torch.cat(cols)


def top_patches(m: dict, feats, dev):
    """(used code columns, top-10 image indices (U, 10), rows (U, 10), cols (U, 10))."""
    used = torch.nonzero(m["w"].max(0).values > MIN_WEIGHT).flatten()
    pooled, r, c = activations(m, feats, dev)
    pooled, r, c = pooled[:, used], r[:, used], c[:, used]
    top = torch.sort(pooled, dim=0, descending=True, stable=True).indices[:TOPK]
    return used, top.t(), r.gather(0, top).t().long(), c.gather(0, top).t().long()


# ---- purity --------------------------------------------------------------------------------------------------
def load_parts(root: Path, ids: list[int]):
    """Part positions x, y (N, P, S) and visibility: left / right parts are two slots of one merged part."""
    names = dict(l.split(" ", 1) for l in (root / "parts" / "parts.txt").read_text().splitlines())
    merged = sorted({n.strip().replace("left", "right") for n in names.values()})
    slot = {i: (merged.index(n.strip().replace("left", "right")), int("left" in n)) for i, n in names.items()}
    where = {i: n for n, i in enumerate(ids)}
    xy, vis = torch.zeros(len(ids), len(merged), 2, 2).double(), torch.zeros(len(ids), len(merged), 2).bool()
    for line in (root / "parts" / "part_locs.txt").read_text().splitlines():
        img, part, x, y, v = line.split()
        if v == "1" and int(img) in where:
            p, s = slot[part]
            xy[where[int(img)], p, s] = torch.tensor([float(x), float(y)]).double()
            vis[where[int(img)], p, s] = True
    return xy, vis, merged


def purity(img, row, col, grid: int, widths, heights, xy, vis):
    """Purity of each code (U,) and its best part, from its patches: the authors' eval_prototypes_cub_parts_csv."""
    skip = round((IMG - PATCH) / (grid - 1))

    def span(i):
        lo = i.double() * skip
        hi = torch.clamp(lo + PATCH, max=IMG)
        hi = torch.where(i == grid - 1, torch.full_like(hi, IMG), hi)
        return torch.where(hi == IMG, torch.full_like(lo, IMG - PATCH), lo), hi

    (y0, y1), (x0, x1) = span(row), span(col)
    sy, sx = heights[img] / IMG, widths[img] / IMG
    x, y = xy[img][..., 0], xy[img][..., 1]                              # U, 10, P, 2
    e = lambda t: t[..., None, None]
    inside = (y >= e(sy * y0)) & (y <= e(sy * y1)) & (x >= e(sx * x0)) & (x <= e(sx * x1)) & vis[img]
    seen, hit = vis[img].any(-1), inside.any(-1)                         # U, 10, P (left or right)
    n = seen.sum(1).double()
    share = torch.where(n > 0, hit.sum(1).double() / n.clamp(min=1), torch.zeros_like(n))
    return share.max(1)


# ---- explanation size ----------------------------------------------------------------------------------------
@torch.no_grad()
def sizes(m: dict, feats, labels, dev) -> dict:
    s = activations(m, feats, dev)[0]
    w = m["w"]
    top1 = ((s @ w.t()).argmax(1) == labels).double().mean().item() * 100
    s_inf = torch.where(s < 0.1, torch.zeros_like(s), s)                 # the authors' inference rule
    pred_inf = (s_inf @ w.t()).argmax(1)
    per_image = ((s_inf * w[pred_inf]) > 1e-3).sum(1).double()
    per_image_all = ((s_inf * w.max(0).values) > 1e-3).sum(1).double()
    contrib = s * w[(s @ w.t()).argmax(1)]
    srt = contrib.sort(1, descending=True).values
    n90 = ((srt.cumsum(1) < 0.9 * contrib.sum(1, keepdim=True) - 1e-6).sum(1) + 1).double()
    return {"test_top1": top1, "edges_per_class": (w > 0).sum(1).double().mean().item(),
            "local_size_w0.1": (w > 0.1).sum(1).double().mean().item(),
            "active_codes": int((w > 0).any(0).sum()), "active_codes_w0.1": int((w > 0.1).any(0).sum()),
            "codes_per_image_pred": per_image.mean().item(), "codes_per_image_all": per_image_all.mean().item(),
            "codes_for_90pct": n90.mean().item()}


# ---- comparison with the authors' functions ------------------------------------------------------------------
def check_authors(m: dict, backbone, root: Path, n_images: int, n_codes: int, dev) -> None:
    sys.path.insert(0, str(L.CODE_DIR))
    from other_benchmarks.purity_benchmark.util import eval_cub_csv as A
    ds = RawCUB(root, train=True, limit=n_images)
    sub = {**m, "codes": m["codes"][:n_codes], "w": m["w"][:, :n_codes], "ids": m["ids"][:n_codes]}
    feats, widths, heights = extract(backbone, ds, dev, 16, 0)

    c = sub["codes"].to(dev)

    class Net:
        def __init__(self):
            self.module = SimpleNamespace(_classification=SimpleNamespace(weight=sub["w"]), _num_prototypes=n_codes)

        def eval(self):
            return self

        def __call__(self, x):
            z = backbone(x.to(dev)).float()
            z = z / (z.norm(dim=1, keepdim=True) + 1e-6)
            u = c / (c.norm(dim=-1, keepdim=True) + 1e-6)
            p = F.softmax(torch.einsum("bdhw,md->bmhw", z, u) / sub["T"], dim=1)
            return p, p.amax((2, 3)), None

    class Loader:
        dataset = SimpleNamespace(imgs=[(p, 0) for p in ds.paths])

        def __len__(self):
            return len(ds)

        def __iter__(self):
            return ((ds[i][0][None], torch.zeros(1)) for i in range(len(ds)))

    class Items:
        imgs = Loader.dataset.imgs

        def __getitem__(self, i):
            return ds[i][0], 0

    Loader.dataset = Items()
    with tempfile.TemporaryDirectory() as tmp:
        a = SimpleNamespace(image_size=IMG, wshape=feats.shape[-1], log_dir=tmp)
        csv = A.get_topk_cub(Net(), Loader(), TOPK, 0, dev, a)
        theirs = A.eval_prototypes_cub_parts_csv(csv, str(root / "parts/part_locs.txt"), str(root / "parts/parts.txt"),
                                                 str(root / "images.txt"), 0, a,
                                                 SimpleNamespace(log_values=lambda *x: None))
    xy, vis, _ = load_parts(root, ds.ids)
    _, img, r, col = top_patches(sub, feats, dev)
    ours, _ = purity(img, r, col, feats.shape[-1], widths, heights, xy, vis)
    got = (ours.mean().item(), ours.std(unbiased=False).item(), int((ours > 0.5).sum()), len(ours))
    ref = (theirs["purity_mean_purest_part"], theirs["purity_std_purest_part"],
           theirs["purity_num_part_related_gt_0.5"], theirs["purity_num_prototypes_in_csv"])
    print(f"[CHECK] mean / std / #>0.5 / #codes: authors {ref} | this script {got}")
    # batching changes float scores slightly, which can swap near-tied images: 0.005 allows a few such swaps
    if abs(got[0] - ref[0]) > 0.005 or abs(got[1] - ref[1]) > 0.005 or got[3] != ref[3]:
        raise SystemExit("[CHECK] FAIL: this script disagrees with the authors' purity code")
    print("[CHECK] PASS", flush=True)


# ---- report --------------------------------------------------------------------------------------------------
def ms(vals, d=2) -> str:
    vals = [v for v in vals if v is not None]
    if not vals:
        return "–"
    return f"{np.mean(vals):.{d}f}" + (f" ± {np.std(vals, ddof=1):.{d}f}" if len(vals) > 1 else "")


def write_md(path: Path, rows: list[dict], cub: bool, seeds: list[int]) -> None:
    names = list(dict.fromkeys(r["model"] for r in rows))
    by = {n: [r for r in rows if r["model"] == n] for n in names}
    title = lambda n: {"dense": "Dense", "random patches": "Chance (random patches)"}.get(n) or label(*n.split("|"))
    g = lambda n, k: [r.get(k) for r in by[n]]
    md = ["# Interpretability", "", f"Dataset {L.DATASET}, seeds {', '.join(map(str, seeds))}, mean ± std over seeds "
          "(n = seeds with the model). Definitions: `scripts/interpretability.py`. No model is trained or selected "
          "here.", ""]
    if cub:
        md += ["## Part purity", "", "Share of a code's top-10 patches showing the same part; higher is better. "
               "Std over codes in brackets. \"Purity change\" compares the same code with the dense model (train "
               "images), so it is not affected by which codes were kept.", "",
               "| Model | n | Codes | Purity train | Purity test | Codes with purity > 0.5 (train) | "
               "Purity change, same code |", "|---|---|---|---|---|---|---|"]
        for n in names:
            md.append(f"| {title(n)} | {len(by[n])} | {ms(g(n, 'used_codes'), 0)} | {ms(g(n, 'purity_train'), 3)} "
                      f"[{np.mean(g(n, 'purity_train_std')):.2f}] | {ms(g(n, 'purity_test'), 3)} "
                      f"[{np.mean(g(n, 'purity_test_std')):.2f}] | {ms(g(n, 'share_pure_train'), 1)} % | "
                      f"{ms(g(n, 'purity_change'), 3)} |")
        md.append("")
    md += ["## Explanation size", "", "Per class: nonzero edges, and edges with weight > 0.1 (the repository's local "
           "size). Per test image: codes contributing > 0.001 to the predicted class (the repository's rule), and "
           "codes that make up 90 % of its score (independent of the weight scale).", "",
           "| Model | n | Edges / class | Edges > 0.1 / class | Codes used | Codes / image (predicted class) | "
           "Codes for 90 % of the score | Test top-1 (check) |", "|---|---|---|---|---|---|---|---|"]
    for n in names:
        if n == "random patches":
            continue
        md.append(f"| {title(n)} | {len(by[n])} | {ms(g(n, 'edges_per_class'), 1)} | {ms(g(n, 'local_size_w0.1'), 1)} "
                  f"| {ms(g(n, 'active_codes'), 0)} | {ms(g(n, 'codes_per_image_pred'), 2)} | "
                  f"{ms(g(n, 'codes_for_90pct'), 2)} | {ms(g(n, 'test_top1'))} |")
    md += ["", "## Meaning kept (same code, compared with the dense model)", "",
           "Top-10 training patches of each used code: share that are the same image and position, and the same "
           "image. Cosine between the code vectors.", "",
           "| Model | n | Same patches (%) | Same images (%) | Code cosine |", "|---|---|---|---|---|"]
    for n in names:
        if n in ("dense", "random patches"):
            continue
        md.append(f"| {title(n)} | {len(by[n])} | {ms(g(n, 'same_patches'), 1)} | {ms(g(n, 'same_images'), 1)} | "
                  f"{ms(g(n, 'code_cosine'), 3)} |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(md) + "\n")
    print(f"wrote {path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone_ckpt", required=True)
    ap.add_argument("--runs_root", required=True, help="holds phase1/ and phase2/")
    ap.add_argument("--cub_raw", default=None, help="uncropped CUB_200_2011 with parts/ (CUB only)")
    ap.add_argument("--code_key", default="CP1-M1000_K10")
    ap.add_argument("--keys", nargs="+", default=None)
    ap.add_argument("--seeds", type=int, nargs="+", default=[40, 41, 42])
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--save_dir", required=True, help="per-code purity and top patches (for figures)")
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--limit_images", type=int, default=0, help="smoke tests only: images per split")
    ap.add_argument("--check_authors", type=int, default=0, metavar="N", help="only compare with the authors' code")
    ap.add_argument("--check_codes", type=int, default=200)
    args = ap.parse_args()
    if args.keys is None:
        args.keys = ["B1_K10", "P1_K10", "CB1-eqCP1M1000_K10", args.code_key, "random_K10"]
    args.keys = list(dict.fromkeys(args.keys))
    if L.protocol_info()["protocol"] != "strict":
        raise SystemExit("runs under PQ_CUB_PROTOCOL=strict")
    cub = L.DATASET == "cub200"
    dev = L.device()
    bb_sha = L.sha256(args.backbone_ckpt)
    backbone = L.get_backbone(L.load_backbone_classifier(args.backbone_ckpt, dev)).eval()
    if cub:
        if not args.cub_raw or not (Path(args.cub_raw) / "parts" / "part_locs.txt").is_file():
            raise SystemExit(f"CUB purity needs --cub_raw with parts/part_locs.txt, got {args.cub_raw}")
        root = Path(args.cub_raw)
    if args.check_authors:
        if not cub:
            raise SystemExit("--check_authors is for CUB")
        check_authors(collect_models(args, args.seeds[0])["dense"], backbone, root, args.check_authors,
                      args.check_codes, dev)
        return

    cache = Path(args.runs_root) / "phase1" / "feature_cache"
    test = load_split(cache, "test", bb_sha)
    proj = {}   # split -> (feats, widths, heights, xy, vis)
    if cub:
        for split in ("train", "test"):
            t0 = time.time()
            ds = RawCUB(root, split == "train", args.limit_images)
            f, w, h = extract(backbone, ds, dev, args.batch_size, args.num_workers)
            xy, vis, parts = load_parts(root, ds.ids)
            proj[split] = (f, w, h, xy, vis)
            print(f"[INTERP] {split}: {len(ds)} uncropped images, {tuple(f.shape)} in {time.time() - t0:.0f} s",
                  flush=True)
    else:   # no part labels: the official training images, for the top-patch comparison only
        proj["train"] = (torch.cat([load_split(cache, s, bb_sha)["features"] for s in ("train", "val")]),)

    save = Path(args.save_dir)
    save.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in args.seeds:
        ref, per_code = None, {}
        models = collect_models(args, seed)
        for name, m in models.items():
            t0 = time.time()
            row = {"seed": seed, "model": name, "codebook_rows": len(m["codes"])}
            row.update(sizes(m, test["features"], test["labels"], dev))
            for split, data in proj.items():
                used, img, r, c = top_patches(m, data[0], dev)
                ids = m["ids"][used]
                if cub:
                    pur, part = purity(img, r, c, data[0].shape[-1], *data[1:])
                    row.update({f"purity_{split}": pur.mean().item(),
                                f"purity_{split}_std": pur.std(unbiased=False).item(),
                                f"share_pure_{split}": 100.0 * (pur > 0.5).double().mean().item()})
                    per_code[f"{name}|{split}|purity"] = pur.numpy()
                    per_code[f"{name}|{split}|part"] = part.numpy()
                    if name == "dense":
                        g = torch.Generator().manual_seed(seed)
                        H = data[0].shape[-1]
                        ri = torch.randint(len(data[0]), img.shape, generator=g)
                        rp = purity(ri, torch.randint(H, img.shape, generator=g),
                                    torch.randint(H, img.shape, generator=g), H, *data[1:])[0]
                        chance = {} if split == "train" else chance
                        chance.update({f"purity_{split}": rp.mean().item(),
                                       f"purity_{split}_std": rp.std(unbiased=False).item(),
                                       f"share_pure_{split}": 100.0 * (rp > 0.5).double().mean().item()})
                if split != "train":
                    continue
                row["used_codes"] = len(used)
                per_code.update({f"{name}|codes": ids.numpy(), f"{name}|train|img": img.numpy().astype(np.int32),
                                 f"{name}|train|pos": (r * data[0].shape[-1] + c).numpy().astype(np.uint8)})
                if name == "dense":   # every dense code, so each surviving code has a reference
                    all_used = dict(m, w=torch.ones_like(m["w"]))
                    _, di, dr, dc = top_patches(all_used, data[0], dev)
                    ref = {"img": di, "pos": dr * data[0].shape[-1] + dc, "codes": m["codes"]}
                    if cub:
                        ref["purity"] = purity(di, dr, dc, data[0].shape[-1], *data[1:])[0]
                else:
                    a_img, a_pos = ref["img"][ids], ref["pos"][ids]
                    P = 1 + data[0].shape[-1] ** 2
                    ka, kb = a_img * P + a_pos, img * P + r * data[0].shape[-1] + c
                    row["same_patches"] = 100.0 * (kb[:, :, None] == ka[:, None]).any(2).double().mean().item()
                    row["same_images"] = 100.0 * (img[:, :, None] == a_img[:, None]).any(2).double().mean().item()
                    row["code_cosine"] = F.cosine_similarity(m["codes"][used], ref["codes"][ids]).mean().item()
                    if cub:
                        row["purity_change"] = (pur - ref["purity"][ids]).mean().item()
            rows.append(row)
            print(f"[INTERP] seed {seed} {name:34s} codes {row['used_codes']:5d}  "
                  + (f"purity {row['purity_train']:.3f}/{row['purity_test']:.3f}  " if cub else "")
                  + f"codes/img {row['codes_per_image_pred']:6.2f}  same patches {row.get('same_patches', 100):5.1f} %  "
                  f"test {row['test_top1']:.2f}  {time.time() - t0:.1f} s", flush=True)
        if cub:
            rows.append({"seed": seed, "model": "random patches", "used_codes": None, **chance})
        np.savez_compressed(save / f"codes_s{seed}.npz", **per_code)
    if cub:
        for split in ("train", "test"):
            ds = RawCUB(root, split == "train", args.limit_images)
            (save / f"images_{split}.txt").write_text("\n".join(ds.paths) + "\n")
    out = Path(args.out_dir)
    L.write_json(out / "interpretability.json", {
        **L.protocol_info(), "backbone_sha256": bb_sha, "limit_images": args.limit_images,
        "parts": parts if cub else None, "rows": rows})
    write_md(out / "interpretability.md", rows, cub, args.seeds)


if __name__ == "__main__":
    main()
