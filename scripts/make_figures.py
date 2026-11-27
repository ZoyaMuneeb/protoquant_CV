"""Report figures from the saved results (no GPU, no model). Names and colours follow the report.

    python scripts/make_figures.py            # writes results/figures/*.pdf and *.png

recovery_paths: test top-1 along the recovery paths at K = 10, CUB (improved backbone) and Stanford Cars, mean +-
sample SD over seeds 40-42. B1: one-shot -> B1 + cached H -> B1-CFR (cached CH) -> B1-IR (images, 145 epochs).
P1: one-shot -> P1 + cached CH (P1 was not trained head-only or on images). Dashed line: dense ProtoQuant.
Shows both ablations in one view: H vs CH (ablation 1) and B1 vs P1 before and after cached CH (ablation 2).

fitting_time: CUB, test top-1 against the recorded fitting time per model (log scale), the values of the report's
timing table. Cached methods: the recorded fitting timer (repair_seconds), which excludes cache creation and mask
preparation. B1-IR: mean seconds per epoch of the recorded B1 image-retraining runs x 145, an approximate duration
(about 41 min). Cars is not shown because its image retraining was not timed.

Numbers come from results/backbone2/phase2_results.{json,md} and results/cars/phase2_results.{json,md}.
"""

import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
OUT = RES / "figures"
DATASETS = {"CUB": RES / "backbone2", "Cars": RES / "cars"}
# the report's colours: B1, P1, B1-IR, B1-CFR, dense / reference
C_B1, C_P1, C_IR, C_CFR, C_REF = "#4779a4", "#d3923e", "#234f70", "#299282", "#7f7f7f"
STAGES = ["one_shot", "phys+rtH", "phys+rtCH", "image145"]
TICKS = ["One-shot\n(B1, P1)", "Cached H\n(B1 + cached H)", "Cached CH\n(B1-CFR,\nP1 + cached CH)",
         "Images, 145 ep.\n(B1-IR)"]
POINT_NAME = {("B1", "phys+rtH"): "B1 + cached H", ("B1", "phys+rtCH"): "B1-CFR", ("B1", "image145"): "B1-IR",
              ("P1", "phys+rtCH"): "P1 + cached CH", ("P1", "phys+rtH"): "P1 + cached H"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e6e6"


def mean_sd(rows, key="test_top1"):
    v = np.array([r[key] for r in rows], dtype=float)
    return v.mean(), v.std(ddof=1) if len(v) > 1 else 0.0


def from_md(md: Path):
    """Dense test top-1 and the B1 145-epoch image-retraining row (mean, SD) from the summary table."""
    text = md.read_text()
    dense = float(re.search(r"Dense head \(no pruning\): test ([\d.]+)", text).group(1))
    row = re.search(r"\| B2_K10 one-shot, 145 .*\| ([\d.]+) ± ([\d.]+) \|\s*$", text, re.M)
    return dense, (float(row.group(1)), float(row.group(2)))


def load(folder: Path):
    d = json.loads((folder / "phase2_results.json").read_text())
    dense, ir = from_md(folder / "phase2_results.md")
    acc, secs = {"B1": {}, "P1": {}}, {}
    for m in acc:
        acc[m]["one_shot"] = mean_sd(d["one_shot"][f"{m}_K10"]["rows"])
    for m, stage in (("B1", "phys+rtH"), ("B1", "phys+rtCH"), ("P1", "phys+rtCH")):
        rows = d["repair"][f"{m}+{stage}_K10"]["rows"]
        acc[m][stage] = mean_sd(rows)
        secs[(m, stage)] = float(np.mean([r["repair_seconds"] for r in rows]))
    acc["B1"]["image145"] = ir
    epoch_s = [h["seconds"] for r in d["retrained"].get("R_B1_K10", {}).get("rows", []) for h in r["history"]]
    if epoch_s:
        secs[("B1", "image145")] = float(np.mean(epoch_s)) * 145
    return dense, acc, secs


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=7)
    ax.yaxis.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def point_color(m, stage):
    return {"phys+rtCH": C_CFR, "image145": C_IR}.get(stage, C_B1) if m == "B1" else C_P1


def fig_paths(data):
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.7), sharey=True)
    for ax, (name, (dense, acc, _)) in zip(axes, data.items()):
        style(ax)
        ax.axhline(dense, color=C_REF, ls="--", lw=1)
        ax.text(-0.35, dense + 0.35, f"Dense {dense:.2f}", ha="left", va="bottom", fontsize=6.5, color=INK2)
        # label position (dx, dy in points, ha, va) per point; chosen so no label crosses a line in either panel
        place = {("B1", "one_shot"): (7, -2, "left", "top"), ("P1", "one_shot"): (-7, 1, "right", "bottom"),
                 ("B1", "phys+rtH"): (7, -2, "left", "top"), ("B1", "phys+rtCH"): (-7, 1, "right", "bottom"),
                 ("P1", "phys+rtCH"): (7, -2, "left", "top"), ("B1", "image145"): (0, -8, "center", "top")}
        for m, color, marker, off in (("B1", C_B1, "o", -0.1), ("P1", C_P1, "s", 0.1)):
            pts = [(STAGES.index(s) + off, *acc[m][s], s) for s in STAGES if s in acc[m]]
            xs, ys = np.array([p[0] for p in pts]), np.array([p[1] for p in pts])
            # P1 has no head-only or image run: a dotted line marks the skipped stage
            ax.plot(xs, ys, color=color, lw=1.4, zorder=2, label=m, ls="-" if m == "B1" else (0, (1.5, 1.5)))
            for x, y, e, s in pts:
                ax.errorbar(x, y, yerr=e, fmt=marker, ms=5.5, color=point_color(m, s), mec="white", mew=0.8,
                            elinewidth=1, capsize=0, zorder=3)
                dx, dy, ha, va = place[(m, s)]
                ax.annotate(f"{y:.2f}", (x, y), xytext=(dx, dy), textcoords="offset points", ha=ha, va=va,
                            fontsize=5.8, color=INK)
        ax.set_xticks(range(len(STAGES)), TICKS, fontsize=6.5)
        ax.set_xlim(-0.4, len(STAGES) - 0.6)
        ax.set_title(name, fontsize=8.5, color=INK)
        ax.set_ylim(68, 93)
    axes[0].set_ylabel("Test top-1 accuracy (%)", fontsize=7.5, color=INK)
    axes[0].legend(frameon=False, fontsize=6.5, loc="lower right", title="Mask", title_fontsize=6.5)
    fig.tight_layout(w_pad=1.0)
    return fig


def fig_time(dense, acc, secs):
    fig, ax = plt.subplots(figsize=(3.5, 2.35))
    style(ax)
    ax.xaxis.grid(True, color=GRID, lw=0.6)
    ax.axhline(dense, color=C_REF, ls="--", lw=1)
    ax.text(1.15, dense + 0.25, f"Dense {dense:.2f}", fontsize=6.5, color=INK2, va="bottom")
    place = {("B1", "phys+rtH"): (6, -3, "left"), ("P1", "phys+rtCH"): (7, -4, "left"),
             ("B1", "phys+rtCH"): (-7, -3, "right"), ("B1", "image145"): (0, -13, "center")}
    for (m, s), t in secs.items():
        y, e = acc[m][s]
        ax.errorbar(t, y, yerr=e, fmt="s" if m == "P1" else "o", ms=5.5, color=point_color(m, s), mec="white",
                    mew=0.8, elinewidth=1, capsize=0, zorder=3)
        dx, dy, ha = place[(m, s)]
        when = f"≈{t / 60:.0f} min" if s == "image145" else f"{t:.1f} s"
        ax.annotate(f"{POINT_NAME[(m, s)]} ({when})", (t, y), xytext=(dx, dy), textcoords="offset points", ha=ha,
                    fontsize=6, color=INK)
    ax.set_xscale("log")
    ax.set_xlim(1, 1e4)
    ax.set_xticks([1, 10, 60, 600, 3600], ["1 s", "10 s", "1 min", "10 min", "1 h"])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_ylim(78.5, 87)
    ax.set_xlabel("Recorded fitting time per model (log scale)", fontsize=7.5, color=INK)
    ax.set_ylabel("Test top-1 accuracy (%)", fontsize=7.5, color=INK)
    ax.set_title("CUB, K = 10", fontsize=8.5, color=INK)
    fig.tight_layout()
    return fig


def main() -> None:
    plt.rcParams.update({"font.family": "serif", "pdf.fonttype": 42, "ps.fonttype": 42})
    OUT.mkdir(parents=True, exist_ok=True)
    for old in ("recovery_stages", "accuracy_vs_time"):      # names used by an earlier version of this script
        for ext in ("pdf", "png"):
            (OUT / f"{old}.{ext}").unlink(missing_ok=True)
    data = {name: load(folder) for name, folder in DATASETS.items()}
    figs = {"recovery_paths": fig_paths(data), "fitting_time": fig_time(*data["CUB"])}
    for name, fig in figs.items():
        for ext in ("pdf", "png"):
            fig.savefig(OUT / f"{name}.{ext}", dpi=300, bbox_inches="tight")
        print(f"wrote {OUT / name}.pdf/.png")
    for name, (dense, acc, secs) in data.items():
        print(name, "dense", dense)
        for m in acc:
            print(" ", m, {s: tuple(round(x, 2) for x in v) for s, v in acc[m].items()})
        print("  seconds", {POINT_NAME[k]: round(v, 2) for k, v in secs.items()})


if __name__ == "__main__":
    main()
