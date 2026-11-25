"""Collect Phase 1 results into results/phase1_baseline.{md,json}.

Reads runs/phase1/<protocol>_s<seed>/{stage1,b0,stage2}/train.log, eval/eval_{val,test}.json,
tests/tests.json and backbone_<protocol>/eval_test.json. Missing pieces are shown as "-".
"""

import argparse
import json
import re
import statistics
from pathlib import Path

PAPER_TOP1 = 87.7      # ProtoQuant, CUB-200, ConvNeXt-T (paper)
PAPER_BACKBONE = 88.4  # black-box ConvNeXt-T (paper)
if __import__("os").environ.get("PQ_DATASET", "cub200") != "cub200":   # paper numbers are for CUB only
    PAPER_TOP1 = PAPER_BACKBONE = "–"
VAL_IMAGES = {"cub200": 600, "cars196": 815}[__import__("os").environ.get("PQ_DATASET", "cub200")]
NUM = r"([0-9]+(?:\.[0-9]+)?(?:e-?[0-9]+)?)"


def read(path: Path) -> str:
    return path.read_text(errors="replace") if path.is_file() else ""


def load(path: Path) -> dict:
    return json.loads(path.read_text()) if path.is_file() else {}


def stage1(log: str) -> dict:
    accs = [float(x) for x in re.findall(r"'Validation Top1 Accuracy': " + NUM, log)]
    dead = [float(x) for x in re.findall(r"'Validation dead_ratio': tensor\(" + NUM, log)]
    if not accs:
        return {}
    best = max(range(len(accs)), key=lambda i: accs[i])
    return {"s1_best_val": accs[best], "s1_best_epoch": best,
            "s1_dead_ratio": dead[best] if best < len(dead) else None, "s1_epochs": len(accs)}


def b0(log: str) -> dict:
    accs = [float(x) for x in re.findall(r"'val_Top1 Accuracy': " + NUM, log)]
    dead = [float(x) for x in re.findall(r"'val_dead_ratio': tensor\(" + NUM, log)]
    sizes = re.findall(r"Pruned codebook from (\d+) to (\d+) entries", log)
    if not accs:
        return {}
    return {"b0_val_before": accs[0], "b0_val_after": accs[-1],
            "b0_dead_before": dead[0] if dead else None, "b0_dead_after": dead[-1] if dead else None,
            "b0_codes": f"{sizes[0][0]}->{sizes[-1][1]}" if sizes else None}


def stage2(log: str) -> dict:
    accs = [float(x) for x in re.findall(r"'Validation Top1 Accuracy': " + NUM, log)]
    if not accs:
        return {}
    best = max(range(len(accs)), key=lambda i: accs[i])
    return {"s2_best_val": accs[best], "s2_best_epoch": best, "s2_epochs": len(accs)}


def fmt(v, nd=2):
    if v is None:
        return "-"
    return f"{v:.{nd}f}" if isinstance(v, float) else str(v)


def mean_sd(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return "-"
    if len(vals) == 1:
        return f"{vals[0]:.2f} (1 seed)"
    return f"{statistics.mean(vals):.2f} ± {statistics.stdev(vals):.2f} (n={len(vals)})"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--phase1_root", required=True)
    p.add_argument("--out_md", required=True)
    p.add_argument("--out_json", required=True)
    args = p.parse_args()
    root = Path(args.phase1_root)

    runs = []
    for d in sorted(root.glob("*_s*")):
        m = re.fullmatch(r"(strict|paper)_s(\d+)", d.name)
        if not m:
            continue
        r = {"protocol": m.group(1), "seed": int(m.group(2))}
        r.update(stage1(read(d / "stage1" / "train.log")))
        r.update(b0(read(d / "b0" / "train.log")))
        r.update(stage2(read(d / "stage2" / "train.log")))
        ev, et = load(d / "eval" / "eval_val.json"), load(d / "eval" / "eval_test.json")
        r["val_top1"], r["test_top1"], r["test_top5"] = ev.get("top1"), et.get("top1"), et.get("top5")
        for k in ("codebook_entries", "nonzero_edges", "active_codes", "codebook_mb_fp32"):
            r[k] = et.get(k, ev.get(k))
        tests = load(d / "tests" / "tests.json")
        r["tests"] = {t["test"]: t["status"] for t in tests.get("tests", [])}
        t2 = next((t for t in tests.get("tests", []) if t["test"] == "2_physical_vs_logical"), None)
        if t2:
            r["t2_changed"] = f"{t2['top_k']['changed_predictions']}/{t2['top_k']['n_images']}"
        runs.append(r)

    backbone = {pr: load(root / f"backbone_{pr}" / "eval_test.json").get("top1") for pr in ("strict", "paper")}

    md = ["# Phase 1 baseline", "",
          "Pipeline per seed: Stage 1 (4,096 codes) → B0 (usage pruning to 3,000 on the protocol's validation "
          "loader) → Stage 2 (30 epochs, best validation epoch) → test read once.", "",
          "## Final accuracy", "",
          "| Protocol | Seeds | Test top-1 (%) | Paper |", "|---|---|---|---|"]
    for pr in ("paper", "strict"):
        sel = [r for r in runs if r["protocol"] == pr]
        if sel:
            md.append(f"| {pr} | {', '.join(str(r['seed']) for r in sel)} | "
                      f"{mean_sd([r['test_top1'] for r in sel])} | {PAPER_TOP1} |")
    for pr, v in backbone.items():
        if v is not None:
            md.append(f"| backbone only ({pr}) | – | {v:.2f} | {PAPER_BACKBONE} |")
    md += ["", "The paper selects checkpoints and prunes codes on the test set. The strict baseline trains on "
           f"10 % fewer images and selects on a {VAL_IMAGES}-image validation split, so a gap to the paper is a finding, "
           "not a failed reproduction.", "", "## Per run", "",
           "| Protocol | Seed | S1 best val (ep) | S1 dead ratio | B0 codes | B0 val before→after | "
           "S2 best val (ep) | Test top-1 | Test top-5 | Nonzero edges | Active codes |",
           "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in runs:
        md.append(
            f"| {r['protocol']} | {r['seed']} | {fmt(r.get('s1_best_val'))} ({fmt(r.get('s1_best_epoch'))}) | "
            f"{fmt(r.get('s1_dead_ratio'), 3)} | {fmt(r.get('b0_codes'))} | "
            f"{fmt(r.get('b0_val_before'))}→{fmt(r.get('b0_val_after'))} | "
            f"{fmt(r.get('s2_best_val'))} ({fmt(r.get('s2_best_epoch'))}) | {fmt(r.get('test_top1'))} | "
            f"{fmt(r.get('test_top5'))} | {fmt(r.get('nonzero_edges'))} | {fmt(r.get('active_codes'))} |")
    md += ["", "S1 and B0 validation accuracy is the Stage 1 signal (quantized features through the backbone's "
           "classifier); S2 is the ProtoQuant head.", "", "## Implementation tests (strict runs)", "",
           "| Seed | 1 masked head | 2 physical vs logical (changed preds, K=10) | 3 feature cache | 4 B0 split |",
           "|---|---|---|---|---|"]
    for r in runs:
        if r["protocol"] == "strict" and r["tests"]:
            t = r["tests"]
            md.append(f"| {r['seed']} | {t.get('1_masked_head_invariance', '-')} | "
                      f"{r.get('t2_changed', '-')} | {t.get('3_feature_cache_agreement', '-')} | "
                      f"{t.get('4_b0_usage_split', '-')} |")

    Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_md).write_text("\n".join(md) + "\n")
    Path(args.out_json).write_text(json.dumps({"runs": runs, "backbone_test_top1": backbone}, indent=2))
    print("\n".join(md))
    print(f"\nwrote {args.out_md} and {args.out_json}")


if __name__ == "__main__":
    main()
