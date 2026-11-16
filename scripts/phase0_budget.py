#!/usr/bin/env python
"""Turn Phase 0 timing runs into seconds/epoch, peak memory and a GPU-hour budget.

Per-epoch time = gap between consecutive "Epoch" log lines (timestamped by common.sh);
the first epoch is dropped as warm-up when more than one is available. Peak memory is
the maximum nvidia-smi memory.used sample during the run (includes CUDA context).
"""
import argparse
import csv
import re
import statistics
from pathlib import Path

EPOCH_RE = re.compile(r"(Epoch: \d+|--- Epoch \d+/\d+ ---)")

# Run list for the budget (counts from the methodology; edit if the plan changes).
# (label, stage, count, epochs)
RUN_LIST = [
    ("Backbone fine-tune (strict, once)", "backbone", 1, 30),
    ("Paper-protocol sanity: Stage 1 @4096 (1 seed)", "stage1_K4096", 1, 30),
    ("Paper-protocol sanity: Stage 2 (1 seed)", "stage2_K4096", 1, 30),
    ("Strict baseline: Stage 1 @4096 (3 seeds)", "stage1_K4096", 3, 30),
    ("Strict baseline: Stage 2 after B0 (3 seeds)", "stage2_K4096", 3, 30),
    ("B2 retrain, K=5/10/20 x 3 seeds", "stage2_K4096", 9, 30),
    ("Retrain at K=10: P1, P3, random, P1-contrast x 3 seeds", "stage2_K4096", 12, 30),
    ("One-shot vs iterative, best method K=10 (~5 rounds x 3 seeds)", "stage2_K4096", 15, 30),
    ("Optional: Stage 1 @8192 (codebook-size track, 1 seed)", "stage1_K8192", 1, 30),
]
OVERHEAD = 1.2  # validation, checkpointing, data loading start-up, reruns


def parse_run(d: Path) -> dict:
    log = d / "train.log"
    m = re.search(r"_bs(\d+)$", d.name)
    row = {"run": d.name, "status": "missing", "stage": re.sub(r"_bs\d+$", "", d.name),
           "batch_size": int(m.group(1)) if m else None}
    if not log.is_file():
        return row
    text = log.read_text(errors="ignore")
    if re.search(r"out of memory|OutOfMemoryError|signal SIGKILL", text, re.I):
        row["status"] = "OOM"
    stamps = [float(l.split(" ", 1)[0]) for l in text.splitlines() if EPOCH_RE.search(l) and l[:1].isdigit()]
    lines = [l for l in text.splitlines() if l[:1].isdigit()]
    end = float(lines[-1].split(" ", 1)[0]) if lines else None
    if stamps and end:
        bounds = stamps + [end]
        epochs = [b - a for a, b in zip(bounds, bounds[1:])]
        steady = epochs[1:] if len(epochs) > 1 else epochs
        row.update(
            status=row["status"] if row["status"] == "OOM" else "ok",
            epochs_timed=len(epochs),
            sec_per_epoch=round(statistics.mean(steady), 1),
            sec_first_epoch=round(epochs[0], 1),
            startup_sec=round(stamps[0] - float(lines[0].split(" ", 1)[0]), 1),
        )
    mem = d / "gpu_mem.csv"
    if mem.is_file():
        used = []
        for r in csv.reader(mem.open()):
            try:
                used.append(float(r[1]))
                row["gpu_total_mib"] = float(r[2])
            except (IndexError, ValueError):
                pass
        if used:
            row["peak_mem_gib"] = round(max(used) / 1024, 2)
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--timing_root", required=True, type=Path)
    ap.add_argument("--out_csv", required=True, type=Path)
    ap.add_argument("--out_md", required=True, type=Path)
    args = ap.parse_args()

    rows = [parse_run(d) for d in sorted(args.timing_root.iterdir()) if d.is_dir() and "_bs" in d.name]
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    keys = ["stage", "batch_size", "status", "sec_per_epoch", "sec_first_epoch", "startup_sec",
            "epochs_timed", "peak_mem_gib", "gpu_total_mib", "run"]
    with args.out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    chosen = {}
    for r in rows:  # largest batch size that ran fine, per stage
        if r.get("status") == "ok" and (r["stage"] not in chosen or r["batch_size"] > chosen[r["stage"]]["batch_size"]):
            chosen[r["stage"]] = r

    gpu = (args.timing_root / "gpu.txt").read_text() if (args.timing_root / "gpu.txt").is_file() else "unknown"
    out = ["# Phase 0 timing and GPU-hour budget", "", "```", gpu.strip(), "```", "",
           "| Stage | Batch | s/epoch | Peak GPU mem (GiB) | Status |", "|---|---|---|---|---|"]
    for r in rows:
        out.append(f"| {r['stage']} | {r.get('batch_size')} | {r.get('sec_per_epoch', '')} | {r.get('peak_mem_gib', '')} | {r['status']} |")
    out += ["", f"Budget (x{OVERHEAD} overhead), using the largest batch that fit per stage:", "",
            "| Block | Runs | Epochs | s/epoch | GPU-h |", "|---|---|---|---|---|"]
    total = 0.0
    for label, stage, n, ep in RUN_LIST:
        r = chosen.get(stage)
        if not r:
            out.append(f"| {label} | {n} | {ep} | not measured | ? |")
            continue
        h = n * ep * r["sec_per_epoch"] * OVERHEAD / 3600
        total += h
        out.append(f"| {label} | {n} | {ep} | {r['sec_per_epoch']} (bs {r['batch_size']}) | {h:.1f} |")
    out += ["", f"**Total ≈ {total:.1f} GPU-hours** (measured blocks only). "
            "Cheap evaluations on cached features (P1/P3 scoring, masking, Tests 1-4) are not included.", ""]
    args.out_md.write_text("\n".join(out))
    print("\n".join(out))


if __name__ == "__main__":
    main()
