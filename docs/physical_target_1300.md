# Improved-backbone 1,300-prototype comparison

Run from the repository root. This experiment compares exactly three configurations:
the existing dense seed-40 ProtoQuant model (3,000 physically stored prototypes), P1-only physical
pruning to 1,300, and batched loss-aware physical pruning to 1,300. There is no training, iterative P1,
greedy experiment, or test-set evaluation. All choices use training data; assessment uses validation.

## 1. Inputs

Use the selected ConvNeXt-Tiny checkpoint from `results/backbone_sweep.md` on `main`:
`continue_lr_2e-4_lr_5e-5`, validation 88.67%, checkpoint ending `20261118_223411.pth`.
The previous pilot's original-backbone artifacts are incompatible. The new runner rejects the original
backbone filename, mismatched backbone tensors/hashes, wrong cache splits, IDs, labels or shapes.
It retains the bounded numerical-drift diagnostic and strict compact-inference checks from the pilot.

For Shaikha's layout, reset existing variables explicitly in the same terminal:

```bash
export VENV=/home/shaikha/.venv
export BACKBONE_CKPT=/home/shaikha/runs/backbone_sweep/continue_lr_2e-4_lr_5e-5/ckpt/\
cub200_convnext_tiny_full_20261118_223411.pth
export BB2_RUNS_ROOT=/home/shaikha/runs_bb2
export PHASE1_ROOT="$BB2_RUNS_ROOT/phase1"
export CACHE_DIR="$PHASE1_ROOT/feature_cache"
export CUB_PATH=/home/shaikha/data/cub200_cropped
unset PQ_CKPT P1_REUSE_DIR DENSE_EVAL_PREFIX TARGET_OUTPUT_DIR
ls -lh "$BACKBONE_CKPT" "$PHASE1_ROOT"/strict_s40/stage2/pipnet_*.pth \
  "$CACHE_DIR/train.pt" "$CACHE_DIR/val.pt"
test -d "$CUB_PATH/CUB_200_2011/images" && echo "Dataset found"
```

The launcher requires exactly one `pipnet_*.pth` under that seed-40 Stage 2 directory. If several
exist, set `export PQ_CKPT="/full/path/to/the/correct/improved-backbone/seed40.pth"` explicitly.
The exact basename is not recorded in the repository. Preserve the selected backbone's basename;
paths/directories may be configured. The checkpoint must have 3,000 codes, and its backbone tensors
must equal the selected backbone. Checkpoint seed provenance is the user's responsibility when
supplying an arbitrary copied checkpoint path; input hashes are recorded.

No compatible full-head P1-only 1,300-code result was found in the tracked reports. CP1 results with
top-K edge masks are a different experiment. The inexpensive P1-only baseline therefore runs by default.
Dense reuses its existing checkpoint without retraining; its validation pass is repeated to obtain
matching cross-entropy, predictions and memory metrics, which the existing accuracy summary lacks.
If `strict_s40/eval/eval_val.json` and `.npz` are present under `PHASE1_ROOT`, the launcher also reads that
existing dense validation evaluation as a checked reference, records its original accuracy and derives
CE from its saved logits. Paths, protocol, split and image IDs must match. Prediction/logit differences
from fresh cached inference are logged; all primary comparison metrics consistently use the current
cache. Override the prefix using `DENSE_EVAL_PREFIX=/full/path/to/eval_val`, or explicitly set it to an
empty string to disable reuse of this optional reference. No test evaluation file is loaded.

## 2. Run

Synthetic correctness tests (no checkpoint, cache or dataset needed):

```bash
bash scripts/run_physical_target.sh --self-test
```

Real-input smoke check: tests plus all three configurations on 200 training and 32 validation images,
removing only two prototypes. Pruned smoke models contain **2,998**, not 1,300, prototypes.

```bash
bash scripts/run_physical_target.sh --smoke
```

Complete experiment (only on the teammate's machine, after smoke succeeds):

```bash
bash scripts/run_physical_target.sh --target-count 1300
```

Each invocation creates a new `runs_bb2/physical_target_1300_<timestamp>_<mode>_<pid>/` directory.
Set `TARGET_OUTPUT_DIR` to override it; it must not exist. This runner cannot resume. After interruption,
keep the failed directory and rerun with a new directory. It never writes into Phase 1/2 or old pilot outputs.

Defaults: seed 40; strict 5,394-train / 600-validation split; two calibration images per species (400 total);
shortlist 64; deletion batch 8; image chunk 32; candidate chunk 4. P1 is summed over species using all
training activations and remains fixed throughout pruning. Deletion losses are recalculated against
the current surviving codebook. Each joint batch recomputes spatial softmax and max-pooling. Batch
acceptance uses the existing additional-damage rule (`0.005 + 0.25 * sum(abs(individual deltas))`),
halving and logging failures. This rule does not bound cumulative loss or guarantee unchanged accuracy.

Set `SHORTLIST_SIZE`, `PRUNE_BATCH_SIZE`, `PILOT_IMAGE_BATCH`, `CANDIDATE_CHUNK_SIZE`,
`BATCH_EXTRA_LOSS_TOL`, `BATCH_EXTRA_LOSS_REL`, `CALIBRATION_PER_CLASS`, or `PILOT_DEVICE`
to override those settings; retain defaults for the agreed comparison. Full runs require CUDA; CPU
is supported for smoke. With insufficient GPU memory, use a new output directory and:

```bash
PILOT_IMAGE_BATCH=8 CANDIDATE_CHUNK_SIZE=1 bash scripts/run_physical_target.sh
```

Deterministic feature caches are held in CPU memory and copied in chunks. No test cache is opened.
Previous 160-deletion pruning took 1.76 seconds and about 1.15 GB peak allocated memory on an RTX 5070 Ti;
the 1,700-deletion run has not been measured. Expect more rounds and possible fallbacks; use recorded
timings, not this earlier result, for comparisons on another GPU. Free host RAM is needed for the
train/validation caches, model/checkpoint loading and hashing.

## 3. Check / share results

The printed `[TARGET] output=...` path contains `run.log`, `state.json`, `config.json`,
`cache_feature_checks.json`, `calibration.json`, `fixed_p1_ranking.pt`, `starting_checks.json`,
and root `results.md`, `results.csv`, `results.json`. A `DONE` file and `completed` state indicate success.

| Configuration | Stored code rows / classifier columns | Saved model |
|---|---:|---|
| dense | 3,000 / 3,000 | Original checkpoint reused; its path is recorded |
| p1 | 1,300 / 1,300 | `p1/pruned.pth` |
| batched | 1,300 / 1,300 | `batched/pruned.pth` |

Every method directory contains `results.json` and `validation_predictions.npz`; pruned methods
also have `decisions.jsonl`. Results include validation accuracy/CE and changes from dense, nonzero
effective classifier connections, physically stored and active-code counts, codebook size, pruning
seconds, validation seconds, GPU allocated/reserved peaks and inference/reload checks. A physically
stored prototype may have no positive classifier connection; these counts are reported separately.

Pruning timing includes deletion selection, cached-state updates and final physical compaction.
It excludes shared P1/calibration setup, input loading/hashing, validation, checks and saving. These
other timings are separate. Dense has zero pruning time. GPU peaks are absolute allocator readings,
not incremental memory or total GPU usage: per-method peaks include resident tensors and verification/
evaluation; `pruning_memory` in JSON records the peak before those checks. They are conservative:
the allocator may retain reserved blocks between methods. CPU smoke reports null GPU memory.

```bash
# In another terminal, substitute the printed output path:
export TARGET_OUTPUT_DIR=/home/shaikha/runs_bb2/physical_target_1300_TIMESTAMP_MODE_PID
tail -f "$TARGET_OUTPUT_DIR/run.log"
cat "$TARGET_OUTPUT_DIR/results.md"
```

Send the three root results files, config, calibration, cache diagnostics, starting checks and run log,
plus `p1/results.json`, `batched/results.json` and `batched/decisions.jsonl`. Keep checkpoints locally.
On failure, send `state.json`, available diagnostics and the last 30 lines of `run.log` (or terminal output
if logging did not start). An accepted bounded cache-drift warning does not require rebuilding caches.
For a real mismatch, verify the entire matched artifact bundle before rebuilding train/val caches in a
fresh directory with `scripts/cache_features.py --splits train val`; retain the strict protocol.

## Optional P1 reuse

After a successful full run of this runner, set `P1_REUSE_DIR=/path/to/that/run/p1` to reuse its P1 selection
and recorded pruning time in another full run. Only exact matching input/cache/code/split fingerprints,
the same retained IDs, verified checkpoint metadata, and tensor-identical physical compaction are
accepted. Reused checkpoints are checked and validation is measured again; historical timing and original
memory are clearly labeled in JSON. Invalid explicit reuse fails rather than silently trusting another
experiment. Leave `P1_REUSE_DIR` unset for smoke or if no matching result is available.
