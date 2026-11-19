# Physical prototype pruning pilot

Run from the repository root. The pilot compares P1-only, greedy loss-aware, and batched loss-aware
physical deletion, without retraining or reading the test split.

- Pilot 1 uses the same dense seed-40 checkpoint for all methods: 3,000 -> 2,840 codes.
- Pilot 2 physically retains the best 1,700 codes using summed P1 from Pilot 1's starting model.
  It recomputes P1 using that compact model, fixes this new ranking, and runs 1,700 -> 1,540.
- Calibration uses 400 training images, two per species, with a saved seed-40 image-ID manifest.
- P1 scoring uses all 5,394 deterministic training features. Evaluation uses the 600 validation features.
- Retained codes keep their original raw classifier columns; there is no top-K edge mask.

## Get the implementation

These commands apply after the implementation branch has been published with approval:

```bash
git remote -v
git fetch origin
git switch pilot/physical-lossaware
git pull --ff-only origin pilot/physical-lossaware
git branch --show-current
```

`origin` should be `https://github.com/ZoyaMuneeb/protoquant_CV.git` (the `.git` suffix is optional).
The branch command should print `pilot/physical-lossaware`. On the first checkout, `git switch` can
create a local branch tracking the uniquely matching remote branch. Resolve local working changes
before switching if Git reports that they would be overwritten.

## Configure the environment and input files

Use the existing repository environment. Set `VENV` if it is somewhere other than `../.venv`.
The launcher chooses its Python executable automatically; `PILOT_PYTHON` can override it.
If the environment has not been prepared, use the repository's existing `bash scripts/setup_env.sh`.
No additional package is required by this pilot.

Replace the four example paths with paths on the GPU machine. Use the matching original seed-40
checkpoint/backbone bundle for comparison with the existing main results. An improved-backbone bundle
is a separate experiment and must use its own matching cache and outputs.

```bash
export VENV="/absolute/path/to/.venv"
export BACKBONE_CKPT="/absolute/path/to/backbone.pth"
export PQ_CKPT="/absolute/path/to/strict_s40/stage2/pipnet_checkpoint.pth"
export CACHE_DIR="/absolute/path/to/phase1/feature_cache"
export CUB_PATH="/absolute/path/to/cub200_cropped"

ls -lh "$BACKBONE_CKPT" "$PQ_CKPT" "$CACHE_DIR/train.pt" "$CACHE_DIR/val.pt"
test -d "$CUB_PATH/CUB_200_2011/images"
test -f "$CUB_PATH/CUB_200_2011/images.txt"
test -f splits/cub200_val10_seed0.txt
"$VENV/bin/python" -c 'import torch; print(torch.__version__, torch.cuda.is_available())'
```

`CUB_PATH` is the directory containing `CUB_200_2011`, not the `images` directory. Both cache files must
come from this backbone and the strict split. The runner verifies image IDs, labels, backbone tensors,
backbone hash, feature dimensions, and sample features against actual image inference. It also checks
physical deletion against compact-model inference on an actual training image.

If train/validation caches are missing, create only those splits using the existing cache utility:

```bash
export PQ_CUB_PROTOCOL=strict
export PQ_CUB_SPLIT_FILE="$PWD/splits/cub200_val10_seed0.txt"
"$VENV/bin/python" scripts/cache_features.py \
  --backbone_ckpt "$BACKBONE_CKPT" --cub_path "$CUB_PATH" \
  --out_dir "$CACHE_DIR" --splits train val
```

If the existing files have correct hashes but fail the stricter ID/feature checks, choose a new
`CACHE_DIR` before rebuilding. The existing cache utility can skip files based on their backbone hash.

If required inputs are elsewhere, set the paths explicitly. `PHASE1_ROOT` can supply the default cache
location and discover `PQ_CKPT` when exactly one checkpoint exists under `strict_s40/stage2`.
`BACKBONE_CKPT` is always required for real-data runs.

## Run checks, then the pilot

Fast synthetic correctness tests, without data/checkpoint requirements:

```bash
bash scripts/run_physical_pilot.sh --self-test
```

Real-data smoke test (runs correctness tests, both starting models, and all three methods, but uses
200 training images, 32 validation images, and only two removals per method):

```bash
export PILOT_OUTPUT_DIR="$PWD/../runs/physical_pilot_smoke_$(date +%Y%m%d_%H%M%S)"
bash scripts/run_physical_pilot.sh --smoke
```

Full pilot, after the smoke test passes:

```bash
export PILOT_OUTPUT_DIR="$PWD/../runs/physical_pilot_$(date +%Y%m%d_%H%M%S)"
bash scripts/run_physical_pilot.sh
```

A complete pilot requires CUDA. The small smoke test can run with `PILOT_DEVICE=cpu`.
Output directories must be new. A failed/finished run is never silently overwritten or resumed.

## Settings

| Environment variable | Default | Meaning |
|---|---:|---|
| `SHORTLIST_SIZE` | 64 | Lowest-P1 surviving candidates for both loss-aware methods |
| `PRUNE_BATCH_SIZE` | 8 | Maximum simultaneous removal count for batched pruning |
| `BATCH_EXTRA_LOSS_TOL` | 0.005 | Absolute allowance for additional interaction damage, CE per image |
| `BATCH_EXTRA_LOSS_REL` | 0.25 | Relative allowance multiplied by the sum of absolute individual loss changes |
| `PILOT_IMAGE_BATCH` | 32 | Images evaluated at a time |
| `CANDIDATE_CHUNK_SIZE` | 4 | Candidates evaluated together on GPU |
| `PILOT_DEVICE` | auto | `auto`, `cuda`, or `cpu` |
| `PILOT_OUTPUT_DIR` | timestamped folder under `RUNS_ROOT` | All pilot artifacts |

For a proposed batch, calculate each individual loss change against the current model, their sum,
and the actual joint change. Extra damage is `joint change - sum(individual changes)`.
Accept when the joint change is nonpositive, or extra damage is at most
`0.005 + 0.25 * sum(abs(individual changes))` with the defaults above. This is an experimental
interaction-damage allowance, not a numerical-equality tolerance or a guarantee on total loss.
If it fails, halve the batch (8 -> 4 -> 2 -> 1), reevaluate the joint effect, and log why.
Each later shortlist uses the current surviving codes, and every loss estimate uses the current model.
The P1 ranking stays fixed within each pilot.

For example, a more conservative batching run can be launched with:

```bash
BATCH_EXTRA_LOSS_TOL=0.002 BATCH_EXTRA_LOSS_REL=0.1 bash scripts/run_physical_pilot.sh
```

Use a new `PILOT_OUTPUT_DIR` for each settings comparison. To reduce GPU workspace, lower
`PILOT_IMAGE_BATCH` and `CANDIDATE_CHUNK_SIZE`; changing these chunk sizes preserves the algorithm
up to floating-point rounding. Shortlist 16/128 and removal batch 16 are configurable alternatives.

## Inspect progress and results

The output path is printed at startup. In another terminal, set `PILOT_OUTPUT_DIR` to that printed path:

```bash
tail -f "$PILOT_OUTPUT_DIR/run.log"
cat "$PILOT_OUTPUT_DIR/state.json"
find "$PILOT_OUTPUT_DIR" -name decisions.jsonl -o -name results.json -o -name pruned.pth
```

After completion:

```bash
cat "$PILOT_OUTPUT_DIR/results.md"
cat "$PILOT_OUTPUT_DIR/results.csv"
test -f "$PILOT_OUTPUT_DIR/DONE" && echo "Pilot completed"
```

The root contains configuration and input hashes, calibration IDs, the shared 1,700-code checkpoint,
combined JSON/CSV/Markdown results, state, and a completion marker. Each pilot contains its starting
metrics and fresh/fixed P1 ranking. Its `p1`, `greedy`, and `batched` directories contain `pruned.pth`,
validation predictions, a result JSON, and flushed `decisions.jsonl` records. Batch records include
candidate IDs, individual/joint loss changes, allowed extra damage, acceptance, and fallback reasons.

Validation metrics use the deterministic feature cache, as in the existing Phase 2 scoring scripts.
Optimized deletion is checked against actual compact-model forward calls on identical spatial inputs,
and a direct image-to-compact-model check runs during preflight. FP16 cache quantization can still
slightly differ from fresh full-precision feature inference; sample cache checks run before selection.

Pruning timing includes selection, state updates, and compact-model construction, with CUDA
synchronization. It excludes shared P1/cache setup, verification, validation, and checkpoint writing.
Shared setup is recorded in each `starting_model.json`. GPU peak allocation includes resident model
and cache tensors. No speed or accuracy result is asserted by the implementation itself.
