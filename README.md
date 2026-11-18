# ProtoQuant pruning project (MLR 512)

Can activation-aware and globally budgeted pruning improve ProtoQuant's accuracy–sparsity trade-off over
its original weight-based pruning, and can validation-guided codebook reduction keep that trade-off at
lower cost?

This repository is built on the official ProtoQuant code
([gmum/protoquant](https://github.com/gmum/protoquant), commit `b80d7e9`, paper
[arXiv 2602.06592](https://arxiv.org/abs/2602.06592)). Everything runs on **CUB-200-2011** with a
**ConvNeXt-Tiny** backbone on a single NVIDIA GPU.

> The upstream repository has no license file. Ask the authors before including their code in a public
> release or a submission.

## Status

| Phase | What | State |
|---|---|---|
| 0 | Data, split, protocol, environment, GPU timing | **Done** (RTX 5070 Ti; Stage 1/2 at batch 1024; ≈ 33 GPU-hours; backbone val top-1 85.5 %) |
| 1 | Official pipeline + strict-protocol baseline | **Done**: strict 84.43 ± 0.33 % (3 seeds), paper protocol 84.74 %, backbone 84.05 %; Tests 1–4 pass ([results](results/phase1_baseline.md)) |
| 2 | Pruning methods (B1, B2, random, P1, P1-contrast, P3) + deletion-effect control, one-shot and retrained | **Done** ([results](results/phase2_results.md)) |
| 3 | Evaluation: paired bootstrap, decision rules, capacity measures | **Done** ([results](results/phase2_results.md), [capacity](results/capacity.md)) |
| 4 | Ablations: iterative vs one-shot (+ matched B1 and 145-epoch controls), with vs without retraining, P1 vs P1-contrast vs random, P3 with vs without the minimum | **Done** ([results](results/phase2_results.md)) |
| 5 | Figures | Not started |
| – | Adaptive codebook size (A1 shrinkage, A3 effective size) | Not started |

## Results so far

CUB-200-2011, ConvNeXt-Tiny, strict protocol, seeds 40–42, test top-1 (mean of 3 seeds). Dense
ProtoQuant head: **84.43 %**, 3,000 codes, ≈ 447k edges. "Better" and "no meaningful loss" follow the
paired-bootstrap rules in [docs/phase0_protocol.md](docs/phase0_protocol.md), section 6.

**Reproduction.** Our baseline is 3.3 points below the paper's 87.7 %, mostly because our fine-tuned
backbone is weaker (84.05 % vs 88.4 %). Our magnitude pruning (B1) keeps almost exactly the paper's code
counts (788 vs 786 at K = 5, 1,305 vs 1,286 at K = 10).

**One-shot pruning (no retraining).** Activation-aware P1 beats magnitude pruning at every K:

| K | B1 (paper) | P1 | P1-contrast | P3 (global budget) | random |
|---|---|---|---|---|---|
| 5 | 66.64 | **72.48** (+5.85) | 72.67 | 60.76 | 3.77 |
| 10 | 71.78 | **75.35** (+3.57) | 75.42 | 68.25 | 4.74 |
| 20 | 75.60 | **77.63** (+2.03) | 77.90 | 74.04 | 8.73 |

**With retraining (K = 10).** Retraining length matters more than the pruning score or schedule:

| Model | Codes | Training epochs | Test top-1 | vs dense |
|---|---|---|---|---|
| B2 (B1 mask, retrained; the paper's "Pruned & Tuned") | 1,305 | 30 | 81.94 | |
| P1, retrained | 1,456 | 30 | 82.25 (≈ B2) | |
| **B2, retrained longer** | **1,305** | **145** | **84.08** | **no meaningful loss** (−0.36) |
| Iterative B1 (20 % per round) | 1,564 | 145 | 83.89 | loss not excluded |
| Iterative P1 (20 % per round) | 1,650 | 145 | 84.47 | **no meaningful loss** (+0.03) |

* At 30 epochs, retraining erases P1's one-shot advantage at every K.
* Training 145 instead of 30 epochs adds +2.1 points; gradual (iterative) pruning adds nothing beyond that
  (iterative B1 vs B2 at 145 epochs: −0.18, not significant).
* Under the same iterative schedule P1 beats magnitude (+0.58, all seeds), but iterative P1 does not
  significantly beat B2 at 145 epochs (+0.39, one seed negative).
* So at K = 10, plain magnitude pruning plus long retraining keeps dense accuracy with 0.45 % of the
  edges and 57 % fewer codes.

**Ablations.** P1-contrast ≈ P1 (gaps ≤ 0.27 points), so P1 is not just rewarding codes every class
shares. P3 loses to B1 because its global ranking starves classes; dropping the one-edge minimum changes
little (−0.4 at K = 5, nothing at K ≥ 10). Only P1-contrast predicts single-edge deletion damage
(Spearman ρ ≈ 0.24 vs ≈ −0.08 for magnitude and P1).

**Capacity (K = 10, [details](results/capacity.md)).** Explanations shrink about 40× (≈ 300 codes for
90 % of a dense prediction's evidence → 7–8). The codebook shrinks 45–57 % (8.8 → 3.8–4.8 MB). The head
is 2–2.4× faster (4× at K = 5), but the full model only ≈ 2 % (93.8 → 92 ms per batch of 128), and peak
GPU memory is unchanged (1.65 GB): the backbone dominates. Iterative P1 costs ≈ 44 GPU-minutes per model
vs ≈ 27 for B2 at 145 epochs.

Runs added after earlier results were seen (P1 retrained at K = 5/20, the 145-epoch control, the
iterative-vs-dense comparisons) are recorded as post hoc in the protocol's change log.

The frozen protocol (split, recipe, operating points, metrics, decision criteria) is in
[docs/phase0_protocol.md](docs/phase0_protocol.md). Read it before running anything.

## What changed from the official code

* **Held-out validation split.** The original CUB loader uses the *test* set as its validation set
  (for checkpoint selection and codebook pruning). [`codebook_train/src/datasets/cub.py`](codebook_train/src/datasets/cub.py)
  now has two modes, set with the environment variable `PQ_CUB_PROTOCOL`:
  * `strict` (default in our scripts): train on 5,394 images, select on a held-out 600-image validation
    split ([`splits/cub200_val10_seed0.txt`](splits/cub200_val10_seed0.txt), 3 images per class, seed 0).
    The test set (5,794) is only read through `get_cub200_test()`.
  * `paper`: the authors' original behaviour, for the one-seed sanity check.

  Every run logs a line like `[CUB200] protocol=strict | train loader split='train' (5394 imgs) | validation loader split='val' (600 imgs)`.
* **No SLURM, no wandb.** The cluster scripts in `slurm/` are replaced by plain shell scripts in
  [`scripts/`](scripts/) (single GPU, `WANDB_MODE=disabled`). The `slurm/` folder is kept for reference.
* **Stage 2 validates on every image.** The repository's data loaders drop the last incomplete batch,
  also for validation. With 600 validation images and batch 1024 that would leave no validation batch
  at all (accuracy 0, no checkpoint saved), so our Stage 1, B0 and Stage 2 scripts set
  `val_dataloader.drop_last=False`.

## Requirements

* Linux with one NVIDIA GPU (the Stage 1/2 Hydra pipelines require CUDA; macOS is not supported).
* Python 3.11 (the authors' version). [`uv`](https://github.com/astral-sh/uv) is used if installed, otherwise `venv` + `pip`.
* About 3 GB free disk for CUB (raw + cropped).

## How to run

All commands are run from the repository root. By default the scripts put the virtual environment,
data, runs and checkpoints in the **parent folder** of the repository (`../.venv`, `../data`, `../runs`,
`../checkpoints`). Override with `PROJECT_ROOT`, `VENV`, `DATA_ROOT`, `RUNS_ROOT` or `CKPT_ROOT`.

### 1. Environment

```bash
bash scripts/setup_env.sh
```

Installs CUDA 12.8 PyTorch wheels, then `requirements.txt`. For an older driver pick another wheel index,
e.g. `TORCH_INDEX=https://download.pytorch.org/whl/cu121 bash scripts/setup_env.sh`.

### 2. Data

```bash
bash scripts/prepare_cub.sh
```

Downloads CUB-200-2011 from Caltech (1.1 GB, checksum verified), crops every image to its ground-truth
bounding box with the authors' `crop_cub.py`, and runs the data checks
([`scripts/check_phase0_data.py`](scripts/check_phase0_data.py)): counts, crop sizes, split stratification,
split reproducibility and loader behaviour under both protocols. All checks must print `PASS`.
Results go to `results/phase0_data_check.json`.

Note: train and test images are oracle-cropped with annotation boxes, as in the paper's code.

### 3. GPU timing and budget (Phase 0, days 1–3)

```bash
bash scripts/phase0_timing.sh
```

Short (3-epoch) runs of the backbone fine-tune, Stage 1 at 4,096 and 8,192 codes, and Stage 2, trying
batch sizes from largest to smallest until one fits (Stage 1/2 start at the paper's 1024, then 512, 256, 128;
the backbone starts at 256). Writes `results/phase0_timing.csv` (seconds per epoch,
peak GPU memory) and `results/phase0_budget.md` (GPU-hour budget for the full run list).
To reuse an existing backbone: `BACKBONE_CKPT=/path/backbone.pth bash scripts/phase0_timing.sh`.

### 4. Backbone fine-tune (once)

```bash
bash scripts/run_backbone_finetune.sh
```

ConvNeXt-Tiny from ImageNet weights, 30 epochs, AdamW lr 5e-5, best epoch chosen on the held-out split.
The checkpoint lands in `../checkpoints/backbone/`. It is then frozen for every later stage.

### 5. Phase 1: official pipeline and baseline

One command per protocol. Run the paper-protocol sanity check first (one seed, ~1.5 GPU-hours): it shows
whether the pipeline lands near the paper's 87.7 %. Then the strict baseline (seeds 40, 41, 42, ~4.5 GPU-hours).

```bash
BACKBONE_CKPT="$(realpath ../checkpoints/backbone/<file>.pth)" PROTOCOL=paper bash scripts/run_phase1.sh
```

```bash
BACKBONE_CKPT="$(realpath ../checkpoints/backbone/<file>.pth)" bash scripts/run_phase1.sh
```

Per seed, in `../runs/phase1/<protocol>_s<seed>/`:

| Step | Script | Output |
|---|---|---|
| Stage 1: codebook, 4,096 codes, 30 epochs | `run_stage1_codebook.sh` | `stage1/codebook_*.pth` (best val epoch) |
| B0: prune 4,096 → 3,000 in 8 steps by usage on the validation loader | `run_b0_prune.sh` | `b0/pruned_3000_*.pth`, `b0/b0_check.txt` (Test 4) |
| Stage 2: ProtoQuant head, 30 epochs | `run_stage2_head.sh` | `stage2/pipnet_*.pth` (best val epoch) |
| Evaluation: val, then test **once** | `eval_protoquant.py` | `eval/eval_{val,test}.{json,npz}` (per-image predictions) |
| Tests 1–4 (strict only) | `phase1_tests.py` | `tests/tests.json` |

Also run once: the backbone alone on test (black-box baseline, `backbone_<protocol>/`) and, for strict,
the fp16 feature cache for train/val/test (`feature_cache/`, ~0.9 GB). The script is resumable: a
finished step leaves a `DONE` file and is skipped next time; an unfinished step is restarted. Under the
strict protocol a model's test set can be read only once (a `*.test_read_strict.json` file next to the
checkpoint records it).

At the end it writes [`results/phase1_baseline.md`](results/phase1_baseline.md): test accuracy per seed
and mean ± sd, Stage 1 dead-code ratio, B0 accuracy before/after pruning, codebook size, nonzero edges
and the test results. What to check:

* `b0_check.txt` and Test 4 say **PASS** (B0 counted usage on the 600 validation images, not test).
* Tests 1 and 3 **PASS**. Test 2 is a measurement: how many validation predictions change when unused
  codes are physically removed (the paper says none should).
* Stage 2 `train.log` shows a validation accuracy every epoch, and `stage2/pipnet_*.pth` exists.
* Paper protocol near 87.7 %; strict may be lower (less training data, honest selection). Report the gap.

The stage scripts can also be run on their own; useful variables: `SEED`, `EPOCHS`, `BATCH_SIZE`, `LR`,
`WARMUP_EPOCHS`, `MODEL`, `NUM_WORKERS`, and `PQ_CUB_PROTOCOL=paper`.

### 6. Phase 2: pruning methods

Runs on the Phase 1 strict checkpoints (seeds 40–42) and the feature cache; strict protocol only.

```bash
BACKBONE_CKPT="$(realpath ../checkpoints/backbone/<file>.pth)" STAGES=masking bash scripts/run_phase2.sh
```

```bash
BACKBONE_CKPT="$(realpath ../checkpoints/backbone/<file>.pth)" bash scripts/run_phase2.sh
```

The first command is the one-shot comparison only (a few minutes, no training). The second adds the
retrained models (21 runs, about 0.6 GPU-hours each, ~12 GPU-hours).

| Stage | What | Output in `../runs/phase2/strict_s<seed>/` |
|---|---|---|
| masking | Masks for B1, random, P1, P1-contrast, P3 at K = 5, 10, 20 from the dense head (all scores on ReLU(W); P1 statistics from the training split); val + test accuracy with no retraining | `masking/results.json`, `masks.pt`, `test_preds.npz` |
| masking | Deletion-effect control: 100 edges removed one at a time, Spearman ρ of real damage vs magnitude, P1, P1-contrast | `masking/deletion.json` |
| retrain | B2 (B1 mask) at K = 5, 10, 20 and P1, P3, random, P1-contrast at K = 10: unused codes removed, then the Stage 2 recipe for 30 epochs with the mask held fixed, best val epoch, test read once | `retrain/<method>_K<k>/` |

Change the retrained set with `RETRAIN_SPECS` (default `B1_K10 P1_K10 P3_K10 random_K10 P1contrast_K10 B1_K5 B1_K20`).
[`results/phase2_results.md`](results/phase2_results.md) has, per K: test accuracy (mean ± sd), the
difference to B1 (one-shot) or B2 (retrained) with its 95 % paired-bootstrap interval and per-seed signs,
the verdict under the fixed rules, edges and active codes, and the deletion-effect correlations.

### 7. Phase 4: iterative pruning, controls and ablations

Fixed in [docs/phase0_protocol.md](docs/phase0_protocol.md), section 10, before any run.

```bash
BACKBONE_CKPT="$(realpath ../checkpoints/backbone/<file>.pth)" bash scripts/run_iterative.sh
```

Iterative B1 and iterative P1 at K = 10, seeds 40–42 (6 runs). Each run removes 20 % of each class's edges per
round, retrains 5 epochs with the full codebook and rescores (P1 recomputes its class means through the current
codes), for about 24 rounds. The final mask then goes through exactly the B2 recipe (unused codes removed, 30
epochs, best validation epoch, test read once). Output in `../runs/phase2/strict_s<seed>/iterative/<score>_K10/`
(`iterative.json` logs every round); `results/phase2_results.md` gets an "Iterative pruning" section comparing
iterative vs one-shot (same score) and iterative P1 vs iterative B1. To time one run first:
`SEEDS=40 SCORES=P1 bash scripts/run_iterative.sh`.

Epoch-matched control (added after the iterative results, protocol section 10): the one-shot B1 mask at K = 10
retrained for 145 epochs, the iterative runs' total, to separate gradual pruning from extra training
(3 runs, ~30 min each):

```bash
BACKBONE_CKPT="$(realpath ../checkpoints/backbone/<file>.pth)" bash scripts/run_budget_control.sh
```

The summary then adds: iterative B1 vs this control, the control vs B2 at 30 epochs, and every iterative
model vs the dense head.

Phase 4 ablations that need no training (a few minutes): P3 without the one-edge-per-class minimum
(one-shot masks at K = 5/10/20) and P1-contrast vs P1. The summary gets an "Ablations" section.

```bash
BACKBONE_CKPT="$(realpath ../checkpoints/backbone/<file>.pth)" bash scripts/run_ablations.sh
```

### 8. Capacity metrics

After Phase 2 (needs the GPU to itself, about 30 min):

```bash
BACKBONE_CKPT="$(realpath ../checkpoints/backbone/<file>.pth)" bash scripts/run_capacity.sh
```

For the dense head, every one-shot mask, every retrained model, the iterative models and the 145-epoch control
(seeds 40–42; whatever has finished), [`scripts/capacity.py`](scripts/capacity.py)
reports codebook entries and memory (MB, fp32), nonzero edges, latency in ms per batch (head only and full
model; median of 50 batches of 128 at 224 × 224, fp32), peak GPU memory, explanation size (codes needed for
90 % of the predicted class's evidence, test set, no labels used) and training GPU-minutes (including the iterative pruning rounds). One-shot masks
keep the full codebook, so their latency equals the dense model's. Output: `results/capacity.{md,json}`.

## Repository layout

```
codebook_train/        official ProtoQuant code (CUB loader patched for the strict protocol)
docs/                  phase0_protocol.md — frozen protocol and decision rules
scripts/               SLURM-free run scripts, split generator, data checks, timing/budget,
                       Phase 1 driver, B0, evaluation, feature cache, implementation tests,
                       Phase 2 masking/retraining, iterative pruning, controls, ablations, capacity
splits/                fixed CUB validation split (+ summary json)
results/               data checks, timing and budget, Phase 1 baseline, Phase 2–4 results, capacity
slurm/                 original cluster scripts (reference only)
```

## Fixed protocol at a glance

* Split: 5,394 train / 600 val / 5,794 test; test read once per final model.
* Seeds 40, 41, 42; operating points K ∈ {5, 10, 20} edges per class (P3: same total).
* Temperature T = 0.1 and starting codebook 4,096 → 3,000 (B0) in all main runs.
* A method beats B1 at a given K only if the 95 % paired-bootstrap interval of the test-accuracy
  difference is above zero and the sign agrees in all three seeds. Details in
  [docs/phase0_protocol.md](docs/phase0_protocol.md).
