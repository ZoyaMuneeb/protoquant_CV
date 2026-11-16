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
| 1 | Official pipeline + strict-protocol baseline | **Scripts ready** ([`scripts/run_phase1.sh`](scripts/run_phase1.sh)); needs the team GPU |
| 2–5 | Pruning methods, evaluation, ablations, figures | Not started |

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

## Repository layout

```
codebook_train/        official ProtoQuant code (CUB loader patched for the strict protocol)
docs/                  phase0_protocol.md — frozen protocol and decision rules
scripts/               SLURM-free run scripts, split generator, data checks, timing/budget,
                       Phase 1 driver, B0, evaluation, feature cache, implementation tests
splits/                fixed CUB validation split (+ summary json)
results/               check outputs, timing and budget
slurm/                 original cluster scripts (reference only)
```

## Fixed protocol at a glance

* Split: 5,394 train / 600 val / 5,794 test; test read once per final model.
* Seeds 40, 41, 42; operating points K ∈ {5, 10, 20} edges per class (P3: same total).
* Temperature T = 0.1 and starting codebook 4,096 → 3,000 (B0) in all main runs.
* A method beats B1 at a given K only if the 95 % paired-bootstrap interval of the test-accuracy
  difference is above zero and the sign agrees in all three seeds. Details in
  [docs/phase0_protocol.md](docs/phase0_protocol.md).
