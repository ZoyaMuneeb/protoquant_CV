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
| 0 | Data, split, protocol, environment, GPU timing | Data, split and protocol done; **timing and backbone fine-tune need the team GPU** |
| 1 | Official pipeline + strict-protocol baseline | Not started |
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

### 5. Stage 1 codebook and Stage 2 head

```bash
BACKBONE_CKPT=../checkpoints/backbone/<file>.pth CODEBOOK_ENTRIES=4096 SEED=40 BATCH_SIZE=<from timing> bash scripts/run_stage1_codebook.sh
```

```bash
BACKBONE_CKPT=../checkpoints/backbone/<file>.pth CODEBOOK_CKPT=<runs/.../codebook_*.pth> SEED=40 BATCH_SIZE=<from timing> bash scripts/run_stage2_head.sh
```

Each run gets its own folder in `../runs/` with `train.log` (timestamped), `gpu_mem.csv`,
`protocol.txt` (protocol and split checksum) and the saved weights.

Useful variables for all scripts: `SEED`, `EPOCHS`, `BATCH_SIZE`, `LR`, `WARMUP_EPOCHS`, `MODEL`,
and `PQ_CUB_PROTOCOL=paper` for the paper-protocol sanity check.

## Repository layout

```
codebook_train/        official ProtoQuant code (CUB loader patched for the strict protocol)
docs/                  phase0_protocol.md — frozen protocol and decision rules
scripts/               SLURM-free run scripts, split generator, data checks, timing/budget
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
