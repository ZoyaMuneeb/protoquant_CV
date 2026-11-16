# Phase 0 protocol: data, split, metrics, decision rules

**Status: FROZEN on 2026-11-15, before any pruning run.** Changing anything below after results exist
must be recorded in the "Change log" at the end, with the reason, and flagged in the report.

## 1. Dataset

| Item | Value |
|---|---|
| Dataset | CUB-200-2011, official release (`CUB_200_2011.tgz`, sha256 `0c685df5…bf081`) |
| Source | `https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz` |
| Official split | 5,994 train / 5,794 test, 200 classes |
| Preprocessing | Every image (train **and** test) cropped to its ground-truth bounding box with the authors' `codebook_train/crop_cub.py` (int-truncated box, RGB, PIL default JPEG re-save) |

**Statement for the report:** train and test images are oracle-cropped with dataset annotations, as in the
authors' code. Results describe behaviour on cropped birds, not on uncropped photos.

## 2. Validation split (strict protocol)

| Item | Value |
|---|---|
| File | `splits/cub200_val10_seed0.txt` (one line per image: `<img_id> <train\|val\|test>`) |
| sha256 | `7e837cd64683c06a25e415fd478bbdce69dd5cfad2160b606f357f07ee1d0eaf` |
| Rule | 10 % of the official training images, stratified by class, `sklearn.model_selection.train_test_split(random_state=0)` |
| Counts | train 5,394 / val 600 (exactly 3 per class) / test 5,794 |
| Generator | `scripts/make_cub_val_split.py`; regenerates bit-identically (checked by `scripts/check_phase0_data.py`) |

The same file is used by every run and every training seed. The split seed (0) is deliberately separate
from the training seeds (40, 41, 42).

### How the code uses it

`codebook_train/src/datasets/cub.py` reads `PQ_CUB_PROTOCOL`:

* `strict` (default in `scripts/common.sh`): the training loader is `train` (5,394) and the repository's
  "validation" loader is the held-out `val` (600). Checkpoint selection (`train_full.py`, Stage 1, Stage 2)
  and code-usage pruning (`main_prune.py`, B0) therefore read the held-out split with no other code change.
  The test split is only reachable through `get_cub200_test()`.
* `paper`: the original behaviour (train 5,994, "validation" = official test). Used only for the one-seed
  paper-protocol sanity check.

Every run prints `[CUB200] protocol=… | train loader split=… | validation loader split=…`; this line is
the evidence for Test 4 (B0 counts usage on validation, not test).

## 3. Training recipe

| Item | Value | Source |
|---|---|---|
| Augmentation | RandomResizedCrop 224, horizontal flip, TrivialAugmentWide, MixUp (α 0.8), CutMix (α 1.0), label smoothing 0.1, ImageNet normalization | Paper App. A.1; repository `transforms.py` |
| Backbone | torchvision ConvNeXt-Tiny, ImageNet-1k weights, 224 px → 7 × 7 × 768 feature map (49 positions; verified) | |
| Backbone fine-tune | `train_full.py`, AdamW, lr 5e-5, wd 1e-5, 30 epochs, 5 warm-up, batch 128 (256 does not fit the team GPU), seed 42, best epoch on held-out val | `slurm/run_full.sh` (batch 256) |
| Backbone after fine-tune | Frozen for all later stages; also the black-box baseline (paper: 88.4 %) | |
| Stage 1 codebook | cosine codebook, 768-d, **4,096 codes**, orthogonal init, AdamW lr 0.05, 30 epochs, 5 warm-up | Paper A.1; `slurm/run_cosine.sh` |
| B0 | `prune_codebook.py`: 4,096 → 3,000 codes in 8 steps, usage counted on held-out val | |
| Stage 2 head | `train_protoquant.py`, learned codes (`use_random_codes=False`), no L1 penalty, AdamW, 30 epochs, 5 warm-up | Paper A.1 |
| Temperature | softmax(s / T), **T = 0.1** in every main run | Repository |
| Batch size (Stage 1/2) | **1,024** with lr 0.05, as in the paper; fits the team GPU (Phase 0 timing). Data-loader workers lowered to 4 train / 2 val to stay within host RAM (speed only, no effect on results) | Paper A.1; Phase 0 timing |
| Logging | wandb disabled; single process (`world_size = 1`) | |

## 4. Operating points and seeds

* Edges kept per class: **K ∈ {5, 10, 20}**; P3 uses the same total, N = K × 200.
* Training seeds: **40, 41, 42**, all on the split above.
* Finalists fixed in advance: every kept method (B1, B2, P1, P3, random, P1-contrast) at every K. All are
  reported on test; no winner is picked from test results.
* Main comparison: one-shot masking, no retraining. Retraining only at K = 10. One-shot vs iterative
  (remove 20 %, retrain, repeat) only for the best validation method at K = 10.

## 5. Metrics

* Top-1 accuracy on the official test set, read **once per final model**.
* Capacity, each reported separately: physical codebook entries; nonzero class–code edges of ReLU(W);
  codebook memory (MB); inference latency (ms/batch at fixed batch, resolution, precision, GPU); peak GPU
  memory (GB); training + selection cost (GPU-hours). No speed claim without a latency measurement.
* Explanation size: smallest number of codes reaching 90 % of Σ ReLU(W)_ŷm · s_m for the predicted class.
* All scores and sparsity counts use the effective classifier ReLU(W), not raw W.
* Figures: accuracy vs nonzero edges and vs physical codes (Pareto, with seed spread). No area-under-curve.

## 6. Decision criteria (fixed now)

* **Improvement:** a method beats B1 at a given K if the 95 % paired-bootstrap interval (10,000 resamples
  over test images, same images and paired predictions for all methods) of its test-accuracy difference lies
  above zero **and** the difference has the same sign in all three seeds.
* **No meaningful loss:** "matches accuracy" means the paired difference's lower 95 % bound is above
  −1 percentage point.
* **Capacity reduction:** a compactness claim needs ≥ 25 % fewer edges or codes at no meaningful loss.
* **Resolution:** one test image ≈ 0.017 points; a single model at ~87.7 % has a 95 % half-width of
  ~0.85 points. Smaller differences are reported, not claimed.
* Seed-wise results are always shown next to the mean.

## 7. Checks run in Phase 0 (`results/phase0_data_check.json`)

All passed on 2026-11-15: official counts; 200 classes; all 11,788 crops present; every crop equals its box
size and is RGB; split covers all images; split's test = official test; 3 val images per class; split
regenerates bit-identically; loader counts under both protocols; strict train/val/test disjoint and complete.

## 8. Decisions taken in Phase 0 that the team should confirm

1. **Backbone is fine-tuned on the 90 % split only.** Fine-tuning on all 5,994 images would let the
   backbone see the validation images. The paper-protocol sanity check reuses this backbone (cheaper,
   slightly conservative); a full-train backbone is one extra run if a closer reproduction is wanted.
2. **`convnext_tiny`, not `convnext_tiny_26`.** `run_full.sh` uses the stride-modified 26 × 26 variant and
   continues from an existing CUB checkpoint; the methodology's 7 × 7 × 768 features need the standard model.
   The repository's lr 5e-5 was set for that continued fine-tune and is used here untuned.
3. **Stage 2 starts from the Stage 1 codebook.** The repository's Stage 2 launcher (`slurm/run_change.sh`)
   sets `use_random_codes=True`, which replaces the loaded codes with random normal vectors before Stage 2
   (codes stay trainable). The methodology's pipeline trains and prunes the codebook first, so we load it
   (`use_random_codes=False`). Which setting produced the paper's numbers is unconfirmed; the Phase 1
   paper-protocol sanity check can run both if time allows.
4. **Timing at 4,096 and 8,192 codes** (rather than 4,000 / 8,000), to match the starting codebook size.

## 9. Still to do on the team GPU (Phase 0, days 1–3)

```bash
bash scripts/setup_env.sh          # Python 3.11 venv, CUDA torch, requirements.txt
bash scripts/prepare_cub.sh        # download, crop, split (already in repo), data checks
bash scripts/phase0_timing.sh      # s/epoch + peak memory: backbone, Stage 1 @4096/@8192, Stage 2 → results/phase0_budget.md
bash scripts/run_backbone_finetune.sh   # the real 30-epoch backbone; record val acc and the checkpoint path
```

Then fill in: GPU model and memory, chosen Stage 1/2 batch size and learning rate, GPU-hour budget, and
backbone val accuracy, below.

| Item | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 5070 Ti, 15.9 GiB; torch 2.11.0+cu128 |
| Stage 1 / Stage 2 batch size, lr | **1,024, lr 0.05** (paper setting), 4 train / 2 val data-loader workers |
| Measured speed and peak memory (batch 1,024) | Stage 1 @4,096: 63.3 s/epoch, 15.48 GiB; Stage 1 @8,192: 69.4 s/epoch, 15.52 GiB; Stage 2 @4,096: 75.1 s/epoch, 15.53 GiB |
| GPU-hour budget | **≈ 33 GPU-hours** for the planned runs, ×1.2 overhead (see `results/phase0_budget.md`) |
| Backbone | Batch 128 (256 ran out of GPU memory), otherwise as in section 3. Best epoch 24, held-out val top-1 **85.5 %** (top-5 95.83 %). Checkpoint `checkpoints/backbone/cub200_convnext_tiny_full_20261116_135156.pth` (outside the repo) |

Peak memory is read from `nvidia-smi` and includes memory PyTorch keeps reserved, so the true need is
probably lower (an earlier run measured 12.9 GiB for Stage 1). If a later run runs out of GPU memory,
batch 512 is the fallback and must be recorded in the change log.

## Change log

| Date | Change | Reason |
|---|---|---|
| 2026-11-15 | Protocol frozen | |
| 2026-11-16 | Section 9 filled: Stage 1/2 at batch 1,024 / lr 0.05 (the paper's setting, replacing "one size that fits"); backbone at batch 128 | Phase 0 timing on the team GPU; set before any pruning run |
| 2026-11-16 | Data-loader workers 8 → 4 (train) / 2 (val) | Batch 1,024 with 8 workers exhausted host RAM (process SIGKILLed); changes loading speed only |
