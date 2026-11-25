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
| 2026-11-16 | Stage 2 and B0 validation loaders keep the last incomplete batch (`drop_last=False`, as Stage 1 already did) | With 600 validation images and batch 1,024 the repository default dropped every validation image, so no Stage 2 checkpoint would be selected. Bug fix found before any Phase 1 run; no results affected |
| 2026-11-17 | Retraining extended to P1 at K = 5 and K = 20 | Added after the Phase 2 K = 10 retraining results were seen; reported as post hoc |
| 2026-11-17 | Iterative pruning (Phase 4) specified before any run, see section 10 | The methodology names the comparison but not its schedule; fixed here, with a matched B1 control |
| 2026-11-18 | Epoch-matched one-shot control added to section 10 (B2 retrained for 145 epochs); iterative models also compared with the dense head | Added after the iterative results were seen: iterative pruning trained 145 epochs vs 30, so its gain over one-shot could be extra training. Reported as post hoc |
| 2026-11-18 | Phase 4 ablations implemented as specified in the methodology: P3 without the one-edge-per-class minimum (one-shot, K = 5/10/20) and P1-contrast vs P1 (paired, from the saved predictions) | Planned ablations; no change to methods or rules |
| 2026-11-19 | Improved-backbone rerun added, see section 11 | Added after all main results were seen: our backbone (84.05 % test) is 4.4 points below the paper's (88.4 %), so the claim "longer retraining closes the gap to dense" is retested on a stronger backbone. Reported as a separate, post hoc experiment; main results unchanged |
| 2026-11-19 | Code-level pruning added, see section 12 | New method, added after all main results were seen (exploratory, reported as post hoc): edge methods end at about 1,300 codes, and codes are the capacity that matters |
| 2026-11-19 | Code-matched CB1 control added to section 12 | Added after the code-level results were seen: CB1-M800 ended with fewer active codes than CP1-M800, so the retrained comparison confounded code score with code count. Reported as post hoc |
| 2026-11-23 | Closed-form weight repair added, see section 13 | New method, added after all main results were seen (exploratory, reported as post hoc): tests whether re-fitting the surviving weights in seconds recovers what 145 epochs of retraining recover |
| 2026-11-23 | CE step sweep added to section 13 | Added after the repair results were seen: the 300-step CE refit had not converged, so how much of retraining's gain a head-only refit recovers was understated. Step count chosen on validation; reported as post hoc |
| 2026-11-23 | Retraining recipe on cached features added to section 13 | Added after the CE sweep was seen: splits the remaining gap to retraining into code adaptation (codes + head vs head only, same recipe) and augmentation / image training (vs real retraining). Reported as post hoc |
| 2026-11-23 | Stanford Cars replication added, see section 14 | Added after all CUB results were seen, to test whether the main findings generalise to a second dataset; methods, budget rule and seeds fixed before any Cars run. Reported as post hoc |

## 10. Iterative pruning (Phase 4), fixed before running

Comparison at K = 10, seeds 40–42, from the Phase 1 strict dense Stage 2 checkpoints.

* **Methods:** iterative P1 (the best validation method at K = 10 after retraining) and **iterative B1** as a
  matched control. Iterative runs train for about 5× longer than one-shot ones, so without the control a gain
  could not be attributed to P1's score rather than to the schedule or the extra training. The control is
  added here, before any run, as part of the planned comparison.
* **Schedule:** start from all nonzero edges of ReLU(W). Each round: rescore, keep the top
  max(K, ⌊0.8 · n_c⌋) of each class's current n_c edges (20 % removed per round, as in iterative magnitude
  pruning), then retrain 5 epochs with the mask held fixed (cosine schedule with 1 warm-up epoch per round,
  otherwise the Stage 2 recipe). The round that reaches K edges per class is the last; after it, no round
  training.
* **Rescoring:** B1 uses the current ReLU(W). P1 uses the current ReLU(W) × E[s | y = k], with the class means
  recomputed every round from the training-split feature cache through the current codes (retraining updates
  the codebook, so stale activation statistics would turn P1 into a weight-only score).
* **Codebook:** kept whole during the rounds (only edges are masked, so every code stays in the softmax and
  can be rescored). After the final mask, unused codes are physically removed and the model is retrained with
  exactly the B2 recipe (`phase2_retrain.py`: 30 epochs, 5 warm-up, best validation epoch).
* **Reported:** test top-1 (read once per final model), codes, edges, validation accuracy after every round,
  number of rounds and total training epochs (next to 30 for one-shot + retraining).
* **Comparisons (paired bootstrap, section 6 rules):** iterative vs one-shot with the same score (iterative B1
  vs B2, iterative P1 vs retrained P1): does gradual pruning help? Iterative P1 vs iterative B1: does the
  activation-aware score matter under repeated pruning?
* **Epoch-matched control (added 2026-11-18, after the iterative results):** B2 (the one-shot B1 mask at
  K = 10) retrained with the same recipe for 145 epochs, the iterative runs' total (cosine schedule stretched
  over 145 epochs, 5 warm-up). Iterative B1 vs this model asks whether gradual pruning helps at equal
  training; this model vs B2 at 30 epochs asks whether longer training alone helps. Test read once per model.

## 11. Improved-backbone rerun (post hoc, added 2026-11-19)

Why: the authors' backbone recipe (`slurm/run_full.sh` and `slurm/launch_sweep_full.sh`) differs from ours.
It continues from an already fine-tuned CUB checkpoint at lr 5e-5, after a learning-rate sweep over seven
values, trained on all 5,994 images and selected on test. Ours ran once from ImageNet weights at that
continued-fine-tune lr (5e-5, untuned), on the strict 5,394-image split.

* **Backbone sweep (`scripts/run_backbone_sweep.sh`):** ConvNeXt-Tiny from ImageNet weights, 30 epochs, batch
  128, lr ∈ {1e-4, 2e-4, 4e-4} (5e-5 = the existing backbone). Then the authors' second stage: continue from
  the best checkpoint for 30 more epochs at lr 5e-5. All selection on the 600-image validation split; the test
  set is not read during the sweep. Same strict split, recipe otherwise as in section 3.
* **Rerun (`scripts/run_new_backbone.sh`), on the selected backbone:** the strict Phase 1 baseline (3 seeds,
  backbone test read once), one-shot masking (all methods, K = 5/10/20), B2 at K = 10 retrained for 30 and for
  145 epochs. Same rules as section 6. Runs in `../runs_bb2/`, summaries in `results/backbone2/`.
* **Question:** with a backbone closer to the paper's, does B2 at 145 epochs still keep dense accuracy, and is
  30-epoch retraining still short of it?
* Under the strict protocol the paper's 88.4 % is not expected (less training data, honest selection).

## 12. Code-level pruning (post hoc, exploratory, added 2026-11-19)

Run on the improved backbone (section 11), K = 10 edges per class, seeds 40–42.

* **Method:** score every code by its summed importance over classes and keep the M best: **CP1** uses the P1
  score, Σ_k ReLU(W)_km · E[s_m | y = k] (training split); **CB1** (control) uses magnitude, Σ_k ReLU(W)_km.
  Then each class keeps its top K edges among the kept codes, ranked by the matching edge score.
* **Budgets:** M ∈ {1,200, 1,000, 800, 600, 400}, evaluated one-shot. Retrained (B2 recipe, 145 epochs, the
  same budget as the edge-level B2 control): CP1 at M = 1,000 and 800, CB1 at M = 800.
* **Comparisons (section 6 rules):** one-shot vs edge-level B1 at K = 10 and CP1 vs CB1 at each M; retrained
  vs the B2 145-epoch control and vs the dense head. The compactness claim needs no meaningful loss against
  dense with at least 25 % fewer codes than B2 (≈ 1,300 → ≤ 975).
* All choices fixed here before any code-level run; test read once per model.
* **Code-matched CB1 control (post hoc, added 2026-11-19 after the results above):** CB1-M800 kept only ≈ 484
  active codes vs ≈ 728 for CP1-M800, so the retrained CP1 vs CB1 comparison was not at equal codes. Per seed,
  the CB1 budget M is set so its active codes are closest to those of CP1-M1000 (≈ 890; ties: smaller M), key
  `CB1-eqCP1M1000_K10`. Evaluated one-shot (test read once, separate files), then retrained for 145 epochs with
  the B2 recipe. Comparison: retrained CP1-M1000 vs this control (section 6 rules), answering whether the code
  score matters once the head is retrained.

## 13. Closed-form weight repair (post hoc, exploratory, added 2026-11-23)

Run on the improved backbone (section 11), seeds 40–42, on the saved K = 10 masks: B1, P1, random, CB1-M1000,
CP1-M1000 and the code-matched CB1 (`CB1-eqCP1M1000`). No retraining; `scripts/phase2_repair.py`.

* **Method (LS, main):** keep each mask's surviving edges and w ≥ 0, and re-fit only their weights so the pruned
  head reproduces the dense head's logits on the training split: per class, non-negative least squares
  min_{v ≥ 0} mean_i (s_i[S_k]·v − z_ik)² + 10⁻⁴·|v|², with z = s·ReLU(W)ᵀ the dense logits. Inspired by OBS and
  layer-wise reconstruction (SparseGPT), applied to the prototype head.
* **Control (CE):** the same surviving weights re-fitted to the training labels instead: full-batch Adam on
  cross-entropy over the cached pooled scores, lr 0.01, 300 steps, projected onto w ≥ 0 and the mask each step.
* **Settings:** logical (every code stays in the softmax, as one-shot masking) and physical (codes no class uses
  are removed first, as the retraining recipe does; pooled scores recomputed, repair fits on those). Physical is
  the comparison with retraining. Ridge, steps and lr are fixed here; nothing is tuned on validation or test.
* **Comparisons (section 6 rules):** each repaired model vs the same mask one-shot; physical LS and CE vs the same
  mask retrained (B2 / code-level, 145 epochs) and vs the dense head; LS vs CE; P1 vs B1 and CP1-M1000 vs the
  code-matched CB1 after repair (does the score matter once the weights are repaired?).
* Test read once per model (separate files: `repair.json`, `repair_test_preds.npz`).
* **CE step sweep (post hoc, added 2026-11-23 after the 300-step results were seen):** the 300-step CE refit had not
  converged (training cross-entropy still roughly halving), so the same refit (lr 0.01, same masks, logical and
  physical) is run to 10,000 steps and evaluated on validation at 300, 1,000, 3,000 and 10,000 steps. Per model, the
  step count with the best validation top-1 is kept (ties: fewer steps); test is read once for it (separate files,
  `repair_cesweep.*`, keys `<method>+CEsel` / `<method>+phys+CEsel`). The 300-step results stay as reported.
  Comparisons: CEsel vs 300-step CE, vs the same mask retrained and vs dense; P1 vs B1 and CP1 vs the code-matched
  CB1 after CEsel repair.
* **Retraining recipe on cached features (post hoc, added 2026-11-23 after the CE sweep was seen):** the CE refit
  closed only part of the gap to retraining, so the remaining gap is split into code adaptation and augmentation.
  The same masks (physical: unused codes removed) are retrained with the exact retraining recipe (AdamW lr 0.05,
  weight decay 5e-4, 5 warm-up epochs + cosine, label smoothing 0.1, batch 1,024, 145 epochs, mask held fixed, best
  epoch on validation) but on the cached training features instead of augmented images (no MixUp / CutMix / image
  augmentation): head only (`<method>+phys+rtH`) and codes + head (`<method>+phys+rtCH`). Comparisons: codes + head
  vs head only (does adapting the codes help?), each vs the same mask retrained on images (what is left is
  augmentation / image training) and vs dense; P1 vs B1 and CP1 vs the code-matched CB1 after each. Test read once
  per model (`repair_cachedrt.*`).

## 14. Stanford Cars replication (post hoc, added 2026-11-23, fixed before any Cars run)

Goal: check whether the main CUB findings hold on a second fine-grained dataset. Stanford Cars (196 classes,
8,144 train / 8,041 test images) was chosen over Stanford Dogs (its images come from ImageNet, on which the
backbone was pre-trained) and Flowers102 (10 training images per class, accuracy near the ceiling).

* **Data and protocol:** the download is converted once to the CUB layout (`scripts/prepare_cars.py`), full
  images (not cropped to the boxes, as the official Stanford Cars configuration), labels 1–196. The same strict
  protocol as CUB: 10 % of the official training images held out for validation (stratified by class, seed 0,
  `splits/cars196_val10_seed0.txt`), official test images read once per final model. Transforms: the official
  Stanford Cars ones (resize 256, crop 224).
* **Backbone:** ConvNeXt-Tiny from ImageNet, the improved-backbone recipe without a sweep: lr 2e-4 for 30
  epochs, then continued at lr 5e-5 for 30 epochs; the run with the better validation top-1 is used.
* **Dense model:** Stage 1 (4,096 codes) → B0 (3,000) → Stage 2, as on CUB; seed 40 (more seeds if time
  allows; with one seed the "every seed agrees" part of the decision rule cannot apply and is reported as such).
* **Methods (the best CUB methods and their references), K = 10:** one-shot magnitude (B1), activation-aware
  (P1) and random masks (plus the other one-shot masks, which cost nothing); code-level masks; the head refit
  (CE, step count chosen on validation) and the retraining recipe on cached features (head only, codes + head)
  for B1, P1, random and the chosen code-level mask; full retraining on images for 145 epochs of B1 ("Pruned &
  Tuned") and the chosen code-level mask.
* **Code budget:** the code-level activation-aware mask (CP1) whose active codes are closest to 68 % of the
  magnitude mask's at K = 10 (890 / 1,304 on CUB), chosen from code counts only, before any accuracy is seen.
* **Comparisons:** the section 6 rules, as on CUB (one-shot vs magnitude; repaired / cached-retrained vs one-shot,
  vs retrained and vs dense; code-level retrained vs "Pruned & Tuned" and vs dense).
* **Outcome (2026-11-25):** seeds 41 and 42 were added after the seed-40 results were seen, with the same settings.
  The budget rule chose CP1-M1000 on seed 40 (879 of 1,347 codes, 65 %), reused for seeds 41 and 42 (874 ± 5 of
  1,339 ± 21). No code-matched magnitude (CB1) control was run, so the score's effect on code-level pruning after
  retraining is untested here. The backbone drops the last training batch when its size is odd (MixUp needs even
  batches).
* Runs: `bash scripts/prepare_cars.sh`, then `bash scripts/run_cars.sh` (`../runs_cars`, `results/cars`).
