# ProtoQuant pruning project (MLR 512)

**Question.** Can activation-aware or globally budgeted pruning beat ProtoQuant's original magnitude pruning,
and how small can the model get before it loses accuracy?

Built on the official ProtoQuant code ([gmum/protoquant](https://github.com/gmum/protoquant), commit `b80d7e9`;
paper [arXiv 2602.06592](https://arxiv.org/abs/2602.06592)). CUB-200-2011 and Stanford Cars, ConvNeXt-Tiny,
one NVIDIA GPU.

> The upstream repository has no license file. Ask the authors before including their code in a public release.

## Status

| Phase | What | State |
|---|---|---|
| 0 | Data, held-out split, frozen protocol, GPU timing | Done |
| 1 | Official pipeline + strict baseline | Done |
| 2–3 | Pruning methods, paired-bootstrap evaluation, capacity | Done |
| 4 | Iterative pruning, controls, ablations | Done |
| – | Improved-backbone rerun | Done |
| – | Code-level pruning + code-matched CB1 control | Done |
| – | Weight repair and decomposition of retraining (head refit, codes + head on cached features) | Done |
| – | Stanford Cars replication | Done (3 seeds, incl. code-matched CB1 control) |
| – | Interpretability (purity on CUB, explanation size and code change on CUB and Cars) | Done (3 seeds) |
| 5 | Figures, prototype visualisations, demo | Not started |

## Key findings

1. **One-shot, activation-aware pruning (P1) beats magnitude pruning (B1) at every K**, on both backbones:
   +2 to +6 points, better in every seed.
2. **Retraining for 30 epochs erases that advantage.** Retrained P1 ≈ retrained B1 (B2, the paper's
   "Pruned & Tuned").
3. **Retraining length matters more than the pruning score.** B2 retrained for 145 instead of 30 epochs gains
   +1.1 to +2.1 points and shows no meaningful loss against the dense head, with 0.45 % of its edges.
4. **Gradual (iterative) pruning adds nothing** once training length is matched.
5. **Pruning whole codes (new) gives a 32 % smaller codebook than B2.** Keeping 890 codes instead of B2's
   1,304, then retraining for 145 epochs, shows no meaningful loss against B2. Against dense, a 1-point loss
   is just not excluded (lower bound −1.07).
6. **The score again matters only before retraining.** At equal codes (890), the activation-aware code score
   (CP1) beats magnitude (CB1) by +8.2 points one-shot. After 145 epochs of retraining the two are the same
   (+0.14, CI [−0.28, +0.56]). The same pattern holds for edges (findings 1–2) and for codes, and for codes on
   Cars (finding 14).
7. **The global budget (P3) and the "contrast" variant don't help.** P3 starves classes at small K.
   P1-contrast ≈ P1 in accuracy. On CUB it is the only score that predicts single-edge deletion damage, but
   this did not replicate on Cars (ρ = 0.08).
8. **Pruning shrinks explanations 40× and the codebook by half, but not the model.** The backbone dominates
   latency and memory, so the full model is only about 2 % faster.
9. **Why retraining erases the score: the codes adapt.** We split retraining's gain into steps (K = 10,
   improved backbone). Re-fitting only the head recovers about 40–60 % of it in seconds. Letting the codes
   adapt too, with the retraining recipe run on cached features, recovers almost all the rest. Training on
   augmented images adds only the last 0.8–1.3 points.
10. **On CUB, a head refit alone already removes the score's advantage.** After re-fitting the kept weights, P1 vs B1
    is +0.21 [−0.25, +0.68] and CP1 vs CB1 (890 codes) is +0.40 [−0.31, +1.09]. One-shot they were +2.46 and
    +8.19.
11. **Retraining on cached features is a cheap near-substitute.** Codes + head trained for 145 epochs on
    cached features take about 8–17 s per model instead of about 41 min on images. They land 0.8–1.3 points
    below full retraining. Even random edges reach 80.8 % this way (7.4 % with a head refit only).
12. **Matching the dense model's logits by least squares fails** (a negative result): −6.6 to −10.3 points
    against one-shot for B1, P1 and CP1 (physical). It restores the logits' scale but not the ranking of classes.
13. **Removing unused codes alone helps code-level pruning** (physical vs logical, one-shot): +2.3 points for
    CP1-M1000 and +16.3 for CB1-M1000, because the softmax renormalises over the kept codes.
14. **The main results hold on Stanford Cars** (3 seeds). B2 at 145 epochs shows no meaningful loss against dense
    (88.49 vs 88.99; −0.50 [−0.96, −0.05], a small drop) with 0.54 % of the edges. Code-level CP1 uses 35 % fewer
    codes than B2 with no meaningful loss (−0.32 [−0.66, +0.00]), as on CUB. P1 beats B1 one-shot at every K. Unlike
    on CUB, a small advantage survives a head refit (+0.63 [+0.17, +1.07]); it is gone once the codes adapt
    (−0.23 [−0.52, +0.07]). At equal codes (874), CP1 beats CB1 by +9.88 [+9.01, +10.74] one-shot, but after 145
    epochs of retraining no difference is detectable (−0.29 [−0.65, +0.06]).
15. **Pruning keeps more part-consistent codes and makes explanations 37–50× shorter** (post hoc). CUB purity is 0.340
    for dense, 0.418 for the codes magnitude pruning keeps (before any retraining) and 0.322 for random ones. Codes that
    make up 90 % of a prediction's score fall from 247 to 6–7 (CUB) and from 290 to 6–7 (Cars); by the repository's
    thresholded rule the reduction is only about 2×.
16. **Retraining on images preserves average purity but not what each code shows** (same-code change −0.007 to
    +0.023). Only 21–31 % (CUB) and 15–23 % (Cars) of a code's top-10 patches stay the same as in the dense model.
17. **Activation-aware codes stay slightly purer after retraining** (CUB, 890 codes): CP1 − CB1 is +0.018, +0.021,
    +0.029 (train) and +0.027, +0.020, +0.024 (test) in seeds 40–42, at about equal accuracy. Small, no interval
    computed, and not testable on Cars (no part labels).
18. **Accuracy does not guarantee interpretability.** Random edges with codes + head trained on cached features reach
    80.8 % on CUB, but their purity is 0.256 and almost none of the dense model's top patches remain (0.8 %).

Verdicts use the rules fixed before any run ([protocol](docs/phase0_protocol.md), section 6):
- **better**: the 95 % paired-bootstrap interval is above 0 and every seed agrees;
- **no meaningful loss**: the lower bound is above −1 point;
- **loss not excluded**: otherwise.

Runs added after earlier results were seen are labelled post hoc in the protocol's change log.

## Results

Test top-1 %, strict protocol. Mean of seeds 40–42. CUB unless stated; Cars has its own section.

### Baselines

| | Backbone alone | ProtoQuant (dense, 3,000 codes) |
|---|---|---|
| Paper | 88.4 | 87.7 |
| Ours, main backbone | 84.05 | 84.43 ± 0.33 |
| Ours, improved backbone (lr sweep + continued fine-tune) | 86.95 | 85.65 ± 0.14 |

Our B1 masks keep almost exactly the paper's code counts (1,305 vs 1,286 at K = 10).

### One-shot pruning (no retraining), K edges per class

| K | B1 (paper) | P1 | P1-contrast | P3 (global) | random |
|---|---|---|---|---|---|
| **Main backbone** | | | | | |
| 5 | 66.64 | **72.48** | 72.67 | 60.76 | 3.77 |
| 10 | 71.78 | **75.35** | 75.42 | 68.25 | 4.74 |
| 20 | 75.60 | **77.63** | 77.90 | 74.04 | 8.73 |
| **Improved backbone** | | | | | |
| 5 | 73.66 | **78.09** | 78.24 | 72.51 | 4.24 |
| 10 | 77.06 | **79.52** | 79.65 | 76.72 | 4.93 |
| 20 | 79.41 | **81.15** | 81.31 | 79.55 | 7.71 |

### With retraining, K = 10

| Model | Codes | Epochs | Test | vs dense |
|---|---|---|---|---|
| **Main backbone** (dense 84.43) | | | | |
| B2 (B1 mask, retrained) | 1,305 | 30 | 81.94 | |
| P1, retrained | 1,456 | 30 | 82.25 | |
| B2, retrained longer | 1,305 | 145 | **84.08** | no meaningful loss (−0.36) |
| Iterative B1 | 1,564 | 145 | 83.89 | loss not excluded |
| Iterative P1 | 1,650 | 145 | 84.47 | no meaningful loss (+0.03) |
| **Improved backbone** (dense 85.65) | | | | |
| B2 | 1,304 | 30 | 84.36 | |
| B2, retrained longer | 1,304 | 145 | **85.47** | no meaningful loss (−0.18) |
| Code-level CP1, M = 1,000 | **890** | 145 | 85.15 | loss not excluded (−0.50, CI [−1.07, +0.07]) |
| Code-level CB1, matched to 890 codes (control) | 890 | 145 | 85.01 | loss not excluded (−0.64, CI [−1.20, −0.09]) |
| Code-level CP1, M = 800 | 728 | 145 | 84.53 | loss not excluded |
| Code-level CB1, M = 800 (magnitude control) | 484 | 145 | 83.82 | loss not excluded |

**Iterative pruning.** Iterative P1 beats iterative B1 (+0.58, all seeds). But neither differs significantly
from B2 trained for the same 145 epochs.

**Code-level pruning, one-shot.** At the same budget M, CP1 beats CB1 by 27–47 points, but CB1 also ends
with far fewer active codes (e.g. 484 vs 728 at M = 800). At equal active codes (890), CP1 still wins by
+8.19 [+7.21, +9.16], better in every seed.

**Code-level pruning, retrained (145 epochs, 890 codes).**
- CP1 vs B2: −0.32 [−0.71, +0.07], no meaningful loss.
- CB1 vs B2: −0.46 [−0.82, −0.10], no meaningful loss.
- CP1 vs CB1: +0.14 [−0.28, +0.56], no difference.

So the smaller codebook comes from pruning at the code level plus long retraining, not from the CP1 score.

### Where retraining's gain comes from (improved backbone, K = 10, post hoc)

Every model has unused codes physically removed. Steps, left to right:
- **head refit:** the kept weights re-fitted to the labels on cached features (Adam, step count chosen on
  validation);
- **recipe, head:** the full retraining recipe on cached features, codes frozen (145 epochs, best epoch on
  validation);
- **recipe, codes + head:** the same with the codes trained too;
- **retrained:** the real retraining on augmented images (145 epochs).

| Mask | Codes | One-shot | Head refit | Recipe, head | Recipe, codes + head | Retrained |
|---|---|---|---|---|---|---|
| B1 | 1,304 | 77.06 | 80.53 | 80.46 | **84.64** | 85.47 |
| P1 | 1,574 | 79.52 | 80.74 | 80.66 | **83.99** | – |
| CP1-M1000 | 890 | 71.14 | 77.08 | 76.73 | **83.88** | 85.15 |
| CB1 matched to 890 codes | 890 | 62.95 | 76.68 | 75.99 | **84.13** | 85.01 |
| CB1-M1000 | 579 | 35.80 | 68.52 | 65.63 | **82.87** | – |
| random | 1,432 | 4.93 | 7.88 | 7.44 | **80.80** | – |
| Time per model | | – | ≈ 5–6 s | ≈ 5–9 s | ≈ 8–17 s | ≈ 41 min |

- **Codes + head vs head only** (same recipe): +3.3 to +17.2 points (+73 for random edges), better in every
  seed for every mask.
- **Codes + head on cached features vs retrained on images:** −0.83 (B1), −1.27 (CP1), −0.87 (CB1).
- **Against dense**, codes + head on cached features is −1.0 to −1.8 for B1, P1, CP1 and the matched CB1
  (loss not excluded; −2.8 for CB1-M1000 at 579 codes). B2 retrained on
  images for 145 epochs shows no meaningful loss.

How the score's advantage shrinks (Δ test points, 3 seeds):

| Comparison | One-shot | Head refit | Recipe, codes + head | Retrained 145 epochs |
|---|---|---|---|---|
| P1 vs B1 | **+2.46** (better) | +0.21 | −0.64 | – |
| CP1 vs CB1, 890 codes | **+8.19** (better) | +0.40 | −0.25 | +0.14 |

Least-squares repair against the dense logits loses 6.6–10.3 points against one-shot for B1, P1 and CP1, a
negative result. A CE
refit with a fixed 300 steps recovers less than the step count chosen on validation. The full repair table is
in `results/backbone2/phase2_results.md`.

### Stanford Cars (3 seeds, post hoc)

Backbone alone 91.90, dense ProtoQuant 88.99 ± 0.06 (362,768 edges). Split: 7,329 train / 815 validation / 8,041
test. Seeds 41 and 42 were added after the seed-40 results were seen, with the same settings.

One-shot:

| K | B1 | P1 | P1-contrast | P3 (global) | random | P1 vs B1 |
|---|---|---|---|---|---|---|
| 5 | 68.25 | **71.50** | 71.49 | 66.53 | 3.51 | +3.24 [+2.68, +3.80] |
| 10 | 72.73 | **75.33** | 75.21 | 72.75 | 5.25 | +2.61 [+2.12, +3.09] |
| 20 | 77.67 | **79.55** | 79.56 | 78.37 | 9.16 | +1.88 [+1.43, +2.32] |

Where retraining's gain comes from (K = 10, unused codes removed, columns as in the CUB table above):

| Mask | Codes | One-shot | Head refit | Recipe, head | Recipe, codes + head | Retrained |
|---|---|---|---|---|---|---|
| B1 | 1,339 | 72.73 | 79.90 | 79.19 | **86.85** | 88.49 |
| P1 | 1,486 | 75.33 | 80.52 | 80.05 | **86.62** | – |
| CP1-M1000 | 874 | 69.85 | 78.33 | 77.44 | **86.88** | 88.16 |
| CB1, code-matched | 874 | 59.97 | – | – | – | 88.46 |
| random | 1,427 | 5.25 | 7.64 | 7.15 | **83.77** | – |
| Time per model | | – | ≈ 5–6 s | ≈ 8–12 s | ≈ 14–22 s | not timed here |

- **Against dense:** B2 −0.50 [−0.96, −0.05], no meaningful loss (but a small real drop). CP1-M1000
  −0.82 [−1.29, −0.37], loss not excluded. Code-matched CB1 −0.53 [−0.99, −0.08], no meaningful loss, narrowly.
- **CP1-M1000 vs B2, both retrained:** −0.32 [−0.66, +0.00], no meaningful loss (seeds −0.55, −0.76, +0.34).
- **Codes + head on cached features vs retrained on images:** −1.64 (B1), −1.28 (CP1-M1000).
- **P1 vs B1:** +0.63 [+0.17, +1.07] after the head refit and +0.85 [+0.42, +1.29] after the head-only recipe (both
  better; about +0.2 and no difference on CUB), then −0.23 [−0.52, +0.07] after codes + head.
- **Code-level, one-shot:** CP1 beats CB1 by 23–45 points at the same budget M, but CB1 keeps fewer active codes. With
  fewer codes, CP1 still wins: 61.91 at 573 codes vs 48.81 for CB1 at 680.
- **CP1-M1000 vs code-matched CB1 (874 codes each):** one-shot +9.88 [+9.01, +10.74], better. After 145 epochs
  −0.29 [−0.65, +0.06], no meaningful loss (seeds −0.46, −0.31, −0.10), so no retained advantage for the
  activation-aware score, as on CUB, and no detectable advantage for magnitude either.
- **Code-matched CB1 vs B2, both retrained:** −0.03 [−0.33, +0.26], no meaningful loss with 35 % fewer codes.

### Interpretability (3 seeds, post hoc)

Purity, as in the authors' code (PIP-Net protocol): share of a code's top-10 patches showing the same bird part, CUB
only (Cars has no part labels). Explanation size: codes making up 90 % of the predicted class's score / the
repository's rule (score × weight > 0.001, scores below 0.1 ignored). Same patches: share of a code's top-10 training
patches unchanged from the dense model. K = 10, means over seeds 40–42.

CUB:

| Model | Codes | Purity train | Purity test | Same-code purity change | Codes for 90 % / per image | Same patches |
|---|---|---|---|---|---|---|
| Dense | 3,000 | 0.340 | 0.345 | – | 247 / 6.9 | – |
| B1, no retraining | 1,304 | 0.418 | 0.420 | 0.000 | 6.3 / 2.4 | 100 % |
| B1, unused codes removed | 1,304 | 0.408 | 0.407 | −0.011 | 6.4 / 4.3 | 85.2 % |
| B1, cached codes + head | 1,304 | 0.391 | 0.384 | −0.027 | 5.6 / 3.4 | 39.9 % |
| B2 (145 epochs) | 1,304 | 0.415 | 0.413 | −0.003 | 6.8 / 3.4 | 30.5 % |
| CB1, code-matched, no retraining | 890 | 0.378 | 0.377 | 0.000 | 5.9 / 1.1 | 100 % |
| CB1, code-matched, retrained | 889 | 0.401 | 0.395 | +0.023 | 5.8 / 3.3 | 21.2 % |
| CP1-M1000, no retraining | 890 | 0.431 | 0.426 | 0.000 | 6.6 / 3.3 | 100 % |
| CP1-M1000, cached codes + head | 890 | 0.387 | 0.384 | −0.043 | 4.5 / 3.0 | 33.5 % |
| CP1-M1000, retrained | 890 | 0.424 | 0.418 | −0.007 | 5.8 / 3.7 | 23.2 % |
| random, no retraining | 1,432 | 0.322 | 0.327 | 0.000 | 1.4 / 0.9 | 100 % |
| random, cached codes + head | 878 | 0.256 | 0.263 | −0.027 | 2.4 / 2.1 | 0.8 % |
| Chance (random patches) | – | 0.108 | 0.109 | – | – | – |

Cars:

| Model | Codes | Codes for 90 % / per image | Same patches |
|---|---|---|---|
| Dense | 3,000 | 290 / 4.4 | – |
| B2 (145 epochs) | 1,339 | 6.5 / 2.5 | 23.2 % |
| CB1, code-matched, retrained | 873 | 5.7 / 2.5 | 15.2 % |
| CP1-M1000, retrained | 873 | 5.8 / 2.9 | 19.8 % |
| random, cached codes + head | 968 | 2.6 / 1.9 | 0.8 % |

- **The dense mean covers all 3,000 codes**, so dense vs pruned purity reflects which codes are kept. Use the same-code
  change to judge retraining.
- **Not a replication of the paper's purity (0.47):** different backbone checkpoint, split and code count, and 7 × 7
  maps. Compare our models with each other.
- **Deviations from the authors' code:** protocol section 15. Full tables below.

### Ablations and controls

CUB, unless a row says otherwise.

| Question | Answer |
|---|---|
| Does contrasting against other classes help (P1-contrast vs P1)? | Slightly: +0.07 to +0.27 one-shot (significant at some K), never more |
| Does P3's one-edge-per-class minimum matter? | −0.4 at K = 5 without it (11 classes get no edge); none at K ≥ 10 |
| Which score predicts real single-edge deletion damage (Spearman ρ)? | P1-contrast 0.24 / 0.42 (main / improved backbone); magnitude and P1 ≈ 0. Cars: none (−0.05 magnitude, −0.10 P1, 0.08 P1-contrast) |
| Does longer retraining alone help (B2 at 145 vs 30 epochs)? | Yes: +2.13 (main), +1.11 (improved), both better |

### Capacity, K = 10, main backbone ([details](results/capacity.md))

| | Dense | Pruned |
|---|---|---|
| Codes needed for 90 % of a prediction's evidence | ≈ 308 | 7–8 |
| Codebook | 8.8 MB | 3.8–4.8 MB |
| Head latency | 1× | 1.85–2.4× faster |
| Full-model latency (batch 128) | 93.8 ms | ≈ 92 ms |
| Peak GPU memory | 1.65 GB | 1.65 GB |

Training cost: iterative P1 ≈ 44 GPU-minutes per model vs ≈ 27 for B2 at 145 epochs.

Full tables:
- main backbone: [`results/phase2_results.md`](results/phase2_results.md);
- improved backbone: [`results/backbone2/phase2_results.md`](results/backbone2/phase2_results.md);
- Stanford Cars: [`results/cars/phase2_results.md`](results/cars/phase2_results.md);
- interpretability: [CUB](results/backbone2/interpretability.md), [Cars](results/cars/interpretability.md).

## What we changed in the official code

- **Held-out validation.** The original CUB loader selects checkpoints and prunes codes on the *test* set.
  `PQ_CUB_PROTOCOL=strict` (our default) trains on 5,394 images and validates on a fixed 600-image split
  ([`splits/`](splits/)). Test (5,794) is read once per final model, and a marker file next to the checkpoint
  enforces this. `paper` keeps the original behaviour.
- **Validation keeps the last batch.** With 600 validation images and batch 1,024, the original
  `drop_last=True` would leave no validation batch at all.
- **Stanford Cars with the same strict protocol.** `scripts/prepare_cars.py` converts the download to the CUB
  layout, and dataset `cars196` reuses the strict CUB loader (held-out validation split, test read once).
  `PQ_DATASET=cars196` switches every script; the default stays CUB.
- **MixUp with an odd last batch.** MixUp rejects odd batch sizes, so the backbone fine-tune drops the last training
  batch when its size is odd (Cars only; CUB is unchanged).
- **No SLURM or wandb.** Plain single-GPU shell scripts are in [`scripts/`](scripts/); `slurm/` is kept for
  reference.

## How to run

You need Linux with one NVIDIA GPU, Python 3.11 and about 3 GB of disk space. Run everything from the repository root.

The virtual environment, data, runs and checkpoints go in the parent folder (`../.venv`, `../data`,
`../runs`, `../checkpoints`). Override them with `PROJECT_ROOT`, `RUNS_ROOT`, `RESULTS_DIR` and similar variables.

Every driver can resume: a finished step leaves a `DONE` file and is skipped next time.

Set `BB=$(realpath ../checkpoints/backbone/<file>.pth)` first.

| Step | Command | GPU time |
|---|---|---|
| Environment | `bash scripts/setup_env.sh` | – |
| Data (download, crop, checks must print PASS) | `bash scripts/prepare_cub.sh` | – |
| Backbone fine-tune | `bash scripts/run_backbone_finetune.sh` | < 1 h |
| Phase 1 baseline (3 seeds) | `BACKBONE_CKPT=$BB bash scripts/run_phase1.sh` | ~4.5 h |
| Phase 2 one-shot + retraining | `BACKBONE_CKPT=$BB bash scripts/run_phase2.sh` | ~12 h |
| Iterative pruning | `BACKBONE_CKPT=$BB bash scripts/run_iterative.sh` | ~4.5 h |
| 145-epoch B2 control | `BACKBONE_CKPT=$BB bash scripts/run_budget_control.sh` | ~1.5 h |
| Ablations (no training) | `BACKBONE_CKPT=$BB bash scripts/run_ablations.sh` | minutes |
| Capacity (GPU to itself) | `BACKBONE_CKPT=$BB bash scripts/run_capacity.sh` | ~30 min |
| Improved backbone: lr sweep | `bash scripts/run_backbone_sweep.sh` | ~2–3 h |
| Improved backbone: rerun | `bash scripts/run_new_backbone.sh` | ~8 h |
| Code-level pruning (improved backbone) | `bash scripts/run_codeprune.sh` | ~6 h |
| Code-matched CB1 control | `bash scripts/run_codeprune_control.sh` | ~2 h |
| Weight repair: LS and CE refit (no retraining) | `bash scripts/run_repair.sh` | minutes |
| Weight repair: CE step count chosen on validation | `CE_SWEEP="300 1000 3000 10000" bash scripts/run_repair.sh` | minutes |
| Retraining recipe on cached features (head only, codes + head) | `CACHED_RETRAIN=1 bash scripts/run_repair.sh` | ~10–30 min |
| Stanford Cars: data + split (once) | `CARS_SRC=/path/to/download bash scripts/prepare_cars.sh` | – |
| Stanford Cars: replication (1 seed) | `bash scripts/run_cars.sh` | ~7 h (estimate) |
| Stanford Cars: 3 seeds | `SEEDS="40 41 42" RETRAIN_SEEDS="40 41 42" bash scripts/run_cars.sh` | – |
| Interpretability: agreement with the authors' purity code | `CHECK=1 bash scripts/run_interpretability.sh` | minutes |
| Interpretability, CUB | `bash scripts/run_interpretability.sh` | ~15–30 min (estimate) |
| Interpretability, Cars (no purity) | `PQ_DATASET=cars196 bash scripts/run_interpretability.sh` | ~15–30 min (estimate) |

Improved-backbone runs write to `../runs_bb2/` and `results/backbone2/`, so the main results are untouched.
Each step's exact settings are fixed in [`docs/phase0_protocol.md`](docs/phase0_protocol.md):
- section 10: iterative pruning;
- section 11: the improved backbone;
- section 12: code-level pruning;
- section 13: weight repair and the retraining recipe on cached features;
- section 14: the Stanford Cars replication;
- section 15: interpretability (purity, explanation size, meaning kept).

## Repository layout

```
codebook_train/   official ProtoQuant code (CUB loader patched for the strict protocol)
docs/             phase0_protocol.md: frozen protocol, decision rules, change log
scripts/          run drivers, pruning methods, evaluation, summaries, capacity
splits/           fixed CUB and Cars validation splits
results/          data checks, baseline, Phase 2–4 results, capacity (backbone2/: improved backbone, cars/: Stanford Cars)
slurm/            original cluster scripts (reference only)
```
