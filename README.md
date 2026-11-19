# ProtoQuant pruning project (MLR 512)

**Question.** Can activation-aware or globally budgeted pruning beat ProtoQuant's original magnitude pruning,
and how small can the model get before it loses accuracy?

Built on the official ProtoQuant code ([gmum/protoquant](https://github.com/gmum/protoquant), commit `b80d7e9`;
paper [arXiv 2602.06592](https://arxiv.org/abs/2602.06592)). CUB-200-2011, ConvNeXt-Tiny, one NVIDIA GPU.

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
| – | Loss-aware physical pruning pilot (this branch) | Code ready, not run |
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
   (+0.14, CI [−0.28, +0.56]). The same pattern holds for edges (findings 1–2) and for codes.
7. **The global budget (P3) and the "contrast" variant don't help.** P3 starves classes at small K.
   P1-contrast ≈ P1 in accuracy, but it is the only score that predicts single-edge deletion damage.
8. **Pruning shrinks explanations 40× and the codebook by half, but not the model.** The backbone dominates
   latency and memory, so the full model is only about 2 % faster.

Verdicts use the rules fixed before any run ([protocol](docs/phase0_protocol.md), section 6):
- **better**: the 95 % paired-bootstrap interval is above 0 and every seed agrees;
- **no meaningful loss**: the lower bound is above −1 point;
- **loss not excluded**: otherwise.

Runs added after earlier results were seen are labelled post hoc in the protocol's change log.

## Results

Test top-1 %, strict protocol, mean of seeds 40–42.

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

### Ablations and controls

| Question | Answer |
|---|---|
| Does contrasting against other classes help (P1-contrast vs P1)? | Slightly: +0.07 to +0.27 one-shot (significant at some K), never more |
| Does P3's one-edge-per-class minimum matter? | −0.4 at K = 5 without it (11 classes get no edge); none at K ≥ 10 |
| Which score predicts real single-edge deletion damage (Spearman ρ)? | P1-contrast 0.24 / 0.42 (main / improved backbone); magnitude and P1 ≈ 0 |
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
- improved backbone: `results/backbone2/phase2_results.md`, written on the GPU machine.

## What we changed in the official code

- **Held-out validation.** The original CUB loader selects checkpoints and prunes codes on the *test* set.
  `PQ_CUB_PROTOCOL=strict` (our default) trains on 5,394 images and validates on a fixed 600-image split
  ([`splits/`](splits/)). Test (5,794) is read once per final model, and a marker file next to the checkpoint
  enforces this. `paper` keeps the original behaviour.
- **Validation keeps the last batch.** With 600 validation images and batch 1,024, the original
  `drop_last=True` would leave no validation batch at all.
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

Improved-backbone runs write to `../runs_bb2/` and `results/backbone2/`, so the main results are untouched.
Each step's exact settings are fixed in [`docs/phase0_protocol.md`](docs/phase0_protocol.md):
- section 10: iterative pruning;
- section 11: the improved backbone;
- section 12: code-level pruning.

## Repository layout

```
codebook_train/   official ProtoQuant code (CUB loader patched for the strict protocol)
docs/             phase0_protocol.md: frozen protocol, decision rules, change log
scripts/          run drivers, pruning methods, evaluation, summaries, capacity
splits/           fixed CUB validation split
results/          data checks, baseline, Phase 2–4 results, capacity (backbone2/: improved backbone)
slurm/            original cluster scripts (reference only)
```

## Loss-aware physical pruning pilot (branch `pilot/physical-lossaware`)

**Idea.** Instead of ranking codes by a score and cutting the lowest, delete codes from the model one at a
time (or in small batches), each time picking the code whose removal raises the training loss least.
Every step measures the real loss change on the current, already-pruned model. No retraining, and test is
never read.

**Three methods compared** (seed 40, validation accuracy):

| Method | How it chooses the next code to delete |
|---|---|
| P1-only | Lowest summed P1 score (fixed ranking) |
| Greedy loss-aware | Among the 64 lowest-P1 codes, the one whose deletion hurts the loss least |
| Batched loss-aware | Same, but deletes up to 8 at once if their combined damage stays close to the sum of their individual damages; otherwise halves the batch |

**Two pilots:**
- Pilot 1 starts from the dense model: 3,000 → 2,840 codes.
- Pilot 2 starts from a compact 1,700-code model (top codes by P1): 1,700 → 1,540 codes.

Loss is measured on 400 training images (2 per class). Removed codes are physically deleted, and the
remaining codes keep all their classifier weights (no top-K edge mask).

**Run it** (on the GPU machine; full details, inputs and settings in
[the pilot guide](docs/physical_pruning_pilot.md)):

```bash
bash scripts/run_physical_pilot.sh --self-test   # synthetic tests, no data needed
bash scripts/run_physical_pilot.sh --smoke       # quick real-data check
bash scripts/run_physical_pilot.sh               # full pilot
```

Real-data runs need `BACKBONE_CKPT`, `PQ_CKPT`, `CACHE_DIR` and `CUB_PATH` set. Results go to
`results.md` in the printed output folder.
