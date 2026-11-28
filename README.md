# Pruning and Recovering ProtoQuant: Accuracy, Training Cost, and Interpretability

MLR 512 course project. Code for the report of the same name.

ProtoQuant classifies an image by matching its patches to 3,000 learned codes (prototypes). A sparse linear head then
turns the code activations into class scores. We prune the head to K = 10 edges per class (about 0.5 % of its
edges) and compare ways of recovering the accuracy lost, both in accuracy and training cost and in how
interpretable the recovered model stays.

Built on the official ProtoQuant code ([gmum/protoquant](https://github.com/gmum/protoquant), commit `b80d7e9`;
paper [arXiv 2602.06592](https://arxiv.org/abs/2602.06592)), which is in [`codebook_train/`](codebook_train/).
Datasets: CUB-200-2011 and Stanford Cars. Backbone: ConvNeXt-Tiny (frozen). Hardware: one NVIDIA GPU.

**Contents:** [Methods](#methods-in-the-report) · [Results](#main-results) · [Setup](#setup) ·
[Demo](#demo-inference-and-training-on-the-trained-models) · [Reproducing the report](#reproducing-the-report) ·
[Where the code is](#where-the-code-is) · [Layout](#repository-layout)

## Methods in the report

All pruned models keep K = 10 edges per class. Names in the code are given so the result files can be read.

| Report name | What it is | Name in code and result files |
|---|---|---|
| Dense | ProtoQuant before pruning, 3,000 codes | `dense` |
| B1 | Magnitude pruning: keep the 10 largest head weights per class (ProtoQuant's own rule), no recovery | `B1_K10` (one-shot) |
| P1 | Activation-aware pruning: keep the 10 largest W × mean activation per class, no recovery | `P1_K10` (one-shot) |
| Random | 10 random edges per class, no recovery | `random_K10` |
| **B1-IR** (image recovery) | B1 mask, unused codes removed, codes + head retrained on augmented images for 145 epochs (ProtoQuant's "Pruned & Tuned" recipe, run longer) | `B2_K10 … 145 epochs`, `BUD_B1_K10_E145`, `budget/B1_K10_E145/` |
| **B1-CFR** (cached-feature recovery) | B1 mask, unused codes removed, codes + head trained for 145 epochs on backbone features cached once | `B1+phys+rtCH_K10` |
| B1 + cached H (ablation) | As B1-CFR but only the head is trained | `B1+phys+rtH_K10` |
| P1 + cached CH (ablation) | As B1-CFR with the P1 mask | `P1+phys+rtCH_K10` |

The shared recipe is AdamW, learning rate 0.05, weight decay 5e-4, batch 1,024, a 5-epoch warm-up then a cosine
schedule, and label smoothing 0.1. The mask is reapplied after every step, and the best epoch is chosen on a held-out
validation split. B1-IR also uses MixUp (α = 0.8) and CutMix (α = 1.0), with switch probability 0.5.

## Main results

Test top-1 accuracy (%), mean ± SD over seeds 40–42. CUB uses the improved backbone.

| Model | CUB | Cars | Fitting time per model (CUB) |
|---|---|---|---|
| Dense | 85.65 ± 0.14 | 88.99 ± 0.06 | – |
| Random | 4.93 ± 0.39 | 5.25 ± 1.49 | – |
| B1 | 77.06 ± 0.18 | 72.73 ± 0.47 | – |
| P1 | 79.52 ± 0.16 | 75.33 ± 0.24 | – |
| B1 + cached H | 80.46 ± 0.51 | 79.19 ± 0.60 | 8.9 s |
| P1 + cached CH | 83.99 ± 0.27 | 86.62 ± 0.19 | 16.6 s |
| **B1-CFR** | **84.64 ± 0.15** | **86.85 ± 0.25** | **14.4 s** |
| **B1-IR** | **85.47 ± 0.42** | **88.49 ± 0.50** | ≈ 41 min (extrapolated) |

- **B1-IR is within 1 point of dense on both datasets.** On CUB the gap is −0.18; on Cars it is −0.50, with an
  interval of [−0.96, −0.05].
- **B1-CFR recovers most of the lost accuracy in seconds instead of minutes.** It ends 0.8 points below B1-IR on
  CUB and 1.6 below on Cars.
- **The ablations show what matters for recovery.** Training the codes as well as the head matters far more than
  the choice of mask: after cached CH training, P1 is no better than B1.
- **Interpretability:**
  - Pruning shrinks the explanation of a prediction from about 247 codes to about 6 on CUB, and from about 290 to
    about 6 on Cars. This is N₉₀: the codes making up 90 % of the predicted class's score.
  - Average part purity is kept by B1-IR (0.413 on CUB test vs 0.345 for dense), and is slightly lower for B1-CFR
    (0.384).

The fitting time covers recovery only: cache creation and mask preparation are not included. B1-IR's time is the
mean epoch time of the 30-epoch B1 retraining runs × 145, because the 145-epoch runs were not timed.

Result files:
- CUB: [`results/backbone2/phase2_results.md`](results/backbone2/phase2_results.md) and
  [`interpretability.md`](results/backbone2/interpretability.md);
- Cars: [`results/cars/phase2_results.md`](results/cars/phase2_results.md) and
  [`interpretability.md`](results/cars/interpretability.md);
- figures: [`results/figures/`](results/figures/).

Each `.md` has a `.json` beside it with the per-seed values. The verdict rules ("better", "no meaningful loss") were
fixed before any run; see [the protocol](docs/phase0_protocol.md), section 6.

## Setup

You need Linux with one NVIDIA GPU (CUDA 12.8 wheels by default; we used an RTX 5070 Ti, 16 GB) and Python 3.11.
Run every command from the repository root.

The environment, data and runs go in the parent folder: `../.venv`, `../data`, `../runs`, `../runs_bb2` and
`../runs_cars`. Override these with `PROJECT_ROOT`, `RUNS_ROOT`, `DATA_ROOT`, `CUB_PATH` and similar variables.

```bash
bash scripts/setup_env.sh                      # creates ../.venv (PyTorch 2.11, CUDA 12.8)
source ../.venv/bin/activate
bash scripts/prepare_cub.sh                    # downloads CUB-200-2011, crops to the boxes; checks must print PASS
CARS_SRC=/path/to/stanford_cars bash scripts/prepare_cars.sh   # Stanford Cars (optional)
```

All runs use our *strict* protocol, which the scripts set for you:
- 10 % of the official training images are held out for validation (fixed files in [`splits/`](splits/));
- the official test set is read once per final model.

The original CUB loader selected checkpoints on the test set; see
[What we changed](#what-we-changed-in-the-official-code).

Trained checkpoints are not in the repository: they are too large for GitHub. They come from the training commands
below.

## Demo: inference and training on the trained models

[`scripts/demo.sh`](scripts/demo.sh) runs every step on one seed (40) of the finished runs. Use `SEED=41` for another
seed and `PQ_DATASET=cars196` for Stanford Cars. Training commands write into a new folder under `../demo/`, so the
report's runs are never changed.

**Inference.** For each model the demo prints:
- the top-5 classes;
- N₉₀;
- the three codes that contribute most, with their share of the class score and where they match in the image.

```bash
bash scripts/demo.sh predict                                  # test image 0, all five models
bash scripts/demo.sh predict --test_index 0 12 345 --out ../demo   # also saves a PNG per image (top-3 patches boxed)
bash scripts/demo.sh predict --random 3 --models Dense B1-IR B1-CFR
bash scripts/demo.sh predict --image path/to/bird.jpg         # any image file
```

**Training.** Each command runs the same code and settings as the report:

```bash
bash scripts/demo.sh train-cfr        # B1 + cached H and B1-CFR, 145 epochs each on cached features: under a minute
bash scripts/demo.sh train-ir         # B1-IR on augmented images, 145 epochs (about 41 min; Ctrl-C stops it)
EPOCHS=3 bash scripts/demo.sh train-ir   # a short B1-IR run (a few minutes; 1 warm-up epoch)
bash scripts/demo.sh prune            # rebuild the one-shot B1, P1 and random masks from the dense model
```

`train-ir` prints the validation accuracy after every epoch. `train-cfr` prints the best epoch and the validation and
test accuracy of both models.

The demo commands read the test set again. This is fine for a demo, but the report's numbers come only from the runs
below, where each final model reads the test set once.

**Figures** (no GPU needed for the first command):

```bash
python scripts/make_figures.py        # recovery paths and fitting-time figures, from the result files
python scripts/make_prototype_figures.py --save_dir ../runs_bb2/interpretability --seed 40   # prototype examples
python scripts/make_prototype_figures.py --save_dir ../runs_bb2/interpretability --seed 40 --test_index 0 \
    --explain --backbone_ckpt "$BB" --runs_root ../runs_bb2 --cub_raw ../data/raw/CUB_200_2011   # one explanation
```

## Reproducing the report

Every driver can resume: a finished step leaves a `DONE` file and is skipped the next time. The GPU times are for an
RTX 5070 Ti.

| Step | Command | GPU time |
|---|---|---|
| 1. Backbone fine-tune | `bash scripts/run_backbone_finetune.sh` | < 1 h |
| 2. Improved backbone (lr sweep, continued fine-tune) | `bash scripts/run_backbone_sweep.sh` | ~2–3 h |
| 3. Dense ProtoQuant (3 seeds), feature cache, one-shot masks (B1, P1, Random), **B1-IR** | `bash scripts/run_new_backbone.sh` | ~8 h |
| 4. **B1-CFR**, B1 + cached H, P1 + cached CH | `CACHED_RETRAIN=1 bash scripts/run_repair.sh` | ~10–30 min |
| 5. Interpretability: purity, N₉₀, code change | `bash scripts/run_interpretability.sh` | ~15–30 min |
| 6. Stanford Cars, all of the above (3 seeds) | `SEEDS="40 41 42" RETRAIN_SEEDS="40 41 42" bash scripts/run_cars.sh` | ~7 h per seed |
| 7. Cars interpretability | `PQ_DATASET=cars196 bash scripts/run_interpretability.sh` | ~15–30 min |
| 8. Figures | see the Demo section | minutes |

Steps 3 and 4 write the result tables, `results/backbone2/phase2_results.{md,json}`. Step 2 picks the backbone and
writes its path to `../runs/backbone_sweep/SELECTED`, and later steps read it from there. The exact settings of each
step are fixed in [`docs/phase0_protocol.md`](docs/phase0_protocol.md): section 11 covers the improved backbone,
section 13 cached-feature recovery, section 14 Cars and section 15 interpretability.

The project also ran experiments that are not in the report:
- the main-backbone runs;
- iterative pruning;
- code-level pruning;
- least-squares and cross-entropy weight repair;
- capacity and latency.

Their commands and results are in [`docs/research_log.md`](docs/research_log.md).

## Where the code is

| Concept | File and function |
|---|---|
| ProtoQuant forward pass (cosine similarity to codes, softmax over codes with T = 0.1, max-pool, ReLU(W) head) | [`proto_quantnet.py`](codebook_train/src/models/proto_quantnet.py) `ProtoQuantNet.forward`, `_detect_prototypes` |
| Pooled code activations from cached features | [`pq_lib.py`](scripts/pq_lib.py) `scores_from_features`, `effective_weight` |
| Feature cache (fp16 768 × 7 × 7, un-augmented images) | [`cache_features.py`](scripts/cache_features.py) |
| B1 and P1 edge scores (W, W · μ) | [`phase2_lib.py`](scripts/phase2_lib.py) `edge_scores`, `class_means` |
| Keeping K edges per class; random mask | [`phase2_lib.py`](scripts/phase2_lib.py) `topk_per_class`, `random_per_class`, `build_masks` |
| Building and evaluating the one-shot masks | [`phase2_masking.py`](scripts/phase2_masking.py) `main` |
| Removing unused codes | [`proto_quantnet.py`](codebook_train/src/models/proto_quantnet.py) `prune_inactive_prototypes` |
| **B1-IR**: retraining on images (MixUp/CutMix, mask reapplied after each step) | [`phase2_retrain.py`](scripts/phase2_retrain.py) `main` (training loop) |
| **B1-CFR** and cached H: training on cached features | [`phase2_repair.py`](scripts/phase2_repair.py) `retrain_cached`, `cached_logits` |
| Paired bootstrap and decision rule | [`phase2_summary.py`](scripts/phase2_summary.py) `bootstrap`, `verdict` |
| N₉₀ and threshold-based explanation sizes | [`interpretability.py`](scripts/interpretability.py) `sizes` |
| Part purity (checked against the authors' code) | [`interpretability.py`](scripts/interpretability.py) `top_patches`, `purity`, `check_authors` |
| Top-patch retention and code cosine | [`interpretability.py`](scripts/interpretability.py) `main` (the dense-model comparison) |
| Loading every trained model of a seed | [`interpretability.py`](scripts/interpretability.py) `collect_models` |
| Single-image inference and explanation | [`predict.py`](scripts/predict.py) `explain` |
| Report figures | [`make_figures.py`](scripts/make_figures.py), [`make_prototype_figures.py`](scripts/make_prototype_figures.py) |

## What we changed in the official code

- **Held-out validation.** The original CUB loader selects checkpoints and prunes codes on the *test* set. With
  `PQ_CUB_PROTOCOL=strict`, the default in every script, the model trains on 5,394 images and is validated on a fixed
  600-image split. The test set (5,794 images) is read once per final model.
- **Validation keeps the last batch.** With 600 validation images and batch 1,024, `drop_last=True` would leave no
  validation batch.
- **Stanford Cars.** `scripts/prepare_cars.py` converts the download to the CUB layout, and `PQ_DATASET=cars196`
  switches every script.
- **No SLURM or wandb.** We use plain single-GPU shell scripts in [`scripts/`](scripts/). The original `slurm/`
  folder is kept for reference.

## Repository layout

```
codebook_train/   official ProtoQuant code (CUB loader patched for the strict protocol)
scripts/          our run drivers, pruning and recovery, evaluation, interpretability, figures, demo
docs/             phase0_protocol.md (frozen protocol, decision rules, change log), research_log.md (all experiments)
splits/           fixed validation splits for CUB and Cars
results/          result tables (.md) and per-seed values (.json); backbone2/ = CUB report runs, cars/ = Cars
slurm/            original cluster scripts (reference only)
```

## Acknowledgement and license

The model and training code in `codebook_train/` are by the ProtoQuant authors
([gmum/protoquant](https://github.com/gmum/protoquant)). Their repository has no license file, so it is included here
for course assessment only. Ask the authors before reusing or redistributing it.
