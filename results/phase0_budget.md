# Phase 0 timing and GPU-hour budget

Run on the team GPU on 2026-11-16 with
`S1_BATCHES=1024 S2_BATCHES="1024 512" bash scripts/phase0_timing.sh` (existing backbone checkpoint,
4 train / 2 val data-loader workers).

```
[GPU-CHECK] torch=2.11.0+cu128 cuda_build=12.8 available=True
[GPU-CHECK] device_0=NVIDIA GeForce RTX 5070 Ti memory=15.9GiB
```

| Stage | Batch | s/epoch | Peak GPU mem (GiB) | Status |
|---|---|---|---|---|
| stage1_K4096 | 1024 | 63.3 | 15.48 | ok |
| stage1_K8192 | 1024 | 69.4 | 15.52 | ok |
| stage2_K4096 | 1024 | 75.1 | 15.53 | ok |
| backbone (earlier run) | 128 | — | — | ok; batch 256 out of memory; val top-1 85.5 % |

Budget (x1.2 overhead), using the largest batch that fit per stage:

| Block | Runs | Epochs | s/epoch | GPU-h |
|---|---|---|---|---|
| Backbone fine-tune (strict, once) | 1 | 30 | done | — |
| Paper-protocol sanity: Stage 1 @4096 (1 seed) | 1 | 30 | 63.3 (bs 1024) | 0.6 |
| Paper-protocol sanity: Stage 2 (1 seed) | 1 | 30 | 75.1 (bs 1024) | 0.8 |
| Strict baseline: Stage 1 @4096 (3 seeds) | 3 | 30 | 63.3 (bs 1024) | 1.9 |
| Strict baseline: Stage 2 after B0 (3 seeds) | 3 | 30 | 75.1 (bs 1024) | 2.3 |
| B2 retrain, K=5/10/20 x 3 seeds | 9 | 30 | 75.1 (bs 1024) | 6.8 |
| Retrain at K=10: P1, P3, random, P1-contrast x 3 seeds | 12 | 30 | 75.1 (bs 1024) | 9.0 |
| One-shot vs iterative, best method K=10 (~5 rounds x 3 seeds) | 15 | 30 | 75.1 (bs 1024) | 11.3 |
| Optional: Stage 1 @8192 (codebook-size track, 1 seed) | 1 | 30 | 69.4 (bs 1024) | 0.7 |

**Total ≈ 33.3 GPU-hours** (measured blocks only). Cheap evaluations on cached features (P1/P3 scoring,
masking, Tests 1-4) are not included.

Notes: peak memory is from `nvidia-smi` and includes PyTorch's reserved cache, so the real need is likely
lower (an earlier Stage 1 run read 12.9 GiB). Epochs are slower than an earlier 8-worker run (~38 s for
Stage 1), probably because of data loading with 4 workers; this affects time only.
