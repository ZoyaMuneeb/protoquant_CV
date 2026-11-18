# Capacity metrics

Device: NVIDIA GeForce RTX 5070 Ti. Latency: median of 50 batches of 128 at 224x224, fp32, after 10 warm-up batches; head = backbone features in, logits out; full = images in. Peak GPU memory: full-model forward at that batch size. Explanation size: codes needed for 90 % of the predicted class's evidence, test set (mean / median). One-shot masks keep the full codebook, so their latency and memory equal the dense model's (marked =dense). Speed-up = dense latency / model latency, same seed. Mean ± sd over seeds.

## Dense head (no pruning)

| Model | Kind | Codebook entries | Codebook MB | Edges | Head ms | Head speed-up | Full ms | Full speed-up | Peak GPU GB | Expl. size mean | Expl. size median | Training GPU-min |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| dense | dense | 3,000 ± 0 | 8.79 ± 0.00 | 447,430 ± 13,892 | 3.62 ± 0.01 | 1.00 ± 0.00 | 93.84 ± 0.36 | 1.00 ± 0.00 | 1.65 ± 0.00 | 308.2 ± 14.2 | 298.7 ± 12.5 | 0.0 ± 0.0 |

## K = 5 edges per class

| Model | Kind | Codebook entries | Codebook MB | Edges | Head ms | Head speed-up | Full ms | Full speed-up | Peak GPU GB | Expl. size mean | Expl. size median | Training GPU-min |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B1 | one-shot | 3,000 ± 0 | 8.79 ± 0.00 | 1,000 ± 0 | =dense | =dense | =dense | =dense | =dense | 3.8 ± 0.0 | 4.0 ± 0.0 | 0.0 ± 0.0 |
| P1 | one-shot | 3,000 ± 0 | 8.79 ± 0.00 | 1,000 ± 0 | =dense | =dense | =dense | =dense | =dense | 4.3 ± 0.0 | 4.0 ± 0.0 | 0.0 ± 0.0 |
| P1contrast | one-shot | 3,000 ± 0 | 8.79 ± 0.00 | 1,000 ± 0 | =dense | =dense | =dense | =dense | =dense | 4.3 ± 0.0 | 4.0 ± 0.0 | 0.0 ± 0.0 |
| P3 | one-shot | 3,000 ± 0 | 8.79 ± 0.00 | 1,000 ± 0 | =dense | =dense | =dense | =dense | =dense | 5.1 ± 0.0 | 5.0 ± 0.0 | 0.0 ± 0.0 |
| random | one-shot | 3,000 ± 0 | 8.79 ± 0.00 | 1,000 ± 0 | =dense | =dense | =dense | =dense | =dense | 1.3 ± 0.0 | 1.0 ± 0.0 | 0.0 ± 0.0 |
| B2 | retrained | 788 ± 17 | 2.31 ± 0.05 | 1,000 ± 0 | 0.90 ± 0.01 | 4.01 ± 0.04 | 91.34 ± 0.30 | 1.03 ± 0.00 | 1.64 ± 0.00 | 4.1 ± 0.0 | 4.0 ± 0.0 | 6.0 ± 0.2 |
| P1 | retrained | 857 ± 4 | 2.51 ± 0.01 | 1,000 ± 0 | 0.99 ± 0.05 | 3.66 ± 0.18 | 91.48 ± 0.14 | 1.03 ± 0.00 | 1.64 ± 0.00 | 4.3 ± 0.0 | 4.0 ± 0.0 | 5.3 ± 0.2 |

## K = 10 edges per class

| Model | Kind | Codebook entries | Codebook MB | Edges | Head ms | Head speed-up | Full ms | Full speed-up | Peak GPU GB | Expl. size mean | Expl. size median | Training GPU-min |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B1 | one-shot | 3,000 ± 0 | 8.79 ± 0.00 | 2,000 ± 0 | =dense | =dense | =dense | =dense | =dense | 6.8 ± 0.1 | 7.0 ± 0.0 | 0.0 ± 0.0 |
| P1 | one-shot | 3,000 ± 0 | 8.79 ± 0.00 | 2,000 ± 0 | =dense | =dense | =dense | =dense | =dense | 7.5 ± 0.0 | 8.0 ± 0.0 | 0.0 ± 0.0 |
| P1contrast | one-shot | 3,000 ± 0 | 8.79 ± 0.00 | 2,000 ± 0 | =dense | =dense | =dense | =dense | =dense | 7.5 ± 0.0 | 8.0 ± 0.0 | 0.0 ± 0.0 |
| P3 | one-shot | 3,000 ± 0 | 8.79 ± 0.00 | 2,000 ± 0 | =dense | =dense | =dense | =dense | =dense | 8.4 ± 0.1 | 8.0 ± 0.0 | 0.0 ± 0.0 |
| random | one-shot | 3,000 ± 0 | 8.79 ± 0.00 | 2,000 ± 0 | =dense | =dense | =dense | =dense | =dense | 1.6 ± 0.1 | 1.0 ± 0.0 | 0.0 ± 0.0 |
| B2 | retrained | 1,305 ± 17 | 3.82 ± 0.05 | 2,000 ± 0 | 1.50 ± 0.02 | 2.42 ± 0.03 | 91.78 ± 0.39 | 1.02 ± 0.00 | 1.65 ± 0.00 | 7.0 ± 0.1 | 7.0 ± 0.0 | 6.1 ± 0.1 |
| P1 | retrained | 1,456 ± 21 | 4.27 ± 0.06 | 2,000 ± 0 | 1.72 ± 0.04 | 2.11 ± 0.05 | 92.16 ± 0.16 | 1.02 ± 0.00 | 1.65 ± 0.00 | 7.6 ± 0.0 | 8.0 ± 0.0 | 6.0 ± 0.2 |
| P1contrast | retrained | 1,463 ± 20 | 4.29 ± 0.06 | 2,000 ± 0 | 1.72 ± 0.03 | 2.11 ± 0.04 | 92.20 ± 0.10 | 1.02 ± 0.00 | 1.65 ± 0.00 | 7.6 ± 0.0 | 8.0 ± 0.0 | 5.8 ± 0.3 |
| P3 | retrained | 1,452 ± 2 | 4.25 ± 0.01 | 2,000 ± 0 | 1.70 ± 0.01 | 2.13 ± 0.02 | 92.24 ± 0.12 | 1.02 ± 0.00 | 1.65 ± 0.00 | 7.7 ± 0.1 | 8.0 ± 0.0 | 6.0 ± 0.2 |
| random | retrained | 1,440 ± 8 | 4.22 ± 0.02 | 1,161 ± 24 | 1.70 ± 0.03 | 2.13 ± 0.03 | 92.20 ± 0.15 | 1.02 ± 0.00 | 1.65 ± 0.00 | 2.5 ± 0.0 | 2.7 ± 0.6 | 6.0 ± 0.2 |
| B2 145 epochs | control | 1,305 ± 17 | 3.82 ± 0.05 | 2,000 ± 0 | 1.50 ± 0.02 | 2.41 ± 0.04 | 92.03 ± 0.05 | 1.02 ± 0.00 | 1.65 ± 0.00 | 7.0 ± 0.1 | 7.0 ± 0.0 | 27.2 ± 4.9 |
| B1 iterative | iterative | 1,564 ± 20 | 4.58 ± 0.06 | 2,000 ± 0 | 1.87 ± 0.02 | 1.93 ± 0.02 | 92.44 ± 0.09 | 1.02 ± 0.00 | 1.65 ± 0.00 | 7.1 ± 0.1 | 7.0 ± 0.0 | 26.7 ± 1.4 |
| P1 iterative | iterative | 1,650 ± 8 | 4.83 ± 0.02 | 2,000 ± 0 | 1.96 ± 0.01 | 1.85 ± 0.01 | 92.52 ± 0.05 | 1.01 ± 0.00 | 1.65 ± 0.00 | 7.7 ± 0.0 | 8.0 ± 0.0 | 44.1 ± 3.3 |

## K = 20 edges per class

| Model | Kind | Codebook entries | Codebook MB | Edges | Head ms | Head speed-up | Full ms | Full speed-up | Peak GPU GB | Expl. size mean | Expl. size median | Training GPU-min |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B1 | one-shot | 3,000 ± 0 | 8.79 ± 0.00 | 4,000 ± 0 | =dense | =dense | =dense | =dense | =dense | 12.1 ± 0.2 | 12.3 ± 0.6 | 0.0 ± 0.0 |
| P1 | one-shot | 3,000 ± 0 | 8.79 ± 0.00 | 4,000 ± 0 | =dense | =dense | =dense | =dense | =dense | 13.6 ± 0.0 | 14.0 ± 0.0 | 0.0 ± 0.0 |
| P1contrast | one-shot | 3,000 ± 0 | 8.79 ± 0.00 | 4,000 ± 0 | =dense | =dense | =dense | =dense | =dense | 13.6 ± 0.1 | 14.0 ± 0.0 | 0.0 ± 0.0 |
| P3 | one-shot | 3,000 ± 0 | 8.79 ± 0.00 | 4,000 ± 0 | =dense | =dense | =dense | =dense | =dense | 14.4 ± 0.1 | 14.7 ± 0.6 | 0.0 ± 0.0 |
| random | one-shot | 3,000 ± 0 | 8.79 ± 0.00 | 4,000 ± 0 | =dense | =dense | =dense | =dense | =dense | 2.4 ± 0.1 | 2.0 ± 0.0 | 0.0 ± 0.0 |
| B2 | retrained | 1,928 ± 23 | 5.65 ± 0.07 | 4,000 ± 0 | 2.38 ± 0.01 | 1.52 ± 0.01 | 92.66 ± 0.24 | 1.01 ± 0.00 | 1.65 ± 0.00 | 12.4 ± 0.2 | 12.7 ± 0.6 | 6.0 ± 0.2 |
| P1 | retrained | 2,116 ± 26 | 6.20 ± 0.07 | 4,000 ± 0 | 2.56 ± 0.03 | 1.41 ± 0.02 | 92.97 ± 0.23 | 1.01 ± 0.00 | 1.65 ± 0.00 | 13.7 ± 0.1 | 14.0 ± 0.0 | 5.3 ± 0.0 |
