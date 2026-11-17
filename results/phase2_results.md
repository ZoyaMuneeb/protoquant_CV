# Phase 2 results

Seeds: 40, 41, 42. Test top-1 is mean ± sd over seeds. Δ = method − reference (points), with the 95 % paired-bootstrap interval over test images (10,000 resamples, correctness averaged over seeds). Verdict rules fixed in Phase 0: better = interval above 0 and same sign in every seed; no meaningful loss = lower bound above −1; otherwise a loss of 1 point or more is not excluded.

Dense head (no pruning): test 84.43 ± 0.33 %, 447,430 nonzero edges.

## One-shot masking (no retraining)

### K = 5 edges per class (N = 1,000)

| Method | Val top-1 | Test top-1 | Δ vs B1 | 95 % CI | Seeds (Δ) | Verdict | Edges | Active codes |
|---|---|---|---|---|---|---|---|---|
| B1 | 65.33 ± 0.93 | 66.64 ± 1.26 | ref |  |  |  | 1,000 ± 0 | 788 ± 17 |
| random | 3.72 ± 0.95 | 3.77 ± 0.48 | -62.86 | [-63.92, -61.81] | -62.93, -63.67, -62.00 | loss not excluded | 1,000 ± 0 | 850 ± 8 |
| P1 | 72.39 ± 0.35 | 72.48 ± 0.42 | +5.85 | [+5.19, +6.50] | +5.37, +5.30, +6.87 | **better** | 1,000 ± 0 | 857 ± 4 |
| P1contrast | 72.50 ± 0.29 | 72.67 ± 0.49 | +6.03 | [+5.38, +6.67] | +5.38, +5.61, +7.09 | **better** | 1,000 ± 0 | 858 ± 6 |
| P3 (min-1 classes: 28 ± 3) | 61.17 ± 0.29 | 60.76 ± 0.48 | -5.87 | [-6.67, -5.09] | -6.49, -6.32, -4.82 | loss not excluded | 1,000 ± 0 | 868 ± 4 |

### K = 10 edges per class (N = 2,000)

| Method | Val top-1 | Test top-1 | Δ vs B1 | 95 % CI | Seeds (Δ) | Verdict | Edges | Active codes |
|---|---|---|---|---|---|---|---|---|
| B1 | 70.83 ± 1.42 | 71.78 ± 1.55 | ref |  |  |  | 2,000 ± 0 | 1,305 ± 17 |
| random | 4.39 ± 0.69 | 4.74 ± 0.34 | -67.03 | [-68.07, -65.96] | -67.45, -68.00, -65.65 | loss not excluded | 2,000 ± 0 | 1,440 ± 8 |
| P1 | 74.89 ± 0.82 | 75.35 ± 0.47 | +3.57 | [+2.99, +4.16] | +3.07, +2.78, +4.87 | **better** | 2,000 ± 0 | 1,456 ± 21 |
| P1contrast | 75.56 ± 0.86 | 75.42 ± 0.34 | +3.64 | [+3.05, +4.22] | +3.14, +2.71, +5.07 | **better** | 2,000 ± 0 | 1,463 ± 20 |
| P3 (min-1 classes: 2 ± 1) | 68.72 ± 0.79 | 68.25 ± 1.15 | -3.53 | [-4.21, -2.84] | -3.99, -3.66, -2.93 | loss not excluded | 2,000 ± 0 | 1,452 ± 2 |

### K = 20 edges per class (N = 4,000)

| Method | Val top-1 | Test top-1 | Δ vs B1 | 95 % CI | Seeds (Δ) | Verdict | Edges | Active codes |
|---|---|---|---|---|---|---|---|---|
| B1 | 75.94 ± 0.92 | 75.60 ± 0.55 | ref |  |  |  | 4,000 ± 0 | 1,928 ± 23 |
| random | 7.61 ± 1.25 | 8.73 ± 0.55 | -66.87 | [-67.92, -65.82] | -66.67, -67.07, -66.86 | loss not excluded | 4,000 ± 0 | 2,205 ± 31 |
| P1 | 78.17 ± 0.76 | 77.63 ± 0.19 | +2.03 | [+1.53, +2.54] | +1.86, +1.78, +2.45 | **better** | 4,000 ± 0 | 2,116 ± 26 |
| P1contrast | 78.56 ± 0.98 | 77.90 ± 0.51 | +2.30 | [+1.81, +2.81] | +1.88, +2.47, +2.55 | **better** | 4,000 ± 0 | 2,149 ± 30 |
| P3 (min-1 classes: 0 ± 0) | 74.11 ± 1.07 | 74.04 ± 0.65 | -1.56 | [-2.16, -0.97] | -1.55, -1.47, -1.67 | loss not excluded | 4,000 ± 0 | 2,107 ± 26 |

## Deletion-effect control

100 class–code edges (a third each from low, medium and high P1 scores) removed one at a time from the dense head; damage = change in validation cross-entropy. Spearman ρ between damage and each score (higher = the score predicts real damage better).

| Seed | Magnitude (B1) | P1 | P1-contrast |
|---|---|---|---|
| 40 | -0.208 | -0.236 | 0.197 |
| 41 | -0.045 | 0.038 | 0.251 |
| 42 | 0.007 | -0.017 | 0.285 |
| mean | -0.08 ± 0.11 | -0.07 ± 0.14 | 0.24 ± 0.04 |
