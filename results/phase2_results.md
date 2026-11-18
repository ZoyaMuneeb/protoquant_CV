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
| P3nomin (classes with no edge: 11 ± 1) | 61.28 ± 0.54 | 60.36 ± 0.94 | -6.28 | [-7.09, -5.48] | -7.15, -6.18, -5.51 | loss not excluded | 1,000 ± 0 | 865 ± 2 |

### K = 10 edges per class (N = 2,000)

| Method | Val top-1 | Test top-1 | Δ vs B1 | 95 % CI | Seeds (Δ) | Verdict | Edges | Active codes |
|---|---|---|---|---|---|---|---|---|
| B1 | 70.83 ± 1.42 | 71.78 ± 1.55 | ref |  |  |  | 2,000 ± 0 | 1,305 ± 17 |
| random | 4.39 ± 0.69 | 4.74 ± 0.34 | -67.03 | [-68.07, -65.96] | -67.45, -68.00, -65.65 | loss not excluded | 2,000 ± 0 | 1,440 ± 8 |
| P1 | 74.89 ± 0.82 | 75.35 ± 0.47 | +3.57 | [+2.99, +4.16] | +3.07, +2.78, +4.87 | **better** | 2,000 ± 0 | 1,456 ± 21 |
| P1contrast | 75.56 ± 0.86 | 75.42 ± 0.34 | +3.64 | [+3.05, +4.22] | +3.14, +2.71, +5.07 | **better** | 2,000 ± 0 | 1,463 ± 20 |
| P3 (min-1 classes: 2 ± 1) | 68.72 ± 0.79 | 68.25 ± 1.15 | -3.53 | [-4.21, -2.84] | -3.99, -3.66, -2.93 | loss not excluded | 2,000 ± 0 | 1,452 ± 2 |
| P3nomin (classes with no edge: 0 ± 1) | 68.78 ± 0.82 | 68.24 ± 1.15 | -3.53 | [-4.22, -2.85] | -4.00, -3.66, -2.93 | loss not excluded | 2,000 ± 0 | 1,452 ± 2 |

### K = 20 edges per class (N = 4,000)

| Method | Val top-1 | Test top-1 | Δ vs B1 | 95 % CI | Seeds (Δ) | Verdict | Edges | Active codes |
|---|---|---|---|---|---|---|---|---|
| B1 | 75.94 ± 0.92 | 75.60 ± 0.55 | ref |  |  |  | 4,000 ± 0 | 1,928 ± 23 |
| random | 7.61 ± 1.25 | 8.73 ± 0.55 | -66.87 | [-67.92, -65.82] | -66.67, -67.07, -66.86 | loss not excluded | 4,000 ± 0 | 2,205 ± 31 |
| P1 | 78.17 ± 0.76 | 77.63 ± 0.19 | +2.03 | [+1.53, +2.54] | +1.86, +1.78, +2.45 | **better** | 4,000 ± 0 | 2,116 ± 26 |
| P1contrast | 78.56 ± 0.98 | 77.90 ± 0.51 | +2.30 | [+1.81, +2.81] | +1.88, +2.47, +2.55 | **better** | 4,000 ± 0 | 2,149 ± 30 |
| P3 (min-1 classes: 0 ± 0) | 74.11 ± 1.07 | 74.04 ± 0.65 | -1.56 | [-2.16, -0.97] | -1.55, -1.47, -1.67 | loss not excluded | 4,000 ± 0 | 2,107 ± 26 |
| P3nomin (classes with no edge: 0 ± 0) | 74.11 ± 1.07 | 74.04 ± 0.65 | -1.56 | [-2.16, -0.97] | -1.55, -1.47, -1.67 | loss not excluded | 4,000 ± 0 | 2,107 ± 26 |

## With retraining (codes no class uses are removed, then 30 epochs, mask held fixed)

B2 = B1 mask retrained (the paper's Pruned & Tuned). Methods are compared with B2 at the same K.

| Model | Codes | Val before → best | Test top-1 | Δ vs B2 | 95 % CI | Seeds (Δ) | Verdict |
|---|---|---|---|---|---|---|---|
| B2_K5 | 788 ± 17 | 63.94 ± 2.34 → 80.72 ± 0.35 | 81.01 ± 0.56 | ref |  |  |  |
| P1_K5 | 857 ± 4 | 71.44 ± 0.38 → 81.44 ± 1.11 | 81.01 ± 0.32 | +0.00 | [-0.46, +0.45] | -0.29, -0.17, +0.47 | no meaningful loss |
| B2_K10 | 1,305 ± 17 | 70.17 ± 1.32 → 83.00 ± 0.76 | 81.94 ± 0.64 | ref |  |  |  |
| P1_K10 | 1,456 ± 21 | 74.22 ± 0.98 → 83.28 ± 0.84 | 82.25 ± 0.24 | +0.31 | [-0.10, +0.72] | +0.91, -0.10, +0.12 | no meaningful loss |
| P1contrast_K10 | 1,463 ± 20 | 74.39 ± 0.92 → 83.22 ± 0.25 | 82.27 ± 0.21 | +0.33 | [-0.07, +0.73] | +0.81, -0.43, +0.62 | no meaningful loss |
| P3_K10 | 1,452 ± 2 | 67.17 ± 0.73 → 81.72 ± 0.84 | 80.72 ± 0.43 | -1.23 | [-1.72, -0.74] | -0.12, -2.21, -1.35 | loss not excluded |
| random_K10 | 1,440 ± 8 | 5.17 ± 0.60 → 68.56 ± 1.00 | 68.31 ± 0.97 | -13.63 | [-14.41, -12.88] | -13.44, -14.98, -12.48 | loss not excluded |
| B2_K20 | 1,928 ± 23 | 74.83 ± 1.04 → 85.33 ± 0.60 | 83.47 ± 0.37 | ref |  |  |  |
| P1_K20 | 2,116 ± 26 | 78.22 ± 0.10 → 84.78 ± 0.35 | 83.10 ± 0.40 | -0.37 | [-0.73, -0.01] | -0.24, -0.91, +0.05 | no meaningful loss |

## Iterative pruning (Phase 4)

20 % of each class's edges removed per round, 5 epochs of retraining and rescoring between rounds, full codebook during the rounds; then unused codes removed and 30 epochs (the B2 recipe). Same rules as above.

| Model | Rounds | Total epochs | Codes | Best val | Test top-1 |
|---|---|---|---|---|---|
| B1_K10 iterative | 24 ± 0 | 145 ± 0 | 1,564 ± 20 | 84.56 ± 0.42 | 83.89 ± 0.47 |
| P1_K10 iterative | 24 ± 0 | 145 ± 0 | 1,650 ± 8 | 85.72 ± 0.25 | 84.47 ± 0.34 |
| B2_K10 one-shot, 145 ± 0 epochs (control) | - | 145 ± 0 | 1,305 ± 17 | 86.06 ± 0.92 | 84.08 ± 0.54 |

| Comparison | Question | Δ | 95 % CI | Seeds (Δ) | Verdict |
|---|---|---|---|---|---|
| iterative B1 vs one-shot B2 (retrained), K = 10 | does gradual pruning help? | +1.95 | [+1.48, +2.42] | +2.14, +1.79, +1.92 | **better** |
| iterative P1 vs one-shot P1 (retrained), K = 10 | does gradual pruning help? | +2.21 | [+1.75, +2.70] | +2.21, +2.31, +2.12 | **better** |
| iterative P1 vs iterative B1, K = 10 | does the score matter under repeated pruning? | +0.58 | [+0.20, +0.95] | +0.98, +0.41, +0.33 | **better** |
| iterative B1 vs one-shot B2 at 145 epochs, K = 10 | gradual pruning, at equal training? | -0.18 | [-0.58, +0.20] | -1.10, +0.24, +0.31 | no meaningful loss |
| iterative P1 vs one-shot B2 at 145 epochs, K = 10 | does iterative P1 beat the simplest strong baseline? | +0.39 | [+0.00, +0.78] | -0.12, +0.66, +0.64 | no meaningful loss |
| one-shot B2 at 145 vs 30 epochs, K = 10 | does longer training alone help? | +2.13 | [+1.71, +2.55] | +3.24, +1.55, +1.61 | **better** |
| one-shot B2 at 145 epochs (K = 10) vs dense head | is dense accuracy kept? | -0.36 | [-0.95, +0.23] | -0.17, -0.31, -0.59 | no meaningful loss |
| iterative B1 (K = 10) vs dense head | is dense accuracy kept? | -0.54 | [-1.16, +0.07] | -1.28, -0.07, -0.28 | loss not excluded |
| iterative P1 (K = 10) vs dense head | is dense accuracy kept? | +0.03 | [-0.59, +0.65] | -0.29, +0.35, +0.05 | no meaningful loss |

## Ablations (Phase 4)

Same rules as above; the reference is the second method named.

| Comparison | Question | Δ | 95 % CI | Seeds (Δ) | Verdict |
|---|---|---|---|---|---|
| P1-contrast vs P1, one-shot, K = 5 | does contrasting against other classes help? | +0.18 | [+0.01, +0.36] | +0.02, +0.31, +0.22 | **better** |
| P1-contrast vs P1, one-shot, K = 10 | does contrasting against other classes help? | +0.07 | [-0.12, +0.26] | +0.07, -0.07, +0.21 | no meaningful loss |
| P1-contrast vs P1, one-shot, K = 20 | does contrasting against other classes help? | +0.27 | [+0.05, +0.50] | +0.02, +0.69, +0.10 | **better** |
| P1-contrast vs P1, retrained, K = 10 | does contrasting against other classes help? | +0.02 | [-0.21, +0.27] | -0.10, -0.33, +0.50 | no meaningful loss |
| P3 without vs with the one-edge minimum, K = 5 | does the minimum help P3? | -0.40 | [-0.61, -0.20] | -0.66, +0.14, -0.69 | no meaningful loss |
| P3 without vs with the one-edge minimum, K = 10 | does the minimum help P3? | -0.01 | [-0.03, +0.02] | -0.02, +0.00, +0.00 | no meaningful loss |
| P3 without vs with the one-edge minimum, K = 20 | does the minimum help P3? | +0.00 | [+0.00, +0.00] | +0.00, +0.00, +0.00 | no meaningful loss |

## Deletion-effect control

100 class–code edges (a third each from low, medium and high P1 scores) removed one at a time from the dense head; damage = change in validation cross-entropy. Spearman ρ between damage and each score (higher = the score predicts real damage better).

| Seed | Magnitude (B1) | P1 | P1-contrast |
|---|---|---|---|
| 40 | -0.208 | -0.236 | 0.197 |
| 41 | -0.045 | 0.038 | 0.251 |
| 42 | 0.007 | -0.017 | 0.285 |
| mean | -0.08 ± 0.11 | -0.07 ± 0.14 | 0.24 ± 0.04 |
