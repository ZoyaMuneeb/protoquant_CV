# Phase 2 results

Seeds: 40, 41, 42. Test top-1 is mean ± sd over seeds. Δ = method − reference (points), with the 95 % paired-bootstrap interval over test images (10,000 resamples, correctness averaged over seeds). Verdict rules fixed in Phase 0: better = interval above 0 and same sign in every seed; no meaningful loss = lower bound above −1; otherwise a loss of 1 point or more is not excluded.

Dense head (no pruning): test 85.65 ± 0.14 %, 422,868 nonzero edges.

## One-shot masking (no retraining)

### K = 5 edges per class (N = 1,000)

| Method | Val top-1 | Test top-1 | Δ vs B1 | 95 % CI | Seeds (Δ) | Verdict | Edges | Active codes |
|---|---|---|---|---|---|---|---|---|
| B1 | 71.56 ± 1.17 | 73.66 ± 1.03 | ref |  |  |  | 1,000 ± 0 | 801 ± 20 |
| random | 4.11 ± 1.50 | 4.24 ± 1.67 | -69.42 | [-70.42, -68.39] | -69.49, -70.04, -68.73 | loss not excluded | 1,000 ± 0 | 849 ± 9 |
| P1 | 74.83 ± 0.60 | 78.09 ± 0.16 | +4.44 | [+3.85, +5.02] | +4.04, +5.56, +3.71 | **better** | 1,000 ± 0 | 924 ± 6 |
| P1contrast | 74.94 ± 0.63 | 78.24 ± 0.15 | +4.58 | [+3.99, +5.17] | +4.16, +5.78, +3.80 | **better** | 1,000 ± 0 | 926 ± 6 |
| P3 (min-1 classes: 18 ± 2) | 71.17 ± 1.17 | 72.51 ± 0.54 | -1.14 | [-1.78, -0.51] | -1.93, -0.02, -1.48 | loss not excluded | 1,000 ± 0 | 943 ± 3 |

### K = 10 edges per class (N = 2,000)

| Method | Val top-1 | Test top-1 | Δ vs B1 | 95 % CI | Seeds (Δ) | Verdict | Edges | Active codes |
|---|---|---|---|---|---|---|---|---|
| B1 | 73.83 ± 0.58 | 77.06 ± 0.18 | ref |  |  |  | 2,000 ± 0 | 1,304 ± 31 |
| random | 4.44 ± 1.02 | 4.93 ± 0.39 | -72.13 | [-73.11, -71.12] | -71.66, -72.01, -72.73 | loss not excluded | 2,000 ± 0 | 1,432 ± 8 |
| P1 | 77.56 ± 1.26 | 79.52 ± 0.16 | +2.46 | [+1.94, +2.99] | +2.54, +2.43, +2.42 | **better** | 2,000 ± 0 | 1,574 ± 17 |
| P1contrast | 77.44 ± 1.36 | 79.65 ± 0.40 | +2.59 | [+2.08, +3.10] | +2.74, +2.28, +2.74 | **better** | 2,000 ± 0 | 1,591 ± 14 |
| P3 (min-1 classes: 0 ± 1) | 75.83 ± 0.50 | 76.72 ± 0.77 | -0.35 | [-0.90, +0.22] | +0.09, -1.10, -0.02 | no meaningful loss | 2,000 ± 0 | 1,571 ± 10 |

### K = 20 edges per class (N = 4,000)

| Method | Val top-1 | Test top-1 | Δ vs B1 | 95 % CI | Seeds (Δ) | Verdict | Edges | Active codes |
|---|---|---|---|---|---|---|---|---|
| B1 | 77.28 ± 0.48 | 79.41 ± 0.69 | ref |  |  |  | 4,000 ± 0 | 1,907 ± 26 |
| random | 7.28 ± 2.43 | 7.71 ± 1.27 | -71.69 | [-72.66, -70.70] | -71.42, -72.44, -71.23 | loss not excluded | 4,000 ± 0 | 2,185 ± 12 |
| P1 | 78.44 ± 0.38 | 81.15 ± 0.24 | +1.74 | [+1.29, +2.20] | +1.83, +2.16, +1.24 | **better** | 4,000 ± 0 | 2,197 ± 16 |
| P1contrast | 78.06 ± 0.69 | 81.31 ± 0.24 | +1.90 | [+1.45, +2.35] | +1.97, +2.33, +1.40 | **better** | 4,000 ± 0 | 2,233 ± 14 |
| P3 (min-1 classes: 0 ± 0) | 77.39 ± 0.79 | 79.55 ± 0.70 | +0.14 | [-0.40, +0.68] | -0.16, +0.41, +0.16 | no meaningful loss | 4,000 ± 0 | 2,196 ± 10 |

## With retraining (codes no class uses are removed, then 30 epochs, mask held fixed)

B2 = B1 mask retrained (the paper's Pruned & Tuned). Methods are compared with B2 at the same K.

| Model | Codes | Val before → best | Test top-1 | Δ vs B2 | 95 % CI | Seeds (Δ) | Verdict |
|---|---|---|---|---|---|---|---|
| B2_K10 | 1,304 ± 31 | 73.22 ± 0.54 → 82.78 ± 0.67 | 84.36 ± 0.08 | ref |  |  |  |

## Iterative pruning and longer retraining (Phase 4)

Iterative: 20 % of each class's edges removed per round, 5 epochs of retraining and rescoring between rounds, full codebook during the rounds; then unused codes removed and 30 epochs (the B2 recipe). Control: the one-shot B1 mask retrained with the B2 recipe for as many epochs in total. Same rules as above.

| Model | Rounds | Total epochs | Codes | Best val | Test top-1 |
|---|---|---|---|---|---|
| B2_K10 one-shot, 145 ± 0 epochs (control) | - | 145 ± 0 | 1,304 ± 31 | 84.61 ± 0.69 | 85.47 ± 0.42 |
| CB1-M800_K10 code-level, 145 ± 0 epochs | - | 145 ± 0 | 484 ± 10 | 82.94 ± 0.25 | 83.82 ± 0.10 |
| CB1-eqCP1M1000_K10 code-level, 145 ± 0 epochs | - | 145 ± 0 | 890 ± 4 | 84.61 ± 0.67 | 85.01 ± 0.03 |
| CP1-M1000_K10 code-level, 145 ± 0 epochs | - | 145 ± 0 | 890 ± 4 | 85.00 ± 0.44 | 85.15 ± 0.17 |
| CP1-M800_K10 code-level, 145 ± 0 epochs | - | 145 ± 0 | 728 ± 5 | 84.33 ± 0.44 | 84.53 ± 0.74 |

| Comparison | Question | Δ | 95 % CI | Seeds (Δ) | Verdict |
|---|---|---|---|---|---|
| one-shot B2 at 145 vs 30 epochs, K = 10 | does longer training alone help? | +1.11 | [+0.78, +1.44] | +0.83, +1.57, +0.93 | **better** |
| one-shot B2 at 145 epochs (K = 10) vs dense head | is dense accuracy kept? | -0.18 | [-0.77, +0.38] | -0.62, +0.45, -0.38 | no meaningful loss |
| code-level CB1-M800 vs one-shot B2, both 145 epochs, K = 10 | fewer codes at no loss? | -1.65 | [-2.11, -1.20] | -1.36, -2.00, -1.59 | loss not excluded |
| code-level CB1-M800 at 145 epochs (K = 10) vs dense head | is dense accuracy kept? | -1.84 | [-2.39, -1.28] | -1.98, -1.55, -1.97 | loss not excluded |
| code-level CB1-eqCP1M1000 vs one-shot B2, both 145 epochs, K = 10 | fewer codes at no loss? | -0.46 | [-0.82, -0.10] | -0.07, -0.95, -0.36 | no meaningful loss |
| code-level CP1-M1000 vs CB1 with as many codes, both 145 epochs, K = 10 | does the code score matter after retraining? | +0.14 | [-0.28, +0.56] | -0.05, +0.16, +0.33 | no meaningful loss |
| code-level CB1-eqCP1M1000 at 145 epochs (K = 10) vs dense head | is dense accuracy kept? | -0.64 | [-1.20, -0.09] | -0.69, -0.50, -0.74 | loss not excluded |
| code-level CP1-M1000 vs one-shot B2, both 145 epochs, K = 10 | fewer codes at no loss? | -0.32 | [-0.71, +0.07] | -0.12, -0.79, -0.03 | no meaningful loss |
| code-level CP1-M1000 at 145 epochs (K = 10) vs dense head | is dense accuracy kept? | -0.50 | [-1.07, +0.07] | -0.74, -0.35, -0.41 | loss not excluded |
| code-level CP1-M800 vs one-shot B2, both 145 epochs, K = 10 | fewer codes at no loss? | -0.94 | [-1.35, -0.52] | -0.97, -1.86, +0.02 | loss not excluded |
| code-level CP1-M800 at 145 epochs (K = 10) vs dense head | is dense accuracy kept? | -1.12 | [-1.68, -0.56] | -1.59, -1.42, -0.36 | loss not excluded |

## Code-level pruning (one-shot, section 12)

The M best codes by summed score (CB1: magnitude, CP1: activation-aware), then the top K edges of each class among them. Δ vs edge-level B1 at the same K (which keeps the codes its edges touch).

| Model | Codes kept (active) | Edges | Classes < K | Val top-1 | Test top-1 | Δ vs B1 | 95 % CI | Seeds (Δ) | Verdict |
|---|---|---|---|---|---|---|---|---|---|
| CB1-M1200_K10 | 683 ± 11 | 2,000 ± 0 | 0 ± 0 | 44.56 ± 1.26 | 47.64 ± 0.95 | -29.42 | [-30.50, -28.34] | -28.60, -28.96, -30.70 | loss not excluded |
| CB1-M1000_K10 | 579 ± 12 | 2,000 ± 0 | 0 ± 0 | 32.33 ± 1.74 | 35.80 ± 2.56 | -41.26 | [-42.37, -40.14] | -38.26, -43.04, -42.47 | loss not excluded |
| CB1-eqCP1M1000_K10 | 890 ± 4 | 2,000 ± 0 | 0 ± 0 | 59.89 ± 1.64 | 62.95 ± 2.76 | -14.11 | [-14.98, -13.23] | -13.07, -17.10, -12.17 | loss not excluded |
| CB1-M800_K10 | 484 ± 10 | 2,000 ± 0 | 0 ± 0 | 19.78 ± 2.31 | 21.38 ± 2.53 | -55.68 | [-56.78, -54.59] | -56.32, -57.65, -53.07 | loss not excluded |
| CB1-M600_K10 | 384 ± 7 | 2,000 ± 0 | 0 ± 0 | 11.44 ± 2.66 | 11.98 ± 2.33 | -65.08 | [-66.13, -64.01] | -64.64, -67.45, -63.15 | loss not excluded |
| CB1-M400_K10 | 305 ± 5 | 2,000 ± 0 | 0 ± 0 | 5.39 ± 1.13 | 5.17 ± 0.85 | -71.89 | [-72.89, -70.85] | -72.09, -72.44, -71.14 | loss not excluded |
| CP1-M1200_K10 | 1,034 ± 7 | 2,000 ± 0 | 0 ± 0 | 72.50 ± 0.50 | 74.34 ± 0.48 | -2.73 | [-3.41, -2.04] | -2.16, -3.09, -2.93 | loss not excluded |
| CP1-M1000_K10 | 890 ± 4 | 2,000 ± 0 | 0 ± 0 | 69.33 ± 0.17 | 71.14 ± 0.44 | -5.92 | [-6.66, -5.17] | -5.90, -6.20, -5.66 | loss not excluded |
| CP1-M800_K10 | 728 ± 5 | 2,000 ± 0 | 0 ± 0 | 62.89 ± 1.44 | 66.85 ± 0.74 | -10.21 | [-11.06, -9.38] | -9.89, -10.93, -9.82 | loss not excluded |
| CP1-M600_K10 | 569 ± 3 | 2,000 ± 0 | 0 ± 0 | 55.50 ± 1.74 | 58.55 ± 1.70 | -18.51 | [-19.46, -17.54] | -18.97, -19.74, -16.81 | loss not excluded |
| CP1-M400_K10 | 389 ± 3 | 2,000 ± 0 | 0 ± 0 | 41.94 ± 3.15 | 44.13 ± 2.38 | -32.93 | [-34.02, -31.83] | -32.26, -35.43, -31.10 | loss not excluded |

| Comparison | Question | Δ | 95 % CI | Seeds (Δ) | Verdict |
|---|---|---|---|---|---|
| CP1-M1000_K10 vs CB1-eqCP1M1000_K10 | does the code score help at equal active codes? | +8.19 | [+7.21, +9.16] | +7.16, +10.91, +6.51 | **better** |
| CP1-M1200_K10 vs CB1-M1200_K10 | does the activation-aware code score help? | +26.69 | [+25.56, +27.82] | +26.44, +25.87, +27.77 | **better** |
| CP1-M1000_K10 vs CB1-M1000_K10 | does the activation-aware code score help? | +35.34 | [+34.16, +36.49] | +32.36, +36.85, +36.81 | **better** |
| CP1-M800_K10 vs CB1-M800_K10 | does the activation-aware code score help? | +45.47 | [+44.29, +46.63] | +46.43, +46.72, +43.25 | **better** |
| CP1-M600_K10 vs CB1-M600_K10 | does the activation-aware code score help? | +46.57 | [+45.43, +47.70] | +45.67, +47.70, +46.34 | **better** |
| CP1-M400_K10 vs CB1-M400_K10 | does the activation-aware code score help? | +38.96 | [+37.87, +40.05] | +39.83, +37.00, +40.04 | **better** |

## Ablations (Phase 4)

Same rules as above; the reference is the second method named.

| Comparison | Question | Δ | 95 % CI | Seeds (Δ) | Verdict |
|---|---|---|---|---|---|
| P1-contrast vs P1, one-shot, K = 5 | does contrasting against other classes help? | +0.14 | [+0.02, +0.26] | +0.12, +0.22, +0.09 | **better** |
| P1-contrast vs P1, one-shot, K = 10 | does contrasting against other classes help? | +0.13 | [-0.03, +0.29] | +0.21, -0.16, +0.33 | no meaningful loss |
| P1-contrast vs P1, one-shot, K = 20 | does contrasting against other classes help? | +0.16 | [-0.03, +0.35] | +0.14, +0.17, +0.16 | no meaningful loss |

## Deletion-effect control

100 class–code edges (a third each from low, medium and high P1 scores) removed one at a time from the dense head; damage = change in validation cross-entropy. Spearman ρ between damage and each score (higher = the score predicts real damage better).

| Seed | Magnitude (B1) | P1 | P1-contrast |
|---|---|---|---|
| 40 | 0.076 | 0.049 | 0.377 |
| 41 | -0.102 | -0.164 | 0.430 |
| 42 | -0.008 | -0.140 | 0.449 |
| mean | -0.01 ± 0.09 | -0.08 ± 0.12 | 0.42 ± 0.04 |
