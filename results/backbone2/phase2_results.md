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

## Closed-form weight repair (no retraining, section 13)

The surviving weights of a pruned head re-fitted on the cached training features, mask and w ≥ 0 kept. LS: per-class non-negative least squares against the dense model's logits. CE: the same weights re-fitted to the labels (projected Adam). phys: codes no class uses are removed first, so the softmax renormalises over the kept codes (as in retraining). CEsel: CE refit with the step count (300 / 1,000 / 3,000 / 10,000) chosen per model on validation (post hoc); the chosen counts are in brackets. rtH / rtCH: the retraining recipe (145 epochs, best epoch on validation) run on the cached features, head only / codes + head (post hoc); best epochs in brackets. Δ vs the same mask one-shot (no repair, every code in the softmax).

| Model | Codes in softmax | Edges | Val top-1 | Test top-1 | Δ vs one-shot | 95 % CI | Seeds (Δ) | Verdict | Fit (s) |
|---|---|---|---|---|---|---|---|---|---|
| B1 one-shot | 1,304 ± 31 (all in softmax) | 2,000 ± 0 | 73.83 ± 0.58 | 77.06 ± 0.18 | ref |  |  |  |  |
| B1+phys | 1,304 ± 31 | 2,000 ± 0 | 73.22 ± 0.54 | 76.69 ± 0.29 | -0.37 | [-0.70, -0.05] | -0.07, -0.59, -0.47 | no meaningful loss | 0.00 ± 0.00 |
| B1+LS | 3,000 ± 0 | 1,948 ± 14 | 72.33 ± 1.76 | 75.42 ± 0.56 | -1.65 | [-2.26, -1.03] | -2.16, -1.00, -1.78 | loss not excluded | 0.21 ± 0.00 |
| B1+CE | 3,000 ± 0 | 2,000 ± 1 | 74.44 ± 0.95 | 77.78 ± 0.13 | +0.71 | [+0.51, +0.93] | +0.66, +0.86, +0.62 | **better** | 0.39 ± 0.02 |
| B1+phys+LS | 1,304 ± 31 | 1,664 ± 22 | 67.83 ± 1.89 | 70.42 ± 1.19 | -6.64 | [-7.38, -5.91] | -6.97, -5.18, -7.78 | loss not excluded | 0.18 ± 0.00 |
| B1+phys+CE | 1,304 ± 31 | 1,998 ± 2 | 75.72 ± 1.06 | 79.08 ± 0.19 | +2.01 | [+1.61, +2.42] | +2.02, +2.00, +2.02 | **better** | 0.16 ± 0.01 |
| B1+CEsel (10000, 3000, 10000 steps) | 3,000 ± 0 | 1,822 ± 133 | 78.50 ± 0.76 | 80.36 ± 0.70 | +3.30 | [+2.70, +3.89] | +2.59, +3.97, +3.33 | **better** | 7.64 ± 0.02 |
| B1+phys+CEsel (3000, 3000, 1000 steps) | 1,304 ± 31 | 1,906 ± 50 | 78.39 ± 0.54 | 80.53 ± 0.37 | +3.47 | [+2.91, +4.03] | +3.16, +3.92, +3.33 | **better** | 5.47 ± 0.59 |
| B1+phys+rtH (best epoch 81, 74, 44) | 1,304 ± 31 | 1,974 ± 11 | 77.83 ± 0.29 | 80.46 ± 0.51 | +3.40 | [+2.84, +3.96] | +2.93, +3.97, +3.30 | **better** | 8.93 ± 0.66 |
| B1+phys+rtCH (best epoch 102, 117, 60) | 1,304 ± 31 | 1,997 ± 3 | 83.83 ± 1.45 | 84.64 ± 0.15 | +7.58 | [+6.90, +8.27] | +7.78, +7.75, +7.20 | **better** | 14.43 ± 0.48 |
| P1 one-shot | 1,574 ± 17 (all in softmax) | 2,000 ± 0 | 77.56 ± 1.26 | 79.52 ± 0.16 | ref |  |  |  |  |
| P1+phys | 1,574 ± 17 | 2,000 ± 0 | 77.72 ± 0.86 | 79.32 ± 0.10 | -0.20 | [-0.49, +0.09] | -0.07, -0.10, -0.43 | no meaningful loss | 0.00 ± 0.00 |
| P1+LS | 3,000 ± 0 | 1,724 ± 23 | 68.28 ± 1.46 | 70.25 ± 0.90 | -9.28 | [-10.04, -8.51] | -9.98, -9.44, -8.42 | loss not excluded | 0.21 ± 0.01 |
| P1+CE | 3,000 ± 0 | 1,994 ± 2 | 78.72 ± 1.18 | 80.05 ± 0.04 | +0.53 | [+0.26, +0.80] | +0.59, +0.64, +0.36 | **better** | 0.24 ± 0.00 |
| P1+phys+LS | 1,574 ± 17 | 1,485 ± 28 | 67.39 ± 1.77 | 69.23 ± 0.59 | -10.30 | [-11.09, -9.50] | -10.94, -9.67, -10.29 | loss not excluded | 0.19 ± 0.00 |
| P1+phys+CE | 1,574 ± 17 | 1,992 ± 1 | 79.39 ± 1.29 | 80.63 ± 0.15 | +1.10 | [+0.73, +1.48] | +1.09, +1.12, +1.10 | **better** | 0.16 ± 0.01 |
| P1+CEsel (10000, 1000, 3000 steps) | 3,000 ± 0 | 1,873 ± 177 | 79.89 ± 0.75 | 80.38 ± 0.91 | +0.85 | [+0.36, +1.34] | -0.16, +1.28, +1.43 | no meaningful loss | 7.45 ± 0.01 |
| P1+phys+CEsel (3000, 300, 300 steps) | 1,574 ± 17 | 1,958 ± 60 | 79.94 ± 0.92 | 80.74 ± 0.23 | +1.22 | [+0.80, +1.64] | +1.43, +1.12, +1.10 | **better** | 5.86 ± 0.76 |
| P1+phys+rtH (best epoch 38, 21, 17) | 1,574 ± 17 | 1,992 ± 6 | 80.67 ± 1.01 | 80.66 ± 0.04 | +1.13 | [+0.67, +1.59] | +1.10, +1.31, +0.98 | **better** | 8.89 ± 0.04 |
| P1+phys+rtCH (best epoch 49, 62, 50) | 1,574 ± 17 | 2,000 ± 1 | 83.56 ± 0.19 | 83.99 ± 0.27 | +4.47 | [+3.86, +5.08] | +4.18, +4.80, +4.44 | **better** | 16.58 ± 0.17 |
| random one-shot | 1,432 ± 8 (all in softmax) | 2,000 ± 0 | 4.44 ± 1.02 | 4.93 ± 0.39 | ref |  |  |  |  |
| random+phys | 1,432 ± 8 | 2,000 ± 0 | 4.72 ± 1.06 | 5.03 ± 0.32 | +0.10 | [-0.03, +0.24] | +0.07, +0.02, +0.22 | no meaningful loss | 0.00 ± 0.00 |
| random+LS | 3,000 ± 0 | 2,000 ± 1 | 2.33 ± 0.44 | 2.51 ± 0.42 | -2.42 | [-2.68, -2.17] | -2.47, -2.28, -2.50 | loss not excluded | 0.21 ± 0.00 |
| random+CE | 3,000 ± 0 | 462 ± 19 | 4.56 ± 0.84 | 5.22 ± 0.81 | +0.29 | [+0.09, +0.49] | +0.85, -0.17, +0.21 | no meaningful loss | 0.24 ± 0.00 |
| random+phys+LS | 1,432 ± 8 | 1,997 ± 3 | 2.83 ± 0.17 | 3.26 ± 0.30 | -1.67 | [-1.93, -1.43] | -1.83, -1.57, -1.62 | loss not excluded | 0.20 ± 0.01 |
| random+phys+CE | 1,432 ± 8 | 484 ± 22 | 5.61 ± 1.34 | 5.78 ± 0.82 | +0.85 | [+0.63, +1.08] | +1.42, +0.41, +0.72 | **better** | 0.16 ± 0.02 |
| random+CEsel (10000, 10000, 10000 steps) | 3,000 ± 0 | 504 ± 22 | 8.00 ± 1.76 | 8.18 ± 1.10 | +3.25 | [+2.92, +3.57] | +4.09, +2.93, +2.73 | **better** | 7.44 ± 0.02 |
| random+phys+CEsel (10000, 3000, 3000 steps) | 1,432 ± 8 | 513 ± 20 | 7.89 ± 2.06 | 7.88 ± 1.43 | +2.95 | [+2.64, +3.24] | +4.14, +2.57, +2.12 | **better** | 5.15 ± 0.39 |
| random+phys+rtH (best epoch 130, 80, 84) | 1,432 ± 8 | 495 ± 26 | 7.78 ± 1.68 | 7.44 ± 1.11 | +2.51 | [+2.23, +2.78] | +3.33, +2.26, +1.93 | **better** | 8.28 ± 0.06 |
| random+phys+rtCH (best epoch 68, 91, 46) | 1,432 ± 8 | 962 ± 44 | 80.61 ± 1.58 | 80.80 ± 0.98 | +75.87 | [+74.92, +76.79] | +74.65, +76.94, +76.01 | **better** | 15.41 ± 0.02 |
| CB1-M1000 one-shot | 579 ± 12 (all in softmax) | 2,000 ± 0 | 32.33 ± 1.74 | 35.80 ± 2.56 | ref |  |  |  |  |
| CB1-M1000+phys | 579 ± 12 | 2,000 ± 0 | 50.00 ± 2.42 | 52.12 ± 2.47 | +16.32 | [+15.65, +17.02] | +16.21, +17.31, +15.45 | **better** | 0.00 ± 0.00 |
| CB1-M1000+LS | 3,000 ± 0 | 1,993 ± 6 | 34.61 ± 0.84 | 37.39 ± 1.36 | +1.59 | [+0.90, +2.27] | -0.24, +3.94, +1.07 | no meaningful loss | 0.21 ± 0.00 |
| CB1-M1000+CE | 3,000 ± 0 | 1,996 ± 1 | 32.89 ± 1.90 | 35.98 ± 2.81 | +0.18 | [-0.11, +0.47] | +0.50, +0.21, -0.17 | no meaningful loss | 0.24 ± 0.00 |
| CB1-M1000+phys+LS | 579 ± 12 | 1,923 ± 12 | 52.83 ± 1.76 | 54.09 ± 1.51 | +18.29 | [+17.42, +19.18] | +16.62, +20.62, +17.62 | **better** | 0.16 ± 0.00 |
| CB1-M1000+phys+CE | 579 ± 12 | 1,992 ± 2 | 57.67 ± 1.96 | 59.56 ± 1.93 | +23.75 | [+22.97, +24.55] | +22.54, +26.06, +22.66 | **better** | 0.16 ± 0.01 |
| CB1-M1000+CEsel (10000, 10000, 10000 steps) | 3,000 ± 0 | 1,851 ± 3 | 59.89 ± 2.17 | 61.76 ± 2.44 | +25.96 | [+25.07, +26.86] | +25.06, +28.56, +24.25 | **better** | 7.48 ± 0.01 |
| CB1-M1000+phys+CEsel (10000, 10000, 10000 steps) | 579 ± 12 | 1,619 ± 21 | 67.11 ± 1.84 | 68.52 ± 1.31 | +32.72 | [+31.77, +33.68] | +30.82, +35.09, +32.26 | **better** | 5.77 ± 0.41 |
| CB1-M1000+phys+rtH (best epoch 97, 85, 128) | 579 ± 12 | 1,966 ± 3 | 64.28 ± 1.42 | 65.63 ± 1.35 | +29.82 | [+28.91, +30.73] | +28.41, +31.34, +29.72 | **better** | 4.66 ± 0.15 |
| CB1-M1000+phys+rtCH (best epoch 105, 56, 72) | 579 ± 12 | 1,971 ± 8 | 81.94 ± 0.59 | 82.87 ± 0.03 | +47.07 | [+46.02, +48.10] | +44.15, +49.00, +48.05 | **better** | 7.55 ± 0.19 |
| CP1-M1000 one-shot | 890 ± 4 (all in softmax) | 2,000 ± 0 | 69.33 ± 0.17 | 71.14 ± 0.44 | ref |  |  |  |  |
| CP1-M1000+phys | 890 ± 4 | 2,000 ± 0 | 71.39 ± 1.25 | 73.47 ± 0.54 | +2.32 | [+1.89, +2.77] | +2.76, +2.11, +2.11 | **better** | 0.00 ± 0.00 |
| CP1-M1000+LS | 3,000 ± 0 | 1,882 ± 18 | 59.17 ± 1.09 | 60.10 ± 0.58 | -11.04 | [-11.89, -10.17] | -10.30, -10.94, -11.87 | loss not excluded | 0.21 ± 0.00 |
| CP1-M1000+CE | 3,000 ± 0 | 1,979 ± 1 | 71.44 ± 0.51 | 72.40 ± 0.55 | +1.25 | [+0.89, +1.63] | +1.17, +1.19, +1.40 | **better** | 0.24 ± 0.00 |
| CP1-M1000+phys+LS | 890 ± 4 | 1,691 ± 27 | 59.50 ± 2.24 | 61.02 ± 0.47 | -10.12 | [-10.98, -9.26] | -9.53, -10.11, -10.72 | loss not excluded | 0.17 ± 0.00 |
| CP1-M1000+phys+CE | 890 ± 4 | 1,948 ± 7 | 75.67 ± 0.44 | 76.35 ± 0.52 | +5.21 | [+4.67, +5.75] | +5.38, +5.04, +5.20 | **better** | 0.16 ± 0.01 |
| CP1-M1000+CEsel (10000, 10000, 10000 steps) | 3,000 ± 0 | 1,526 ± 34 | 74.89 ± 1.11 | 76.32 ± 0.14 | +5.18 | [+4.47, +5.91] | +5.14, +5.52, +4.87 | **better** | 7.47 ± 0.02 |
| CP1-M1000+phys+CEsel (10000, 1000, 1000 steps) | 890 ± 4 | 1,658 ± 335 | 76.11 ± 0.19 | 77.08 ± 0.71 | +5.94 | [+5.27, +6.63] | +5.25, +6.44, +6.13 | **better** | 5.18 ± 0.10 |
| CP1-M1000+phys+rtH (best epoch 20, 80, 17) | 890 ± 4 | 1,934 ± 61 | 76.22 ± 0.35 | 76.73 ± 0.43 | +5.59 | [+4.96, +6.21] | +5.16, +6.21, +5.40 | **better** | 5.84 ± 0.02 |
| CP1-M1000+phys+rtCH (best epoch 75, 93, 107) | 890 ± 4 | 1,974 ± 1 | 83.11 ± 0.67 | 83.88 ± 0.18 | +12.74 | [+11.93, +13.58] | +12.60, +13.20, +12.41 | **better** | 10.77 ± 0.07 |
| CB1-eqCP1M1000 one-shot | 890 ± 4 (all in softmax) | 2,000 ± 0 | 59.89 ± 1.64 | 62.95 ± 2.76 | ref |  |  |  |  |
| CB1-eqCP1M1000+phys | 890 ± 4 | 2,000 ± 0 | 63.56 ± 1.67 | 66.00 ± 1.06 | +3.05 | [+2.59, +3.50] | +2.11, +5.09, +1.95 | **better** | 0.00 ± 0.00 |
| CB1-eqCP1M1000+LS | 3,000 ± 0 | 1,992 ± 5 | 60.61 ± 2.58 | 63.66 ± 2.35 | +0.71 | [+0.09, +1.31] | +0.71, +1.14, +0.28 | **better** | 0.21 ± 0.00 |
| CB1-eqCP1M1000+CE | 3,000 ± 0 | 1,999 ± 1 | 60.00 ± 1.48 | 63.30 ± 2.38 | +0.35 | [+0.12, +0.58] | +0.17, +0.79, +0.09 | **better** | 0.24 ± 0.00 |
| CB1-eqCP1M1000+phys+LS | 890 ± 4 | 1,843 ± 21 | 63.33 ± 1.45 | 65.33 ± 0.97 | +2.38 | [+1.60, +3.15] | +1.04, +4.78, +1.33 | **better** | 0.17 ± 0.00 |
| CB1-eqCP1M1000+phys+CE | 890 ± 4 | 1,997 ± 2 | 70.22 ± 2.22 | 72.42 ± 0.75 | +9.47 | [+8.88, +10.04] | +8.65, +11.77, +7.99 | **better** | 0.16 ± 0.01 |
| CB1-eqCP1M1000+CEsel (10000, 10000, 10000 steps) | 3,000 ± 0 | 1,798 ± 11 | 74.11 ± 0.35 | 75.01 ± 0.54 | +12.06 | [+11.30, +12.82] | +11.24, +14.58, +10.37 | **better** | 7.46 ± 0.00 |
| CB1-eqCP1M1000+phys+CEsel (3000, 3000, 10000 steps) | 890 ± 4 | 1,778 ± 179 | 75.94 ± 0.86 | 76.68 ± 0.05 | +13.73 | [+12.94, +14.50] | +12.74, +16.90, +11.56 | **better** | 5.28 ± 0.41 |
| CB1-eqCP1M1000+phys+rtH (best epoch 73, 85, 43) | 890 ± 4 | 1,979 ± 8 | 74.94 ± 0.95 | 75.99 ± 0.48 | +13.04 | [+12.30, +13.77] | +12.29, +15.60, +11.24 | **better** | 5.85 ± 0.03 |
| CB1-eqCP1M1000+phys+rtCH (best epoch 92, 103, 21) | 890 ± 4 | 1,990 ± 9 | 82.61 ± 0.67 | 84.13 ± 0.79 | +21.18 | [+20.30, +22.04] | +20.68, +24.75, +18.12 | **better** | 10.76 ± 0.07 |

| Comparison | Question | Δ | 95 % CI | Seeds (Δ) | Verdict |
|---|---|---|---|---|---|
| B1 phys+LS vs B1 retrained 145 epochs, K = 10 | can repair replace retraining? | -15.05 | [-15.82, -14.27] | -15.10, -14.17, -15.88 | loss not excluded |
| B1 phys+LS vs B1 retrained 30 epochs, K = 10 | can repair replace retraining? | -13.94 | [-14.67, -13.22] | -14.27, -12.60, -14.95 | loss not excluded |
| B1 phys+LS (K = 10) vs dense head | is dense accuracy kept? | -15.23 | [-16.05, -14.42] | -15.72, -13.72, -16.26 | loss not excluded |
| B1 phys+CE vs B1 retrained 145 epochs, K = 10 | can repair replace retraining? | -6.39 | [-6.99, -5.80] | -6.11, -6.99, -6.08 | loss not excluded |
| B1 phys+CE vs B1 retrained 30 epochs, K = 10 | can repair replace retraining? | -5.28 | [-5.81, -4.76] | -5.28, -5.42, -5.14 | loss not excluded |
| B1 phys+CE (K = 10) vs dense head | is dense accuracy kept? | -6.58 | [-7.27, -5.90] | -6.73, -6.54, -6.45 | loss not excluded |
| B1 phys+CEsel vs B1 retrained 145 epochs, K = 10 | can repair replace retraining? | -4.94 | [-5.45, -4.42] | -4.97, -5.07, -4.76 | loss not excluded |
| B1 phys+CEsel vs B1 retrained 30 epochs, K = 10 | can repair replace retraining? | -3.83 | [-4.29, -3.37] | -4.14, -3.50, -3.83 | loss not excluded |
| B1 phys+CEsel (K = 10) vs dense head | is dense accuracy kept? | -5.12 | [-5.78, -4.46] | -5.59, -4.63, -5.14 | loss not excluded |
| B1 phys+rtH vs B1 retrained 145 epochs, K = 10 | can repair replace retraining? | -5.01 | [-5.53, -4.47] | -5.20, -5.02, -4.80 | loss not excluded |
| B1 phys+rtH vs B1 retrained 30 epochs, K = 10 | can repair replace retraining? | -3.89 | [-4.35, -3.45] | -4.37, -3.45, -3.87 | loss not excluded |
| B1 phys+rtH (K = 10) vs dense head | is dense accuracy kept? | -5.19 | [-5.86, -4.53] | -5.82, -4.57, -5.18 | loss not excluded |
| B1 phys+rtCH vs B1 retrained 145 epochs, K = 10 | can repair replace retraining? | -0.83 | [-1.24, -0.42] | -0.35, -1.24, -0.90 | loss not excluded |
| B1 phys+rtCH vs B1 retrained 30 epochs, K = 10 | can repair replace retraining? | +0.28 | [-0.12, +0.68] | +0.48, +0.33, +0.03 | no meaningful loss |
| B1 phys+rtCH (K = 10) vs dense head | is dense accuracy kept? | -1.01 | [-1.63, -0.39] | -0.97, -0.79, -1.28 | loss not excluded |
| B1: LS (dense logits) vs CE (labels), physical | which repair target is better? | -8.66 | [-9.34, -8.00] | -8.99, -7.18, -9.80 | loss not excluded |
| B1: CE, steps chosen on val vs 300 steps, physical | do more CE steps help? | +1.46 | [+1.04, +1.87] | +1.14, +1.92, +1.31 | **better** |
| B1: codes + head vs head only, retraining recipe on cached features | does adapting the codes help? | +4.18 | [+3.67, +4.68] | +4.85, +3.78, +3.90 | **better** |
| P1 phys+LS (K = 10) vs dense head | is dense accuracy kept? | -16.43 | [-17.29, -15.58] | -17.16, -15.77, -16.34 | loss not excluded |
| P1 phys+CE (K = 10) vs dense head | is dense accuracy kept? | -5.02 | [-5.70, -4.37] | -5.13, -4.99, -4.95 | loss not excluded |
| P1 phys+CEsel (K = 10) vs dense head | is dense accuracy kept? | -4.91 | [-5.56, -4.26] | -4.78, -4.99, -4.95 | loss not excluded |
| P1 phys+rtH (K = 10) vs dense head | is dense accuracy kept? | -4.99 | [-5.69, -4.33] | -5.11, -4.80, -5.07 | loss not excluded |
| P1 phys+rtCH (K = 10) vs dense head | is dense accuracy kept? | -1.66 | [-2.31, -1.01] | -2.04, -1.31, -1.62 | loss not excluded |
| P1: LS (dense logits) vs CE (labels), physical | which repair target is better? | -11.40 | [-12.16, -10.67] | -12.03, -10.79, -11.39 | loss not excluded |
| P1: CE, steps chosen on val vs 300 steps, physical | do more CE steps help? | +0.12 | [-0.09, +0.32] | +0.35, +0.00, +0.00 | no meaningful loss |
| P1: codes + head vs head only, retraining recipe on cached features | does adapting the codes help? | +3.34 | [+2.84, +3.84] | +3.07, +3.49, +3.45 | **better** |
| random phys+LS (K = 10) vs dense head | is dense accuracy kept? | -82.40 | [-83.28, -81.50] | -82.24, -82.12, -82.83 | loss not excluded |
| random phys+CE (K = 10) vs dense head | is dense accuracy kept? | -79.87 | [-80.77, -78.97] | -79.00, -80.13, -80.48 | loss not excluded |
| random phys+CEsel (K = 10) vs dense head | is dense accuracy kept? | -77.78 | [-78.70, -76.86] | -76.27, -77.98, -79.08 | loss not excluded |
| random phys+rtH (K = 10) vs dense head | is dense accuracy kept? | -78.21 | [-79.13, -77.29] | -77.08, -78.29, -79.27 | loss not excluded |
| random phys+rtCH (K = 10) vs dense head | is dense accuracy kept? | -4.86 | [-5.53, -4.17] | -5.76, -3.61, -5.20 | loss not excluded |
| random: LS (dense logits) vs CE (labels), physical | which repair target is better? | -2.53 | [-2.80, -2.26] | -3.24, -1.98, -2.35 | loss not excluded |
| random: CE, steps chosen on val vs 300 steps, physical | do more CE steps help? | +2.09 | [+1.79, +2.39] | +2.73, +2.16, +1.40 | **better** |
| random: codes + head vs head only, retraining recipe on cached features | does adapting the codes help? | +73.36 | [+72.40, +74.31] | +71.32, +74.68, +74.08 | **better** |
| CB1-M1000 phys+LS (K = 10) vs dense head | is dense accuracy kept? | -31.56 | [-32.53, -30.57] | -30.39, -30.96, -33.33 | loss not excluded |
| CB1-M1000 phys+CE (K = 10) vs dense head | is dense accuracy kept? | -26.10 | [-27.03, -25.16] | -24.47, -25.53, -28.29 | loss not excluded |
| CB1-M1000 phys+CEsel (K = 10) vs dense head | is dense accuracy kept? | -17.13 | [-17.93, -16.30] | -16.19, -16.50, -18.69 | loss not excluded |
| CB1-M1000 phys+rtH (K = 10) vs dense head | is dense accuracy kept? | -20.03 | [-20.89, -19.16] | -18.61, -20.25, -21.23 | loss not excluded |
| CB1-M1000 phys+rtCH (K = 10) vs dense head | is dense accuracy kept? | -2.78 | [-3.39, -2.18] | -2.87, -2.59, -2.90 | loss not excluded |
| CB1-M1000: LS (dense logits) vs CE (labels), physical | which repair target is better? | -5.47 | [-6.17, -4.80] | -5.92, -5.44, -5.04 | loss not excluded |
| CB1-M1000: CE, steps chosen on val vs 300 steps, physical | do more CE steps help? | +8.97 | [+8.27, +9.68] | +8.28, +9.03, +9.60 | **better** |
| CB1-M1000: codes + head vs head only, retraining recipe on cached features | does adapting the codes help? | +17.24 | [+16.45, +18.04] | +15.74, +17.66, +18.33 | **better** |
| CP1-M1000 phys+LS vs CP1-M1000 retrained 145 epochs, K = 10 | can repair replace retraining? | -24.13 | [-25.07, -23.20] | -23.44, -24.51, -24.44 | loss not excluded |
| CP1-M1000 phys+LS (K = 10) vs dense head | is dense accuracy kept? | -24.63 | [-25.59, -23.66] | -24.18, -24.85, -24.85 | loss not excluded |
| CP1-M1000 phys+CE vs CP1-M1000 retrained 145 epochs, K = 10 | can repair replace retraining? | -8.80 | [-9.48, -8.14] | -8.53, -9.35, -8.53 | loss not excluded |
| CP1-M1000 phys+CE (K = 10) vs dense head | is dense accuracy kept? | -9.30 | [-10.05, -8.58] | -9.27, -9.70, -8.94 | loss not excluded |
| CP1-M1000 phys+CEsel vs CP1-M1000 retrained 145 epochs, K = 10 | can repair replace retraining? | -8.07 | [-8.72, -7.45] | -8.66, -7.96, -7.59 | loss not excluded |
| CP1-M1000 phys+CEsel (K = 10) vs dense head | is dense accuracy kept? | -8.57 | [-9.29, -7.85] | -9.41, -8.30, -8.01 | loss not excluded |
| CP1-M1000 phys+rtH vs CP1-M1000 retrained 145 epochs, K = 10 | can repair replace retraining? | -8.42 | [-9.08, -7.77] | -8.75, -8.18, -8.32 | loss not excluded |
| CP1-M1000 phys+rtH (K = 10) vs dense head | is dense accuracy kept? | -8.92 | [-9.66, -8.20] | -9.49, -8.53, -8.73 | loss not excluded |
| CP1-M1000 phys+rtCH vs CP1-M1000 retrained 145 epochs, K = 10 | can repair replace retraining? | -1.27 | [-1.72, -0.84] | -1.31, -1.19, -1.31 | loss not excluded |
| CP1-M1000 phys+rtCH (K = 10) vs dense head | is dense accuracy kept? | -1.77 | [-2.39, -1.16] | -2.05, -1.54, -1.73 | loss not excluded |
| CP1-M1000: LS (dense logits) vs CE (labels), physical | which repair target is better? | -15.33 | [-16.13, -14.49] | -14.91, -15.15, -15.91 | loss not excluded |
| CP1-M1000: CE, steps chosen on val vs 300 steps, physical | do more CE steps help? | +0.73 | [+0.33, +1.14] | -0.14, +1.40, +0.93 | no meaningful loss |
| CP1-M1000: codes + head vs head only, retraining recipe on cached features | does adapting the codes help? | +7.15 | [+6.52, +7.77] | +7.44, +6.99, +7.01 | **better** |
| CB1-eqCP1M1000 phys+LS vs CB1-eqCP1M1000 retrained 145 epochs, K = 10 | can repair replace retraining? | -19.68 | [-20.53, -18.86] | -20.09, -20.37, -18.57 | loss not excluded |
| CB1-eqCP1M1000 phys+LS (K = 10) vs dense head | is dense accuracy kept? | -20.32 | [-21.20, -19.46] | -20.78, -20.87, -19.31 | loss not excluded |
| CB1-eqCP1M1000 phys+CE vs CB1-eqCP1M1000 retrained 145 epochs, K = 10 | can repair replace retraining? | -12.59 | [-13.30, -11.86] | -12.48, -13.38, -11.91 | loss not excluded |
| CB1-eqCP1M1000 phys+CE (K = 10) vs dense head | is dense accuracy kept? | -13.23 | [-14.02, -12.45] | -13.17, -13.88, -12.65 | loss not excluded |
| CB1-eqCP1M1000 phys+CEsel vs CB1-eqCP1M1000 retrained 145 epochs, K = 10 | can repair replace retraining? | -8.32 | [-8.95, -7.70] | -8.39, -8.25, -8.34 | loss not excluded |
| CB1-eqCP1M1000 phys+CEsel (K = 10) vs dense head | is dense accuracy kept? | -8.97 | [-9.69, -8.26] | -9.08, -8.75, -9.08 | loss not excluded |
| CB1-eqCP1M1000 phys+rtH vs CB1-eqCP1M1000 retrained 145 epochs, K = 10 | can repair replace retraining? | -9.02 | [-9.65, -8.36] | -8.84, -9.54, -8.66 | loss not excluded |
| CB1-eqCP1M1000 phys+rtH (K = 10) vs dense head | is dense accuracy kept? | -9.66 | [-10.39, -8.95] | -9.53, -10.04, -9.41 | loss not excluded |
| CB1-eqCP1M1000 phys+rtCH vs CB1-eqCP1M1000 retrained 145 epochs, K = 10 | can repair replace retraining? | -0.87 | [-1.29, -0.45] | -0.45, -0.40, -1.78 | loss not excluded |
| CB1-eqCP1M1000 phys+rtCH (K = 10) vs dense head | is dense accuracy kept? | -1.52 | [-2.11, -0.91] | -1.14, -0.90, -2.52 | loss not excluded |
| CB1-eqCP1M1000: LS (dense logits) vs CE (labels), physical | which repair target is better? | -7.09 | [-7.77, -6.41] | -7.61, -6.99, -6.66 | loss not excluded |
| CB1-eqCP1M1000: CE, steps chosen on val vs 300 steps, physical | do more CE steps help? | +4.26 | [+3.69, +4.84] | +4.09, +5.13, +3.57 | **better** |
| CB1-eqCP1M1000: codes + head vs head only, retraining recipe on cached features | does adapting the codes help? | +8.14 | [+7.54, +8.73] | +8.39, +9.15, +6.89 | **better** |
| P1 vs B1, both physical + LS repair | does the score matter after repair? | -1.19 | [-1.96, -0.41] | -1.43, -2.05, -0.09 | loss not excluded |
| P1 vs B1, both physical + CEsel repair | does the score matter after repair? | +0.21 | [-0.25, +0.68] | +0.81, -0.36, +0.19 | no meaningful loss |
| P1 vs B1, both physical + rtH repair | does the score matter after repair? | +0.20 | [-0.28, +0.65] | +0.71, -0.22, +0.10 | no meaningful loss |
| P1 vs B1, both physical + rtCH repair | does the score matter after repair? | -0.64 | [-0.97, -0.31] | -1.07, -0.52, -0.35 | no meaningful loss |
| CP1-M1000 vs CB1-eqCP1M1000, both physical + LS repair | does the score matter after repair? | -4.31 | [-5.26, -3.34] | -3.40, -3.99, -5.54 | loss not excluded |
| CP1-M1000 vs CB1-eqCP1M1000, both physical + CEsel repair | does the score matter after repair? | +0.40 | [-0.31, +1.09] | -0.33, +0.45, +1.07 | no meaningful loss |
| CP1-M1000 vs CB1-eqCP1M1000, both physical + rtH repair | does the score matter after repair? | +0.74 | [+0.05, +1.44] | +0.03, +1.52, +0.67 | **better** |
| CP1-M1000 vs CB1-eqCP1M1000, both physical + rtCH repair | does the score matter after repair? | -0.25 | [-0.71, +0.18] | -0.91, -0.64, +0.79 | no meaningful loss |

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
