# Phase 2 results

Seeds: 40, 41, 42. Test top-1 is mean ± sd over seeds. Δ = method − reference (points), with the 95 % paired-bootstrap interval over test images (10,000 resamples, correctness averaged over seeds). Verdict rules fixed in Phase 0: better = interval above 0 and same sign in every seed; no meaningful loss = lower bound above −1; otherwise a loss of 1 point or more is not excluded.

Dense head (no pruning): test 88.99 ± 0.06 %, 362,768 nonzero edges.

## One-shot masking (no retraining)

### K = 5 edges per class (N = 980)

| Method | Val top-1 | Test top-1 | Δ vs B1 | 95 % CI | Seeds (Δ) | Verdict | Edges | Active codes |
|---|---|---|---|---|---|---|---|---|
| B1 | 68.38 ± 0.92 | 68.25 ± 0.38 | ref |  |  |  | 980 ± 0 | 819 ± 17 |
| random | 3.68 ± 0.98 | 3.51 ± 0.61 | -64.74 | [-65.62, -63.88] | -63.70, -64.99, -65.54 | loss not excluded | 980 ± 0 | 828 ± 5 |
| P1 | 70.14 ± 0.84 | 71.50 ± 0.35 | +3.24 | [+2.68, +3.80] | +3.72, +3.57, +2.44 | **better** | 980 ± 0 | 873 ± 8 |
| P1contrast | 70.27 ± 0.98 | 71.49 ± 0.52 | +3.23 | [+2.68, +3.79] | +3.76, +3.71, +2.24 | **better** | 980 ± 0 | 873 ± 6 |
| P3 (min-1 classes: 9 ± 3) | 64.70 ± 0.46 | 66.53 ± 0.37 | -1.72 | [-2.31, -1.13] | -0.99, -2.00, -2.18 | loss not excluded | 980 ± 0 | 890 ± 6 |

### K = 10 edges per class (N = 1,960)

| Method | Val top-1 | Test top-1 | Δ vs B1 | 95 % CI | Seeds (Δ) | Verdict | Edges | Active codes |
|---|---|---|---|---|---|---|---|---|
| B1 | 72.02 ± 0.68 | 72.73 ± 0.47 | ref |  |  |  | 1,960 ± 0 | 1,339 ± 21 |
| random | 4.83 ± 1.67 | 5.25 ± 1.49 | -67.48 | [-68.35, -66.61] | -68.20, -67.94, -66.30 | loss not excluded | 1,960 ± 0 | 1,427 ± 22 |
| P1 | 74.68 ± 0.50 | 75.33 ± 0.24 | +2.61 | [+2.12, +3.09] | +3.05, +2.74, +2.04 | **better** | 1,960 ± 0 | 1,486 ± 10 |
| P1contrast | 74.23 ± 0.65 | 75.21 ± 0.22 | +2.48 | [+1.99, +2.97] | +2.90, +2.62, +1.92 | **better** | 1,960 ± 0 | 1,498 ± 10 |
| P3 (min-1 classes: 1 ± 1) | 70.96 ± 0.75 | 72.75 ± 0.72 | +0.02 | [-0.54, +0.58] | +0.95, +0.35, -1.22 | no meaningful loss | 1,960 ± 0 | 1,480 ± 13 |

### K = 20 edges per class (N = 3,920)

| Method | Val top-1 | Test top-1 | Δ vs B1 | 95 % CI | Seeds (Δ) | Verdict | Edges | Active codes |
|---|---|---|---|---|---|---|---|---|
| B1 | 76.89 ± 0.95 | 77.67 ± 0.21 | ref |  |  |  | 3,920 ± 0 | 1,925 ± 18 |
| random | 9.12 ± 3.05 | 9.16 ± 2.08 | -68.51 | [-69.38, -67.67] | -66.04, -69.29, -70.20 | loss not excluded | 3,920 ± 0 | 2,140 ± 15 |
| P1 | 78.65 ± 0.75 | 79.55 ± 0.65 | +1.88 | [+1.43, +2.32] | +2.57, +1.33, +1.73 | **better** | 3,920 ± 0 | 2,153 ± 41 |
| P1contrast | 78.77 ± 0.68 | 79.56 ± 0.83 | +1.89 | [+1.43, +2.34] | +2.80, +1.19, +1.67 | **better** | 3,920 ± 0 | 2,168 ± 44 |
| P3 (min-1 classes: 0 ± 0) | 78.49 ± 0.61 | 78.37 ± 0.48 | +0.70 | [+0.19, +1.19] | +1.34, +0.52, +0.22 | **better** | 3,920 ± 0 | 2,134 ± 39 |

## Iterative pruning and longer retraining (Phase 4)

Iterative: 20 % of each class's edges removed per round, 5 epochs of retraining and rescoring between rounds, full codebook during the rounds; then unused codes removed and 30 epochs (the B2 recipe). Control: the one-shot B1 mask retrained with the B2 recipe for as many epochs in total. Same rules as above.

| Model | Rounds | Total epochs | Codes | Best val | Test top-1 |
|---|---|---|---|---|---|
| B2_K10 one-shot, 145 ± 0 epochs (control) | - | 145 ± 0 | 1,339 ± 21 | 88.43 ± 0.19 | 88.49 ± 0.50 |
| CB1-eqCP1M1000_K10 code-level, 145 ± 0 epochs | - | 145 ± 0 | 874 ± 5 | 88.43 ± 0.78 | 88.46 ± 0.30 |
| CP1-M1000_K10 code-level, 145 ± 0 epochs | - | 145 ± 0 | 874 ± 5 | 87.81 ± 0.43 | 88.16 ± 0.23 |

| Comparison | Question | Δ | 95 % CI | Seeds (Δ) | Verdict |
|---|---|---|---|---|---|
| one-shot B2 at 145 epochs (K = 10) vs dense head | is dense accuracy kept? | -0.50 | [-0.96, -0.05] | -0.17, -0.27, -1.06 | no meaningful loss |
| code-level CB1-eqCP1M1000 vs one-shot B2, both 145 epochs, K = 10 | fewer codes at no loss? | -0.03 | [-0.33, +0.26] | -0.09, -0.45, +0.44 | no meaningful loss |
| code-level CP1-M1000 vs CB1 with as many codes, both 145 epochs, K = 10 | does the code score matter after retraining? | -0.29 | [-0.65, +0.06] | -0.46, -0.31, -0.10 | no meaningful loss |
| code-level CB1-eqCP1M1000 at 145 epochs (K = 10) vs dense head | is dense accuracy kept? | -0.53 | [-0.99, -0.08] | -0.26, -0.72, -0.62 | no meaningful loss |
| code-level CP1-M1000 vs one-shot B2, both 145 epochs, K = 10 | fewer codes at no loss? | -0.32 | [-0.66, +0.00] | -0.55, -0.76, +0.34 | no meaningful loss |
| code-level CP1-M1000 at 145 epochs (K = 10) vs dense head | is dense accuracy kept? | -0.82 | [-1.29, -0.37] | -0.72, -1.03, -0.72 | loss not excluded |

## Code-level pruning (one-shot, section 12)

The M best codes by summed score (CB1: magnitude, CP1: activation-aware), then the top K edges of each class among them. Δ vs edge-level B1 at the same K (which keeps the codes its edges touch).

| Model | Codes kept (active) | Edges | Classes < K | Val top-1 | Test top-1 | Δ vs B1 | 95 % CI | Seeds (Δ) | Verdict |
|---|---|---|---|---|---|---|---|---|---|
| CB1-M1200_K10 | 680 ± 27 | 1,960 ± 0 | 0 ± 0 | 50.67 ± 1.17 | 48.81 ± 1.69 | -23.91 | [-24.86, -22.97] | -21.79, -24.67, -25.28 | loss not excluded |
| CB1-M1000_K10 | 558 ± 27 | 1,960 ± 0 | 0 ± 0 | 42.99 ± 0.95 | 42.30 ± 1.59 | -30.42 | [-31.40, -29.43] | -29.30, -31.91, -30.06 | loss not excluded |
| CB1-eqCP1M1000_K10 | 874 ± 5 | 1,960 ± 0 | 0 ± 0 | 60.78 ± 1.85 | 59.97 ± 1.91 | -12.76 | [-13.55, -11.99] | -10.76, -14.40, -13.12 | loss not excluded |
| CB1-M800_K10 | 448 ± 33 | 1,960 ± 0 | 0 ± 0 | 33.99 ± 1.12 | 33.06 ± 1.78 | -39.67 | [-40.68, -38.67] | -40.38, -40.48, -38.15 | loss not excluded |
| CB1-M600_K10 | 317 ± 23 | 1,960 ± 0 | 0 ± 0 | 21.80 ± 1.86 | 22.02 ± 2.49 | -50.71 | [-51.71, -49.72] | -50.70, -52.75, -48.66 | loss not excluded |
| CB1-M400_K10 | 237 ± 15 | 1,960 ± 0 | 0 ± 0 | 8.51 ± 2.82 | 8.65 ± 2.65 | -64.08 | [-65.00, -63.16] | -63.35, -66.61, -62.27 | loss not excluded |
| CP1-M1200_K10 | 999 ± 6 | 1,960 ± 0 | 0 ± 0 | 70.76 ± 0.79 | 71.83 ± 0.47 | -0.90 | [-1.52, -0.28] | -0.98, +0.00, -1.70 | loss not excluded |
| CP1-M1000_K10 | 874 ± 5 | 1,960 ± 0 | 0 ± 0 | 68.67 ± 1.32 | 69.85 ± 0.53 | -2.88 | [-3.56, -2.20] | -3.26, -2.05, -3.33 | loss not excluded |
| CP1-M800_K10 | 726 ± 5 | 1,960 ± 0 | 0 ± 0 | 66.22 ± 0.61 | 66.87 ± 0.23 | -5.86 | [-6.59, -5.12] | -5.63, -5.30, -6.64 | loss not excluded |
| CP1-M600_K10 | 573 ± 7 | 1,960 ± 0 | 0 ± 0 | 61.19 ± 1.72 | 61.91 ± 1.13 | -10.82 | [-11.63, -9.97] | -10.43, -9.45, -12.56 | loss not excluded |
| CP1-M400_K10 | 395 ± 1 | 1,960 ± 0 | 0 ± 0 | 52.97 ± 2.88 | 53.89 ± 0.08 | -18.84 | [-19.77, -17.90] | -18.70, -18.39, -19.41 | loss not excluded |

| Comparison | Question | Δ | 95 % CI | Seeds (Δ) | Verdict |
|---|---|---|---|---|---|
| CP1-M1000_K10 vs CB1-eqCP1M1000_K10 | does the code score help at equal active codes? | +9.88 | [+9.01, +10.74] | +7.50, +12.35, +9.79 | **better** |
| CP1-M1200_K10 vs CB1-M1200_K10 | does the activation-aware code score help? | +23.02 | [+22.05, +24.00] | +20.81, +24.67, +23.58 | **better** |
| CP1-M1000_K10 vs CB1-M1000_K10 | does the activation-aware code score help? | +27.54 | [+26.52, +28.54] | +26.04, +29.86, +26.73 | **better** |
| CP1-M800_K10 vs CB1-M800_K10 | does the activation-aware code score help? | +33.81 | [+32.80, +34.83] | +34.75, +35.18, +31.51 | **better** |
| CP1-M600_K10 vs CB1-M600_K10 | does the activation-aware code score help? | +39.89 | [+38.88, +40.87] | +40.27, +43.30, +36.10 | **better** |
| CP1-M400_K10 vs CB1-M400_K10 | does the activation-aware code score help? | +45.24 | [+44.30, +46.20] | +44.65, +48.22, +42.86 | **better** |

## Closed-form weight repair (no retraining, section 13)

The surviving weights of a pruned head re-fitted on the cached training features, mask and w ≥ 0 kept. LS: per-class non-negative least squares against the dense model's logits. CE: the same weights re-fitted to the labels (projected Adam). phys: codes no class uses are removed first, so the softmax renormalises over the kept codes (as in retraining). CEsel: CE refit with the step count (300 / 1,000 / 3,000 / 10,000) chosen per model on validation (post hoc); the chosen counts are in brackets. rtH / rtCH: the retraining recipe (145 epochs, best epoch on validation) run on the cached features, head only / codes + head (post hoc); best epochs in brackets. Δ vs the same mask one-shot (no repair, every code in the softmax).

| Model | Codes in softmax | Edges | Val top-1 | Test top-1 | Δ vs one-shot | 95 % CI | Seeds (Δ) | Verdict | Fit (s) |
|---|---|---|---|---|---|---|---|---|---|
| B1 one-shot | 1,339 ± 21 (all in softmax) | 1,960 ± 0 | 72.02 ± 0.68 | 72.73 ± 0.47 | ref |  |  |  |  |
| B1+CEsel (10000, 10000, 10000 steps) | 3,000 ± 0 | 1,739 ± 10 | 79.47 ± 0.99 | 80.03 ± 0.44 | +7.30 | [+6.73, +7.87] | +7.50, +7.20, +7.20 | **better** | 10.69 ± 0.16 |
| B1+phys+CEsel (10000, 10000, 3000 steps) | 1,339 ± 21 | 1,664 ± 174 | 80.00 ± 0.68 | 79.90 ± 0.47 | +7.17 | [+6.60, +7.74] | +7.14, +7.20, +7.18 | **better** | 5.89 ± 0.16 |
| B1+phys+rtH (best epoch 89, 79, 87) | 1,339 ± 21 | 1,927 ± 7 | 79.51 ± 0.21 | 79.19 ± 0.60 | +6.47 | [+5.96, +6.98] | +6.73, +6.18, +6.49 | **better** | 12.20 ± 0.16 |
| B1+phys+rtCH (best epoch 56, 54, 88) | 1,339 ± 21 | 1,959 ± 0 | 85.73 ± 0.07 | 86.85 ± 0.25 | +14.12 | [+13.45, +14.82] | +14.50, +14.19, +13.68 | **better** | 19.86 ± 0.32 |
| P1 one-shot | 1,486 ± 10 (all in softmax) | 1,960 ± 0 | 74.68 ± 0.50 | 75.33 ± 0.24 | ref |  |  |  |  |
| P1+CEsel (10000, 10000, 10000 steps) | 3,000 ± 0 | 1,687 ± 14 | 80.20 ± 0.79 | 80.49 ± 0.37 | +5.16 | [+4.61, +5.73] | +4.53, +5.40, +5.55 | **better** | 10.42 ± 0.05 |
| P1+phys+CEsel (3000, 10000, 10000 steps) | 1,486 ± 10 | 1,630 ± 205 | 80.94 ± 0.43 | 80.52 ± 0.06 | +5.19 | [+4.63, +5.75] | +4.92, +5.36, +5.29 | **better** | 6.12 ± 0.03 |
| P1+phys+rtH (best epoch 81, 113, 64) | 1,486 ± 10 | 1,904 ± 18 | 79.43 ± 0.39 | 80.05 ± 0.06 | +4.71 | [+4.21, +5.22] | +4.53, +4.89, +4.73 | **better** | 11.91 ± 0.03 |
| P1+phys+rtCH (best epoch 111, 49, 97) | 1,486 ± 10 | 1,954 ± 3 | 85.69 ± 0.31 | 86.62 ± 0.19 | +11.29 | [+10.61, +11.97] | +11.25, +11.37, +11.24 | **better** | 21.54 ± 0.13 |
| random one-shot | 1,427 ± 22 (all in softmax) | 1,960 ± 0 | 4.83 ± 1.67 | 5.25 ± 1.49 | ref |  |  |  |  |
| random+CEsel (10000, 10000, 10000 steps) | 3,000 ± 0 | 688 ± 33 | 6.91 ± 1.46 | 7.35 ± 1.27 | +2.10 | [+1.86, +2.35] | +1.88, +2.67, +1.75 | **better** | 10.41 ± 0.03 |
| random+phys+CEsel (10000, 10000, 10000 steps) | 1,427 ± 22 | 706 ± 33 | 7.53 ± 1.60 | 7.64 ± 1.25 | +2.39 | [+2.13, +2.64] | +2.23, +2.90, +2.04 | **better** | 6.01 ± 0.07 |
| random+phys+rtH (best epoch 88, 81, 92) | 1,427 ± 22 | 671 ± 27 | 6.79 ± 1.41 | 7.15 ± 1.23 | +1.90 | [+1.67, +2.13] | +1.75, +2.41, +1.54 | **better** | 11.61 ± 0.08 |
| random+phys+rtCH (best epoch 53, 92, 106) | 1,427 ± 22 | 1,078 ± 33 | 82.49 ± 0.80 | 83.77 ± 0.33 | +78.52 | [+77.78, +79.26] | +79.07, +79.42, +77.08 | **better** | 21.03 ± 0.10 |
| CP1-M1000 one-shot | 874 ± 5 (all in softmax) | 1,960 ± 0 | 68.67 ± 1.32 | 69.85 ± 0.53 | ref |  |  |  |  |
| CP1-M1000+CEsel (10000, 10000, 10000 steps) | 3,000 ± 0 | 1,683 ± 18 | 76.48 ± 1.19 | 77.08 ± 0.07 | +7.24 | [+6.65, +7.84] | +7.85, +6.79, +7.08 | **better** | 10.46 ± 0.01 |
| CP1-M1000+phys+CEsel (3000, 3000, 3000 steps) | 874 ± 5 | 1,772 ± 11 | 77.42 ± 0.68 | 78.33 ± 0.22 | +8.49 | [+7.92, +9.06] | +9.05, +7.79, +8.62 | **better** | 5.10 ± 0.71 |
| CP1-M1000+phys+rtH (best epoch 52, 67, 60) | 874 ± 5 | 1,883 ± 8 | 76.65 ± 1.02 | 77.44 ± 0.14 | +7.59 | [+7.04, +8.15] | +8.02, +7.10, +7.65 | **better** | 8.09 ± 0.01 |
| CP1-M1000+phys+rtCH (best epoch 87, 86, 78) | 874 ± 5 | 1,936 ± 2 | 85.28 ± 0.32 | 86.88 ± 0.33 | +17.04 | [+16.31, +17.80] | +17.83, +16.70, +16.58 | **better** | 14.35 ± 0.07 |

| Comparison | Question | Δ | 95 % CI | Seeds (Δ) | Verdict |
|---|---|---|---|---|---|
| B1 phys+CEsel vs B1 retrained 145 epochs, K = 10 | can repair replace retraining? | -8.59 | [-9.13, -8.06] | -9.20, -9.08, -7.49 | loss not excluded |
| B1 phys+CEsel (K = 10) vs dense head | is dense accuracy kept? | -9.09 | [-9.68, -8.50] | -9.38, -9.35, -8.54 | loss not excluded |
| B1 phys+rtH vs B1 retrained 145 epochs, K = 10 | can repair replace retraining? | -9.29 | [-9.85, -8.73] | -9.61, -10.10, -8.17 | loss not excluded |
| B1 phys+rtH (K = 10) vs dense head | is dense accuracy kept? | -9.80 | [-10.41, -9.19] | -9.79, -10.37, -9.23 | loss not excluded |
| B1 phys+rtCH vs B1 retrained 145 epochs, K = 10 | can repair replace retraining? | -1.64 | [-2.03, -1.25] | -1.84, -2.09, -0.98 | loss not excluded |
| B1 phys+rtCH (K = 10) vs dense head | is dense accuracy kept? | -2.14 | [-2.64, -1.65] | -2.01, -2.36, -2.04 | loss not excluded |
| B1: codes + head vs head only, retraining recipe on cached features | does adapting the codes help? | +7.66 | [+7.15, +8.18] | +7.77, +8.01, +7.19 | **better** |
| P1 phys+CEsel (K = 10) vs dense head | is dense accuracy kept? | -8.46 | [-9.08, -7.86] | -8.54, -8.46, -8.39 | loss not excluded |
| P1 phys+rtH (K = 10) vs dense head | is dense accuracy kept? | -8.94 | [-9.58, -8.32] | -8.94, -8.93, -8.95 | loss not excluded |
| P1 phys+rtCH (K = 10) vs dense head | is dense accuracy kept? | -2.37 | [-2.88, -1.85] | -2.21, -2.45, -2.44 | loss not excluded |
| P1: codes + head vs head only, retraining recipe on cached features | does adapting the codes help? | +6.57 | [+6.05, +7.12] | +6.73, +6.48, +6.52 | **better** |
| random phys+CEsel (K = 10) vs dense head | is dense accuracy kept? | -81.35 | [-82.08, -80.63] | -82.49, -81.59, -79.98 | loss not excluded |
| random phys+rtH (K = 10) vs dense head | is dense accuracy kept? | -81.84 | [-82.56, -81.13] | -82.96, -82.08, -80.48 | loss not excluded |
| random phys+rtCH (K = 10) vs dense head | is dense accuracy kept? | -5.22 | [-5.77, -4.67] | -5.65, -5.07, -4.94 | loss not excluded |
| random: codes + head vs head only, retraining recipe on cached features | does adapting the codes help? | +76.62 | [+75.86, +77.35] | +77.32, +77.01, +75.54 | **better** |
| CP1-M1000 phys+CEsel vs CP1-M1000 retrained 145 epochs, K = 10 | can repair replace retraining? | -9.83 | [-10.44, -9.24] | -10.00, -9.79, -9.71 | loss not excluded |
| CP1-M1000 phys+CEsel (K = 10) vs dense head | is dense accuracy kept? | -10.66 | [-11.31, -10.01] | -10.72, -10.82, -10.43 | loss not excluded |
| CP1-M1000 phys+rtH vs CP1-M1000 retrained 145 epochs, K = 10 | can repair replace retraining? | -10.73 | [-11.35, -10.09] | -11.03, -10.47, -10.68 | loss not excluded |
| CP1-M1000 phys+rtH (K = 10) vs dense head | is dense accuracy kept? | -11.55 | [-12.22, -10.89] | -11.75, -11.50, -11.40 | loss not excluded |
| CP1-M1000 phys+rtCH vs CP1-M1000 retrained 145 epochs, K = 10 | can repair replace retraining? | -1.28 | [-1.68, -0.88] | -1.22, -0.87, -1.75 | loss not excluded |
| CP1-M1000 phys+rtCH (K = 10) vs dense head | is dense accuracy kept? | -2.11 | [-2.60, -1.60] | -1.94, -1.90, -2.47 | loss not excluded |
| CP1-M1000: codes + head vs head only, retraining recipe on cached features | does adapting the codes help? | +9.45 | [+8.86, +10.06] | +9.81, +9.60, +8.93 | **better** |
| P1 vs B1, both physical + CEsel repair | does the score matter after repair? | +0.63 | [+0.17, +1.07] | +0.83, +0.90, +0.15 | **better** |
| P1 vs B1, both physical + rtH repair | does the score matter after repair? | +0.85 | [+0.42, +1.29] | +0.85, +1.44, +0.27 | **better** |
| P1 vs B1, both physical + rtCH repair | does the score matter after repair? | -0.23 | [-0.52, +0.07] | -0.20, -0.09, -0.40 | no meaningful loss |

## Ablations (Phase 4)

Same rules as above; the reference is the second method named.

| Comparison | Question | Δ | 95 % CI | Seeds (Δ) | Verdict |
|---|---|---|---|---|---|
| P1-contrast vs P1, one-shot, K = 5 | does contrasting against other classes help? | -0.01 | [-0.15, +0.13] | +0.04, +0.14, -0.20 | no meaningful loss |
| P1-contrast vs P1, one-shot, K = 10 | does contrasting against other classes help? | -0.13 | [-0.25, -0.01] | -0.15, -0.11, -0.12 | no meaningful loss |
| P1-contrast vs P1, one-shot, K = 20 | does contrasting against other classes help? | +0.01 | [-0.14, +0.16] | +0.22, -0.14, -0.06 | no meaningful loss |

## Deletion-effect control

100 class–code edges (a third each from low, medium and high P1 scores) removed one at a time from the dense head; damage = change in validation cross-entropy. Spearman ρ between damage and each score (higher = the score predicts real damage better).

| Seed | Magnitude (B1) | P1 | P1-contrast |
|---|---|---|---|
| 40 | -0.132 | -0.122 | 0.062 |
| 41 | 0.005 | -0.079 | 0.076 |
| 42 | -0.013 | -0.108 | 0.113 |
| mean | -0.05 ± 0.07 | -0.10 ± 0.02 | 0.08 ± 0.03 |
