# Phase 2 results

Seeds: 40. Test top-1 is mean ± sd over seeds. Δ = method − reference (points), with the 95 % paired-bootstrap interval over test images (10,000 resamples, correctness averaged over seeds). Verdict rules fixed in Phase 0: better = interval above 0 and same sign in every seed; no meaningful loss = lower bound above −1; otherwise a loss of 1 point or more is not excluded.

Dense head (no pruning): test 89.06 %, 353,640 nonzero edges.

## One-shot masking (no retraining)

### K = 5 edges per class (N = 1,000)

| Method | Val top-1 | Test top-1 | Δ vs B1 | 95 % CI | Seeds (Δ) | Verdict | Edges | Active codes |
|---|---|---|---|---|---|---|---|---|
| B1 | 67.48 | 67.91 | ref |  |  |  | 980 | 830 |
| random | 4.66 | 4.22 | -63.70 | [-64.79, -62.59] | -63.70 | loss not excluded | 980 | 831 |
| P1 | 69.20 | 71.63 | +3.72 | [+2.91, +4.55] | +3.72 | **better** | 980 | 872 |
| P1contrast | 69.33 | 71.67 | +3.76 | [+2.95, +4.59] | +3.76 | **better** | 980 | 876 |
| P3 (min-1 classes: 6) | 65.03 | 66.92 | -0.99 | [-1.85, -0.14] | -0.99 | loss not excluded | 980 | 893 |

### K = 10 edges per class (N = 2,000)

| Method | Val top-1 | Test top-1 | Δ vs B1 | 95 % CI | Seeds (Δ) | Verdict | Edges | Active codes |
|---|---|---|---|---|---|---|---|---|
| B1 | 72.15 | 72.54 | ref |  |  |  | 1,960 | 1,347 |
| random | 3.80 | 4.34 | -68.20 | [-69.26, -67.14] | -68.20 | loss not excluded | 1,960 | 1,409 |
| P1 | 74.97 | 75.59 | +3.05 | [+2.31, +3.79] | +3.05 | **better** | 1,960 | 1,481 |
| P1contrast | 74.48 | 75.44 | +2.90 | [+2.19, +3.64] | +2.90 | **better** | 1,960 | 1,497 |
| P3 (min-1 classes: 1) | 70.80 | 73.49 | +0.95 | [+0.14, +1.75] | +0.95 | **better** | 1,960 | 1,483 |

### K = 20 edges per class (N = 4,000)

| Method | Val top-1 | Test top-1 | Δ vs B1 | 95 % CI | Seeds (Δ) | Verdict | Edges | Active codes |
|---|---|---|---|---|---|---|---|---|
| B1 | 77.67 | 77.58 | ref |  |  |  | 3,920 | 1,905 |
| random | 12.64 | 11.54 | -66.04 | [-67.17, -64.94] | -66.04 | loss not excluded | 3,920 | 2,127 |
| P1 | 79.51 | 80.15 | +2.57 | [+1.92, +3.23] | +2.57 | **better** | 3,920 | 2,167 |
| P1contrast | 79.39 | 80.38 | +2.80 | [+2.13, +3.46] | +2.80 | **better** | 3,920 | 2,174 |
| P3 (min-1 classes: 0) | 78.77 | 78.92 | +1.34 | [+0.62, +2.06] | +1.34 | **better** | 3,920 | 2,149 |

## Iterative pruning and longer retraining (Phase 4)

Iterative: 20 % of each class's edges removed per round, 5 epochs of retraining and rescoring between rounds, full codebook during the rounds; then unused codes removed and 30 epochs (the B2 recipe). Control: the one-shot B1 mask retrained with the B2 recipe for as many epochs in total. Same rules as above.

| Model | Rounds | Total epochs | Codes | Best val | Test top-1 |
|---|---|---|---|---|---|
| B2_K10 one-shot, 145 epochs (control) | - | 145 | 1,347 | 88.47 | 88.88 |
| CP1-M1000_K10 code-level, 145 epochs | - | 145 | 879 | 87.85 | 88.33 |

| Comparison | Question | Δ | 95 % CI | Seeds (Δ) | Verdict |
|---|---|---|---|---|---|
| one-shot B2 at 145 epochs (K = 10) vs dense head | is dense accuracy kept? | -0.17 | [-0.73, +0.39] | -0.17 | no meaningful loss |
| code-level CP1-M1000 vs one-shot B2, both 145 epochs, K = 10 | fewer codes at no loss? | -0.55 | [-1.06, -0.02] | -0.55 | loss not excluded |
| code-level CP1-M1000 at 145 epochs (K = 10) vs dense head | is dense accuracy kept? | -0.72 | [-1.28, -0.16] | -0.72 | loss not excluded |

## Code-level pruning (one-shot, section 12)

The M best codes by summed score (CB1: magnitude, CP1: activation-aware), then the top K edges of each class among them. Δ vs edge-level B1 at the same K (which keeps the codes its edges touch).

| Model | Codes kept (active) | Edges | Classes < K | Val top-1 | Test top-1 | Δ vs B1 | 95 % CI | Seeds (Δ) | Verdict |
|---|---|---|---|---|---|---|---|---|---|
| CB1-M1200_K10 | 665 | 1,960 | 0 | 51.90 | 50.75 | -21.79 | [-23.03, -20.52] | -21.79 | loss not excluded |
| CB1-M1000_K10 | 550 | 1,960 | 0 | 42.21 | 43.24 | -29.30 | [-30.61, -27.99] | -29.30 | loss not excluded |
| CB1-M800_K10 | 445 | 1,960 | 0 | 33.74 | 32.16 | -40.38 | [-41.67, -39.07] | -40.38 | loss not excluded |
| CB1-M600_K10 | 320 | 1,960 | 0 | 21.47 | 21.84 | -50.70 | [-51.93, -49.43] | -50.70 | loss not excluded |
| CB1-M400_K10 | 229 | 1,960 | 0 | 8.59 | 9.19 | -63.35 | [-64.49, -62.24] | -63.35 | loss not excluded |
| CP1-M1200_K10 | 994 | 1,960 | 0 | 70.43 | 71.56 | -0.98 | [-1.87, -0.10] | -0.98 | loss not excluded |
| CP1-M1000_K10 | 879 | 1,960 | 0 | 67.73 | 69.28 | -3.26 | [-4.22, -2.31] | -3.26 | loss not excluded |
| CP1-M800_K10 | 728 | 1,960 | 0 | 66.63 | 66.91 | -5.63 | [-6.62, -4.66] | -5.63 | loss not excluded |
| CP1-M600_K10 | 580 | 1,960 | 0 | 61.72 | 62.11 | -10.43 | [-11.52, -9.36] | -10.43 | loss not excluded |
| CP1-M400_K10 | 396 | 1,960 | 0 | 53.01 | 53.84 | -18.70 | [-19.89, -17.54] | -18.70 | loss not excluded |

| Comparison | Question | Δ | 95 % CI | Seeds (Δ) | Verdict |
|---|---|---|---|---|---|
| CP1-M1200_K10 vs CB1-M1200_K10 | does the activation-aware code score help? | +20.81 | [+19.50, +22.09] | +20.81 | **better** |
| CP1-M1000_K10 vs CB1-M1000_K10 | does the activation-aware code score help? | +26.04 | [+24.70, +27.36] | +26.04 | **better** |
| CP1-M800_K10 vs CB1-M800_K10 | does the activation-aware code score help? | +34.75 | [+33.38, +36.05] | +34.75 | **better** |
| CP1-M600_K10 vs CB1-M600_K10 | does the activation-aware code score help? | +40.27 | [+38.94, +41.55] | +40.27 | **better** |
| CP1-M400_K10 vs CB1-M400_K10 | does the activation-aware code score help? | +44.65 | [+43.43, +45.86] | +44.65 | **better** |

## Closed-form weight repair (no retraining, section 13)

The surviving weights of a pruned head re-fitted on the cached training features, mask and w ≥ 0 kept. LS: per-class non-negative least squares against the dense model's logits. CE: the same weights re-fitted to the labels (projected Adam). phys: codes no class uses are removed first, so the softmax renormalises over the kept codes (as in retraining). CEsel: CE refit with the step count (300 / 1,000 / 3,000 / 10,000) chosen per model on validation (post hoc); the chosen counts are in brackets. rtH / rtCH: the retraining recipe (145 epochs, best epoch on validation) run on the cached features, head only / codes + head (post hoc); best epochs in brackets. Δ vs the same mask one-shot (no repair, every code in the softmax).

| Model | Codes in softmax | Edges | Val top-1 | Test top-1 | Δ vs one-shot | 95 % CI | Seeds (Δ) | Verdict | Fit (s) |
|---|---|---|---|---|---|---|---|---|---|
| B1 one-shot | 1,347 (all in softmax) | 1,960 | 72.15 | 72.54 | ref |  |  |  |  |
| B1+CEsel (10000 steps) | 3,000 | 1,751 | 78.90 | 80.04 | +7.50 | [+6.70, +8.29] | +7.50 | **better** | 10.87 |
| B1+phys+CEsel (10000 steps) | 1,347 | 1,574 | 79.26 | 79.68 | +7.14 | [+6.31, +7.97] | +7.14 | **better** | 5.71 |
| B1+phys+rtH (best epoch 89) | 1,347 | 1,920 | 79.63 | 79.27 | +6.73 | [+5.98, +7.49] | +6.73 | **better** | 12.35 |
| B1+phys+rtCH (best epoch 56) | 1,347 | 1,959 | 85.77 | 87.04 | +14.50 | [+13.63, +15.41] | +14.50 | **better** | 20.10 |
| P1 one-shot | 1,481 (all in softmax) | 1,960 | 74.97 | 75.59 | ref |  |  |  |  |
| P1+CEsel (10000 steps) | 3,000 | 1,676 | 79.63 | 80.11 | +4.53 | [+3.77, +5.29] | +4.53 | **better** | 10.43 |
| P1+phys+CEsel (3000 steps) | 1,481 | 1,867 | 80.49 | 80.51 | +4.92 | [+4.22, +5.62] | +4.92 | **better** | 6.08 |
| P1+phys+rtH (best epoch 81) | 1,481 | 1,904 | 79.14 | 80.11 | +4.53 | [+3.84, +5.19] | +4.53 | **better** | 11.94 |
| P1+phys+rtCH (best epoch 111) | 1,481 | 1,954 | 85.40 | 86.84 | +11.25 | [+10.45, +12.08] | +11.25 | **better** | 21.48 |
| random one-shot | 1,409 (all in softmax) | 1,960 | 3.80 | 4.34 | ref |  |  |  |  |
| random+CEsel (10000 steps) | 3,000 | 675 | 5.28 | 6.22 | +1.88 | [+1.49, +2.28] | +1.88 | **better** | 10.41 |
| random+phys+CEsel (10000 steps) | 1,409 | 687 | 5.89 | 6.57 | +2.23 | [+1.82, +2.65] | +2.23 | **better** | 5.99 |
| random+phys+rtH (best epoch 88) | 1,409 | 654 | 5.40 | 6.09 | +1.75 | [+1.38, +2.14] | +1.75 | **better** | 11.70 |
| random+phys+rtCH (best epoch 53) | 1,409 | 1,076 | 83.31 | 83.41 | +79.07 | [+78.16, +79.97] | +79.07 | **better** | 20.92 |
| CP1-M1000 one-shot | 879 (all in softmax) | 1,960 | 67.73 | 69.28 | ref |  |  |  |  |
| CP1-M1000+CEsel (10000 steps) | 3,000 | 1,668 | 75.46 | 77.13 | +7.85 | [+7.00, +8.71] | +7.85 | **better** | 10.47 |
| CP1-M1000+phys+CEsel (3000 steps) | 879 | 1,765 | 76.81 | 78.34 | +9.05 | [+8.26, +9.89] | +9.05 | **better** | 4.86 |
| CP1-M1000+phys+rtH (best epoch 52) | 879 | 1,890 | 76.32 | 77.30 | +8.02 | [+7.23, +8.83] | +8.02 | **better** | 8.07 |
| CP1-M1000+phys+rtCH (best epoch 87) | 879 | 1,938 | 85.40 | 87.12 | +17.83 | [+16.89, +18.83] | +17.83 | **better** | 14.39 |

| Comparison | Question | Δ | 95 % CI | Seeds (Δ) | Verdict |
|---|---|---|---|---|---|
| B1 phys+CEsel vs B1 retrained 145 epochs, K = 10 | can repair replace retraining? | -9.20 | [-9.96, -8.47] | -9.20 | loss not excluded |
| B1 phys+CEsel (K = 10) vs dense head | is dense accuracy kept? | -9.38 | [-10.16, -8.61] | -9.38 | loss not excluded |
| B1 phys+rtH vs B1 retrained 145 epochs, K = 10 | can repair replace retraining? | -9.61 | [-10.37, -8.85] | -9.61 | loss not excluded |
| B1 phys+rtH (K = 10) vs dense head | is dense accuracy kept? | -9.79 | [-10.56, -9.04] | -9.79 | loss not excluded |
| B1 phys+rtCH vs B1 retrained 145 epochs, K = 10 | can repair replace retraining? | -1.84 | [-2.36, -1.31] | -1.84 | loss not excluded |
| B1 phys+rtCH (K = 10) vs dense head | is dense accuracy kept? | -2.01 | [-2.62, -1.42] | -2.01 | loss not excluded |
| B1: codes + head vs head only, retraining recipe on cached features | does adapting the codes help? | +7.77 | [+7.06, +8.49] | +7.77 | **better** |
| P1 phys+CEsel (K = 10) vs dense head | is dense accuracy kept? | -8.54 | [-9.31, -7.81] | -8.54 | loss not excluded |
| P1 phys+rtH (K = 10) vs dense head | is dense accuracy kept? | -8.94 | [-9.70, -8.20] | -8.94 | loss not excluded |
| P1 phys+rtCH (K = 10) vs dense head | is dense accuracy kept? | -2.21 | [-2.82, -1.60] | -2.21 | loss not excluded |
| P1: codes + head vs head only, retraining recipe on cached features | does adapting the codes help? | +6.73 | [+6.04, +7.44] | +6.73 | **better** |
| random phys+CEsel (K = 10) vs dense head | is dense accuracy kept? | -82.49 | [-83.34, -81.62] | -82.49 | loss not excluded |
| random phys+rtH (K = 10) vs dense head | is dense accuracy kept? | -82.96 | [-83.81, -82.12] | -82.96 | loss not excluded |
| random phys+rtCH (K = 10) vs dense head | is dense accuracy kept? | -5.65 | [-6.37, -4.95] | -5.65 | loss not excluded |
| random: codes + head vs head only, retraining recipe on cached features | does adapting the codes help? | +77.32 | [+76.35, +78.26] | +77.32 | **better** |
| CP1-M1000 phys+CEsel vs CP1-M1000 retrained 145 epochs, K = 10 | can repair replace retraining? | -10.00 | [-10.78, -9.22] | -10.00 | loss not excluded |
| CP1-M1000 phys+CEsel (K = 10) vs dense head | is dense accuracy kept? | -10.72 | [-11.54, -9.92] | -10.72 | loss not excluded |
| CP1-M1000 phys+rtH vs CP1-M1000 retrained 145 epochs, K = 10 | can repair replace retraining? | -11.03 | [-11.84, -10.21] | -11.03 | loss not excluded |
| CP1-M1000 phys+rtH (K = 10) vs dense head | is dense accuracy kept? | -11.75 | [-12.59, -10.93] | -11.75 | loss not excluded |
| CP1-M1000 phys+rtCH vs CP1-M1000 retrained 145 epochs, K = 10 | can repair replace retraining? | -1.22 | [-1.77, -0.67] | -1.22 | loss not excluded |
| CP1-M1000 phys+rtCH (K = 10) vs dense head | is dense accuracy kept? | -1.94 | [-2.56, -1.32] | -1.94 | loss not excluded |
| CP1-M1000: codes + head vs head only, retraining recipe on cached features | does adapting the codes help? | +9.81 | [+9.04, +10.61] | +9.81 | **better** |
| P1 vs B1, both physical + CEsel repair | does the score matter after repair? | +0.83 | [+0.12, +1.55] | +0.83 | **better** |
| P1 vs B1, both physical + rtH repair | does the score matter after repair? | +0.85 | [+0.17, +1.50] | +0.85 | **better** |
| P1 vs B1, both physical + rtCH repair | does the score matter after repair? | -0.20 | [-0.67, +0.29] | -0.20 | no meaningful loss |

## Ablations (Phase 4)

Same rules as above; the reference is the second method named.

| Comparison | Question | Δ | 95 % CI | Seeds (Δ) | Verdict |
|---|---|---|---|---|---|
| P1-contrast vs P1, one-shot, K = 5 | does contrasting against other classes help? | +0.04 | [-0.19, +0.26] | +0.04 | no meaningful loss |
| P1-contrast vs P1, one-shot, K = 10 | does contrasting against other classes help? | -0.15 | [-0.40, +0.09] | -0.15 | no meaningful loss |
| P1-contrast vs P1, one-shot, K = 20 | does contrasting against other classes help? | +0.22 | [-0.05, +0.49] | +0.22 | no meaningful loss |

## Deletion-effect control

100 class–code edges (a third each from low, medium and high P1 scores) removed one at a time from the dense head; damage = change in validation cross-entropy. Spearman ρ between damage and each score (higher = the score predicts real damage better).

| Seed | Magnitude (B1) | P1 | P1-contrast |
|---|---|---|---|
| 40 | -0.132 | -0.122 | 0.062 |
| mean | -0.13 | -0.12 | 0.06 |
