# Interpretability

Dataset cub200, seeds 40, 41, 42, mean ± std over seeds (n = seeds with the model). Definitions: `scripts/interpretability.py`. No model is trained or selected here.

## Part purity

Share of a code's top-10 patches showing the same part; higher is better. Std over codes in brackets. "Purity change" compares the same code with the dense model (train images), so it is not affected by which codes were kept.

| Model | n | Codes | Purity train | Purity test | Codes with purity > 0.5 (train) | Purity change, same code |
|---|---|---|---|---|---|---|
| Dense | 3 | 3000 ± 0 | 0.340 ± 0.003 [0.28] | 0.345 ± 0.003 [0.27] | 27.0 ± 0.6 % | – |
| Magnitude: pruned, no retraining | 3 | 1304 ± 31 | 0.418 ± 0.006 [0.26] | 0.420 ± 0.004 [0.26] | 34.9 ± 1.2 % | 0.000 ± 0.000 |
| Magnitude: pruned, unused codes removed | 3 | 1304 ± 31 | 0.408 ± 0.004 [0.26] | 0.407 ± 0.004 [0.26] | 33.8 ± 1.1 % | -0.011 ± 0.002 |
| Magnitude: cached features, head only | 3 | 1297 ± 34 | 0.407 ± 0.003 [0.26] | 0.407 ± 0.003 [0.26] | 33.8 ± 1.1 % | -0.011 ± 0.002 |
| Magnitude: cached features, codes + head | 3 | 1304 ± 31 | 0.391 ± 0.003 [0.25] | 0.384 ± 0.004 [0.25] | 30.0 ± 0.5 % | -0.027 ± 0.004 |
| Magnitude: retrained on images, 145 epochs | 3 | 1304 ± 31 | 0.415 ± 0.003 [0.27] | 0.413 ± 0.002 [0.26] | 34.6 ± 1.0 % | -0.003 ± 0.004 |
| Activation-aware: pruned, no retraining | 3 | 1574 ± 17 | 0.415 ± 0.010 [0.27] | 0.415 ± 0.009 [0.27] | 35.9 ± 2.2 % | 0.000 ± 0.000 |
| Activation-aware: pruned, unused codes removed | 3 | 1574 ± 17 | 0.408 ± 0.010 [0.27] | 0.405 ± 0.008 [0.27] | 34.9 ± 2.0 % | -0.006 ± 0.001 |
| Activation-aware: cached features, head only | 3 | 1573 ± 17 | 0.408 ± 0.010 [0.27] | 0.405 ± 0.008 [0.27] | 34.9 ± 2.0 % | -0.006 ± 0.001 |
| Activation-aware: cached features, codes + head | 3 | 1574 ± 17 | 0.389 ± 0.008 [0.26] | 0.387 ± 0.007 [0.26] | 31.2 ± 0.6 % | -0.025 ± 0.004 |
| Code-level magnitude: pruned, no retraining | 3 | 890 ± 4 | 0.378 ± 0.004 [0.26] | 0.377 ± 0.002 [0.26] | 28.5 ± 0.5 % | 0.000 ± 0.000 |
| Code-level magnitude: pruned, unused codes removed | 3 | 890 ± 4 | 0.376 ± 0.009 [0.26] | 0.370 ± 0.004 [0.26] | 29.4 ± 1.7 % | -0.002 ± 0.005 |
| Code-level magnitude: cached features, head only | 3 | 886 ± 5 | 0.378 ± 0.009 [0.26] | 0.371 ± 0.004 [0.26] | 29.5 ± 1.8 % | -0.002 ± 0.005 |
| Code-level magnitude: cached features, codes + head | 3 | 890 ± 4 | 0.370 ± 0.003 [0.26] | 0.367 ± 0.005 [0.25] | 26.1 ± 1.2 % | -0.008 ± 0.005 |
| Code-level magnitude: retrained on images, 145 epochs | 3 | 889 ± 4 | 0.401 ± 0.004 [0.27] | 0.395 ± 0.001 [0.27] | 32.8 ± 1.5 % | 0.023 ± 0.003 |
| Code-level activation-aware: pruned, no retraining | 3 | 890 ± 4 | 0.431 ± 0.012 [0.29] | 0.426 ± 0.007 [0.28] | 40.2 ± 3.1 % | 0.000 ± 0.000 |
| Code-level activation-aware: pruned, unused codes removed | 3 | 890 ± 4 | 0.417 ± 0.010 [0.28] | 0.414 ± 0.008 [0.28] | 37.9 ± 3.0 % | -0.014 ± 0.001 |
| Code-level activation-aware: cached features, head only | 3 | 887 ± 5 | 0.417 ± 0.010 [0.28] | 0.414 ± 0.007 [0.28] | 38.0 ± 3.0 % | -0.014 ± 0.001 |
| Code-level activation-aware: cached features, codes + head | 3 | 890 ± 4 | 0.387 ± 0.006 [0.26] | 0.384 ± 0.002 [0.26] | 30.7 ± 1.5 % | -0.043 ± 0.006 |
| Code-level activation-aware: retrained on images, 145 epochs | 3 | 890 ± 4 | 0.424 ± 0.008 [0.28] | 0.418 ± 0.004 [0.28] | 38.7 ± 1.1 % | -0.007 ± 0.008 |
| Random: pruned, no retraining | 3 | 1432 ± 8 | 0.322 ± 0.003 [0.28] | 0.327 ± 0.003 [0.27] | 25.0 ± 0.4 % | 0.000 ± 0.000 |
| Random: pruned, unused codes removed | 3 | 1432 ± 8 | 0.325 ± 0.003 [0.28] | 0.329 ± 0.002 [0.27] | 25.5 ± 0.6 % | 0.003 ± 0.001 |
| Random: cached features, head only | 3 | 444 ± 25 | 0.275 ± 0.016 [0.28] | 0.283 ± 0.014 [0.27] | 20.8 ± 1.4 % | 0.006 ± 0.001 |
| Random: cached features, codes + head | 3 | 878 ± 33 | 0.256 ± 0.002 [0.27] | 0.263 ± 0.001 [0.26] | 18.7 ± 1.0 % | -0.027 ± 0.006 |
| Chance (random patches) | 3 | – | 0.108 ± 0.002 [0.07] | 0.109 ± 0.002 [0.07] | 0.0 ± 0.0 % | – |

## Explanation size

Per class: nonzero edges, and edges with weight > 0.1 (the repository's local size). Per test image: codes contributing > 0.001 to the predicted class (the repository's rule), and codes that make up 90 % of its score (independent of the weight scale).

| Model | n | Edges / class | Edges > 0.1 / class | Codes used | Codes / image (predicted class) | Codes for 90 % of the score | Test top-1 (check) |
|---|---|---|---|---|---|---|---|
| Dense | 3 | 2114.3 ± 17.6 | 1974.0 ± 24.2 | 3000 ± 0 | 6.92 ± 0.38 | 247.16 ± 7.03 | 85.65 ± 0.14 |
| Magnitude: pruned, no retraining | 3 | 10.0 ± 0.0 | 10.0 ± 0.0 | 1304 ± 31 | 2.44 ± 0.04 | 6.31 ± 0.09 | 77.06 ± 0.18 |
| Magnitude: pruned, unused codes removed | 3 | 10.0 ± 0.0 | 10.0 ± 0.0 | 1304 ± 31 | 4.30 ± 0.08 | 6.43 ± 0.08 | 76.69 ± 0.29 |
| Magnitude: cached features, head only | 3 | 9.9 ± 0.1 | 9.9 ± 0.1 | 1297 ± 34 | 4.20 ± 0.07 | 6.30 ± 0.08 | 80.46 ± 0.51 |
| Magnitude: cached features, codes + head | 3 | 10.0 ± 0.0 | 10.0 ± 0.0 | 1304 ± 31 | 3.41 ± 0.08 | 5.63 ± 0.13 | 84.64 ± 0.15 |
| Magnitude: retrained on images, 145 epochs | 3 | 10.0 ± 0.0 | 10.0 ± 0.0 | 1304 ± 31 | 3.40 ± 0.29 | 6.77 ± 0.06 | 85.47 ± 0.42 |
| Activation-aware: pruned, no retraining | 3 | 10.0 ± 0.0 | 10.0 ± 0.0 | 1574 ± 17 | 4.22 ± 0.12 | 7.27 ± 0.02 | 79.52 ± 0.16 |
| Activation-aware: pruned, unused codes removed | 3 | 10.0 ± 0.0 | 10.0 ± 0.0 | 1574 ± 17 | 5.82 ± 0.09 | 7.41 ± 0.03 | 79.32 ± 0.10 |
| Activation-aware: cached features, head only | 3 | 10.0 ± 0.0 | 10.0 ± 0.0 | 1573 ± 17 | 5.76 ± 0.07 | 7.29 ± 0.06 | 80.66 ± 0.04 |
| Activation-aware: cached features, codes + head | 3 | 10.0 ± 0.0 | 10.0 ± 0.0 | 1574 ± 17 | 4.02 ± 0.07 | 6.43 ± 0.04 | 83.99 ± 0.27 |
| Code-level magnitude: pruned, no retraining | 3 | 10.0 ± 0.0 | 10.0 ± 0.0 | 890 ± 4 | 1.06 ± 0.07 | 5.88 ± 0.04 | 62.95 ± 2.76 |
| Code-level magnitude: pruned, unused codes removed | 3 | 10.0 ± 0.0 | 10.0 ± 0.0 | 890 ± 4 | 3.52 ± 0.13 | 6.04 ± 0.06 | 66.00 ± 1.06 |
| Code-level magnitude: cached features, head only | 3 | 9.9 ± 0.0 | 9.9 ± 0.1 | 886 ± 5 | 3.37 ± 0.08 | 5.95 ± 0.04 | 75.99 ± 0.48 |
| Code-level magnitude: cached features, codes + head | 3 | 10.0 ± 0.0 | 9.9 ± 0.0 | 890 ± 4 | 3.18 ± 0.40 | 5.00 ± 0.41 | 84.13 ± 0.79 |
| Code-level magnitude: retrained on images, 145 epochs | 3 | 10.0 ± 0.0 | 10.0 ± 0.0 | 889 ± 4 | 3.28 ± 0.09 | 5.82 ± 0.02 | 85.01 ± 0.03 |
| Code-level activation-aware: pruned, no retraining | 3 | 10.0 ± 0.0 | 10.0 ± 0.0 | 890 ± 4 | 3.25 ± 0.12 | 6.59 ± 0.08 | 71.14 ± 0.44 |
| Code-level activation-aware: pruned, unused codes removed | 3 | 10.0 ± 0.0 | 10.0 ± 0.0 | 890 ± 4 | 5.58 ± 0.14 | 6.74 ± 0.07 | 73.47 ± 0.54 |
| Code-level activation-aware: cached features, head only | 3 | 9.7 ± 0.3 | 9.6 ± 0.3 | 887 ± 5 | 5.36 ± 0.19 | 6.19 ± 0.25 | 76.73 ± 0.43 |
| Code-level activation-aware: cached features, codes + head | 3 | 9.9 ± 0.0 | 9.9 ± 0.0 | 890 ± 4 | 3.00 ± 0.11 | 4.45 ± 0.09 | 83.88 ± 0.18 |
| Code-level activation-aware: retrained on images, 145 epochs | 3 | 9.9 ± 0.0 | 9.9 ± 0.0 | 890 ± 4 | 3.65 ± 0.10 | 5.80 ± 0.04 | 85.15 ± 0.17 |
| Random: pruned, no retraining | 3 | 10.0 ± 0.0 | 9.3 ± 0.1 | 1432 ± 8 | 0.93 ± 0.01 | 1.36 ± 0.06 | 4.93 ± 0.39 |
| Random: pruned, unused codes removed | 3 | 10.0 ± 0.0 | 9.3 ± 0.1 | 1432 ± 8 | 1.09 ± 0.02 | 1.39 ± 0.05 | 5.03 ± 0.32 |
| Random: cached features, head only | 3 | 2.5 ± 0.1 | 2.5 ± 0.1 | 444 ± 25 | 0.96 ± 0.02 | 1.26 ± 0.01 | 7.44 ± 1.11 |
| Random: cached features, codes + head | 3 | 4.8 ± 0.2 | 4.8 ± 0.2 | 878 ± 33 | 2.05 ± 0.07 | 2.42 ± 0.06 | 80.80 ± 0.98 |

## Meaning kept (same code, compared with the dense model)

Top-10 training patches of each used code: share that are the same image and position, and the same image. Cosine between the code vectors.

| Model | n | Same patches (%) | Same images (%) | Code cosine |
|---|---|---|---|---|
| Magnitude: pruned, no retraining | 3 | 100.0 ± 0.0 | 100.0 ± 0.0 | 1.000 ± 0.000 |
| Magnitude: pruned, unused codes removed | 3 | 85.2 ± 0.3 | 88.5 ± 0.1 | 1.000 ± 0.000 |
| Magnitude: cached features, head only | 3 | 85.2 ± 0.2 | 88.6 ± 0.1 | 1.000 ± 0.000 |
| Magnitude: cached features, codes + head | 3 | 39.9 ± 1.8 | 50.9 ± 1.6 | 0.730 ± 0.013 |
| Magnitude: retrained on images, 145 epochs | 3 | 30.5 ± 1.8 | 44.4 ± 1.6 | 0.781 ± 0.012 |
| Activation-aware: pruned, no retraining | 3 | 100.0 ± 0.0 | 100.0 ± 0.0 | 1.000 ± 0.000 |
| Activation-aware: pruned, unused codes removed | 3 | 88.7 ± 0.3 | 91.5 ± 0.4 | 1.000 ± 0.000 |
| Activation-aware: cached features, head only | 3 | 88.7 ± 0.3 | 91.5 ± 0.4 | 1.000 ± 0.000 |
| Activation-aware: cached features, codes + head | 3 | 49.7 ± 0.7 | 61.4 ± 0.5 | 0.766 ± 0.006 |
| Code-level magnitude: pruned, no retraining | 3 | 100.0 ± 0.0 | 100.0 ± 0.0 | 1.000 ± 0.000 |
| Code-level magnitude: pruned, unused codes removed | 3 | 79.3 ± 0.1 | 82.9 ± 0.1 | 1.000 ± 0.000 |
| Code-level magnitude: cached features, head only | 3 | 79.4 ± 0.1 | 82.9 ± 0.1 | 1.000 ± 0.000 |
| Code-level magnitude: cached features, codes + head | 3 | 26.2 ± 2.6 | 35.8 ± 2.6 | 0.668 ± 0.037 |
| Code-level magnitude: retrained on images, 145 epochs | 3 | 21.2 ± 1.3 | 32.9 ± 1.4 | 0.702 ± 0.015 |
| Code-level activation-aware: pruned, no retraining | 3 | 100.0 ± 0.0 | 100.0 ± 0.0 | 1.000 ± 0.000 |
| Code-level activation-aware: pruned, unused codes removed | 3 | 78.3 ± 0.3 | 82.9 ± 0.2 | 1.000 ± 0.000 |
| Code-level activation-aware: cached features, head only | 3 | 78.3 ± 0.3 | 82.9 ± 0.3 | 1.000 ± 0.000 |
| Code-level activation-aware: cached features, codes + head | 3 | 33.5 ± 1.4 | 44.7 ± 1.2 | 0.677 ± 0.011 |
| Code-level activation-aware: retrained on images, 145 epochs | 3 | 23.2 ± 1.0 | 36.3 ± 1.0 | 0.715 ± 0.009 |
| Random: pruned, no retraining | 3 | 100.0 ± 0.0 | 100.0 ± 0.0 | 1.000 ± 0.000 |
| Random: pruned, unused codes removed | 3 | 86.2 ± 0.1 | 88.5 ± 0.2 | 1.000 ± 0.000 |
| Random: cached features, head only | 3 | 85.8 ± 0.6 | 87.6 ± 0.8 | 1.000 ± 0.000 |
| Random: cached features, codes + head | 3 | 0.8 ± 0.1 | 1.5 ± 0.3 | 0.155 ± 0.008 |
