# Interpretability

Dataset cub200, seeds 40, mean ± std over seeds (n = seeds with the model). Definitions: `scripts/interpretability.py`. No model is trained or selected here.

## Part purity

Share of a code's top-10 patches showing the same part; higher is better. Std over codes in brackets. "Purity change" compares the same code with the dense model (train images), so it is not affected by which codes were kept.

| Model | n | Codes | Purity train | Purity test | Codes with purity > 0.5 (train) | Purity change, same code |
|---|---|---|---|---|---|---|
| Dense | 1 | 3000 | 0.338 [0.28] | 0.343 [0.27] | 26.4 % | – |
| Magnitude: pruned, no retraining | 1 | 1270 | 0.417 [0.26] | 0.421 [0.26] | 34.3 % | 0.000 |
| Magnitude: pruned, unused codes removed | 1 | 1270 | 0.407 [0.26] | 0.408 [0.26] | 33.1 % | -0.011 |
| Magnitude: cached features, head only | 1 | 1261 | 0.406 [0.26] | 0.408 [0.26] | 33.0 % | -0.011 |
| Magnitude: cached features, codes + head | 1 | 1270 | 0.393 [0.25] | 0.384 [0.25] | 30.2 % | -0.024 |
| Magnitude: retrained on images, 145 epochs | 1 | 1270 | 0.412 [0.27] | 0.411 [0.26] | 33.5 % | -0.005 |
| Activation-aware: pruned, no retraining | 1 | 1569 | 0.404 [0.27] | 0.406 [0.27] | 34.5 % | 0.000 |
| Activation-aware: pruned, unused codes removed | 1 | 1569 | 0.398 [0.27] | 0.397 [0.27] | 33.5 % | -0.006 |
| Activation-aware: cached features, head only | 1 | 1566 | 0.399 [0.27] | 0.398 [0.27] | 33.5 % | -0.006 |
| Activation-aware: cached features, codes + head | 1 | 1569 | 0.379 [0.26] | 0.379 [0.26] | 30.7 % | -0.025 |
| Code-level magnitude: pruned, no retraining | 1 | 885 | 0.374 [0.26] | 0.378 [0.26] | 28.2 % | 0.000 |
| Code-level magnitude: pruned, unused codes removed | 1 | 885 | 0.367 [0.26] | 0.369 [0.26] | 27.8 % | -0.006 |
| Code-level magnitude: cached features, head only | 1 | 881 | 0.368 [0.26] | 0.370 [0.26] | 27.8 % | -0.007 |
| Code-level magnitude: cached features, codes + head | 1 | 885 | 0.371 [0.25] | 0.373 [0.25] | 27.5 % | -0.002 |
| Code-level magnitude: retrained on images, 145 epochs | 1 | 885 | 0.397 [0.27] | 0.394 [0.27] | 32.1 % | 0.024 |
| Code-level activation-aware: pruned, no retraining | 1 | 885 | 0.423 [0.29] | 0.421 [0.28] | 38.9 % | 0.000 |
| Code-level activation-aware: pruned, unused codes removed | 1 | 885 | 0.411 [0.28] | 0.412 [0.28] | 35.9 % | -0.013 |
| Code-level activation-aware: cached features, head only | 1 | 885 | 0.411 [0.28] | 0.412 [0.28] | 35.9 % | -0.013 |
| Code-level activation-aware: cached features, codes + head | 1 | 885 | 0.382 [0.26] | 0.385 [0.27] | 31.2 % | -0.042 |
| Code-level activation-aware: retrained on images, 145 epochs | 1 | 885 | 0.415 [0.28] | 0.421 [0.29] | 37.4 % | -0.009 |
| Random: pruned, no retraining | 1 | 1437 | 0.321 [0.28] | 0.331 [0.27] | 24.5 % | 0.000 |
| Random: pruned, unused codes removed | 1 | 1437 | 0.324 [0.28] | 0.331 [0.27] | 25.6 % | 0.003 |
| Random: cached features, head only | 1 | 445 | 0.275 [0.28] | 0.290 [0.28] | 21.1 % | 0.004 |
| Random: cached features, codes + head | 1 | 846 | 0.257 [0.27] | 0.264 [0.27] | 18.9 % | -0.022 |
| Chance (random patches) | 1 | – | 0.107 [0.07] | 0.107 [0.07] | 0.0 % | – |

## Explanation size

Per class: nonzero edges, and edges with weight > 0.1 (the repository's local size). Per test image: codes contributing > 0.001 to the predicted class (the repository's rule), and codes that make up 90 % of its score (independent of the weight scale).

| Model | n | Edges / class | Edges > 0.1 / class | Codes used | Codes / image (predicted class) | Codes for 90 % of the score | Test top-1 (check) |
|---|---|---|---|---|---|---|---|
| Dense | 1 | 2094.3 | 1946.8 | 3000 | 7.21 | 239.24 | 85.73 |
| Magnitude: pruned, no retraining | 1 | 10.0 | 10.0 | 1270 | 2.40 | 6.25 | 76.98 |
| Magnitude: pruned, unused codes removed | 1 | 10.0 | 10.0 | 1270 | 4.27 | 6.39 | 76.91 |
| Magnitude: cached features, head only | 1 | 9.8 | 9.8 | 1261 | 4.18 | 6.22 | 79.91 |
| Magnitude: cached features, codes + head | 1 | 10.0 | 10.0 | 1270 | 3.34 | 5.51 | 84.76 |
| Magnitude: retrained on images, 145 epochs | 1 | 10.0 | 10.0 | 1270 | 3.73 | 6.71 | 85.11 |
| Activation-aware: pruned, no retraining | 1 | 10.0 | 10.0 | 1569 | 4.32 | 7.28 | 79.51 |
| Activation-aware: pruned, unused codes removed | 1 | 10.0 | 10.0 | 1569 | 5.90 | 7.42 | 79.44 |
| Activation-aware: cached features, head only | 1 | 9.9 | 9.9 | 1566 | 5.80 | 7.22 | 80.62 |
| Activation-aware: cached features, codes + head | 1 | 10.0 | 10.0 | 1569 | 4.10 | 6.48 | 83.69 |
| Code-level magnitude: pruned, no retraining | 1 | 10.0 | 10.0 | 885 | 1.12 | 5.88 | 63.91 |
| Code-level magnitude: pruned, unused codes removed | 1 | 10.0 | 10.0 | 885 | 3.64 | 6.05 | 66.02 |
| Code-level magnitude: cached features, head only | 1 | 9.9 | 9.9 | 881 | 3.44 | 5.91 | 76.20 |
| Code-level magnitude: cached features, codes + head | 1 | 9.9 | 9.9 | 885 | 2.96 | 4.77 | 84.59 |
| Code-level magnitude: retrained on images, 145 epochs | 1 | 10.0 | 10.0 | 885 | 3.34 | 5.82 | 85.04 |
| Code-level activation-aware: pruned, no retraining | 1 | 10.0 | 10.0 | 885 | 3.35 | 6.63 | 71.07 |
| Code-level activation-aware: pruned, unused codes removed | 1 | 10.0 | 10.0 | 885 | 5.70 | 6.78 | 73.84 |
| Code-level activation-aware: cached features, head only | 1 | 9.8 | 9.8 | 885 | 5.57 | 6.34 | 76.23 |
| Code-level activation-aware: cached features, codes + head | 1 | 9.9 | 9.8 | 885 | 3.12 | 4.56 | 83.67 |
| Code-level activation-aware: retrained on images, 145 epochs | 1 | 9.9 | 9.9 | 885 | 3.75 | 5.79 | 84.98 |
| Random: pruned, no retraining | 1 | 10.0 | 9.2 | 1437 | 0.93 | 1.29 | 5.32 |
| Random: pruned, unused codes removed | 1 | 10.0 | 9.2 | 1437 | 1.07 | 1.33 | 5.38 |
| Random: cached features, head only | 1 | 2.5 | 2.4 | 445 | 0.96 | 1.24 | 8.65 |
| Random: cached features, codes + head | 1 | 4.6 | 4.5 | 846 | 2.01 | 2.35 | 79.96 |

## Meaning kept (same code, compared with the dense model)

Top-10 training patches of each used code: share that are the same image and position, and the same image. Cosine between the code vectors.

| Model | n | Same patches (%) | Same images (%) | Code cosine |
|---|---|---|---|---|
| Magnitude: pruned, no retraining | 1 | 100.0 | 100.0 | 1.000 |
| Magnitude: pruned, unused codes removed | 1 | 84.9 | 88.4 | 1.000 |
| Magnitude: cached features, head only | 1 | 84.9 | 88.5 | 1.000 |
| Magnitude: cached features, codes + head | 1 | 38.7 | 49.9 | 0.724 |
| Magnitude: retrained on images, 145 epochs | 1 | 31.6 | 45.4 | 0.785 |
| Activation-aware: pruned, no retraining | 1 | 100.0 | 100.0 | 1.000 |
| Activation-aware: pruned, unused codes removed | 1 | 88.6 | 91.5 | 1.000 |
| Activation-aware: cached features, head only | 1 | 88.6 | 91.5 | 1.000 |
| Activation-aware: cached features, codes + head | 1 | 49.6 | 61.6 | 0.769 |
| Code-level magnitude: pruned, no retraining | 1 | 100.0 | 100.0 | 1.000 |
| Code-level magnitude: pruned, unused codes removed | 1 | 79.2 | 82.8 | 1.000 |
| Code-level magnitude: cached features, head only | 1 | 79.3 | 82.8 | 1.000 |
| Code-level magnitude: cached features, codes + head | 1 | 24.8 | 34.1 | 0.653 |
| Code-level magnitude: retrained on images, 145 epochs | 1 | 22.0 | 33.4 | 0.703 |
| Code-level activation-aware: pruned, no retraining | 1 | 100.0 | 100.0 | 1.000 |
| Code-level activation-aware: pruned, unused codes removed | 1 | 78.4 | 82.9 | 1.000 |
| Code-level activation-aware: cached features, head only | 1 | 78.4 | 82.9 | 1.000 |
| Code-level activation-aware: cached features, codes + head | 1 | 35.1 | 46.1 | 0.689 |
| Code-level activation-aware: retrained on images, 145 epochs | 1 | 24.3 | 37.5 | 0.716 |
| Random: pruned, no retraining | 1 | 100.0 | 100.0 | 1.000 |
| Random: pruned, unused codes removed | 1 | 86.3 | 88.7 | 1.000 |
| Random: cached features, head only | 1 | 86.3 | 87.9 | 1.000 |
| Random: cached features, codes + head | 1 | 0.9 | 1.7 | 0.162 |
