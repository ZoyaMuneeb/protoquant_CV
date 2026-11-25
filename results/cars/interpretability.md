# Interpretability

Dataset cars196, seeds 40, 41, 42, mean ± std over seeds (n = seeds with the model). Definitions: `scripts/interpretability.py`. No model is trained or selected here.

## Explanation size

Per class: nonzero edges, and edges with weight > 0.1 (the repository's local size). Per test image: codes contributing > 0.001 to the predicted class (the repository's rule), and codes that make up 90 % of its score (independent of the weight scale).

| Model | n | Edges / class | Edges > 0.1 / class | Codes used | Codes / image (predicted class) | Codes for 90 % of the score | Test top-1 (check) |
|---|---|---|---|---|---|---|---|
| Dense | 3 | 1850.9 ± 46.0 | 1749.0 ± 42.9 | 3000 ± 0 | 4.36 ± 0.07 | 289.58 ± 11.02 | 88.99 ± 0.06 |
| Magnitude: pruned, no retraining | 3 | 10.0 ± 0.0 | 10.0 ± 0.0 | 1339 ± 21 | 1.75 ± 0.03 | 5.89 ± 0.03 | 72.73 ± 0.47 |
| Magnitude: pruned, unused codes removed | 3 | 10.0 ± 0.0 | 10.0 ± 0.0 | 1339 ± 21 | 3.31 ± 0.04 | 6.02 ± 0.04 | 72.98 ± 0.59 |
| Magnitude: cached features, head only | 3 | 9.8 ± 0.0 | 9.8 ± 0.0 | 1328 ± 19 | 3.20 ± 0.04 | 5.90 ± 0.04 | 79.19 ± 0.60 |
| Magnitude: cached features, codes + head | 3 | 10.0 ± 0.0 | 10.0 ± 0.0 | 1339 ± 21 | 2.69 ± 0.11 | 5.48 ± 0.09 | 86.85 ± 0.25 |
| Magnitude: retrained on images, 145 epochs | 3 | 10.0 ± 0.0 | 10.0 ± 0.0 | 1339 ± 21 | 2.50 ± 0.10 | 6.46 ± 0.01 | 88.48 ± 0.51 |
| Activation-aware: pruned, no retraining | 3 | 10.0 ± 0.0 | 10.0 ± 0.0 | 1486 ± 10 | 2.92 ± 0.01 | 6.53 ± 0.02 | 75.33 ± 0.24 |
| Activation-aware: pruned, unused codes removed | 3 | 10.0 ± 0.0 | 10.0 ± 0.0 | 1486 ± 10 | 4.51 ± 0.01 | 6.69 ± 0.02 | 75.76 ± 0.10 |
| Activation-aware: cached features, head only | 3 | 9.7 ± 0.1 | 9.7 ± 0.1 | 1467 ± 18 | 4.31 ± 0.04 | 6.17 ± 0.11 | 80.05 ± 0.06 |
| Activation-aware: cached features, codes + head | 3 | 10.0 ± 0.0 | 10.0 ± 0.0 | 1486 ± 10 | 2.80 ± 0.20 | 5.59 ± 0.17 | 86.62 ± 0.19 |
| Code-level magnitude: pruned, no retraining | 3 | 10.0 ± 0.0 | 10.0 ± 0.0 | 874 ± 5 | 0.58 ± 0.04 | 5.71 ± 0.04 | 59.97 ± 1.91 |
| Code-level magnitude: pruned, unused codes removed | 3 | 10.0 ± 0.0 | 10.0 ± 0.0 | 874 ± 5 | 2.59 ± 0.02 | 5.89 ± 0.05 | 63.45 ± 1.72 |
| Code-level magnitude: retrained on images, 145 epochs | 3 | 10.0 ± 0.0 | 10.0 ± 0.0 | 873 ± 5 | 2.53 ± 0.09 | 5.65 ± 0.04 | 88.46 ± 0.30 |
| Code-level activation-aware: pruned, no retraining | 3 | 10.0 ± 0.0 | 10.0 ± 0.0 | 874 ± 5 | 2.16 ± 0.00 | 5.94 ± 0.01 | 69.85 ± 0.53 |
| Code-level activation-aware: pruned, unused codes removed | 3 | 10.0 ± 0.0 | 10.0 ± 0.0 | 874 ± 5 | 4.45 ± 0.00 | 6.19 ± 0.03 | 71.45 ± 0.29 |
| Code-level activation-aware: cached features, head only | 3 | 9.6 ± 0.0 | 9.6 ± 0.0 | 870 ± 3 | 4.22 ± 0.03 | 5.61 ± 0.07 | 77.44 ± 0.14 |
| Code-level activation-aware: cached features, codes + head | 3 | 9.9 ± 0.0 | 9.9 ± 0.0 | 874 ± 5 | 2.54 ± 0.06 | 4.61 ± 0.06 | 86.88 ± 0.33 |
| Code-level activation-aware: retrained on images, 145 epochs | 3 | 10.0 ± 0.0 | 9.9 ± 0.0 | 873 ± 5 | 2.94 ± 0.06 | 5.75 ± 0.05 | 88.16 ± 0.23 |
| Random: pruned, no retraining | 3 | 10.0 ± 0.0 | 9.4 ± 0.1 | 1427 ± 22 | 0.80 ± 0.03 | 1.50 ± 0.02 | 5.25 ± 1.49 |
| Random: pruned, unused codes removed | 3 | 10.0 ± 0.0 | 9.4 ± 0.1 | 1427 ± 22 | 1.06 ± 0.00 | 1.52 ± 0.02 | 5.42 ± 1.24 |
| Random: cached features, head only | 3 | 3.4 ± 0.1 | 3.4 ± 0.1 | 590 ± 23 | 0.92 ± 0.02 | 1.33 ± 0.04 | 7.15 ± 1.23 |
| Random: cached features, codes + head | 3 | 5.5 ± 0.2 | 5.4 ± 0.2 | 968 ± 23 | 1.85 ± 0.06 | 2.62 ± 0.06 | 83.77 ± 0.33 |

## Meaning kept (same code, compared with the dense model)

Top-10 training patches of each used code: share that are the same image and position, and the same image. Cosine between the code vectors.

| Model | n | Same patches (%) | Same images (%) | Code cosine |
|---|---|---|---|---|
| Magnitude: pruned, no retraining | 3 | 100.0 ± 0.0 | 100.0 ± 0.0 | 1.000 ± 0.000 |
| Magnitude: pruned, unused codes removed | 3 | 84.9 ± 0.1 | 87.8 ± 0.1 | 1.000 ± 0.000 |
| Magnitude: cached features, head only | 3 | 84.9 ± 0.1 | 87.8 ± 0.1 | 1.000 ± 0.000 |
| Magnitude: cached features, codes + head | 3 | 28.3 ± 0.4 | 35.2 ± 0.4 | 0.703 ± 0.008 |
| Magnitude: retrained on images, 145 epochs | 3 | 23.2 ± 0.6 | 35.3 ± 1.0 | 0.760 ± 0.005 |
| Activation-aware: pruned, no retraining | 3 | 100.0 ± 0.0 | 100.0 ± 0.0 | 1.000 ± 0.000 |
| Activation-aware: pruned, unused codes removed | 3 | 86.9 ± 0.2 | 89.7 ± 0.1 | 1.000 ± 0.000 |
| Activation-aware: cached features, head only | 3 | 86.9 ± 0.2 | 89.7 ± 0.1 | 1.000 ± 0.000 |
| Activation-aware: cached features, codes + head | 3 | 30.7 ± 0.4 | 38.7 ± 0.5 | 0.701 ± 0.011 |
| Code-level magnitude: pruned, no retraining | 3 | 100.0 ± 0.0 | 100.0 ± 0.0 | 1.000 ± 0.000 |
| Code-level magnitude: pruned, unused codes removed | 3 | 79.3 ± 0.6 | 82.2 ± 0.6 | 1.000 ± 0.000 |
| Code-level magnitude: retrained on images, 145 epochs | 3 | 15.2 ± 0.3 | 23.9 ± 0.2 | 0.662 ± 0.005 |
| Code-level activation-aware: pruned, no retraining | 3 | 100.0 ± 0.0 | 100.0 ± 0.0 | 1.000 ± 0.000 |
| Code-level activation-aware: pruned, unused codes removed | 3 | 77.0 ± 0.5 | 80.8 ± 0.1 | 1.000 ± 0.000 |
| Code-level activation-aware: cached features, head only | 3 | 77.0 ± 0.5 | 80.8 ± 0.2 | 1.000 ± 0.000 |
| Code-level activation-aware: cached features, codes + head | 3 | 19.4 ± 0.3 | 26.3 ± 0.4 | 0.645 ± 0.006 |
| Code-level activation-aware: retrained on images, 145 epochs | 3 | 19.8 ± 0.4 | 29.3 ± 0.5 | 0.711 ± 0.006 |
| Random: pruned, no retraining | 3 | 100.0 ± 0.0 | 100.0 ± 0.0 | 1.000 ± 0.000 |
| Random: pruned, unused codes removed | 3 | 85.7 ± 0.7 | 87.9 ± 0.6 | 1.000 ± 0.000 |
| Random: cached features, head only | 3 | 86.1 ± 0.9 | 87.8 ± 0.7 | 1.000 ± 0.000 |
| Random: cached features, codes + head | 3 | 0.8 ± 0.1 | 1.2 ± 0.1 | 0.180 ± 0.002 |
