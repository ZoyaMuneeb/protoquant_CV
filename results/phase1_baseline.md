# Phase 1 baseline

Pipeline per seed: Stage 1 (4,096 codes) → B0 (usage pruning to 3,000 on the protocol's validation loader) → Stage 2 (30 epochs, best validation epoch) → test read once.

## Final accuracy

| Protocol | Seeds | Test top-1 (%) | Paper |
|---|---|---|---|
| paper | 40 | 84.74 (1 seed) | 87.7 |
| strict | 40, 41, 42 | 84.43 ± 0.33 (n=3) | 87.7 |
| backbone only (strict) | – | 84.05 | 88.4 |
| backbone only (paper) | – | 84.05 | 88.4 |

The paper selects checkpoints and prunes codes on the test set. The strict baseline trains on 10 % fewer images and selects on a 600-image validation split, so a gap to the paper is a finding, not a failed reproduction.

## Per run

| Protocol | Seed | S1 best val (ep) | S1 dead ratio | B0 codes | B0 val before→after | S2 best val (ep) | Test top-1 | Test top-5 | Nonzero edges | Active codes |
|---|---|---|---|---|---|---|---|---|---|---|
| paper | 40 | 79.32 (20) | 0.231 | 4096->3000 | 79.32→79.34 | 84.74 (26) | 84.74 | 97.13 | 436540 | 3000 |
| strict | 40 | 79.67 (20) | 0.343 | 4096->3000 | 79.67→79.67 | 85.50 (26) | 84.74 | 97.12 | 435694 | 3000 |
| strict | 41 | 80.00 (24) | 0.336 | 4096->3000 | 79.83→79.83 | 85.83 (24) | 84.47 | 97.05 | 443828 | 3000 |
| strict | 42 | 80.83 (14) | 0.357 | 4096->3000 | 80.83→80.83 | 86.17 (20) | 84.09 | 97.05 | 462768 | 3000 |

S1 and B0 validation accuracy is the Stage 1 signal (quantized features through the backbone's classifier); S2 is the ProtoQuant head.

## Implementation tests (strict runs)

| Seed | 1 masked head | 2 physical vs logical (changed preds, K=10) | 3 feature cache | 4 B0 split |
|---|---|---|---|---|
| 40 | PASS | 57/600 | PASS | PASS |
| 41 | PASS | 60/600 | PASS | PASS |
| 42 | PASS | 63/600 | PASS | PASS |
