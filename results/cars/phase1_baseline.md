# Phase 1 baseline

Pipeline per seed: Stage 1 (4,096 codes) → B0 (usage pruning to 3,000 on the protocol's validation loader) → Stage 2 (30 epochs, best validation epoch) → test read once.

## Final accuracy

| Protocol | Seeds | Test top-1 (%) | Paper |
|---|---|---|---|
| strict | 40, 41, 42 | 88.99 ± 0.06 (n=3) | – |
| backbone only (strict) | – | 91.90 | – |

The paper selects checkpoints and prunes codes on the test set. The strict baseline trains on 10 % fewer images and selects on a 815-image validation split, so a gap to the paper is a finding, not a failed reproduction.

## Per run

| Protocol | Seed | S1 best val (ep) | S1 dead ratio | B0 codes | B0 val before→after | S2 best val (ep) | Test top-1 | Test top-5 | Nonzero edges | Active codes |
|---|---|---|---|---|---|---|---|---|---|---|
| strict | 40 | 86.01 (21) | 0.301 | 4096->3000 | 86.01→86.01 | 87.61 (25) | 89.06 | 98.31 | 353640 | 3000 |
| strict | 41 | 85.15 (21) | 0.302 | 4096->3000 | 85.15→85.15 | 88.34 (25) | 88.94 | 98.42 | 362982 | 3000 |
| strict | 42 | 85.15 (27) | 0.291 | 4096->3000 | 85.15→85.15 | 87.98 (24) | 88.98 | 98.36 | 371682 | 3000 |

S1 and B0 validation accuracy is the Stage 1 signal (quantized features through the backbone's classifier); S2 is the ProtoQuant head.

## Implementation tests (strict runs)

| Seed | 1 masked head | 2 physical vs logical (changed preds, K=10) | 3 feature cache | 4 B0 split |
|---|---|---|---|---|
| 40 | PASS | 52/815 | PASS | PASS |
| 41 | PASS | 46/815 | PASS | PASS |
| 42 | PASS | 45/815 | PASS | PASS |
