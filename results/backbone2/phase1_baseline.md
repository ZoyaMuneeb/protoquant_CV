# Phase 1 baseline

Pipeline per seed: Stage 1 (4,096 codes) → B0 (usage pruning to 3,000 on the protocol's validation loader) → Stage 2 (30 epochs, best validation epoch) → test read once.

## Final accuracy

| Protocol | Seeds | Test top-1 (%) | Paper |
|---|---|---|---|
| strict | 40, 41, 42 | 85.65 ± 0.14 (n=3) | 87.7 |
| backbone only (strict) | – | 86.95 | 88.4 |

The paper selects checkpoints and prunes codes on the test set. The strict baseline trains on 10 % fewer images and selects on a 600-image validation split, so a gap to the paper is a finding, not a failed reproduction.

## Per run

| Protocol | Seed | S1 best val (ep) | S1 dead ratio | B0 codes | B0 val before→after | S2 best val (ep) | Test top-1 | Test top-5 | Nonzero edges | Active codes |
|---|---|---|---|---|---|---|---|---|---|---|
| strict | 40 | 83.67 (18) | 0.409 | 4096->3000 | 83.67→83.67 | 84.50 (18) | 85.73 | 97.00 | 418866 | 3000 |
| strict | 41 | 84.33 (22) | 0.390 | 4096->3000 | 84.50→84.50 | 84.83 (18) | 85.48 | 97.13 | 425517 | 3000 |
| strict | 42 | 83.83 (11) | 0.438 | 4096->3000 | 83.83→83.83 | 84.67 (28) | 85.74 | 97.24 | 424220 | 3000 |

S1 and B0 validation accuracy is the Stage 1 signal (quantized features through the backbone's classifier); S2 is the ProtoQuant head.

## Implementation tests (strict runs)

| Seed | 1 masked head | 2 physical vs logical (changed preds, K=10) | 3 feature cache | 4 B0 split |
|---|---|---|---|---|
| 40 | PASS | 42/600 | PASS | PASS |
| 41 | PASS | 32/600 | PASS | PASS |
| 42 | PASS | 43/600 | PASS | PASS |
