# Backbone sweep (strict protocol, selected on the 600-image validation split)

Batch 128, 30 epochs per run, seed 42. Previous backbone (lr 5e-5, from ImageNet): val 85.50 %.

| Run | Best val top-1 | Checkpoint |
|---|---|---|
| continue_lr_2e-4_lr_5e-5 | 88.67 | `cub200_convnext_tiny_full_20261118_223411.pth` |
| lr_1e-4 | 88.17 | `cub200_convnext_tiny_full_20261118_215012.pth` |
| lr_2e-4 | 88.33 | `cub200_convnext_tiny_full_20261118_220456.pth` |
| lr_4e-4 | 87.67 | `cub200_convnext_tiny_full_20261118_221935.pth` |

Selected: **continue_lr_2e-4_lr_5e-5** -> `/home/shaikha/runs/backbone_sweep/continue_lr_2e-4_lr_5e-5/ckpt/cub200_convnext_tiny_full_20261118_223411.pth`
