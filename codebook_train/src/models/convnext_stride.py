import torch.nn as nn
from torchvision import models


def _replace_convlayers_convnext(model: nn.Module, threshold: int) -> nn.Module:
    """Reduce selected ConvNeXt strides to mimic PIPNet latent grid variants.

    Threshold behavior matches purity benchmark implementation:
    - threshold=100 -> ~26x26 latent grid for 224 input
    - threshold=300 -> ~13x13 latent grid for 224 input
    """

    for module in model.modules():
        if isinstance(module, nn.Conv2d) and module.stride[0] == 2:
            if module.in_channels > threshold:
                module.stride = tuple(s // 2 for s in module.stride)

    return model


def convnext_tiny_26(**kwargs) -> nn.Module:
    """Torchvision ConvNeXt-Tiny with stride edits for ~26x26 feature maps.

    This mirrors PIPNet's convnext_tiny_26 stride policy while preserving
    parameter shapes so regular ConvNeXt checkpoints can be loaded.
    """

    model = models.convnext_tiny(**kwargs)
    _replace_convlayers_convnext(model, threshold=100)
    return model
