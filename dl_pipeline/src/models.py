"""
Model builders for the COMP9517 DL pipeline.

Baseline architecture: ResNet-18 (He et al., CVPR 2016) via torchvision.
Same architecture for from-scratch and pretrained runs — only the weight
initialisation differs — so the comparison isolates transfer learning.
"""

from __future__ import annotations

from typing import Literal, Optional

import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights


ArchitectureName = Literal["resnet18"]


def build_resnet18(
    num_classes: int = 500,
    *,
    pretrained: bool = False,
) -> nn.Module:
    """
    Build a ResNet-18 with a custom final classifier.

    Parameters
    ----------
    num_classes :
        Number of species in our subset (500).
    pretrained :
        False → random Kaiming init (from-scratch run).
        True  → ImageNet-1K weights, then replace ``fc`` (pretrained run).

    Citation
    --------
    He, Kaiming, et al. "Deep Residual Learning for Image Recognition."
    CVPR 2016. https://doi.org/10.1109/CVPR.2016.90
    """
    if pretrained:
        weights = ResNet18_Weights.IMAGENET1K_V1
        model = models.resnet18(weights=weights)
    else:
        model = models.resnet18(weights=None)

    in_features = model.fc.in_features  # 512 for ResNet-18
    model.fc = nn.Linear(in_features, num_classes)
    return model


def build_model(
    architecture: ArchitectureName = "resnet18",
    num_classes: int = 500,
    *,
    pretrained: bool = False,
) -> nn.Module:
    """Factory so notebooks 03/04 share one call site (easy to add EfficientNet later)."""
    if architecture == "resnet18":
        return build_resnet18(num_classes, pretrained=pretrained)
    raise ValueError(f"Unknown architecture: {architecture!r}")


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """Return (total_params, trainable_params)."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
