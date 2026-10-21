"""CIFAR-10 ResNet-18.

The torchvision ResNet-18 stem is built for ImageNet (7×7 stride-2 convolution
and a 3×3 stride-2 max pool). On 32×32 CIFAR images that stem discards too much
spatial resolution. This module replaces it with a 3×3 stride-1 convolution and
removes the max pool, which is the usual small-image adaptation.
"""

from __future__ import annotations

import torch.nn as nn
from torchvision.models import resnet18


def resnet18_cifar(num_classes: int = 10) -> nn.Module:
    """Return a ResNet-18 whose stem keeps a 32×32 feature map."""
    model = resnet18(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
