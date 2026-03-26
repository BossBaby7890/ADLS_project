"""
src/models/resnet.py
====================
ResNet-20 for CIFAR-10/100 — the default APQ-Lite benchmark architecture.

Follows He et al. (2016) "Deep Residual Learning for Image Recognition",
adapted for 32x32 inputs.  Register additional variants (ResNet-32, ResNet-56)
by adding entries to the ``_CONFIGS`` dict and calling ``@register_model``.
"""

from __future__ import annotations

from typing import List, Type

import torch.nn as nn
import torch.nn.functional as F

from . import register_model


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


class ResNet(nn.Module):
    def __init__(
        self,
        block: Type[BasicBlock],
        num_blocks: List[int],
        num_classes: int = 10,
    ) -> None:
        super().__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        # nn.AdaptiveAvgPool2d instead of x.mean([2,3]) — MASE's FX graph
        # analyser only recognises nn.Module submodules, not bare tensor
        # method calls, so the raw .mean() call causes KeyError: 'mase'.
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(64 * block.expansion, num_classes)

    def _make_layer(
        self,
        block: Type[BasicBlock],
        planes: int,
        num_blocks: int,
        stride: int,
    ) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avgpool(out)             # global average pooling
        out = out.flatten(1)
        return self.linear(out)


# ---------------------------------------------------------------------------
# Registered factories
# ---------------------------------------------------------------------------

@register_model("resnet20")
def resnet20(num_classes: int = 10, **_) -> ResNet:
    """ResNet-20: 3 groups x 3 blocks (≈0.27M params on CIFAR-10)."""
    return ResNet(BasicBlock, [3, 3, 3], num_classes=num_classes)


@register_model("resnet32")
def resnet32(num_classes: int = 10, **_) -> ResNet:
    """ResNet-32: 3 groups x 5 blocks."""
    return ResNet(BasicBlock, [5, 5, 5], num_classes=num_classes)


@register_model("resnet56")
def resnet56(num_classes: int = 10, **_) -> ResNet:
    """ResNet-56: 3 groups x 9 blocks."""
    return ResNet(BasicBlock, [9, 9, 9], num_classes=num_classes)
