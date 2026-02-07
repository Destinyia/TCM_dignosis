from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn

try:
    from torchvision.models import (
        ResNet50_Weights,
        ResNet101_Weights,
        Swin_T_Weights,
        Swin_S_Weights,
        resnet50,
        resnet101,
        swin_t,
        swin_s,
    )
except Exception as exc:  # pragma: no cover
    raise RuntimeError("torchvision is required for backbone models") from exc


class BackboneBase(nn.Module):
    """Backbone base class."""

    def __init__(self, out_channels: List[int]):
        super().__init__()
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        raise NotImplementedError


class ResNetBackbone(BackboneBase):
    """ResNet backbone."""

    def __init__(
        self,
        depth: int = 50,
        pretrained: bool = True,
        frozen_stages: int = 1,
    ):
        if depth == 50:
            weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
            backbone = _safe_resnet(resnet50, weights)
            out_channels = [256, 512, 1024, 2048]
        elif depth == 101:
            weights = ResNet101_Weights.IMAGENET1K_V2 if pretrained else None
            backbone = _safe_resnet(resnet101, weights)
            out_channels = [256, 512, 1024, 2048]
        else:
            raise ValueError(f"Unsupported ResNet depth: {depth}")

        super().__init__(out_channels)

        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        self._freeze_stages(frozen_stages)

    def _freeze_stages(self, num_stages: int):
        if num_stages <= 0:
            return
        if num_stages >= 1:
            for module in [self.conv1, self.bn1]:
                for param in module.parameters():
                    param.requires_grad = False
        if num_stages >= 2:
            for param in self.layer1.parameters():
                param.requires_grad = False
        if num_stages >= 3:
            for param in self.layer2.parameters():
                param.requires_grad = False
        if num_stages >= 4:
            for param in self.layer3.parameters():
                param.requires_grad = False
        if num_stages >= 5:
            for param in self.layer4.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        feat0 = self.layer1(x)
        feat1 = self.layer2(feat0)
        feat2 = self.layer3(feat1)
        feat3 = self.layer4(feat2)

        return {
            "feat0": feat0,
            "feat1": feat1,
            "feat2": feat2,
            "feat3": feat3,
        }


class SwinBackbone(BackboneBase):
    """Swin Transformer backbone."""

    def __init__(
        self,
        variant: str = "tiny",
        pretrained: bool = True,
    ):
        if variant == "tiny":
            weights = Swin_T_Weights.IMAGENET1K_V1 if pretrained else None
            model = _safe_swin(swin_t, weights)
            out_channels = [96, 192, 384, 768]
        elif variant == "small":
            weights = Swin_S_Weights.IMAGENET1K_V1 if pretrained else None
            model = _safe_swin(swin_s, weights)
            out_channels = [96, 192, 384, 768]
        else:
            raise ValueError(f"Unsupported Swin variant: {variant}")

        super().__init__(out_channels)
        self.model = model

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = self.model.features
        x = features[0](x)
        x = features[1](x)
        feat0 = x

        x = features[2](x)
        x = features[3](x)
        feat1 = x

        x = features[4](x)
        x = features[5](x)
        feat2 = x

        x = features[6](x)
        x = features[7](x)
        feat3 = x

        return {
            "feat0": _to_nchw(feat0),
            "feat1": _to_nchw(feat1),
            "feat2": _to_nchw(feat2),
            "feat3": _to_nchw(feat3),
        }


def create_backbone(
    name: str,
    pretrained: bool = True,
    **kwargs,
) -> BackboneBase:
    """Backbone factory."""
    backbones = {
        "resnet50": lambda: ResNetBackbone(50, pretrained, **kwargs),
        "resnet101": lambda: ResNetBackbone(101, pretrained, **kwargs),
        "swin_t": lambda: SwinBackbone("tiny", pretrained),
        "swin_s": lambda: SwinBackbone("small", pretrained),
    }
    if name not in backbones:
        raise ValueError(f"Unknown backbone: {name}")
    return backbones[name]()


def _to_nchw(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 4:
        return x.permute(0, 3, 1, 2).contiguous()
    return x


def _safe_resnet(fn, weights):
    try:
        return fn(weights=weights)
    except Exception:
        return fn(weights=None)


def _safe_swin(fn, weights):
    try:
        return fn(weights=weights)
    except Exception:
        return fn(weights=None)
