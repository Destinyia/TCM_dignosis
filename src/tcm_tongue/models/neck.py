from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F


class FPN(nn.Module):
    """Feature Pyramid Network."""

    def __init__(
        self,
        in_channels: List[int],
        out_channels: int = 256,
        num_outs: int = 5,
    ):
        super().__init__()
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for in_ch in in_channels:
            self.lateral_convs.append(nn.Conv2d(in_ch, out_channels, 1))
            self.fpn_convs.append(nn.Conv2d(out_channels, out_channels, 3, padding=1))

        self.num_outs = num_outs
        self.extra_convs = nn.ModuleList()
        if num_outs > len(in_channels):
            for _ in range(num_outs - len(in_channels)):
                self.extra_convs.append(
                    nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1)
                )

    def forward(self, features: Dict[str, torch.Tensor]) -> List[torch.Tensor]:
        feat_keys = sorted(features.keys())
        feats = [features[k] for k in feat_keys]

        laterals = [l_conv(f) for l_conv, f in zip(self.lateral_convs, feats)]

        for i in range(len(laterals) - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[-2:]
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i], size=prev_shape, mode="nearest"
            )

        outs = [fpn_conv(lat) for fpn_conv, lat in zip(self.fpn_convs, laterals)]

        if self.extra_convs:
            last = outs[-1]
            for conv in self.extra_convs:
                last = conv(last)
                outs.append(last)

        return outs


class BiFPN(nn.Module):
    """Bidirectional Feature Pyramid Network."""

    def __init__(self, *args, **kwargs):
        super().__init__()
        raise NotImplementedError("BiFPN is not implemented yet")


class PAFPN(nn.Module):
    """Path Aggregation Feature Pyramid Network."""

    def __init__(self, *args, **kwargs):
        super().__init__()
        raise NotImplementedError("PAFPN is not implemented yet")


def create_neck(
    name: str,
    in_channels: List[int],
    out_channels: int = 256,
    **kwargs,
) -> nn.Module:
    """Neck factory."""
    necks = {
        "fpn": FPN,
        "bifpn": BiFPN,
        "pafpn": PAFPN,
    }
    if name not in necks:
        raise ValueError(f"Unknown neck: {name}")
    return necks[name](in_channels, out_channels, **kwargs)
