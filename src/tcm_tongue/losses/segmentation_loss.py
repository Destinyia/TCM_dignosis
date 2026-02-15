"""Segmentation losses for mask supervision."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        preds = preds.float()
        targets = targets.float()
        preds = preds.view(preds.size(0), -1)
        targets = targets.view(targets.size(0), -1)
        intersection = (preds * targets).sum(dim=1)
        union = preds.sum(dim=1) + targets.sum(dim=1)
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


class BCEDiceLoss(nn.Module):
    """BCE + Dice loss，使用 binary_cross_entropy_with_logits 以支持 AMP"""

    def __init__(self, bce_weight: float = 0.5, smooth: float = 1.0, from_logits: bool = False):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice = DiceLoss(smooth=smooth)
        self.from_logits = from_logits

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float()

        if self.from_logits:
            # preds 是 logits，使用 with_logits 版本（AMP 安全）
            bce_loss = F.binary_cross_entropy_with_logits(preds, targets)
            preds_sigmoid = torch.sigmoid(preds)
        else:
            # preds 已经过 sigmoid，手动计算 BCE 以支持 AMP
            # BCE = -[y*log(p) + (1-y)*log(1-p)]
            preds_clamped = preds.float().clamp(min=1e-7, max=1 - 1e-7)
            bce_loss = -(targets * torch.log(preds_clamped) +
                        (1 - targets) * torch.log(1 - preds_clamped)).mean()
            preds_sigmoid = preds

        dice_loss = self.dice(preds_sigmoid, targets)
        return self.bce_weight * bce_loss + (1.0 - self.bce_weight) * dice_loss
