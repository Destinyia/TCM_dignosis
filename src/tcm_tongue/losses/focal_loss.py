from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal loss for multi-class classification."""

    def __init__(
        self,
        alpha: Optional[float | Sequence[float] | torch.Tensor] = 0.25,
        gamma: float = 2.0,
        reduction: str = "mean",
        ignore_index: Optional[int] = None,
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if logits.dim() < 2:
            raise ValueError("Logits must have shape (N, C, ...) for focal loss")

        num_classes = logits.size(1)
        logits_flat, targets_flat = _flatten_logits_targets(logits, targets)

        if self.ignore_index is not None:
            valid_mask = targets_flat != self.ignore_index
            logits_flat = logits_flat[valid_mask]
            targets_flat = targets_flat[valid_mask]

        if targets_flat.numel() == 0:
            return logits_flat.sum() * 0.0

        log_probs = F.log_softmax(logits_flat, dim=1)
        log_pt = log_probs.gather(1, targets_flat.unsqueeze(1)).squeeze(1)
        pt = log_pt.exp()

        focal_factor = (1.0 - pt) ** self.gamma
        loss = -focal_factor * log_pt

        alpha_factor = _compute_alpha(self.alpha, num_classes, logits_flat, targets_flat)
        if alpha_factor is not None:
            loss = loss * alpha_factor

        return _reduce(loss, self.reduction)


def _flatten_logits_targets(
    logits: torch.Tensor, targets: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    if logits.dim() == 2:
        return logits, targets.view(-1)

    permute_dims = [0] + list(range(2, logits.dim())) + [1]
    logits_perm = logits.permute(*permute_dims).contiguous()
    logits_flat = logits_perm.view(-1, logits.size(1))
    targets_flat = targets.view(-1)
    return logits_flat, targets_flat


def _compute_alpha(
    alpha: Optional[float | Sequence[float] | torch.Tensor],
    num_classes: int,
    logits_flat: torch.Tensor,
    targets_flat: torch.Tensor,
) -> Optional[torch.Tensor]:
    if alpha is None:
        return None

    if isinstance(alpha, torch.Tensor):
        alpha_t = alpha.to(device=logits_flat.device, dtype=logits_flat.dtype)
    elif isinstance(alpha, (list, tuple)):
        alpha_t = torch.tensor(alpha, device=logits_flat.device, dtype=logits_flat.dtype)
    else:
        alpha_t = torch.tensor(float(alpha), device=logits_flat.device, dtype=logits_flat.dtype)

    if alpha_t.numel() == 1:
        return alpha_t
    if alpha_t.numel() != num_classes:
        raise ValueError("Alpha length must match number of classes")

    return alpha_t[targets_flat]


def _reduce(loss: torch.Tensor, reduction: str) -> torch.Tensor:
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    if reduction == "none":
        return loss
    raise ValueError(f"Unsupported reduction: {reduction}")
