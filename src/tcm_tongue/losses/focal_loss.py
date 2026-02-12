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


class ClassBalancedFocalLoss(nn.Module):
    """Class-Balanced Focal Loss based on effective number of samples.

    Reference: CVPR 2019 "Class-Balanced Loss Based on Effective Number of Samples"

    The effective number of samples is computed as:
        E_n = (1 - beta^n) / (1 - beta)
    where n is the number of samples for each class.

    The class weight is then:
        w = 1 / E_n = (1 - beta) / (1 - beta^n)
    """

    def __init__(
        self,
        class_counts: Sequence[int],
        beta: float = 0.9999,
        gamma: float = 2.0,
        reduction: str = "mean",
        ignore_index: Optional[int] = None,
    ):
        """Initialize ClassBalancedFocalLoss.

        Args:
            class_counts: Number of samples for each class.
            beta: Hyperparameter for effective number calculation.
                  Higher beta gives more weight to rare classes.
                  Typical values: 0.9, 0.99, 0.999, 0.9999
            gamma: Focal loss focusing parameter.
            reduction: Reduction method ('mean', 'sum', 'none').
            ignore_index: Target value to ignore.
        """
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.beta = beta

        # Compute class-balanced weights
        weights = self._compute_cb_weights(class_counts, beta)
        self.register_buffer("class_weights", weights)

    def _compute_cb_weights(
        self, class_counts: Sequence[int], beta: float
    ) -> torch.Tensor:
        """Compute class-balanced weights based on effective number of samples."""
        counts = torch.tensor(class_counts, dtype=torch.float32)
        # Avoid division by zero for classes with no samples
        counts = torch.clamp(counts, min=1.0)

        # Effective number: E_n = (1 - beta^n) / (1 - beta)
        effective_num = 1.0 - torch.pow(beta, counts)

        # Weight: w = (1 - beta) / E_n
        weights = (1.0 - beta) / effective_num

        # Normalize weights to sum to num_classes
        weights = weights / weights.sum() * len(weights)

        return weights

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

        # Focal factor
        focal_factor = (1.0 - pt) ** self.gamma
        loss = -focal_factor * log_pt

        # Apply class-balanced weights
        weights = self.class_weights.to(logits_flat.device)
        if weights.numel() != num_classes:
            raise ValueError(
                f"Number of class weights ({weights.numel()}) must match "
                f"number of classes ({num_classes})"
            )
        alpha_factor = weights[targets_flat]
        loss = loss * alpha_factor

        return _reduce(loss, self.reduction)
