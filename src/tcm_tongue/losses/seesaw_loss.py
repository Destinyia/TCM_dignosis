from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SeesawLoss(nn.Module):
    """Seesaw Loss for long-tailed instance segmentation.

    Reference: CVPR 2021 "Seesaw Loss for Long-Tailed Instance Segmentation"

    The loss dynamically re-balances gradients of positive and negative samples
    for each category with two complementary factors:
    1. Mitigation factor: reduces penalty for tail classes from head classes
    2. Compensation factor: increases gradients for misclassified tail samples
    """

    def __init__(
        self,
        num_classes: int,
        p: float = 0.6,
        q: float = 1.5,
        eps: float = 1e-2,
        reduction: str = "mean",
        ignore_index: Optional[int] = None,
    ):
        """Initialize SeesawLoss.

        Args:
            num_classes: Number of classes.
            p: Mitigation factor exponent. Controls how much to reduce
               negative gradients from head to tail classes.
            q: Compensation factor exponent. Controls how much to boost
               gradients for misclassified samples.
            eps: Small constant for numerical stability.
            reduction: Reduction method ('mean', 'sum', 'none').
            ignore_index: Target value to ignore.
        """
        super().__init__()
        self.num_classes = num_classes
        self.p = p
        self.q = q
        self.eps = eps
        self.reduction = reduction
        self.ignore_index = ignore_index

        # Cumulative class counts for mitigation factor
        self.register_buffer(
            "cum_counts", torch.zeros(num_classes, dtype=torch.float32)
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute Seesaw Loss.

        Args:
            logits: Predicted logits of shape (N, C) or (N, C, ...).
            targets: Ground truth labels of shape (N,) or (N, ...).

        Returns:
            Computed loss value.
        """
        if logits.dim() < 2:
            raise ValueError("Logits must have shape (N, C, ...) for seesaw loss")

        logits_flat, targets_flat = self._flatten(logits, targets)

        if self.ignore_index is not None:
            valid_mask = targets_flat != self.ignore_index
            logits_flat = logits_flat[valid_mask]
            targets_flat = targets_flat[valid_mask]

        if targets_flat.numel() == 0:
            return logits_flat.sum() * 0.0

        # Update cumulative counts during training
        if self.training:
            self._update_counts(targets_flat)

        # Compute seesaw weights
        seesaw_weights = self._compute_seesaw_weights(logits_flat, targets_flat)

        # Apply weights to logits
        weighted_logits = logits_flat + torch.log(seesaw_weights + self.eps)

        # Cross entropy with weighted logits
        loss = F.cross_entropy(
            weighted_logits, targets_flat, reduction="none"
        )

        return self._reduce(loss)

    def _flatten(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Flatten logits and targets for batch processing."""
        if logits.dim() == 2:
            return logits, targets.view(-1)

        permute_dims = [0] + list(range(2, logits.dim())) + [1]
        logits_perm = logits.permute(*permute_dims).contiguous()
        logits_flat = logits_perm.view(-1, logits.size(1))
        targets_flat = targets.view(-1)
        return logits_flat, targets_flat

    def _update_counts(self, targets: torch.Tensor) -> None:
        """Update cumulative class counts."""
        unique, counts = torch.unique(targets, return_counts=True)
        for cls_id, count in zip(unique.tolist(), counts.tolist()):
            if 0 <= cls_id < self.num_classes:
                self.cum_counts[cls_id] += count

    def _compute_seesaw_weights(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute seesaw weights for each sample.

        Returns:
            Weight tensor of shape (N, C).
        """
        batch_size = logits.size(0)
        num_classes = logits.size(1)

        # Initialize weights to 1
        weights = torch.ones_like(logits)

        # Get cumulative counts
        cum_counts = self.cum_counts.to(logits.device)

        # Mitigation factor: reduce negative gradients from head to tail
        # For each sample i with label y_i, for class j != y_i:
        # M_j = (N_j / N_{y_i})^p if N_j < N_{y_i} else 1
        for i in range(batch_size):
            target_cls = targets[i].item()
            if target_cls < 0 or target_cls >= num_classes:
                continue

            target_count = cum_counts[target_cls]
            if target_count < 1:
                target_count = 1.0

            for j in range(num_classes):
                if j == target_cls:
                    continue

                other_count = cum_counts[j]
                if other_count < target_count:
                    # Tail class: reduce penalty
                    ratio = other_count / target_count
                    weights[i, j] = torch.pow(ratio + self.eps, self.p)

        # Compensation factor: boost gradients for misclassified samples
        # C_j = (s_j / s_{y_i})^q if s_j > s_{y_i}
        probs = F.softmax(logits, dim=1)
        for i in range(batch_size):
            target_cls = targets[i].item()
            if target_cls < 0 or target_cls >= num_classes:
                continue

            target_prob = probs[i, target_cls]

            for j in range(num_classes):
                if j == target_cls:
                    continue

                other_prob = probs[i, j]
                if other_prob > target_prob:
                    # Misclassified: boost gradient
                    ratio = other_prob / (target_prob + self.eps)
                    weights[i, j] = weights[i, j] * torch.pow(ratio, self.q)

        return weights

    def _reduce(self, loss: torch.Tensor) -> torch.Tensor:
        """Apply reduction to loss."""
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        if self.reduction == "none":
            return loss
        raise ValueError(f"Unsupported reduction: {self.reduction}")
