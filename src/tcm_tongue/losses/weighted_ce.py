from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedCrossEntropyLoss(nn.Module):
    """Weighted cross-entropy loss."""

    def __init__(
        self,
        class_weights: Optional[Sequence[float] | torch.Tensor] = None,
        reduction: str = "mean",
        ignore_index: Optional[int] = None,
    ):
        super().__init__()
        if class_weights is None:
            self.register_buffer("class_weights", None)
        elif isinstance(class_weights, torch.Tensor):
            self.register_buffer("class_weights", class_weights.float())
        else:
            self.register_buffer("class_weights", torch.tensor(class_weights, dtype=torch.float32))
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(
            logits,
            targets,
            weight=self.class_weights,
            reduction=self.reduction,
            ignore_index=-100 if self.ignore_index is None else self.ignore_index,
        )
