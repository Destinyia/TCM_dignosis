"""单目标约束损失

惩罚检测多个目标，鼓励每图只检测一个舌头。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SingleTargetLoss(nn.Module):
    """单目标约束损失 - 惩罚检测多个目标"""

    def __init__(
        self,
        target_count: int = 1,
        loss_weight: float = 0.1,
        use_entropy_loss: bool = True,
        use_margin_loss: bool = True,
    ):
        super().__init__()
        self.target_count = target_count
        self.loss_weight = loss_weight
        self.use_entropy_loss = use_entropy_loss
        self.use_margin_loss = use_margin_loss

    def forward(
        self,
        objectness_logits: torch.Tensor,
        num_detections: int,
    ) -> torch.Tensor:
        """
        Args:
            objectness_logits: RPN 的 objectness 分数 (N,)
            num_detections: 当前检测到的目标数
        Returns:
            单目标约束损失
        """
        device = objectness_logits.device
        loss = torch.tensor(0.0, device=device)

        # 1. 惩罚检测数量偏离目标
        count_penalty = F.relu(
            torch.tensor(num_detections - self.target_count, dtype=torch.float, device=device)
        )
        loss = loss + count_penalty

        if len(objectness_logits) == 0:
            return self.loss_weight * loss

        probs = torch.sigmoid(objectness_logits)

        # 2. 熵最小化，鼓励 objectness 分数集中（非0即1）
        if self.use_entropy_loss:
            entropy = -probs * torch.log(probs + 1e-8) - (1 - probs) * torch.log(1 - probs + 1e-8)
            entropy_loss = entropy.mean()
            loss = loss + entropy_loss

        # 3. Top-1 与其他的 margin loss，鼓励最高分与次高分有明显差距
        if self.use_margin_loss and len(probs) > 1:
            sorted_probs, _ = probs.sort(descending=True)
            # 希望 top1 - top2 > margin
            margin = 0.3
            margin_loss = F.relu(sorted_probs[1] - sorted_probs[0] + margin)
            loss = loss + margin_loss

        return self.loss_weight * loss


class ConsistencyLoss(nn.Module):
    """检测分类与全局分类的一致性约束"""

    def __init__(self, temperature: float = 1.0, loss_weight: float = 0.1):
        super().__init__()
        self.temperature = temperature
        self.loss_weight = loss_weight

    def forward(
        self,
        det_logits: torch.Tensor,
        global_logits: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            det_logits: 检测头的分类 logits (N, C)
            global_logits: 全局分类头的 logits (B, C)
        Returns:
            KL 散度一致性损失
        """
        if det_logits.numel() == 0 or global_logits.numel() == 0:
            return torch.tensor(0.0, device=det_logits.device)

        det_probs = F.softmax(det_logits / self.temperature, dim=-1)
        global_probs = F.softmax(global_logits / self.temperature, dim=-1)

        # 扩展 global_probs 以匹配 det_probs 的数量
        if det_probs.shape[0] != global_probs.shape[0]:
            # 假设每张图的检测结果与对应的全局分类对齐
            # 这里简化处理，取平均
            det_probs_mean = det_probs.mean(dim=0, keepdim=True)
            global_probs_mean = global_probs.mean(dim=0, keepdim=True)
            kl_loss = F.kl_div(
                det_probs_mean.log(),
                global_probs_mean,
                reduction="batchmean"
            )
        else:
            kl_loss = F.kl_div(
                det_probs.log(),
                global_probs,
                reduction="batchmean"
            )

        return self.loss_weight * kl_loss
