"""掩码-边界框约束损失 - 利用bbox标注监督分割掩码质量"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskBBoxLoss(nn.Module):
    """利用bbox标注监督分割掩码质量

    通过三个子损失来约束分割掩码与bbox的对齐：
    1. IoU Loss: 掩码与bbox的交并比
    2. Coverage Loss: bbox内掩码覆盖率
    3. Boundary Loss: 惩罚bbox外的掩码（支持距离加权软惩罚）
    """

    def __init__(
        self,
        iou_weight: float = 1.0,
        coverage_weight: float = 1.0,
        boundary_weight: float = 0.5,
        use_distance_penalty: bool = True,
        distance_scale: float = 5.0,
    ):
        """
        Args:
            iou_weight: IoU损失权重
            coverage_weight: 覆盖率损失权重
            boundary_weight: 边界惩罚权重
            use_distance_penalty: 是否使用基于距离的软惩罚
            distance_scale: 距离惩罚的缩放因子，越大惩罚增长越快
        """
        super().__init__()
        self.iou_weight = iou_weight
        self.coverage_weight = coverage_weight
        self.boundary_weight = boundary_weight
        self.use_distance_penalty = use_distance_penalty
        self.distance_scale = distance_scale

    def forward(
        self, mask: torch.Tensor, bbox: torch.Tensor
    ) -> torch.Tensor:
        """计算掩码-bbox约束损失

        Args:
            mask: 分割掩码 (B, 1, H, W)，值域[0,1]
            bbox: 边界框 (B, 4) [x1, y1, x2, y2] 归一化坐标

        Returns:
            loss: 标量损失值
        """
        # 生成bbox掩码和距离图
        bbox_mask = self._bbox_to_mask(bbox, mask.shape[2:])

        # 计算三个子损失
        iou_loss = self._compute_iou_loss(mask, bbox_mask)
        coverage_loss = self._compute_coverage_loss(mask, bbox_mask)

        if self.use_distance_penalty:
            distance_map = self._compute_distance_map(bbox, mask.shape[2:])
            boundary_loss = self._compute_distance_boundary_loss(mask, bbox_mask, distance_map)
        else:
            boundary_loss = self._compute_boundary_loss(mask, bbox_mask)

        # 加权求和
        total_loss = (
            self.iou_weight * iou_loss +
            self.coverage_weight * coverage_loss +
            self.boundary_weight * boundary_loss
        )

        return total_loss

    def _compute_distance_map(
        self, bbox: torch.Tensor, size: tuple
    ) -> torch.Tensor:
        """计算每个像素到bbox边界的距离图

        Args:
            bbox: (B, 4) [x1, y1, x2, y2] 归一化坐标
            size: (H, W) 目标尺寸

        Returns:
            distance_map: (B, 1, H, W) 距离图，bbox内部为0，外部为到边界的距离
        """
        B = bbox.shape[0]
        H, W = size
        device = bbox.device

        # 创建坐标网格 [0, 1]
        y_coords = torch.linspace(0, 1, H, device=device).view(1, 1, H, 1).expand(B, 1, H, W)
        x_coords = torch.linspace(0, 1, W, device=device).view(1, 1, 1, W).expand(B, 1, H, W)

        # 扩展bbox维度
        x1 = bbox[:, 0].view(B, 1, 1, 1)
        y1 = bbox[:, 1].view(B, 1, 1, 1)
        x2 = bbox[:, 2].view(B, 1, 1, 1)
        y2 = bbox[:, 3].view(B, 1, 1, 1)

        # 计算到各边界的距离（负值表示在bbox内部）
        dist_left = x1 - x_coords    # 正值表示在左边界外
        dist_right = x_coords - x2   # 正值表示在右边界外
        dist_top = y1 - y_coords     # 正值表示在上边界外
        dist_bottom = y_coords - y2  # 正值表示在下边界外

        # 计算到bbox的距离（使用L2距离）
        # 对于bbox外的点，计算到最近边界的距离
        dx = torch.max(torch.max(dist_left, dist_right), torch.zeros_like(dist_left))
        dy = torch.max(torch.max(dist_top, dist_bottom), torch.zeros_like(dist_top))

        # 欧氏距离
        distance = torch.sqrt(dx ** 2 + dy ** 2)

        return distance

    def _compute_distance_boundary_loss(
        self, mask: torch.Tensor, bbox_mask: torch.Tensor, distance_map: torch.Tensor
    ) -> torch.Tensor:
        """计算基于距离的边界惩罚

        距离越远，惩罚越大。使用 mask * distance * scale 作为惩罚。

        Args:
            mask: 预测掩码 (B, 1, H, W)
            bbox_mask: bbox掩码 (B, 1, H, W)
            distance_map: 距离图 (B, 1, H, W)

        Returns:
            loss: 距离加权的边界惩罚
        """
        # bbox外的掩码
        outside_mask = mask * (1 - bbox_mask)

        # 距离加权惩罚：mask值 * 距离 * 缩放因子
        weighted_penalty = outside_mask * distance_map * self.distance_scale

        # 归一化：除以总掩码面积
        total_mask = mask.sum(dim=[2, 3])
        loss = weighted_penalty.sum(dim=[2, 3]) / (total_mask + 1e-6)

        return loss.mean()

    def _bbox_to_mask(
        self, bbox: torch.Tensor, size: tuple
    ) -> torch.Tensor:
        """将bbox转换为二值掩码

        Args:
            bbox: (B, 4) [x1, y1, x2, y2] 归一化坐标
            size: (H, W) 目标尺寸

        Returns:
            mask: (B, 1, H, W) 二值掩码
        """
        B = bbox.shape[0]
        H, W = size
        device = bbox.device

        # 创建坐标网格
        y_coords = torch.linspace(0, 1, H, device=device).view(1, 1, H, 1)
        x_coords = torch.linspace(0, 1, W, device=device).view(1, 1, 1, W)

        # 扩展bbox维度 (B, 4) -> (B, 1, 1, 1) for each coord
        x1 = bbox[:, 0].view(B, 1, 1, 1)
        y1 = bbox[:, 1].view(B, 1, 1, 1)
        x2 = bbox[:, 2].view(B, 1, 1, 1)
        y2 = bbox[:, 3].view(B, 1, 1, 1)

        # 生成掩码：在bbox内部为1，外部为0
        mask = (
            (x_coords >= x1) & (x_coords <= x2) &
            (y_coords >= y1) & (y_coords <= y2)
        ).float()

        return mask

    def _compute_iou_loss(
        self, mask: torch.Tensor, bbox_mask: torch.Tensor
    ) -> torch.Tensor:
        """计算掩码与bbox的IoU损失

        Args:
            mask: 预测掩码 (B, 1, H, W)
            bbox_mask: bbox掩码 (B, 1, H, W)

        Returns:
            loss: 1 - IoU
        """
        # 计算交集和并集
        intersection = (mask * bbox_mask).sum(dim=[2, 3])
        union = mask.sum(dim=[2, 3]) + bbox_mask.sum(dim=[2, 3]) - intersection

        # 计算IoU
        iou = intersection / (union + 1e-6)

        return 1 - iou.mean()

    def _compute_coverage_loss(
        self, mask: torch.Tensor, bbox_mask: torch.Tensor
    ) -> torch.Tensor:
        """计算bbox内的掩码覆盖率损失

        Args:
            mask: 预测掩码 (B, 1, H, W)
            bbox_mask: bbox掩码 (B, 1, H, W)

        Returns:
            loss: 1 - coverage
        """
        # bbox面积
        bbox_area = bbox_mask.sum(dim=[2, 3])

        # bbox内的掩码面积
        inside_mask = mask * bbox_mask
        coverage = inside_mask.sum(dim=[2, 3]) / (bbox_area + 1e-6)

        return 1 - coverage.mean()

    def _compute_boundary_loss(
        self, mask: torch.Tensor, bbox_mask: torch.Tensor
    ) -> torch.Tensor:
        """计算bbox外的掩码惩罚

        Args:
            mask: 预测掩码 (B, 1, H, W)
            bbox_mask: bbox掩码 (B, 1, H, W)

        Returns:
            loss: bbox外掩码占总掩码的比例
        """
        # bbox外的掩码
        outside_mask = mask * (1 - bbox_mask)

        # bbox外掩码占总掩码的比例
        total_mask = mask.sum(dim=[2, 3])
        outside_ratio = outside_mask.sum(dim=[2, 3]) / (total_mask + 1e-6)

        return outside_ratio.mean()

    def get_metrics(
        self, mask: torch.Tensor, bbox: torch.Tensor
    ) -> dict:
        """获取详细指标（用于监控）

        Args:
            mask: 分割掩码 (B, 1, H, W)
            bbox: 边界框 (B, 4)

        Returns:
            metrics: 包含各子损失的字典
        """
        bbox_mask = self._bbox_to_mask(bbox, mask.shape[2:])

        iou_loss = self._compute_iou_loss(mask, bbox_mask)
        coverage_loss = self._compute_coverage_loss(mask, bbox_mask)
        boundary_loss = self._compute_boundary_loss(mask, bbox_mask)

        return {
            "mask_iou_loss": iou_loss.item(),
            "mask_coverage_loss": coverage_loss.item(),
            "mask_boundary_loss": boundary_loss.item(),
            "mask_iou": 1 - iou_loss.item(),
            "mask_coverage": 1 - coverage_loss.item(),
        }
