"""分割注意力模块 - 从预训练分割模型提取注意力掩码"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _get_resunet2_class():
    """动态导入 ResUNet2 类"""
    # 尝试多种导入方式
    try:
        from models.unet import ResUNet2
        return ResUNet2
    except ImportError:
        pass

    # 添加 Tongue_segment 到路径后重试
    tongue_segment_path = Path(__file__).parent.parent.parent.parent / "Tongue_segment"
    if str(tongue_segment_path) not in sys.path:
        sys.path.insert(0, str(tongue_segment_path))

    try:
        from models.unet import ResUNet2
        return ResUNet2
    except ImportError:
        pass

    # 直接导入
    try:
        import importlib.util
        unet_path = tongue_segment_path / "models" / "unet.py"
        if unet_path.exists():
            spec = importlib.util.spec_from_file_location("unet", unet_path)
            unet_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(unet_module)
            return unet_module.ResUNet2
    except Exception:
        pass

    raise ImportError(
        "Cannot import ResUNet2. Please ensure Tongue_segment/models/unet.py exists."
    )


class SegmentationEncoder(nn.Module):
    """从 ResUNet2 提取的编码器，用于生成分割掩码作为注意力。

    该模块加载预训练的分割模型，冻结参数，仅用于生成空间注意力掩码。
    """

    def __init__(
        self,
        weights_path: Optional[str] = None,
        freeze: bool = True,
        num_classes: int = 33,
    ):
        super().__init__()
        ResUNet2 = _get_resunet2_class()
        self.seg_model = ResUNet2(num_classes=num_classes)
        self.frozen = bool(freeze)

        if weights_path is not None:
            self._load_weights(weights_path)

        if freeze:
            self._freeze_parameters()

    def _load_weights(self, weights_path: str) -> None:
        """加载预训练权重"""
        checkpoint = torch.load(weights_path, map_location="cpu")
        if isinstance(checkpoint, dict) and "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint
        self.seg_model.load_state_dict(state_dict, strict=False)
        print(f"Loaded segmentation weights from {weights_path}")

    def _freeze_parameters(self) -> None:
        """冻结所有参数"""
        for param in self.seg_model.parameters():
            param.requires_grad = False
        self.seg_model.eval()
        self.frozen = True

    def _unfreeze_parameters(self) -> None:
        """解冻所有参数"""
        for param in self.seg_model.parameters():
            param.requires_grad = True
        self.frozen = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """生成分割掩码

        Args:
            x: 输入图像 (B, 3, H, W)

        Returns:
            mask: 分割掩码 (B, 1, H, W)，值域 [0, 1]
        """
        if self.frozen:
            with torch.no_grad():
                mask, _ = self.seg_model(x)
        else:
            mask, _ = self.seg_model(x)
        return mask

    def get_encoder_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """获取编码器各层特征（用于双流融合）

        Args:
            x: 输入图像 (B, 3, H, W)

        Returns:
            tuple: (e1, e2, e3, e4, middle) 各层特征
        """
        if self.frozen:
            with torch.no_grad():
                e1 = self.seg_model.enc1(x)
                e2 = self.seg_model.enc2(e1)
                e3 = self.seg_model.enc3(e2)
                e4 = self.seg_model.enc4(e3)
                middle = self.seg_model.middle(F.max_pool2d(e4, kernel_size=2))
        else:
            e1 = self.seg_model.enc1(x)
            e2 = self.seg_model.enc2(e1)
            e3 = self.seg_model.enc3(e2)
            e4 = self.seg_model.enc4(e3)
            middle = self.seg_model.middle(F.max_pool2d(e4, kernel_size=2))
        return e1, e2, e3, e4, middle


class SpatialAttention(nn.Module):
    """空间注意力模块 - 将分割掩码转换为可学习的注意力"""

    def __init__(self, refine: bool = True):
        super().__init__()
        self.refine = refine
        if refine:
            self.attention_conv = nn.Sequential(
                nn.Conv2d(1, 16, 3, padding=1),
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True),
                nn.Conv2d(16, 1, 3, padding=1),
            )

    def forward(
        self, mask: torch.Tensor, target_size: Optional[Tuple[int, int]] = None
    ) -> torch.Tensor:
        """生成空间注意力

        Args:
            mask: 分割掩码 (B, 1, H, W)
            target_size: 目标尺寸 (H', W')

        Returns:
            attention: 注意力图 (B, 1, H', W')
        """
        if self.refine:
            attention = torch.sigmoid(self.attention_conv(mask))
        else:
            attention = mask

        if target_size is not None:
            attention = F.interpolate(
                attention, size=target_size, mode="bilinear", align_corners=False
            )

        return attention


class MaskGuidedAttention(nn.Module):
    """掩码引导的注意力模块

    结合分割掩码和特征图，生成加权特征。
    """

    def __init__(
        self,
        in_channels: int,
        reduction: int = 16,
        use_channel_attention: bool = True,
    ):
        super().__init__()
        self.use_channel_attention = use_channel_attention

        # 空间注意力
        self.spatial_attention = SpatialAttention(refine=True)

        # 通道注意力（可选）
        if use_channel_attention:
            self.channel_attention = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(in_channels, in_channels // reduction, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels // reduction, in_channels, 1),
                nn.Sigmoid(),
            )

    def forward(
        self, features: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """应用掩码引导的注意力

        Args:
            features: 特征图 (B, C, H, W)
            mask: 分割掩码 (B, 1, H_m, W_m)

        Returns:
            attended_features: 加权特征 (B, C, H, W)
        """
        # 空间注意力
        spatial_att = self.spatial_attention(mask, features.shape[2:])
        attended = features * spatial_att

        # 通道注意力
        if self.use_channel_attention:
            channel_att = self.channel_attention(attended)
            attended = attended * channel_att

        return attended


class ResidualSoftAttention(nn.Module):
    """残差软注意力模块

    改进点：
    1. soft_floor 避免完全抑制背景特征
    2. 可学习的残差权重 alpha
    3. 空间注意力精炼 + 通道注意力
    """

    def __init__(
        self,
        in_channels: int,
        soft_floor: float = 0.1,
        reduction: int = 16,
        use_channel_attention: bool = True,
        init_alpha: float = 0.0,  # 初始化值，sigmoid 后约 0.5
        attention_mode: str = "add",  # "add" 或 "gate"
    ):
        super().__init__()
        self.soft_floor = soft_floor
        self.use_channel_attention = use_channel_attention
        self.attention_mode = attention_mode

        # 可学习的残差权重
        # init_alpha=0 -> sigmoid=0.5, init_alpha=-2 -> sigmoid≈0.12
        self.alpha = nn.Parameter(torch.tensor(init_alpha))

        # 空间注意力精炼
        self.spatial_refine = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 3, padding=1),
        )

        # 通道注意力
        if use_channel_attention:
            self.channel_attention = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(in_channels, in_channels // reduction, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels // reduction, in_channels, 1),
                nn.Sigmoid(),
            )

    def forward(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """应用残差软注意力

        Args:
            features: 特征图 (B, C, H, W)
            mask: 分割掩码 (B, 1, H_m, W_m)

        Returns:
            output: 融合后的特征 (B, C, H, W)
        """
        # 1. 调整 mask 尺寸
        mask_resized = F.interpolate(
            mask, size=features.shape[2:], mode="bilinear", align_corners=False
        )

        # 2. 空间注意力精炼，输出 [0, 1]
        spatial_att = torch.sigmoid(self.spatial_refine(mask_resized))

        # 3. 软注意力：[soft_floor, 1.0]，避免完全抑制
        soft_att = self.soft_floor + (1 - self.soft_floor) * spatial_att

        # 4. 应用空间注意力
        attended = features * soft_att

        # 5. 通道注意力
        if self.use_channel_attention:
            channel_att = self.channel_attention(attended)
            attended = attended * channel_att

        # 6. 残差连接
        alpha = torch.sigmoid(self.alpha)

        if self.attention_mode == "add":
            # 加性残差：output = original + alpha * (attended - original)
            # alpha=0 时完全是原始特征，alpha=1 时完全是注意力特征
            output = features + alpha * (attended - features)
        else:  # gate
            # 门控残差：alpha 控制注意力增强的强度
            # output = original + alpha * attended
            output = features + alpha * attended

        return output


class MultiScaleAttention(nn.Module):
    """多尺度注意力模块

    在 ResNet 的多个层级应用软注意力，充分利用高分辨率特征。
    """

    def __init__(
        self,
        feature_dims: list = [256, 512, 1024, 2048],
        soft_floor: float = 0.1,
        reduction: int = 16,
    ):
        super().__init__()
        self.soft_floor = soft_floor

        # 每个层级的注意力模块
        self.attention_modules = nn.ModuleList()
        self.alphas = nn.ParameterList()

        for dim in feature_dims:
            # 空间注意力精炼
            spatial_refine = nn.Sequential(
                nn.Conv2d(1, 16, 3, padding=1),
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True),
                nn.Conv2d(16, 1, 3, padding=1),
            )

            # 通道注意力
            channel_att = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(dim, dim // reduction, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(dim // reduction, dim, 1),
                nn.Sigmoid(),
            )

            self.attention_modules.append(
                nn.ModuleDict({
                    "spatial": spatial_refine,
                    "channel": channel_att,
                })
            )

            # 可学习的残差权重，浅层初始化较小（更依赖原始特征）
            self.alphas.append(nn.Parameter(torch.tensor(0.7)))

    def forward_single(
        self, features: torch.Tensor, mask: torch.Tensor, level: int
    ) -> torch.Tensor:
        """单层级注意力

        Args:
            features: 特征图 (B, C, H, W)
            mask: 分割掩码 (B, 1, H_m, W_m)
            level: 层级索引 (0-3)

        Returns:
            output: 融合后的特征
        """
        att_module = self.attention_modules[level]
        alpha = torch.sigmoid(self.alphas[level])

        # 调整 mask 尺寸
        mask_resized = F.interpolate(
            mask, size=features.shape[2:], mode="bilinear", align_corners=False
        )

        # 空间注意力
        spatial_att = torch.sigmoid(att_module["spatial"](mask_resized))
        soft_att = self.soft_floor + (1 - self.soft_floor) * spatial_att

        # 应用注意力
        attended = features * soft_att
        channel_att = att_module["channel"](attended)
        attended = attended * channel_att

        # 残差连接
        return alpha * features + (1 - alpha) * attended
