"""舌象分类模型 - 基于分割注意力的图像分类"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from .seg_attention import (
    MaskGuidedAttention,
    MultiScaleAttention,
    ResidualSoftAttention,
    SegmentationEncoder,
    SpatialAttention,
)


class SegAttentionClassifier(nn.Module):
    """分割注意力分类器

    使用预训练分割模型生成舌头掩码作为空间注意力，
    引导分类backbone聚焦于舌头区域。
    """

    def __init__(
        self,
        num_classes: int = 8,
        backbone: str = "resnet50",
        pretrained: bool = True,
        seg_weights_path: Optional[str] = None,
        freeze_seg: bool = True,
        use_channel_attention: bool = True,
        dropout: float = 0.5,
        use_mask_refiner: bool = False,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.use_mask_refiner = use_mask_refiner

        # 分割编码器（生成注意力掩码）
        self.seg_encoder = SegmentationEncoder(
            weights_path=seg_weights_path,
            freeze=freeze_seg,
        )

        # 可学习的掩码精炼层（用于bbox监督）
        if use_mask_refiner:
            self.mask_refiner = nn.Sequential(
                nn.Conv2d(1, 16, 3, padding=1),
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True),
                nn.Conv2d(16, 16, 3, padding=1),
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True),
                nn.Conv2d(16, 1, 3, padding=1),
                nn.Sigmoid(),
            )
        else:
            self.mask_refiner = None

        # 分类backbone
        self.backbone, self.feature_dim = self._create_backbone(backbone, pretrained)

        # 掩码引导注意力
        self.mask_attention = MaskGuidedAttention(
            in_channels=self.feature_dim,
            use_channel_attention=use_channel_attention,
        )

        # 分类头
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(self.feature_dim, num_classes),
        )

    def _create_backbone(
        self, name: str, pretrained: bool
    ) -> Tuple[nn.Module, int]:
        """创建分类backbone"""
        if name == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            backbone = models.resnet50(weights=weights)
            feature_dim = 2048
            # 移除最后的全连接层和池化层
            backbone = nn.Sequential(*list(backbone.children())[:-2])
        elif name == "resnet101":
            weights = models.ResNet101_Weights.DEFAULT if pretrained else None
            backbone = models.resnet101(weights=weights)
            feature_dim = 2048
            backbone = nn.Sequential(*list(backbone.children())[:-2])
        elif name == "efficientnet_b0":
            weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            backbone = models.efficientnet_b0(weights=weights)
            feature_dim = 1280
            backbone.classifier = nn.Identity()
            backbone.avgpool = nn.Identity()
        elif name == "efficientnet_b3":
            weights = models.EfficientNet_B3_Weights.DEFAULT if pretrained else None
            backbone = models.efficientnet_b3(weights=weights)
            feature_dim = 1536
            backbone.classifier = nn.Identity()
            backbone.avgpool = nn.Identity()
        else:
            raise ValueError(f"Unsupported backbone: {name}")

        return backbone, feature_dim

    def forward(
        self, x: torch.Tensor, return_attention: bool = False, return_mask: bool = False
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """前向传播

        Args:
            x: 输入图像 (B, 3, H, W)
            return_attention: 是否返回注意力图（与return_mask相同，保持兼容）
            return_mask: 是否返回掩码（用于bbox损失计算）

        Returns:
            logits: 分类logits (B, num_classes)
            mask: (可选) 掩码 (B, 1, H', W')
        """
        # 生成分割掩码
        if self.seg_encoder.frozen:
            with torch.no_grad():
                raw_mask = self.seg_encoder(x)  # (B, 1, H, W)
        else:
            raw_mask = self.seg_encoder(x)

        # 精炼掩码（如果启用）
        if self.mask_refiner is not None:
            mask = self.mask_refiner(raw_mask)
        else:
            mask = raw_mask

        # 提取分类特征
        features = self.backbone(x)  # (B, C, H', W')

        # 应用掩码引导注意力
        attended_features = self.mask_attention(features, mask)

        # 分类
        logits = self.classifier(attended_features)

        if return_attention or return_mask:
            return logits, mask
        return logits


class DualStreamClassifier(nn.Module):
    """双流融合分类器

    同时利用分割编码器特征和分类backbone特征，
    通过特征融合进行分类。
    """

    def __init__(
        self,
        num_classes: int = 8,
        backbone: str = "resnet50",
        pretrained: bool = True,
        seg_weights_path: Optional[str] = None,
        freeze_seg: bool = True,
        fusion_type: str = "concat",  # concat, add, attention
        dropout: float = 0.5,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.fusion_type = fusion_type

        # 分割编码器
        self.seg_encoder = SegmentationEncoder(
            weights_path=seg_weights_path,
            freeze=freeze_seg,
        )

        # 分类backbone
        self.backbone, self.cls_feature_dim = self._create_backbone(backbone, pretrained)

        # 分割特征维度 (来自ASPP middle层)
        self.seg_feature_dim = 512

        # 特征融合
        if fusion_type == "concat":
            fusion_dim = self.cls_feature_dim + self.seg_feature_dim
        else:
            # 需要对齐维度
            self.seg_proj = nn.Conv2d(self.seg_feature_dim, self.cls_feature_dim, 1)
            fusion_dim = self.cls_feature_dim

        if fusion_type == "attention":
            self.fusion_attention = nn.Sequential(
                nn.Conv2d(self.cls_feature_dim * 2, self.cls_feature_dim, 1),
                nn.Sigmoid(),
            )

        # 分类头
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(256, num_classes),
        )

    def _create_backbone(
        self, name: str, pretrained: bool
    ) -> Tuple[nn.Module, int]:
        """创建分类backbone"""
        if name == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            backbone = models.resnet50(weights=weights)
            feature_dim = 2048
            backbone = nn.Sequential(*list(backbone.children())[:-2])
        elif name == "resnet101":
            weights = models.ResNet101_Weights.DEFAULT if pretrained else None
            backbone = models.resnet101(weights=weights)
            feature_dim = 2048
            backbone = nn.Sequential(*list(backbone.children())[:-2])
        else:
            raise ValueError(f"Unsupported backbone: {name}")

        return backbone, feature_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        # 获取分割编码器特征
        _, _, _, _, seg_features = self.seg_encoder.get_encoder_features(x)

        # 获取分类特征
        cls_features = self.backbone(x)

        # 对齐空间尺寸
        if seg_features.shape[2:] != cls_features.shape[2:]:
            seg_features = F.interpolate(
                seg_features, size=cls_features.shape[2:],
                mode="bilinear", align_corners=False
            )

        # 特征融合
        if self.fusion_type == "concat":
            fused = torch.cat([cls_features, seg_features], dim=1)
        elif self.fusion_type == "add":
            seg_proj = self.seg_proj(seg_features)
            fused = cls_features + seg_proj
        elif self.fusion_type == "attention":
            seg_proj = self.seg_proj(seg_features)
            concat = torch.cat([cls_features, seg_proj], dim=1)
            att = self.fusion_attention(concat)
            fused = cls_features * att + seg_proj * (1 - att)
        else:
            raise ValueError(f"Unknown fusion type: {self.fusion_type}")

        # 分类
        logits = self.classifier(fused)
        return logits


class BaselineClassifier(nn.Module):
    """基线分类器 - 不使用分割注意力的纯分类模型"""

    def __init__(
        self,
        num_classes: int = 8,
        backbone: str = "resnet50",
        pretrained: bool = True,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.num_classes = num_classes

        if backbone == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            self.backbone = models.resnet50(weights=weights)
            feature_dim = 2048
        elif backbone == "resnet101":
            weights = models.ResNet101_Weights.DEFAULT if pretrained else None
            self.backbone = models.resnet101(weights=weights)
            feature_dim = 2048
        elif backbone == "efficientnet_b0":
            weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            self.backbone = models.efficientnet_b0(weights=weights)
            feature_dim = 1280
        elif backbone == "efficientnet_b3":
            weights = models.EfficientNet_B3_Weights.DEFAULT if pretrained else None
            self.backbone = models.efficientnet_b3(weights=weights)
            feature_dim = 1536
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # 替换分类头
        if "resnet" in backbone:
            self.backbone.fc = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(feature_dim, num_classes),
            )
        else:
            self.backbone.classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(feature_dim, num_classes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class SegAttentionClassifierV2(nn.Module):
    """改进版分割注意力分类器

    改进点：
    1. 使用 ResidualSoftAttention 替代硬乘法
    2. soft_floor 保留背景信息
    3. 可学习残差权重
    """

    def __init__(
        self,
        num_classes: int = 8,
        backbone: str = "resnet50",
        pretrained: bool = True,
        seg_weights_path: Optional[str] = None,
        freeze_seg: bool = True,
        soft_floor: float = 0.1,
        use_channel_attention: bool = True,
        dropout: float = 0.5,
        init_alpha: float = 0.0,
        attention_mode: str = "add",
    ):
        super().__init__()
        self.num_classes = num_classes

        # 分割编码器
        self.seg_encoder = SegmentationEncoder(
            weights_path=seg_weights_path,
            freeze=freeze_seg,
        )

        # 分类 backbone
        self.backbone, self.feature_dim = self._create_backbone(backbone, pretrained)

        # 残差软注意力
        self.attention = ResidualSoftAttention(
            in_channels=self.feature_dim,
            soft_floor=soft_floor,
            use_channel_attention=use_channel_attention,
            init_alpha=init_alpha,
            attention_mode=attention_mode,
        )

        # 分类头
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(self.feature_dim, num_classes),
        )

    def _create_backbone(
        self, name: str, pretrained: bool
    ) -> Tuple[nn.Module, int]:
        """创建分类 backbone"""
        if name == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            backbone = models.resnet50(weights=weights)
            feature_dim = 2048
            backbone = nn.Sequential(*list(backbone.children())[:-2])
        elif name == "resnet101":
            weights = models.ResNet101_Weights.DEFAULT if pretrained else None
            backbone = models.resnet101(weights=weights)
            feature_dim = 2048
            backbone = nn.Sequential(*list(backbone.children())[:-2])
        elif name == "efficientnet_b0":
            weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            backbone = models.efficientnet_b0(weights=weights)
            feature_dim = 1280
            backbone.classifier = nn.Identity()
            backbone.avgpool = nn.Identity()
        elif name == "efficientnet_b3":
            weights = models.EfficientNet_B3_Weights.DEFAULT if pretrained else None
            backbone = models.efficientnet_b3(weights=weights)
            feature_dim = 1536
            backbone.classifier = nn.Identity()
            backbone.avgpool = nn.Identity()
        else:
            raise ValueError(f"Unsupported backbone: {name}")

        return backbone, feature_dim

    def forward(
        self, x: torch.Tensor, return_mask: bool = False
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """前向传播"""
        # 生成分割掩码
        if self.seg_encoder.frozen:
            with torch.no_grad():
                mask = self.seg_encoder(x)
        else:
            mask = self.seg_encoder(x)

        # 提取特征
        features = self.backbone(x)

        # 应用残差软注意力
        attended_features = self.attention(features, mask)

        # 分类
        logits = self.classifier(attended_features)

        if return_mask:
            return logits, mask
        return logits


class MultiScaleSegAttentionClassifier(nn.Module):
    """多尺度分割注意力分类器

    在 ResNet 的多个层级应用注意力，充分利用高分辨率特征。
    """

    def __init__(
        self,
        num_classes: int = 8,
        backbone: str = "resnet50",
        pretrained: bool = True,
        seg_weights_path: Optional[str] = None,
        freeze_seg: bool = True,
        soft_floor: float = 0.1,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.num_classes = num_classes

        # 分割编码器
        self.seg_encoder = SegmentationEncoder(
            weights_path=seg_weights_path,
            freeze=freeze_seg,
        )

        # 创建分阶段 backbone
        self.stages, self.feature_dims = self._create_staged_backbone(backbone, pretrained)

        # 多尺度注意力
        self.multi_scale_attention = MultiScaleAttention(
            feature_dims=self.feature_dims,
            soft_floor=soft_floor,
        )

        # 分类头
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(self.feature_dims[-1], num_classes),
        )

    def _create_staged_backbone(
        self, name: str, pretrained: bool
    ) -> Tuple[nn.ModuleList, list]:
        """创建分阶段 backbone"""
        if name not in ["resnet50", "resnet101"]:
            raise ValueError(f"MultiScale only supports resnet50/101, got {name}")

        if name == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            resnet = models.resnet50(weights=weights)
        else:
            weights = models.ResNet101_Weights.DEFAULT if pretrained else None
            resnet = models.resnet101(weights=weights)

        # 分阶段：stem, layer1, layer2, layer3, layer4
        stages = nn.ModuleList([
            nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool),
            resnet.layer1,  # 256 channels
            resnet.layer2,  # 512 channels
            resnet.layer3,  # 1024 channels
            resnet.layer4,  # 2048 channels
        ])

        feature_dims = [256, 512, 1024, 2048]
        return stages, feature_dims

    def forward(
        self, x: torch.Tensor, return_attention: bool = False
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """前向传播"""
        # 生成分割掩码
        if self.seg_encoder.frozen:
            with torch.no_grad():
                mask = self.seg_encoder(x)
        else:
            mask = self.seg_encoder(x)

        # 分阶段前向 + 多尺度注意力
        features = self.stages[0](x)  # stem

        for i, stage in enumerate(self.stages[1:]):
            features = stage(features)
            # 在每个 ResNet layer 后应用注意力
            features = self.multi_scale_attention.forward_single(features, mask, level=i)

        # 分类
        logits = self.classifier(features)

        if return_attention:
            return logits, mask
        return logits


def build_classifier(
    model_type: str = "seg_attention",
    num_classes: int = 8,
    backbone: str = "resnet50",
    pretrained: bool = True,
    seg_weights_path: Optional[str] = None,
    **kwargs,
) -> nn.Module:
    """构建分类模型

    Args:
        model_type: 模型类型 (baseline, seg_attention, seg_attention_v2,
                    seg_attention_multiscale, dual_stream)
        num_classes: 类别数
        backbone: backbone名称
        pretrained: 是否使用预训练权重
        seg_weights_path: 分割模型权重路径
        **kwargs: 其他参数
            - dropout: dropout率
            - soft_floor: 软注意力下限 (v2/multiscale)
            - use_channel_attention: 是否使用通道注意力

    Returns:
        分类模型
    """
    if model_type == "baseline":
        return BaselineClassifier(
            num_classes=num_classes,
            backbone=backbone,
            pretrained=pretrained,
            dropout=kwargs.get("dropout", 0.5),
        )
    elif model_type == "seg_attention":
        return SegAttentionClassifier(
            num_classes=num_classes,
            backbone=backbone,
            pretrained=pretrained,
            seg_weights_path=seg_weights_path,
            freeze_seg=kwargs.get("freeze_seg", True),
            use_channel_attention=kwargs.get("use_channel_attention", True),
            dropout=kwargs.get("dropout", 0.5),
            use_mask_refiner=kwargs.get("use_mask_refiner", False),
        )
    elif model_type == "seg_attention_v2":
        return SegAttentionClassifierV2(
            num_classes=num_classes,
            backbone=backbone,
            pretrained=pretrained,
            seg_weights_path=seg_weights_path,
            freeze_seg=kwargs.get("freeze_seg", True),
            soft_floor=kwargs.get("soft_floor", 0.1),
            use_channel_attention=kwargs.get("use_channel_attention", True),
            dropout=kwargs.get("dropout", 0.5),
            init_alpha=kwargs.get("init_alpha", 0.0),
            attention_mode=kwargs.get("attention_mode", "add"),
        )
    elif model_type == "seg_attention_multiscale":
        return MultiScaleSegAttentionClassifier(
            num_classes=num_classes,
            backbone=backbone,
            pretrained=pretrained,
            seg_weights_path=seg_weights_path,
            freeze_seg=kwargs.get("freeze_seg", True),
            soft_floor=kwargs.get("soft_floor", 0.1),
            dropout=kwargs.get("dropout", 0.5),
        )
    elif model_type == "dual_stream":
        return DualStreamClassifier(
            num_classes=num_classes,
            backbone=backbone,
            pretrained=pretrained,
            seg_weights_path=seg_weights_path,
            freeze_seg=kwargs.get("freeze_seg", True),
            fusion_type=kwargs.get("fusion_type", "concat"),
            dropout=kwargs.get("dropout", 0.5),
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
