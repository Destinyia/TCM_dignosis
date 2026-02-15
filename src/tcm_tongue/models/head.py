from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torchvision.models.detection import (
        fasterrcnn_resnet50_fpn,
        fcos_resnet50_fpn,
        retinanet_resnet50_fpn,
        FasterRCNN_ResNet50_FPN_Weights,
        FCOS_ResNet50_FPN_Weights,
        RetinaNet_ResNet50_FPN_Weights,
    )
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    try:
        from torchvision.models.detection import (
            fasterrcnn_resnet50_fpn_v2,
            FasterRCNN_ResNet50_FPN_V2_Weights,
        )
    except Exception:  # pragma: no cover - optional in older torchvision
        fasterrcnn_resnet50_fpn_v2 = None
        FasterRCNN_ResNet50_FPN_V2_Weights = None
except Exception as exc:  # pragma: no cover
    raise RuntimeError("torchvision is required for detection heads") from exc


class MutualExclusivePredictor(nn.Module):
    """互斥分类预测头，使用温度缩放增强分类互斥性"""

    def __init__(self, in_features: int, num_classes: int, temperature: float = 1.0):
        super().__init__()
        self.cls_score = nn.Linear(in_features, num_classes)
        self.bbox_pred = nn.Linear(in_features, num_classes * 4)
        self.temperature = temperature

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if x.dim() == 4:
            x = x.flatten(start_dim=1)
        scores = self.cls_score(x) / self.temperature
        bbox_deltas = self.bbox_pred(x)
        return scores, bbox_deltas


class GlobalClassifier(nn.Module):
    """图像级全局分类头"""

    def __init__(self, in_channels: int = 2048, num_classes: int = 8):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x = self.pool(features).flatten(1)
        return self.fc(x)


class TongueClassifier(nn.Module):
    """舌象8分类器（用于两阶段解耦的Stage 2）"""

    def __init__(self, in_features: int = 1024, num_classes: int = 8):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x.flatten(1))


class DetectionHead(nn.Module):
    """Detection head base class."""

    requires_features: bool = True

    def __init__(self, num_classes: int, in_channels: int = 256):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels

    def forward(
        self,
        features: List[torch.Tensor],
        targets: Optional[List[Dict]] = None,
    ) -> Tuple[Dict[str, torch.Tensor], List[Dict]]:
        raise NotImplementedError


class FasterRCNNHead(DetectionHead):
    """Faster R-CNN detection head (torchvision wrapper)."""

    requires_features = False

    def __init__(
        self,
        num_classes: int,
        in_channels: int = 256,
        pretrained: bool = True,
        classifier_type: str = "default",
        classifier_temperature: float = 1.0,
        **kwargs,
    ):
        super().__init__(num_classes, in_channels)
        _ = kwargs
        weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained else None
        self.model = _build_with_retry(
            lambda: fasterrcnn_resnet50_fpn(weights=weights, weights_backbone=None),
            weights,
        )
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features

        if classifier_type == "mutual_exclusive":
            self.model.roi_heads.box_predictor = MutualExclusivePredictor(
                in_features, num_classes, temperature=classifier_temperature
            )
        else:
            self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    def forward(
        self,
        images: List[torch.Tensor],
        targets: Optional[List[Dict]] = None,
    ) -> Tuple[Dict[str, torch.Tensor], List[Dict]]:
        if targets is not None:
            losses = self.model(images, targets)
            return losses, []
        detections = self.model(images)
        return {}, detections


class FasterRCNNV2Head(DetectionHead):
    """Faster R-CNN v2 detection head (torchvision wrapper)."""

    requires_features = False

    def __init__(
        self,
        num_classes: int,
        in_channels: int = 256,
        pretrained: bool = True,
        classifier_type: str = "default",
        classifier_temperature: float = 1.0,
        **kwargs,
    ):
        super().__init__(num_classes, in_channels)
        _ = kwargs
        if fasterrcnn_resnet50_fpn_v2 is None:
            raise RuntimeError("fasterrcnn_resnet50_fpn_v2 is not available in this torchvision version.")
        weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT if pretrained else None
        self.model = _build_with_retry(
            lambda: fasterrcnn_resnet50_fpn_v2(weights=weights, weights_backbone=None),
            weights,
        )
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features

        if classifier_type == "mutual_exclusive":
            self.model.roi_heads.box_predictor = MutualExclusivePredictor(
                in_features, num_classes, temperature=classifier_temperature
            )
        else:
            self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    def forward(
        self,
        images: List[torch.Tensor],
        targets: Optional[List[Dict]] = None,
    ) -> Tuple[Dict[str, torch.Tensor], List[Dict]]:
        if targets is not None:
            losses = self.model(images, targets)
            return losses, []
        detections = self.model(images)
        return {}, detections


class FCOSHead(DetectionHead):
    """FCOS detection head (torchvision wrapper)."""

    requires_features = False

    def __init__(
        self,
        num_classes: int,
        in_channels: int = 256,
        pretrained: bool = True,
        **kwargs,
    ):
        super().__init__(num_classes, in_channels)
        _ = kwargs
        weights = FCOS_ResNet50_FPN_Weights.DEFAULT if pretrained else None
        self.model = _build_with_retry(
            lambda: fcos_resnet50_fpn(
                weights=weights, weights_backbone=None, num_classes=num_classes
            ),
            weights,
        )

    def forward(
        self,
        images: List[torch.Tensor],
        targets: Optional[List[Dict]] = None,
    ) -> Tuple[Dict[str, torch.Tensor], List[Dict]]:
        if targets is not None:
            losses = self.model(images, targets)
            return losses, []
        detections = self.model(images)
        return {}, detections


class RetinaNetHead(DetectionHead):
    """RetinaNet detection head (torchvision wrapper)."""

    requires_features = False

    def __init__(
        self,
        num_classes: int,
        in_channels: int = 256,
        pretrained: bool = True,
        **kwargs,
    ):
        super().__init__(num_classes, in_channels)
        _ = kwargs
        weights = RetinaNet_ResNet50_FPN_Weights.DEFAULT if pretrained else None
        self.model = _build_with_retry(
            lambda: retinanet_resnet50_fpn(
                weights=weights,
                weights_backbone=None,
                num_classes=num_classes,
            ),
            weights,
        )

    def forward(
        self,
        images: List[torch.Tensor],
        targets: Optional[List[Dict]] = None,
    ) -> Tuple[Dict[str, torch.Tensor], List[Dict]]:
        if targets is not None:
            losses = self.model(images, targets)
            return losses, []
        detections = self.model(images)
        return {}, detections


def create_head(
    name: str,
    num_classes: int,
    in_channels: int = 256,
    classifier_type: str = "default",
    classifier_temperature: float = 1.0,
    **kwargs,
) -> DetectionHead:
    """Detection head factory."""
    heads = {
        "faster_rcnn": FasterRCNNHead,
        "faster_rcnn_v2": FasterRCNNV2Head,
        "fcos": FCOSHead,
        "retinanet": RetinaNetHead,
    }
    if name not in heads:
        raise ValueError(f"Unknown head: {name}")
    if name in ("faster_rcnn_v2", "faster_rcnn"):
        return heads[name](
            num_classes, in_channels,
            classifier_type=classifier_type,
            classifier_temperature=classifier_temperature,
            **kwargs
        )
    return heads[name](num_classes, in_channels, **kwargs)


def _build_with_retry(build_fn, weights):
    try:
        return build_fn()
    except RuntimeError as exc:
        if weights is None or "invalid hash value" not in str(exc):
            raise
        _purge_weight_file(weights)
        return build_fn()


def _purge_weight_file(weights) -> None:
    url = getattr(weights, "url", None)
    if not url:
        return
    filename = os.path.basename(url)
    hub_dir = torch.hub.get_dir()
    ckpt_path = os.path.join(hub_dir, "checkpoints", filename)
    if os.path.isfile(ckpt_path):
        os.remove(ckpt_path)
