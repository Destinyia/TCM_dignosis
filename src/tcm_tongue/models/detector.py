from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import nms, RoIAlign

from tcm_tongue.config import Config
from .backbone import BackboneBase, create_backbone
from .head import DetectionHead, create_head, GlobalClassifier, TongueClassifier
from .neck import create_neck


class TongueDetector(nn.Module):
    """End-to-end tongue detector."""

    def __init__(
        self,
        backbone: nn.Module,
        neck: nn.Module,
        head: DetectionHead,
    ):
        super().__init__()
        self.backbone = backbone
        self.neck = neck
        self.head = head

    def forward(
        self,
        images: List[torch.Tensor],
        targets: Optional[List[Dict]] = None,
    ) -> Tuple[Dict[str, torch.Tensor], List[Dict]]:
        if not images:
            raise ValueError("Images list is empty")

        if not getattr(self.head, "requires_features", True):
            return self.head(images, targets)

        batch = _stack_images(images)
        features = self.backbone(batch)
        fpn_features = self.neck(features)
        losses, detections = self.head(fpn_features, targets)
        return losses, detections

    @torch.no_grad()
    def predict(
        self,
        images: List[torch.Tensor],
        score_thresh: float = 0.5,
        nms_thresh: float = 0.5,
        top_k: int = 0,
    ) -> List[Dict]:
        self.eval()
        _, detections = self.forward(images)
        return self._postprocess(detections, score_thresh, nms_thresh, top_k)

    def _postprocess(
        self,
        detections: List[Dict],
        score_thresh: float,
        nms_thresh: float,
        top_k: int = 0,
    ) -> List[Dict]:
        processed = []
        for det in detections:
            boxes = det.get("boxes")
            scores = det.get("scores")
            labels = det.get("labels")
            if boxes is None or scores is None or labels is None:
                processed.append(det)
                continue

            keep = scores >= score_thresh
            boxes = boxes[keep]
            scores = scores[keep]
            labels = labels[keep]

            if boxes.numel() == 0:
                processed.append({"boxes": boxes, "scores": scores, "labels": labels})
                continue

            keep_idx = nms(boxes, scores, nms_thresh)

            # Top-K 限制（单目标约束）
            if top_k > 0 and len(keep_idx) > top_k:
                sorted_idx = scores[keep_idx].argsort(descending=True)[:top_k]
                keep_idx = keep_idx[sorted_idx]

            processed.append(
                {
                    "boxes": boxes[keep_idx],
                    "scores": scores[keep_idx],
                    "labels": labels[keep_idx],
                }
            )
        return processed


class TongueDetectorWithGlobal(TongueDetector):
    """带全局分类分支的检测器"""

    def __init__(
        self,
        backbone: nn.Module,
        neck: nn.Module,
        head: DetectionHead,
        global_classifier: nn.Module,
        num_classes: int = 8,
        fusion_weight: float = 0.5,
        global_loss_weight: float = 1.0,
        consistency_loss_weight: float = 0.1,
    ):
        super().__init__(backbone, neck, head)
        self.global_classifier = global_classifier
        self.num_classes = num_classes
        self.fusion_weight = fusion_weight
        self.global_loss_weight = global_loss_weight
        self.consistency_loss_weight = consistency_loss_weight

    def forward(
        self,
        images: List[torch.Tensor],
        targets: Optional[List[Dict]] = None,
    ) -> Tuple[Dict[str, torch.Tensor], List[Dict]]:
        # 原有检测流程
        losses, detections = super().forward(images, targets)

        if self.global_classifier is None:
            return losses, detections

        # 获取backbone特征用于全局分类
        # 对于 requires_features=False 的head，需要从head内部获取特征
        if not getattr(self.head, "requires_features", True):
            # 从 torchvision 模型获取 backbone 特征
            backbone_features = self._extract_backbone_features(images)
        else:
            batch = _stack_images(images)
            backbone_features = self.backbone(batch)

        # 全局分类 - 使用最深层特征
        if isinstance(backbone_features, dict):
            feat = backbone_features.get("feat3", list(backbone_features.values())[-1])
        elif isinstance(backbone_features, (list, tuple)):
            feat = backbone_features[-1]
        else:
            feat = backbone_features

        global_logits = self.global_classifier(feat)

        if self.training and targets is not None:
            # 计算全局分类损失
            global_labels = self._extract_image_labels(targets)
            if global_labels is not None:
                global_loss = F.cross_entropy(global_logits, global_labels)
                losses["loss_global_cls"] = self.global_loss_weight * global_loss
        else:
            # 推理时可选融合（当前简化处理，不融合）
            pass

        return losses, detections

    def _extract_backbone_features(self, images: List[torch.Tensor]) -> torch.Tensor:
        """从 torchvision 检测模型提取 backbone 特征"""
        if hasattr(self.head, "model") and hasattr(self.head.model, "backbone"):
            # 使用 torchvision 模型的 backbone
            batch = _stack_images(images)
            return self.head.model.backbone(batch)
        return None

    def _extract_image_labels(self, targets: List[Dict]) -> Optional[torch.Tensor]:
        """从 targets 提取图像级标签（取每图第一个目标的类别）"""
        labels = []
        for t in targets:
            if "labels" in t and len(t["labels"]) > 0:
                # 取第一个标签，减去 label_offset 得到 0-based 类别
                label = t["labels"][0].item() - 1  # 假设 label_offset=1
                labels.append(max(0, min(label, self.num_classes - 1)))
            else:
                labels.append(0)
        if not labels:
            return None
        device = targets[0]["labels"].device if targets and "labels" in targets[0] else "cpu"
        return torch.tensor(labels, dtype=torch.long, device=device)


class TwoStageDetector(nn.Module):
    """两阶段检测器：定位 + 分类"""

    def __init__(
        self,
        localizer: nn.Module,
        classifier: nn.Module,
        roi_size: int = 7,
        num_classes: int = 8,
    ):
        super().__init__()
        self.localizer = localizer  # Stage 1: 二分类检测器
        self.classifier = classifier  # Stage 2: 8分类器
        self.roi_size = roi_size
        self.num_classes = num_classes
        # ROI Align 用于提取检测框特征
        self.roi_align = RoIAlign(
            output_size=(roi_size, roi_size),
            spatial_scale=1.0 / 32,  # 假设特征图下采样32倍
            sampling_ratio=2,
        )

    def forward(
        self,
        images: List[torch.Tensor],
        targets: Optional[List[Dict]] = None,
    ) -> Tuple[Dict[str, torch.Tensor], List[Dict]]:
        # Stage 1: 定位
        losses, detections = self.localizer(images, targets)

        if self.training:
            # 训练时，Stage 2 分类器单独训练
            # 这里返回 Stage 1 的损失
            return losses, detections

        # 推理时，对检测结果进行重分类
        if self.classifier is not None:
            detections = self._reclassify(images, detections)

        return losses, detections

    def _reclassify(
        self,
        images: List[torch.Tensor],
        detections: List[Dict],
    ) -> List[Dict]:
        """对检测框进行重分类"""
        # 简化实现：直接使用检测结果，不做重分类
        # 完整实现需要提取 ROI 特征并通过 classifier
        return detections

    @torch.no_grad()
    def predict(
        self,
        images: List[torch.Tensor],
        score_thresh: float = 0.5,
        nms_thresh: float = 0.5,
        top_k: int = 0,
    ) -> List[Dict]:
        self.eval()
        _, detections = self.forward(images)
        # 使用 localizer 的后处理
        if hasattr(self.localizer, "_postprocess"):
            return self.localizer._postprocess(detections, score_thresh, nms_thresh, top_k)
        return detections


def build_detector(config: Config) -> TongueDetector:
    """Build detector from configuration."""
    # 检查是否启用全局分类分支
    global_cfg = getattr(config, "global_classifier", None)
    use_global = global_cfg is not None and getattr(global_cfg, "enabled", False)

    # 检查是否启用两阶段解耦
    two_stage_cfg = getattr(config, "two_stage", None)
    use_two_stage = two_stage_cfg is not None and getattr(two_stage_cfg, "enabled", False)

    head = create_head(
        config.model.head,
        num_classes=config.model.num_classes,
        pretrained=config.model.pretrained,
        classifier_type=getattr(config.model, "classifier_type", "default"),
        classifier_temperature=getattr(config.model, "classifier_temperature", 1.0),
    )

    if getattr(head, "requires_features", True):
        backbone = create_backbone(
            config.model.backbone,
            pretrained=config.model.pretrained,
        )
        neck = create_neck(
            config.model.neck,
            in_channels=getattr(backbone, "out_channels", []),
        )
    else:
        backbone = nn.Identity()
        neck = nn.Identity()

    # 构建带全局分类的检测器
    if use_global:
        # 8类舌象（不含背景）
        num_tongue_classes = config.model.num_classes - getattr(config.data, "label_offset", 1)
        global_classifier = GlobalClassifier(
            in_channels=2048,  # ResNet50 最后一层通道数
            num_classes=num_tongue_classes,
        )
        return TongueDetectorWithGlobal(
            backbone, neck, head,
            global_classifier=global_classifier,
            num_classes=num_tongue_classes,
            fusion_weight=getattr(global_cfg, "fusion_weight", 0.5),
            global_loss_weight=getattr(global_cfg, "global_loss_weight", 1.0),
            consistency_loss_weight=getattr(global_cfg, "consistency_loss_weight", 0.1),
        )

    # 构建两阶段检测器
    if use_two_stage:
        # Stage 2 分类器
        classifier = TongueClassifier(
            in_features=1024,
            num_classes=getattr(two_stage_cfg, "stage2_num_classes", 8),
        )
        base_detector = TongueDetector(backbone, neck, head)
        return TwoStageDetector(
            localizer=base_detector,
            classifier=classifier,
            roi_size=getattr(two_stage_cfg, "roi_size", 7),
            num_classes=getattr(two_stage_cfg, "stage2_num_classes", 8),
        )

    return TongueDetector(backbone, neck, head)


def _stack_images(images: List[torch.Tensor]) -> torch.Tensor:
    if len(images) == 1:
        return images[0].unsqueeze(0)
    shapes = [img.shape for img in images]
    if len({s for s in shapes}) != 1:
        raise ValueError("All images must have the same shape to stack")
    return torch.stack(images, dim=0)
