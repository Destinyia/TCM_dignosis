from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torchvision.ops import nms

from tcm_tongue.config import Config
from .backbone import BackboneBase, create_backbone
from .head import DetectionHead, create_head
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
    ) -> List[Dict]:
        self.eval()
        _, detections = self.forward(images)
        return self._postprocess(detections, score_thresh, nms_thresh)

    def _postprocess(
        self,
        detections: List[Dict],
        score_thresh: float,
        nms_thresh: float,
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
            processed.append(
                {
                    "boxes": boxes[keep_idx],
                    "scores": scores[keep_idx],
                    "labels": labels[keep_idx],
                }
            )
        return processed


def build_detector(config: Config) -> TongueDetector:
    """Build detector from configuration."""
    head = create_head(
        config.model.head,
        num_classes=config.model.num_classes,
        pretrained=config.model.pretrained,
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

    return TongueDetector(backbone, neck, head)


def _stack_images(images: List[torch.Tensor]) -> torch.Tensor:
    if len(images) == 1:
        return images[0].unsqueeze(0)
    shapes = [img.shape for img in images]
    if len({s for s in shapes}) != 1:
        raise ValueError("All images must have the same shape to stack")
    return torch.stack(images, dim=0)
