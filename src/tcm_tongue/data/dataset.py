from __future__ import annotations

import json
import os
from collections import defaultdict
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image, ImageFile
from torch.utils.data import Dataset
from torchvision.transforms import functional as F


class TongueCocoDataset(Dataset):
    """COCO-style dataset for tongue diagnosis."""

    def __init__(
        self,
        root: str,
        split: str = "train",
        transforms: Optional[Callable] = None,
        label_offset: int = 0,
    ):
        """
        Args:
            root: Dataset root directory.
            split: Dataset split (train/val/test).
            transforms: Optional augmentation callable.
        """
        self.root = root
        self.split = split
        self.transforms = transforms
        self.label_offset = int(label_offset)

        ann_path = os.path.join(root, split, "annotations", f"{split}.json")
        img_dir = os.path.join(root, split, "images")
        if not os.path.isfile(ann_path):
            raise FileNotFoundError(f"Annotation not found: {ann_path}")
        if not os.path.isdir(img_dir):
            raise FileNotFoundError(f"Image dir not found: {img_dir}")

        with open(ann_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.images = list(data.get("images", []))
        self._image_id_to_info = {img["id"]: img for img in self.images}
        self.image_dir = img_dir

        categories = list(data.get("categories", []))
        categories_sorted = sorted(categories, key=lambda c: c["id"])
        self.category_ids = [c["id"] for c in categories_sorted]
        self.category_names = {c["id"]: c["name"] for c in categories_sorted}
        self._cat_id_to_contiguous = {
            cat_id: idx for idx, cat_id in enumerate(self.category_ids)
        }

        self._annotations_by_image_id = defaultdict(list)
        for ann in data.get("annotations", []):
            self._annotations_by_image_id[ann["image_id"]].append(ann)

    def __len__(self) -> int:
        return len(self.images)

    def _load_image(self, file_name: str) -> Image.Image:
        path = os.path.join(self.image_dir, file_name)
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        return Image.open(path).convert("RGB")

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict]:
        """
        Returns:
            image: (C, H, W) tensor
            target: {
                "boxes": (N, 4) in xyxy format
                "labels": (N,) class labels
                "image_id": int
                "area": (N,) bbox areas
                "iscrowd": (N,) crowd flags
            }
        """
        image_info = self.images[idx]
        image_id = image_info["id"]
        image = self._load_image(image_info["file_name"])

        anns = self._annotations_by_image_id.get(image_id, [])
        bboxes_coco: List[List[float]] = []
        labels: List[int] = []
        areas: List[float] = []
        iscrowd: List[int] = []

        width, height = image.size

        for ann in anns:
            bbox = ann.get("bbox", None)
            if not bbox or len(bbox) != 4:
                continue
            x, y, w, h = bbox
            if w <= 0 or h <= 0:
                continue

            x1 = max(0.0, float(x))
            y1 = max(0.0, float(y))
            x2 = min(float(width), float(x) + float(w))
            y2 = min(float(height), float(y) + float(h))
            if x2 <= x1 or y2 <= y1:
                continue

            bw = x2 - x1
            bh = y2 - y1
            bboxes_coco.append([x1, y1, bw, bh])

            cat_id = ann.get("category_id")
            label = self._cat_id_to_contiguous.get(cat_id)
            if label is None:
                continue
            labels.append(int(label) + self.label_offset)
            areas.append(float(bw * bh))
            iscrowd.append(int(ann.get("iscrowd", 0)))

        image_np = np.array(image)
        if self.transforms is not None:
            transformed = self.transforms(image_np, np.array(bboxes_coco), np.array(labels))
            if isinstance(transformed, tuple) and len(transformed) == 3:
                image_tensor, bboxes_coco, labels = transformed
            elif isinstance(transformed, dict):
                image_tensor = transformed["image"]
                bboxes_coco = np.array(transformed.get("bboxes", []))
                labels = np.array(transformed.get("labels", []))
            else:
                raise ValueError("Unsupported transform output")
            areas = [float(b[2] * b[3]) for b in bboxes_coco]
            iscrowd = [0 for _ in range(len(bboxes_coco))]
        else:
            image_tensor = F.to_tensor(image)
            bboxes_coco = np.array(bboxes_coco, dtype=np.float32)
            labels = np.array(labels, dtype=np.int64)

        boxes_xyxy = _coco_to_xyxy(bboxes_coco)
        target = {
            "boxes": torch.as_tensor(boxes_xyxy, dtype=torch.float32),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor(image_id, dtype=torch.int64),
            "area": torch.as_tensor(areas, dtype=torch.float32),
            "iscrowd": torch.as_tensor(iscrowd, dtype=torch.int64),
        }

        return image_tensor, target

    def get_category_counts(self) -> Dict[int, int]:
        """Get annotation counts per class."""
        counts: Dict[int, int] = {idx: 0 for idx in range(len(self.category_ids))}
        for anns in self._annotations_by_image_id.values():
            for ann in anns:
                cat_id = ann.get("category_id")
                label = self._cat_id_to_contiguous.get(cat_id)
                if label is not None:
                    counts[label] += 1
        return counts

    def get_category_names(self) -> Dict[int, str]:
        """Get contiguous class id to name mapping."""
        return {self._cat_id_to_contiguous[k]: v for k, v in self.category_names.items()}

    @staticmethod
    def collate_fn(batch: List) -> Tuple[List[torch.Tensor], List[Dict]]:
        """Batch collation function."""
        images, targets = zip(*batch)
        return list(images), list(targets)


def _coco_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    if boxes.size == 0:
        return np.zeros((0, 4), dtype=np.float32)
    boxes = np.array(boxes, dtype=np.float32)
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 0] + boxes[:, 2]
    y2 = boxes[:, 1] + boxes[:, 3]
    return np.stack([x1, y1, x2, y2], axis=1)
