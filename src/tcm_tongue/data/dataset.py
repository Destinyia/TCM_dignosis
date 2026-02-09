from __future__ import annotations

import json
import os
import random
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
        mosaic_prob: float = 0.0,
        mixup_prob: float = 0.0,
        image_size: Optional[Tuple[int, int]] = None,
        class_filter: Optional[List[str]] = None,
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
        self.mosaic_prob = float(mosaic_prob)
        self.mixup_prob = float(mixup_prob)
        self.image_size = image_size
        self.class_filter = class_filter

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
        if class_filter:
            allowed_ids = _resolve_class_filter(categories_sorted, class_filter)
            categories_sorted = [c for c in categories_sorted if c["id"] in allowed_ids]

        self.category_ids = [c["id"] for c in categories_sorted]
        self.category_names = {c["id"]: c["name"] for c in categories_sorted}
        self._cat_id_to_contiguous = {
            cat_id: idx for idx, cat_id in enumerate(self.category_ids)
        }

        self._annotations_by_image_id = defaultdict(list)
        annotations = data.get("annotations", [])
        if class_filter:
            annotations = [ann for ann in annotations if ann.get("category_id") in set(self.category_ids)]
        for ann in annotations:
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

        if self.mosaic_prob > 0 and random.random() < self.mosaic_prob:
            image_np, bboxes_coco, labels = self._load_mosaic(idx)
        else:
            image_np, bboxes_coco, labels = self._load_image_and_targets(idx)
            if self.mixup_prob > 0 and random.random() < self.mixup_prob:
                image_np, bboxes_coco, labels = self._apply_mixup(
                    image_np, bboxes_coco, labels
                )

        areas: List[float] = [float(b[2] * b[3]) for b in bboxes_coco]
        iscrowd: List[int] = [0 for _ in range(len(bboxes_coco))]

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
            image_tensor = F.to_tensor(Image.fromarray(image_np))
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

    def _load_image_and_targets(
        self, idx: int
    ) -> Tuple[np.ndarray, List[List[float]], List[int]]:
        image_info = self.images[idx]
        image = self._load_image(image_info["file_name"])
        image_id = image_info["id"]

        anns = self._annotations_by_image_id.get(image_id, [])
        bboxes_coco: List[List[float]] = []
        labels: List[int] = []
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

        return np.array(image), bboxes_coco, labels

    def _resize_image_and_boxes(
        self,
        image: np.ndarray,
        bboxes_coco: List[List[float]],
        size: Tuple[int, int],
    ) -> Tuple[np.ndarray, List[List[float]]]:
        if size is None:
            return image, bboxes_coco
        target_h, target_w = size
        h, w = image.shape[:2]
        if (w, h) == (target_w, target_h):
            return image, bboxes_coco
        scale_x = target_w / float(w)
        scale_y = target_h / float(h)
        resized = np.array(Image.fromarray(image).resize((target_w, target_h), resample=Image.BILINEAR))
        scaled_boxes: List[List[float]] = []
        for x, y, bw, bh in bboxes_coco:
            scaled_boxes.append([x * scale_x, y * scale_y, bw * scale_x, bh * scale_y])
        return resized, scaled_boxes

    def _load_mosaic(
        self, idx: int
    ) -> Tuple[np.ndarray, List[List[float]], List[int]]:
        if self.image_size is None:
            return self._load_image_and_targets(idx)
        out_h, out_w = self.image_size
        half_h, half_w = out_h // 2, out_w // 2
        canvas = np.full((out_h, out_w, 3), 114, dtype=np.uint8)

        indices = [idx]
        while len(indices) < 4:
            indices.append(random.randint(0, len(self.images) - 1))

        placements = [(0, 0), (0, half_w), (half_h, 0), (half_h, half_w)]
        all_boxes: List[List[float]] = []
        all_labels: List[int] = []

        for (y0, x0), img_idx in zip(placements, indices):
            img, boxes, labels = self._load_image_and_targets(img_idx)
            img, boxes = self._resize_image_and_boxes(img, boxes, (half_h, half_w))
            canvas[y0 : y0 + half_h, x0 : x0 + half_w] = img
            for x, y, bw, bh in boxes:
                all_boxes.append([x + x0, y + y0, bw, bh])
            all_labels.extend(labels)

        return canvas, all_boxes, all_labels

    def _apply_mixup(
        self,
        image: np.ndarray,
        bboxes_coco: List[List[float]],
        labels: List[int],
    ) -> Tuple[np.ndarray, List[List[float]], List[int]]:
        idx = random.randint(0, len(self.images) - 1)
        img2, boxes2, labels2 = self._load_image_and_targets(idx)
        size = self.image_size
        img1, boxes1 = self._resize_image_and_boxes(image, bboxes_coco, size)
        img2, boxes2 = self._resize_image_and_boxes(img2, boxes2, img1.shape[:2])

        alpha = 0.5
        mixed = (img1.astype(np.float32) * alpha + img2.astype(np.float32) * (1 - alpha)).astype(
            np.uint8
        )
        all_boxes = boxes1 + boxes2
        all_labels = labels + labels2
        return mixed, all_boxes, all_labels

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


def _resolve_class_filter(categories: List[Dict], class_filter: List[str]) -> set[int]:
    id_set = {int(c["id"]) for c in categories}
    name_to_id = {c["name"]: int(c["id"]) for c in categories}
    allowed: set[int] = set()
    missing: List[str] = []

    for item in class_filter:
        if isinstance(item, str):
            if item in name_to_id:
                allowed.add(name_to_id[item])
                continue
            try:
                val = int(item)
                if val in id_set:
                    allowed.add(val)
                    continue
            except ValueError:
                pass
            missing.append(item)
        elif isinstance(item, int):
            if item in id_set:
                allowed.add(item)
            else:
                missing.append(str(item))
        else:
            missing.append(str(item))

    if missing:
        raise ValueError(f"Unknown class_filter entries: {', '.join(missing)}")
    return allowed
