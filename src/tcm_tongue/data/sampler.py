from __future__ import annotations

from typing import Dict, Iterator, List, Optional

import numpy as np
import torch
from torch.utils.data import Sampler, WeightedRandomSampler, Subset

from .dataset import TongueCocoDataset


class ClassBalancedSampler(Sampler):
    """Class-balanced sampler using oversampling."""

    def __init__(
        self,
        dataset: TongueCocoDataset,
        oversample_factor: float = 2.0,
    ):
        self.dataset = dataset
        self.oversample_factor = oversample_factor
        self._compute_weights()

    def _compute_weights(self):
        base, indices = _resolve_dataset_indices(self.dataset)
        counts = _get_category_counts(base, indices)
        max_count = max(counts.values()) if counts else 1
        class_weights = {
            cls_id: max_count / max(count, 1) for cls_id, count in counts.items()
        }

        tail_classes = {cls_id for cls_id, count in counts.items() if count < 100}
        weights: List[float] = []
        image_infos = _select_image_infos(base, indices)
        for img_info in image_infos:
            anns = base._annotations_by_image_id.get(img_info["id"], [])
            labels = [base._cat_id_to_contiguous.get(ann.get("category_id")) for ann in anns]
            labels = [l for l in labels if l is not None]
            if not labels:
                weights.append(1.0)
                continue
            weight = max(class_weights.get(l, 1.0) for l in labels)
            if any(l in tail_classes for l in labels):
                weight *= self.oversample_factor
            weights.append(float(weight))

        weights = np.array(weights, dtype=np.float32)
        if weights.sum() > 0:
            weights = weights / weights.mean()
        self._weights = torch.as_tensor(weights, dtype=torch.double)
        self._sampler = WeightedRandomSampler(self._weights, len(self._weights), replacement=True)

    def __iter__(self) -> Iterator[int]:
        return iter(self._sampler)

    def __len__(self) -> int:
        return len(self._weights)


class StratifiedSampler(Sampler):
    """Stratified sampler to increase per-batch diversity."""

    def __init__(
        self,
        dataset: TongueCocoDataset,
        batch_size: int,
        drop_last: bool = True,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self._indices = self._build_indices()

    def _build_indices(self) -> List[int]:
        label_to_indices: Dict[int, List[int]] = {}
        base, indices = _resolve_dataset_indices(self.dataset)
        image_infos = _select_image_infos(base, indices)
        for pos, img_info in enumerate(image_infos):
            anns = base._annotations_by_image_id.get(img_info["id"], [])
            labels = [base._cat_id_to_contiguous.get(ann.get("category_id")) for ann in anns]
            labels = [l for l in labels if l is not None]
            if not labels:
                label = -1
            else:
                label = labels[0]
            label_to_indices.setdefault(label, []).append(pos)

        # Round-robin interleave across labels to improve diversity per batch.
        buckets = [indices[:] for _, indices in sorted(label_to_indices.items())]
        max_len = max(len(b) for b in buckets) if buckets else 0
        order: List[int] = []
        for i in range(max_len):
            for bucket in buckets:
                if i < len(bucket):
                    order.append(bucket[i])
        return order

    def __iter__(self) -> Iterator[int]:
        total = len(self._indices)
        if self.drop_last:
            total = total - (total % self.batch_size)
        return iter(self._indices[:total])

    def __len__(self) -> int:
        if self.drop_last:
            return len(self._indices) - (len(self._indices) % self.batch_size)
        return len(self._indices)


class UnderSampler(Sampler):
    """Undersampler for head classes."""

    def __init__(
        self,
        dataset: TongueCocoDataset,
        target_ratio: float = 0.5,
    ):
        self.dataset = dataset
        self.target_ratio = target_ratio
        self._compute_weights()

    def _compute_weights(self):
        base, indices = _resolve_dataset_indices(self.dataset)
        counts = _get_category_counts(base, indices)
        if not counts:
            self._weights = torch.ones(len(self.dataset), dtype=torch.double)
            self._sampler = WeightedRandomSampler(self._weights, len(self._weights), replacement=True)
            return

        sorted_counts = sorted(counts.values())
        median = sorted_counts[len(sorted_counts) // 2]
        head_classes = {cls_id for cls_id, count in counts.items() if count >= median}

        weights: List[float] = []
        image_infos = _select_image_infos(base, indices)
        for img_info in image_infos:
            anns = base._annotations_by_image_id.get(img_info["id"], [])
            labels = [base._cat_id_to_contiguous.get(ann.get("category_id")) for ann in anns]
            labels = [l for l in labels if l is not None]
            if not labels:
                weights.append(1.0)
                continue
            if any(l in head_classes for l in labels):
                weights.append(float(self.target_ratio))
            else:
                weights.append(1.0)

        weights = np.array(weights, dtype=np.float32)
        if weights.sum() > 0:
            weights = weights / weights.mean()
        self._weights = torch.as_tensor(weights, dtype=torch.double)
        self._sampler = WeightedRandomSampler(self._weights, len(self._weights), replacement=True)

    def __iter__(self) -> Iterator[int]:
        return iter(self._sampler)

    def __len__(self) -> int:
        return len(self._weights)


def create_sampler(
    sampler_type: str,
    dataset: TongueCocoDataset,
    **kwargs,
) -> Optional[Sampler]:
    """Sampler factory."""
    samplers = {
        "default": None,
        "oversample": ClassBalancedSampler,
        "undersample": UnderSampler,
        "stratified": StratifiedSampler,
        "class_aware": ClassAwareSampler,
    }
    if sampler_type == "default":
        return None
    if sampler_type not in samplers:
        raise ValueError(f"Unknown sampler type: {sampler_type}")
    return samplers[sampler_type](dataset, **kwargs)


def _resolve_dataset_indices(dataset):
    if isinstance(dataset, Subset):
        return dataset.dataset, list(dataset.indices)
    return dataset, None


def _select_image_infos(dataset, indices):
    if hasattr(dataset, "valid_images"):
        images = list(dataset.valid_images)
    else:
        images = list(dataset.images)
    if indices is None:
        return images
    return [images[i] for i in indices]


def _get_category_counts(dataset, indices):
    if indices is None and hasattr(dataset, "get_category_counts"):
        return dataset.get_category_counts()
    counts: Dict[int, int] = {}
    cat_ids = getattr(dataset, "category_ids", [])
    if cat_ids:
        counts = {idx: 0 for idx in range(len(cat_ids))}
    for img_info in _select_image_infos(dataset, indices):
        anns = dataset._annotations_by_image_id.get(img_info["id"], [])
        for ann in anns:
            label = dataset._cat_id_to_contiguous.get(ann.get("category_id"))
            if label is not None:
                counts[label] = counts.get(label, 0) + 1
    return counts


class ClassAwareSampler(Sampler):
    """Class-aware sampler that samples by class first, then by image.

    Unlike ClassBalancedSampler which weights images, this sampler:
    1. First randomly selects a class
    2. Then randomly selects an image containing that class

    This ensures each class appears roughly equally often per epoch.
    """

    def __init__(
        self,
        dataset: TongueCocoDataset,
        num_samples_per_class: int = 4,
    ):
        """Initialize ClassAwareSampler.

        Args:
            dataset: The dataset to sample from.
            num_samples_per_class: Number of samples per class per epoch.
        """
        self.dataset = dataset
        self.num_samples_per_class = num_samples_per_class
        self._build_class_to_images()

    def _build_class_to_images(self) -> None:
        """Build mapping from class to image indices."""
        base, indices = _resolve_dataset_indices(self.dataset)
        image_infos = _select_image_infos(base, indices)

        self._class_to_images: Dict[int, List[int]] = {}
        for pos, img_info in enumerate(image_infos):
            anns = base._annotations_by_image_id.get(img_info["id"], [])
            labels = [
                base._cat_id_to_contiguous.get(ann.get("category_id"))
                for ann in anns
            ]
            labels = [l for l in labels if l is not None]
            for label in set(labels):
                self._class_to_images.setdefault(label, []).append(pos)

        # Get list of classes that have at least one image
        self._classes = [
            cls_id for cls_id, imgs in self._class_to_images.items()
            if len(imgs) > 0
        ]

    def __iter__(self) -> Iterator[int]:
        """Generate indices by class-aware sampling."""
        indices: List[int] = []

        for _ in range(self.num_samples_per_class):
            # Shuffle classes for each round
            class_order = np.random.permutation(self._classes).tolist()
            for cls_id in class_order:
                images = self._class_to_images.get(cls_id, [])
                if images:
                    idx = np.random.choice(images)
                    indices.append(int(idx))

        return iter(indices)

    def __len__(self) -> int:
        return len(self._classes) * self.num_samples_per_class
