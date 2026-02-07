import os

import numpy as np
import pytest

from tcm_tongue.data import (
    ClassBalancedSampler,
    StratifiedSampler,
    TongueCocoDataset,
    UnderSampler,
)


def _dataset_root():
    here = os.path.dirname(__file__)
    return os.path.abspath(os.path.join(here, "..", "datasets", "shezhenv3-coco"))


def _image_labels(dataset, idx):
    image_id = dataset.images[idx]["id"]
    anns = dataset._annotations_by_image_id.get(image_id, [])
    labels = [
        dataset._cat_id_to_contiguous.get(ann.get("category_id"))
        for ann in anns
    ]
    return [l for l in labels if l is not None]


class TestSamplers:

    @pytest.fixture
    def dataset(self):
        root = _dataset_root()
        if not os.path.isdir(root):
            pytest.skip("dataset not found")
        return TongueCocoDataset(root=root, split="train")

    def test_class_balanced_sampler_weights(self, dataset):
        sampler = ClassBalancedSampler(dataset)
        weights = sampler._weights
        assert len(weights) == len(dataset)
        assert float(weights.min()) > 0

    def test_class_balanced_sampler_distribution(self, dataset):
        sampler = ClassBalancedSampler(dataset)
        counts = dataset.get_category_counts()
        tail_classes = {cls_id for cls_id, count in counts.items() if count < 100}
        head_classes = {cls_id for cls_id, count in counts.items() if count > 1000}
        if not tail_classes or not head_classes:
            pytest.skip("insufficient tail/head classes")

        tail_index = next(
            idx for idx in range(len(dataset)) if set(_image_labels(dataset, idx)) & tail_classes
        )
        head_index = next(
            idx for idx in range(len(dataset)) if set(_image_labels(dataset, idx)) & head_classes
        )

        assert sampler._weights[tail_index] > sampler._weights[head_index]

    def test_stratified_sampler_batch_composition(self, dataset):
        sampler = StratifiedSampler(dataset, batch_size=8, drop_last=True)
        indices = list(iter(sampler))[:8]
        labels = [set(_image_labels(dataset, idx)) for idx in indices]
        distinct_labels = set.union(*labels) if labels else set()
        assert len(distinct_labels) >= 2

    def test_undersampler_reduces_head_classes(self, dataset):
        sampler = UnderSampler(dataset, target_ratio=0.5)
        counts = dataset.get_category_counts()
        sorted_counts = sorted(counts.values())
        median = sorted_counts[len(sorted_counts) // 2]
        head_classes = {cls_id for cls_id, count in counts.items() if count >= median}
        tail_classes = {cls_id for cls_id, count in counts.items() if count < median}
        if not tail_classes or not head_classes:
            pytest.skip("insufficient tail/head classes")

        head_weights = []
        tail_weights = []
        for idx in range(len(dataset)):
            labels = set(_image_labels(dataset, idx))
            if labels & head_classes:
                head_weights.append(float(sampler._weights[idx]))
            elif labels & tail_classes:
                tail_weights.append(float(sampler._weights[idx]))

        if not head_weights or not tail_weights:
            pytest.skip("no head/tail weights")
        assert np.mean(head_weights) < np.mean(tail_weights)
