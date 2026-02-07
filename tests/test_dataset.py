import os

import pytest
import torch

from tcm_tongue.data import TongueCocoDataset


def _dataset_root():
    here = os.path.dirname(__file__)
    return os.path.abspath(os.path.join(here, "..", "datasets", "shezhenv3-coco"))


class TestTongueCocoDataset:

    @pytest.fixture
    def dataset(self):
        root = _dataset_root()
        if not os.path.isdir(root):
            pytest.skip("dataset not found")
        return TongueCocoDataset(root=root, split="train")

    def test_dataset_length(self, dataset):
        assert len(dataset) == 5594

    def test_getitem_returns_correct_format(self, dataset):
        image, target = dataset[0]
        assert isinstance(image, torch.Tensor)
        assert image.dim() == 3
        assert "boxes" in target
        assert "labels" in target
        assert target["boxes"].shape[1] == 4

    def test_category_counts(self, dataset):
        counts = dataset.get_category_counts()
        assert sum(counts.values()) == 14677

    def test_collate_fn(self, dataset):
        batch = [dataset[i] for i in range(4)]
        images, targets = TongueCocoDataset.collate_fn(batch)
        assert len(images) == 4
        assert len(targets) == 4
