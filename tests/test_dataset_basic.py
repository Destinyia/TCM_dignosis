import os

import pytest
import torch

from datasets.shezhen_coco import ShezhenCocoDataset


def _dataset_root():
    here = os.path.dirname(__file__)
    return os.path.abspath(os.path.join(here, "..", "datasets", "shezhenv3-coco"))


def test_dataset_basic():
    root = _dataset_root()
    if not os.path.isdir(root):
        pytest.skip("dataset not found")

    ds = ShezhenCocoDataset(root=root, split="train")
    assert len(ds) > 0
    assert len(ds.class_names) > 0

    image, target = ds[0]
    assert isinstance(image, torch.Tensor)
    assert image.dtype == torch.float32
    assert image.ndim == 3 and image.shape[0] == 3
    assert float(image.min()) >= 0.0 and float(image.max()) <= 1.0

    for key in ["boxes", "labels", "image_id", "area", "iscrowd"]:
        assert key in target

    boxes = target["boxes"]
    labels = target["labels"]
    assert boxes.ndim == 2 and boxes.shape[1] == 4
    assert labels.ndim == 1

    image_id = target["image_id"]
    assert image_id.dtype == torch.int64
    assert image_id.ndim == 0

    areas = target["area"]
    iscrowd = target["iscrowd"]
    assert areas.ndim == 1
    assert iscrowd.ndim == 1
