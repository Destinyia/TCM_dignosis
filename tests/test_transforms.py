import random

import numpy as np
import pytest
import torch

try:
    import albumentations as A  # noqa: F401
except Exception:
    A = None

from tcm_tongue.data import TrainTransform, ValTransform


@pytest.mark.skipif(A is None, reason="albumentations not installed")
class TestTransforms:

    def test_train_transform_output_shape(self):
        transform = TrainTransform(image_size=(800, 800))
        image = np.zeros((600, 800, 3), dtype=np.uint8)
        bboxes = np.array([[10, 10, 100, 120]], dtype=np.float32)
        labels = np.array([0], dtype=np.int64)

        out_image, out_bboxes, out_labels = transform(image, bboxes, labels)
        assert isinstance(out_image, torch.Tensor)
        assert out_image.shape == (3, 800, 800)
        assert len(out_bboxes) == len(out_labels) == 1

    def test_val_transform_output_shape(self):
        transform = ValTransform(image_size=(800, 800))
        image = np.zeros((500, 700, 3), dtype=np.uint8)
        bboxes = np.array([[20, 30, 80, 90]], dtype=np.float32)
        labels = np.array([1], dtype=np.int64)

        out_image, out_bboxes, out_labels = transform(image, bboxes, labels)
        assert out_image.shape == (3, 800, 800)
        assert len(out_bboxes) == len(out_labels) == 1

    def test_bbox_preserved_after_transform(self):
        transform = ValTransform(image_size=(800, 800))
        image = np.zeros((400, 600, 3), dtype=np.uint8)
        bboxes = np.array([[50, 60, 120, 140]], dtype=np.float32)
        labels = np.array([2], dtype=np.int64)

        _, out_bboxes, out_labels = transform(image, bboxes, labels)
        assert len(out_bboxes) == 1
        assert len(out_labels) == 1

    def test_augmentation_increases_diversity(self):
        transform = TrainTransform(image_size=(256, 256))
        image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        bboxes = np.array([[20, 20, 80, 80]], dtype=np.float32)
        labels = np.array([0], dtype=np.int64)

        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)
        img1, _, _ = transform(image, bboxes, labels)

        random.seed(1)
        np.random.seed(1)
        torch.manual_seed(1)
        img2, _, _ = transform(image, bboxes, labels)

        assert not torch.allclose(img1, img2)
