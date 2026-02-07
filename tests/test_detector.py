import torch
import pytest

from tcm_tongue.config import Config
from tcm_tongue.models import build_detector


class TestTongueDetector:

    @pytest.fixture
    def detector(self):
        config = Config()
        return build_detector(config)

    def test_detector_forward_train(self, detector):
        images = [torch.rand(3, 800, 800) for _ in range(2)]
        targets = [
            {"boxes": torch.tensor([[100.0, 100.0, 200.0, 200.0]]), "labels": torch.tensor([1])},
            {"boxes": torch.tensor([[50.0, 50.0, 150.0, 150.0]]), "labels": torch.tensor([2])},
        ]
        detector.train()
        losses, _ = detector(images, targets)
        assert "loss_classifier" in losses or "loss_cls" in losses
        assert "loss_box_reg" in losses or "loss_bbox" in losses

    def test_detector_forward_eval(self, detector):
        images = [torch.rand(3, 800, 800) for _ in range(2)]
        detector.eval()
        _, detections = detector(images)
        assert len(detections) == 2
        assert "boxes" in detections[0]
        assert "labels" in detections[0]
        assert "scores" in detections[0]

    def test_detector_predict(self, detector):
        images = [torch.rand(3, 800, 800)]
        results = detector.predict(images)
        assert len(results) == 1
