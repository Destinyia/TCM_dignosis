from pathlib import Path

import numpy as np
import pytest
import torch

from tcm_tongue.api.inference import TongueDetectorInference


class DummyModel(torch.nn.Module):
    def forward(self, images):
        detections = []
        for _ in images:
            detections.append(
                {
                    "boxes": torch.tensor([[1.0, 2.0, 10.0, 12.0]]),
                    "scores": torch.tensor([0.9]),
                    "labels": torch.tensor([0]),
                }
            )
        return {}, detections


def _make_inference(tmp_path: Path):
    model_path = tmp_path / "dummy.pth"
    torch.save({"model": DummyModel().state_dict()}, model_path)

    inf = TongueDetectorInference.__new__(TongueDetectorInference)
    inf.device = "cpu"
    inf.score_thresh = 0.5
    inf.nms_thresh = 0.5
    inf.config = type("cfg", (), {"data": type("d", (), {"root": str(tmp_path), "train_split": "train"})})()
    inf.config.data.label_offset = 0
    inf.model = DummyModel()
    inf.transform = lambda image, bboxes, labels: (
        torch.from_numpy(image).permute(2, 0, 1).float() / 255.0,
        bboxes,
        labels,
    )
    inf.category_names = {0: "baitaishe"}
    return inf


def test_predict_single_image(tmp_path: Path):
    inf = _make_inference(tmp_path)
    img = np.zeros((32, 32, 3), dtype=np.uint8)
    result = TongueDetectorInference.predict(inf, img)
    assert "boxes" in result
    assert "labels" in result
    assert "scores" in result


def test_predict_batch(tmp_path: Path):
    inf = _make_inference(tmp_path)
    imgs = [np.zeros((16, 16, 3), dtype=np.uint8) for _ in range(3)]
    results = TongueDetectorInference.predict_batch(inf, imgs, batch_size=2)
    assert len(results) == 3


def test_health_assessment(tmp_path: Path):
    inf = _make_inference(tmp_path)
    result = {"labels": ["hongshe", "chihenshe"]}
    assessment = TongueDetectorInference._assess_health(inf, result)
    assert assessment["risk_level"] in {"low", "medium", "high"}
    assert len(assessment["findings"]) > 0


def test_visualize_output(tmp_path: Path):
    inf = _make_inference(tmp_path)
    img = np.zeros((32, 32, 3), dtype=np.uint8)
    result = {
        "boxes": [[1, 1, 10, 10]],
        "labels": ["baitaishe"],
        "scores": [0.9],
    }
    out = TongueDetectorInference.visualize(inf, img, result, output_path=None, show=False)
    assert isinstance(out, np.ndarray)
