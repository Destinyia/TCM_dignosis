from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont

from tcm_tongue.config import Config
from tcm_tongue.data import ValTransform
from tcm_tongue.models import build_detector


class TongueDetectorInference:
    """Tongue diagnosis detection inference API."""

    def __init__(
        self,
        model_path: str,
        config_path: Optional[str] = None,
        device: str = "cuda",
        score_thresh: float = 0.5,
        nms_thresh: float = 0.5,
    ):
        self.device = _resolve_device(device)
        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh

        self.config = Config.from_yaml(config_path) if config_path else Config()
        self.model = self._load_model(model_path)
        self.transform = ValTransform(
            normalize=getattr(self.config.data, "normalize", True),
            resize=getattr(self.config.data, "resize_in_dataset", True),
        )
        self.category_names = self._load_category_names()

    def _load_model(self, model_path: str) -> nn.Module:
        model = build_detector(self.config)
        checkpoint = torch.load(model_path, map_location=self.device)
        state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
        model.load_state_dict(state, strict=False)
        model.to(self.device)
        model.eval()
        return model

    @torch.no_grad()
    def predict(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
    ) -> Dict:
        img_array = self._preprocess(image)
        img_tensor = self._apply_transform(img_array)
        img_tensor = img_tensor.unsqueeze(0).to(self.device)

        _, detections = self.model([img_tensor[0]])
        result = self._postprocess(detections[0])
        result["health_assessment"] = self._assess_health(result)
        return result

    def predict_batch(
        self,
        images: List[Union[str, Path, Image.Image, np.ndarray]],
        batch_size: int = 8,
    ) -> List[Dict]:
        results: List[Dict] = []
        for i in range(0, len(images), batch_size):
            batch = images[i : i + batch_size]
            tensors = [self._apply_transform(self._preprocess(img)) for img in batch]
            tensors = [t.to(self.device) for t in tensors]
            _, detections = self.model(tensors)
            for det in detections:
                res = self._postprocess(det)
                res["health_assessment"] = self._assess_health(res)
                results.append(res)
        return results

    def _preprocess(self, image) -> np.ndarray:
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")
        if isinstance(image, Image.Image):
            image = np.array(image)
        if not isinstance(image, np.ndarray):
            raise TypeError("Unsupported image type")
        return image

    def _apply_transform(self, img_array: np.ndarray) -> torch.Tensor:
        try:
            out = self.transform(img_array, np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int64))
            if isinstance(out, tuple):
                return out[0]
            if isinstance(out, dict):
                return out["image"]
        except TypeError:
            return self.transform(img_array)
        raise ValueError("Unsupported transform output")

    def _postprocess(self, detection: Dict) -> Dict:
        boxes = detection.get("boxes", torch.empty((0, 4))).cpu().numpy().tolist()
        scores = detection.get("scores", torch.empty((0,))).cpu().numpy().tolist()
        label_ids = detection.get("labels", torch.empty((0,), dtype=torch.int64)).cpu().numpy().tolist()
        label_offset = getattr(self.config.data, "label_offset", 0)
        mapped_ids = [lid - label_offset for lid in label_ids]
        labels = [self.category_names.get(lid, f"unknown_{lid}") for lid in mapped_ids]

        filtered = [
            (b, l, lid, s)
            for b, l, lid, s in zip(boxes, labels, mapped_ids, scores)
            if s >= self.score_thresh
        ]

        if filtered:
            boxes, labels, label_ids, scores = zip(*filtered)
        else:
            boxes, labels, label_ids, scores = [], [], [], []

        return {
            "boxes": list(boxes),
            "labels": list(labels),
            "label_ids": list(mapped_ids),
            "scores": list(scores),
        }

    def _assess_health(self, detection_result: Dict) -> Dict:
        labels = detection_result.get("labels", [])

        findings: List[str] = []
        suggestions: List[str] = []
        organ_status: Dict[str, str] = {}
        risk_score = 0

        if "hongshe" in labels:
            findings.append("红舌 - 可能存在热证")
            suggestions.append("建议清热降火，多饮水")
            risk_score += 1
        if "zishe" in labels:
            findings.append("紫舌 - 可能存在血瘀")
            suggestions.append("建议活血化瘀，适当运动")
            risk_score += 2

        if "huangtaishe" in labels:
            findings.append("黄苔 - 可能存在里热证")
            suggestions.append("建议清热解毒")
            risk_score += 1
        if "heitaishe" in labels:
            findings.append("黑苔 - 可能存在重症或服药影响")
            suggestions.append("建议及时就医检查")
            risk_score += 3

        if "chihenshe" in labels:
            findings.append("齿痕舌 - 可能存在脾虚湿盛")
            suggestions.append("建议健脾祛湿")
            risk_score += 1
        if "liewenshe" in labels:
            findings.append("裂纹舌 - 可能存在阴虚")
            suggestions.append("建议滋阴润燥")
            risk_score += 1

        for label in labels:
            if "xinfeiao" in label:
                organ_status["心肺"] = "虚"
            if "gandanao" in label:
                organ_status["肝胆"] = "虚"
            if "piweiao" in label:
                organ_status["脾胃"] = "虚"
            if "shenquao" in label:
                organ_status["肾"] = "虚"

        if "jiankangshe" in labels and len(labels) == 1:
            findings.append("健康舌象")
            risk_score = 0

        if risk_score == 0:
            risk_level = "low"
        elif risk_score <= 2:
            risk_level = "medium"
        else:
            risk_level = "high"

        if not findings:
            findings.append("未检测到明显异常舌象")
        if not suggestions:
            suggestions.append("保持良好生活习惯")

        return {
            "risk_level": risk_level,
            "findings": findings,
            "suggestions": suggestions,
            "organ_status": organ_status,
        }

    def visualize(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
        result: Dict,
        output_path: Optional[str] = None,
        show: bool = True,
    ) -> np.ndarray:
        img_array = self._preprocess(image)
        img = Image.fromarray(img_array.copy())
        draw = ImageDraw.Draw(img)
        font = _load_default_font()

        for box, label, score in zip(
            result.get("boxes", []), result.get("labels", []), result.get("scores", [])
        ):
            x1, y1, x2, y2 = box
            draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
            text = f"{label}: {score:.2f}"
            draw.text((x1, max(y1 - 12, 0)), text, fill="red", font=font)

        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            img.save(output_path)

        if show:
            try:
                img.show()
            except Exception:
                pass

        return np.array(img)

    def _load_category_names(self) -> Dict[int, str]:
        root = Path(self.config.data.root)
        split = self.config.data.train_split
        classes_path = root / split / "classes.txt"
        if classes_path.is_file():
            names = [line.strip() for line in classes_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            return {idx: name for idx, name in enumerate(names)}
        return {}


def _resolve_device(device: str) -> str:
    if device.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return device


def _load_default_font() -> ImageFont.ImageFont:
    try:
        return ImageFont.load_default()
    except Exception:
        return None
