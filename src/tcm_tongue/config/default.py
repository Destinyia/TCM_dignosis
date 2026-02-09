from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Any

import yaml


@dataclass
class DataConfig:
    root: str = "datasets/shezhenv3-coco"
    train_split: str = "train"
    val_split: str = "val"
    test_split: str = "test"
    num_classes: int = 21
    label_offset: int = 1
    image_size: List[int] = field(default_factory=lambda: [800, 800])
    class_filter: Optional[List[str]] = None
    batch_size: int = 8
    num_workers: int = 4
    normalize: bool = False
    resize_in_dataset: bool = False


@dataclass
class ModelConfig:
    backbone: str = "resnet50"  # resnet50, resnet101, swin_t, swin_s
    pretrained: bool = True
    neck: str = "fpn"  # fpn, bifpn, panet
    head: str = "faster_rcnn_v2"  # faster_rcnn, faster_rcnn_v2, fcos, retinanet
    num_classes: int = 22


@dataclass
class TrainConfig:
    epochs: int = 50
    lr: float = 0.001
    weight_decay: float = 0.0001
    lr_scheduler: str = "cosine"  # step, cosine, warmup_cosine
    warmup_epochs: int = 3
    early_stop_patience: int = 5
    early_stop_min_delta: float = 0.0
    early_stop_metric: str = "mAP"


@dataclass
class LossConfig:
    type: str = "focal"  # ce, weighted_ce, focal
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    class_weights: Optional[List[float]] = None


@dataclass
class SamplerConfig:
    type: str = "default"  # default, oversample, undersample, stratified
    oversample_factor: float = 2.0


@dataclass
class AugmentationConfig:
    type: str = "basic"  # basic, strong, tcm_prior
    horizontal_flip: bool = True
    brightness_contrast: bool = True
    hue_saturation: bool = True
    gauss_noise: bool = True
    mosaic: bool = False
    mixup: bool = False
    mosaic_prob: float = 0.5
    mixup_prob: float = 0.2
    tcm_prior_prob: float = 0.3


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    sampler: SamplerConfig = field(default_factory=SamplerConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        """Load configuration from a YAML file."""
        path_obj = Path(path)
        if not path_obj.is_file():
            raise FileNotFoundError(f"Config file not found: {path}")

        with path_obj.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        if "_base_" in data:
            base_path = Path(data.pop("_base_"))
            if not base_path.is_absolute():
                base_path = path_obj.parent / base_path
            base_cfg = cls.from_yaml(str(base_path))
            merged = _merge_dicts(asdict(base_cfg), data)
        else:
            merged = data

        return cls(
            data=_load_section(DataConfig, merged.get("data", {})),
            model=_load_section(ModelConfig, merged.get("model", {})),
            train=_load_section(TrainConfig, merged.get("train", {})),
            loss=_load_section(LossConfig, merged.get("loss", {})),
            sampler=_load_section(SamplerConfig, merged.get("sampler", {})),
            augmentation=_load_section(AugmentationConfig, merged.get("augmentation", {})),
        )

    def to_yaml(self, path: str) -> None:
        """Save configuration to a YAML file."""
        path_obj = Path(path)
        payload = {
            "data": asdict(self.data),
            "model": asdict(self.model),
            "train": asdict(self.train),
            "loss": asdict(self.loss),
            "sampler": asdict(self.sampler),
            "augmentation": asdict(self.augmentation),
        }
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        with path_obj.open("w", encoding="utf-8") as f:
            yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=False)


def _load_section(dataclass_type, values: Dict[str, Any]):
    base = dataclass_type()
    for key, value in values.items():
        if hasattr(base, key):
            setattr(base, key, value)
    return base


def _merge_dicts(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged: Dict[str, Any] = {}
    for key in base.keys() | override.keys():
        if key in base and key in override and isinstance(base[key], dict) and isinstance(override[key], dict):
            merged[key] = _merge_dicts(base[key], override[key])
        elif key in override:
            merged[key] = override[key]
        else:
            merged[key] = base[key]
    return merged
