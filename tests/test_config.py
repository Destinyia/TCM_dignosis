from pathlib import Path

from tcm_tongue.config import Config


def test_config_default():
    cfg = Config()
    assert cfg.data.root == "datasets/shezhenv3-coco"
    assert cfg.model.backbone == "resnet50"
    assert cfg.train.epochs == 50
    assert cfg.loss.type == "focal"
    assert cfg.sampler.type == "default"


def test_config_to_yaml(tmp_path: Path):
    cfg = Config()
    cfg.data.batch_size = 4
    cfg.loss.type = "ce"

    out_path = tmp_path / "config.yaml"
    cfg.to_yaml(str(out_path))
    assert out_path.is_file()

    loaded = Config.from_yaml(str(out_path))
    assert loaded.data.batch_size == 4
    assert loaded.loss.type == "ce"
    assert loaded.model.backbone == cfg.model.backbone


def test_config_from_yaml(tmp_path: Path):
    content = {
        "data": {"batch_size": 16},
        "model": {"backbone": "resnet101"},
        "loss": {"type": "weighted_ce", "class_weights": [1.0, 2.0]},
    }
    path = tmp_path / "custom.yaml"
    path.write_text(
        "data:\n  batch_size: 16\n"
        "model:\n  backbone: resnet101\n"
        "loss:\n  type: weighted_ce\n  class_weights: [1.0, 2.0]\n",
        encoding="utf-8",
    )

    cfg = Config.from_yaml(str(path))
    assert cfg.data.batch_size == 16
    assert cfg.model.backbone == "resnet101"
    assert cfg.loss.type == "weighted_ce"
    assert cfg.loss.class_weights == [1.0, 2.0]


def test_config_override(tmp_path: Path):
    base_path = tmp_path / "base.yaml"
    base_path.write_text(
        "data:\n  batch_size: 8\n  num_workers: 2\n"
        "model:\n  backbone: resnet50\n"
        "loss:\n  type: ce\n",
        encoding="utf-8",
    )

    override_path = tmp_path / "override.yaml"
    override_path.write_text(
        f"_base_: {base_path.name}\n"
        "data:\n  batch_size: 12\n"
        "loss:\n  type: focal\n  focal_gamma: 1.5\n",
        encoding="utf-8",
    )

    cfg = Config.from_yaml(str(override_path))
    assert cfg.data.batch_size == 12
    assert cfg.data.num_workers == 2
    assert cfg.model.backbone == "resnet50"
    assert cfg.loss.type == "focal"
    assert cfg.loss.focal_gamma == 1.5
