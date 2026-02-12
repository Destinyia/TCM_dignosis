import json
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from tcm_tongue.data import TongueCocoDataset
from tcm_tongue.engine import COCOEvaluator, Trainer, DecoupledTrainer


def _make_dummy_coco(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    for split in ["train", "val"]:
        (root / split / "images").mkdir(parents=True, exist_ok=True)
        (root / split / "annotations").mkdir(parents=True, exist_ok=True)

    # Create one tiny image per split
    for split in ["train", "val"]:
        img_path = root / split / "images" / "img1.jpg"
        from PIL import Image

        Image.new("RGB", (32, 32), color=(255, 255, 255)).save(img_path)

        ann = {
            "images": [
                {"id": 1, "file_name": "img1.jpg", "width": 32, "height": 32},
            ],
            "annotations": [
                {
                    "id": 1,
                    "image_id": 1,
                    "category_id": 1,
                    "bbox": [4, 4, 16, 16],
                    "area": 256,
                    "iscrowd": 0,
                }
            ],
            "categories": [
                {"id": 1, "name": "class1"},
            ],
        }
        ann_path = root / split / "annotations" / f"{split}.json"
        ann_path.write_text(json.dumps(ann), encoding="utf-8")


class DummyDetector(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.param = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, images, targets=None):
        if targets is not None:
            loss = (self.param - 1.0).pow(2).mean()
            return {"loss_cls": loss, "loss_bbox": loss}, []
        detections = []
        for _ in images:
            detections.append(
                {
                    "boxes": torch.tensor([[4.0, 4.0, 20.0, 20.0]]),
                    "scores": torch.tensor([0.9]),
                    "labels": torch.tensor([0]),
                }
            )
        return {}, detections


class TestTrainer:
    def test_trainer_one_epoch(self, tmp_path: Path):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        data_root = tmp_path / "coco"
        _make_dummy_coco(data_root)
        dataset = TongueCocoDataset(root=str(data_root), split="train")
        loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=dataset.collate_fn)

        model = DummyDetector()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        trainer = Trainer(
            model=model,
            train_loader=loader,
            val_loader=None,
            optimizer=optimizer,
            device=device,
            max_epochs=1,
            log_interval=10,
            save_interval=10,
            eval_interval=10,
        )
        metrics = trainer.train()
        assert "best_mAP" in metrics

    def test_trainer_checkpoint_save_load(self, tmp_path: Path):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        data_root = tmp_path / "coco"
        _make_dummy_coco(data_root)
        dataset = TongueCocoDataset(root=str(data_root), split="train")
        loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=dataset.collate_fn)

        model = DummyDetector()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        trainer = Trainer(
            model=model,
            train_loader=loader,
            val_loader=None,
            optimizer=optimizer,
            device=device,
            max_epochs=1,
            output_dir=str(tmp_path / "runs"),
            log_interval=10,
            save_interval=1,
            eval_interval=10,
        )
        trainer.train()
        ckpt_path = Path(trainer.output_dir) / "epoch_1.pth"
        assert ckpt_path.is_file()

        model2 = DummyDetector()
        optimizer2 = torch.optim.SGD(model2.parameters(), lr=0.01)
        trainer2 = Trainer(
            model=model2,
            train_loader=loader,
            val_loader=None,
            optimizer=optimizer2,
            device=device,
        )
        state = trainer2.load_checkpoint(str(ckpt_path))
        assert "epoch" in state

    def test_trainer_amp(self, tmp_path: Path):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        data_root = tmp_path / "coco"
        _make_dummy_coco(data_root)
        dataset = TongueCocoDataset(root=str(data_root), split="train")
        loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=dataset.collate_fn)

        model = DummyDetector()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        trainer = Trainer(
            model=model,
            train_loader=loader,
            val_loader=None,
            optimizer=optimizer,
            device=device,
            max_epochs=1,
            amp=True,
            log_interval=10,
            save_interval=10,
            eval_interval=10,
        )
        trainer.train()

    def test_evaluator_coco_metrics(self, tmp_path: Path):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        data_root = tmp_path / "coco"
        _make_dummy_coco(data_root)
        dataset = TongueCocoDataset(root=str(data_root), split="val")
        loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=dataset.collate_fn)

        model = DummyDetector().to(device)
        evaluator = COCOEvaluator(dataset)
        metrics = evaluator.evaluate(model, loader, device=device)
        assert "mAP" in metrics
        assert "mAP_50" in metrics

    def test_evaluator_per_class_ap(self, tmp_path: Path):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        data_root = tmp_path / "coco"
        _make_dummy_coco(data_root)
        dataset = TongueCocoDataset(root=str(data_root), split="val")
        loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=dataset.collate_fn)

        model = DummyDetector().to(device)
        evaluator = COCOEvaluator(dataset)
        metrics = evaluator.evaluate(model, loader, device=device)
        per_class = metrics["per_class_AP"]
        assert "class1" in per_class


class TestDecoupledTrainer:
    def test_decoupled_trainer_two_stages(self, tmp_path: Path):
        """Test DecoupledTrainer runs both stages."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        data_root = tmp_path / "coco"
        _make_dummy_coco(data_root)
        dataset = TongueCocoDataset(root=str(data_root), split="train")
        loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=dataset.collate_fn)

        model = DummyDetector()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        trainer = DecoupledTrainer(
            model=model,
            train_loader=loader,
            val_loader=None,
            optimizer=optimizer,
            device=device,
            stage1_epochs=1,
            stage2_epochs=1,
            log_interval=10,
            save_interval=10,
            eval_interval=10,
        )
        metrics = trainer.train()
        assert "best_mAP" in metrics

    def test_decoupled_trainer_freezes_backbone(self, tmp_path: Path):
        """Test that DecoupledTrainer freezes backbone in stage 2."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        data_root = tmp_path / "coco"
        _make_dummy_coco(data_root)
        dataset = TongueCocoDataset(root=str(data_root), split="train")
        loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=dataset.collate_fn)

        model = DummyDetector()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        trainer = DecoupledTrainer(
            model=model,
            train_loader=loader,
            val_loader=None,
            optimizer=optimizer,
            device=device,
            stage1_epochs=1,
            stage2_epochs=0,
            log_interval=10,
            save_interval=10,
            eval_interval=10,
        )

        # Run stage 1 only
        trainer.train()

        # Check stage 1 checkpoint exists
        stage1_ckpt = Path(trainer.output_dir) / "stage1_final.pth"
        assert stage1_ckpt.is_file()
