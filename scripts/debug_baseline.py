from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict
from pathlib import Path

import torch
from torch.optim import AdamW, SGD
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from tcm_tongue.config import Config
from tcm_tongue.data import TrainTransform, ValTransform, TongueCocoDataset
from tcm_tongue.engine import COCOEvaluator, Trainer
from tcm_tongue.models import build_detector


def _pick_indices(total: int, size: int, seed: int, shuffle: bool) -> list[int]:
    size = min(total, size)
    indices = list(range(total))
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(indices)
    return indices[:size]


def build_subset_loader(cfg: Config, split: str, train: bool, size: int, seed: int, shuffle: bool):
    transform = TrainTransform() if train else ValTransform()
    dataset = TongueCocoDataset(
        root=cfg.data.root,
        split=split,
        transforms=transform,
        label_offset=getattr(cfg.data, "label_offset", 0),
    )
    indices = _pick_indices(len(dataset), size, seed, shuffle)
    subset = Subset(dataset, indices)

    loader = DataLoader(
        subset,
        batch_size=cfg.data.batch_size,
        shuffle=train,
        num_workers=cfg.data.num_workers,
        collate_fn=dataset.collate_fn,
    )
    return loader


def _print_env_config(cfg: Config, args: argparse.Namespace, device: str) -> None:
    print("=== Environment ===")
    print(f"python: {sys.executable}")
    print(f"torch: {torch.__version__}")
    print(f"torch.version.cuda: {torch.version.cuda}")
    print(f"cuda available: {torch.cuda.is_available()}")
    print(f"device count: {torch.cuda.device_count()}")
    print(f"selected device: {device}")
    print("")
    print("=== Run Args ===")
    print(f"config: {args.config}")
    print(f"output_dir: {args.output_dir}")
    print(f"epochs: {args.epochs}")
    print(f"train_size: {args.train_size}")
    print(f"val_size: {args.val_size}")
    print(f"batch_size: {args.batch_size}")
    print(f"num_workers: {args.num_workers}")
    print(f"optimizer: {args.optimizer}")
    print("")
    print("=== Config ===")
    print(json.dumps(asdict(cfg), indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiments/baseline.yaml")
    parser.add_argument("--output-dir", default="runs/debug_baseline")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--train-size", type=int, default=200)
    parser.add_argument("--val-size", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--optimizer", choices=["adamw", "sgd"], default="adamw")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-shuffle", action="store_true")
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config)
    cfg.train.epochs = args.epochs
    cfg.data.batch_size = args.batch_size
    cfg.data.num_workers = args.num_workers

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    _print_env_config(cfg, args, device)

    train_loader = build_subset_loader(
        cfg,
        cfg.data.train_split,
        train=True,
        size=args.train_size,
        seed=args.seed,
        shuffle=not args.no_shuffle,
    )
    val_loader = build_subset_loader(
        cfg,
        cfg.data.val_split,
        train=False,
        size=args.val_size,
        seed=args.seed,
        shuffle=False,
    )

    model = build_detector(cfg)
    params = [p for p in model.parameters() if p.requires_grad]
    if args.optimizer == "sgd":
        optimizer = SGD(params, lr=cfg.train.lr, momentum=0.9, weight_decay=cfg.train.weight_decay)
    else:
        optimizer = AdamW(params, lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=None,
        device=device,
        output_dir=args.output_dir,
        max_epochs=cfg.train.epochs,
        eval_interval=1,
        save_interval=1,
    )
    trainer.train()

    evaluator = COCOEvaluator(val_loader.dataset.dataset)
    metrics = evaluator.evaluate(model, val_loader, device=device)
    print("=== Metrics ===")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
