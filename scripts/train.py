from __future__ import annotations

import argparse
import json
import random
import sys
import datetime
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import math
import torch
from torch.optim import SGD, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torch.utils.data import DataLoader, Subset

from tcm_tongue.config import Config
from tcm_tongue.data import TrainTransform, ValTransform, TongueCocoDataset, create_sampler
from tcm_tongue.engine import COCOEvaluator, Trainer
from tcm_tongue.utils import plot_per_class_ap_ar, plot_loss_curves
from tcm_tongue.models import build_detector


def _subset_dataset(dataset, size: int | None, seed: int):
    if size is None:
        return dataset, None
    size = min(size, len(dataset))
    indices = list(range(len(dataset)))
    rng = random.Random(seed)
    rng.shuffle(indices)
    indices = indices[:size]
    return Subset(dataset, indices), indices


def build_dataloader(cfg: Config, split: str, train: bool, subset_size: int | None, seed: int):
    normalize = getattr(cfg.data, "normalize", True)
    resize = getattr(cfg.data, "resize_in_dataset", True)
    image_size = getattr(cfg.data, "image_size", (800, 800))
    if isinstance(image_size, list):
        image_size = tuple(image_size)
    augmentation = getattr(cfg, "augmentation", None)
    use_strong = False
    use_tcm_prior = False
    if augmentation is not None:
        use_strong = getattr(augmentation, "type", "") == "strong"
        use_tcm_prior = getattr(augmentation, "type", "") == "tcm_prior"
    aug_flip = getattr(augmentation, "horizontal_flip", True) if augmentation is not None else True
    aug_bc = getattr(augmentation, "brightness_contrast", True) if augmentation is not None else True
    aug_hs = getattr(augmentation, "hue_saturation", True) if augmentation is not None else True
    aug_noise = getattr(augmentation, "gauss_noise", True) if augmentation is not None else True
    tcm_prior_prob = (
        getattr(augmentation, "tcm_prior_prob", 0.3) if augmentation is not None else 0.3
    )
    transform = (
        TrainTransform(
            image_size=image_size,
            normalize=normalize,
            resize=resize,
            horizontal_flip=aug_flip,
            brightness_contrast=aug_bc,
            hue_saturation=aug_hs,
            gauss_noise=aug_noise,
            strong=use_strong,
            tcm_prior=use_tcm_prior,
            tcm_prior_prob=tcm_prior_prob,
        )
        if train
        else ValTransform(image_size=image_size, normalize=normalize, resize=resize)
    )
    mosaic_prob = 0.0
    mixup_prob = 0.0
    if train and augmentation is not None:
        if getattr(augmentation, "mosaic", False):
            mosaic_prob = float(getattr(augmentation, "mosaic_prob", 0.5))
        if getattr(augmentation, "mixup", False):
            mixup_prob = float(getattr(augmentation, "mixup_prob", 0.2))
    dataset = TongueCocoDataset(
        root=cfg.data.root,
        split=split,
        transforms=transform,
        label_offset=getattr(cfg.data, "label_offset", 0),
        mosaic_prob=mosaic_prob,
        mixup_prob=mixup_prob,
        image_size=image_size,
        class_filter=getattr(cfg.data, "class_filter", None),
    )
    dataset, _ = _subset_dataset(dataset, subset_size, seed)

    sampler = None
    if train and cfg.sampler.type != "default":
        kwargs = {}
        if cfg.sampler.type == "oversample":
            kwargs["oversample_factor"] = cfg.sampler.oversample_factor
        if cfg.sampler.type == "stratified":
            kwargs["batch_size"] = cfg.data.batch_size
        sampler = create_sampler(cfg.sampler.type, dataset, **kwargs)

    loader = DataLoader(
        dataset,
        batch_size=cfg.data.batch_size,
        shuffle=sampler is None and train,
        sampler=sampler,
        num_workers=cfg.data.num_workers,
        collate_fn=dataset.dataset.collate_fn if isinstance(dataset, Subset) else dataset.collate_fn,
    )
    return loader


def build_optimizer(cfg: Config, model: torch.nn.Module):
    params = [p for p in model.parameters() if p.requires_grad]
    if cfg.train.lr_scheduler in {"cosine", "warmup_cosine"}:
        return SGD(params, lr=cfg.train.lr, momentum=0.9, weight_decay=cfg.train.weight_decay)
    return AdamW(params, lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)


def build_scheduler(cfg: Config, optimizer):
    if cfg.train.lr_scheduler == "cosine":
        return CosineAnnealingLR(optimizer, T_max=cfg.train.epochs)
    if cfg.train.lr_scheduler == "step":
        return StepLR(optimizer, step_size=max(cfg.train.epochs // 3, 1), gamma=0.1)
    if cfg.train.lr_scheduler == "warmup_cosine":
        warmup_epochs = max(int(cfg.train.warmup_epochs), 0)
        total_epochs = max(int(cfg.train.epochs), 1)

        def lr_lambda(epoch: int) -> float:
            if warmup_epochs > 0 and epoch < warmup_epochs:
                return float(epoch + 1) / float(warmup_epochs)
            if total_epochs <= warmup_epochs:
                return 1.0
            progress = (epoch - warmup_epochs) / float(total_epochs - warmup_epochs)
            return 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    return None


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
    print(f"epochs override: {args.epochs}")
    print(f"batch_size override: {args.batch_size}")
    print(f"num_workers override: {args.num_workers}")
    print(f"train_size: {args.train_size}")
    print(f"val_size: {args.val_size}")
    print(f"seed: {args.seed}")
    print(f"image_size override: {args.image_size}")
    print(f"log_interval: {args.log_interval}")
    print(f"train_metric_samples: {args.train_metric_samples}")
    print(f"early_stop_patience: {cfg.train.early_stop_patience}")
    print(f"early_stop_min_delta: {cfg.train.early_stop_min_delta}")
    print(f"early_stop_metric: {cfg.train.early_stop_metric}")
    print("")
    print("=== Config ===")
    print(json.dumps(asdict(cfg), indent=2))
    print("")
    print("=== Key Settings ===")
    print(f"data.num_classes: {cfg.data.num_classes}")
    print(f"model.num_classes: {cfg.model.num_classes}")
    print(f"label_offset: {getattr(cfg.data, 'label_offset', 0)}")
    print(f"data.normalize: {getattr(cfg.data, 'normalize', True)}")
    print(f"data.resize_in_dataset: {getattr(cfg.data, 'resize_in_dataset', True)}")
    print(f"data.image_size: {getattr(cfg.data, 'image_size', None)}")
    print(f"model.head: {cfg.model.head}")
    print(f"model.pretrained: {cfg.model.pretrained}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiments/baseline.yaml")
    timestamp=datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    parser.add_argument("--output-dir", default=f"runs/experiments/baseline_8cls_4000_{timestamp}")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--train-size", type=int, default=4000)
    parser.add_argument("--val-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-size", nargs=2, type=int, default=[800, 800], help="Image size, e.g. --image-size 800 800 (default from config).")
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--train-metric-samples", type=int, default=200)
    parser.add_argument("--early-stop-patience", type=int, default=8)
    parser.add_argument("--early-stop-min-delta", type=float, default=0.001)
    parser.add_argument("--early-stop-metric", default="mAP")
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config)
    if args.epochs is not None:
        cfg.train.epochs = args.epochs
    if args.batch_size is not None:
        cfg.data.batch_size = args.batch_size
    if args.num_workers is not None:
        cfg.data.num_workers = args.num_workers
    if args.image_size is not None:
        cfg.data.image_size = list(args.image_size)
    if args.early_stop_patience is not None:
        cfg.train.early_stop_patience = args.early_stop_patience
    if args.early_stop_min_delta is not None:
        cfg.train.early_stop_min_delta = args.early_stop_min_delta
    if args.early_stop_metric is not None:
        cfg.train.early_stop_metric = args.early_stop_metric
    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    _print_env_config(cfg, args, device)

    train_loader = build_dataloader(
        cfg, cfg.data.train_split, train=True, subset_size=args.train_size, seed=args.seed
    )
    val_loader = build_dataloader(
        cfg, cfg.data.val_split, train=False, subset_size=args.val_size, seed=args.seed
    )

    # Auto-adjust num_classes based on actual dataset categories (supports class_filter)
    base_dataset = train_loader.dataset
    if isinstance(base_dataset, Subset):
        base_dataset = base_dataset.dataset
    actual_num_classes = len(base_dataset.category_ids) + cfg.data.label_offset
    if cfg.model.num_classes != actual_num_classes:
        print(f"[INFO] Auto-adjusting model.num_classes: {cfg.model.num_classes} -> {actual_num_classes}")
        cfg.model.num_classes = actual_num_classes

    model = build_detector(cfg)
    optimizer = build_optimizer(cfg, model)
    scheduler = build_scheduler(cfg, optimizer)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        output_dir=args.output_dir,
        max_epochs=cfg.train.epochs,
        log_interval=args.log_interval,
        train_metric_samples=args.train_metric_samples,
        early_stop_patience=cfg.train.early_stop_patience,
        early_stop_min_delta=cfg.train.early_stop_min_delta,
        early_stop_metric=cfg.train.early_stop_metric,
    )
    train_summary = trainer.train()

    evaluator = COCOEvaluator(val_loader.dataset)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pr_dir = output_dir / "per_class_pr_curves"
    metrics = evaluator.evaluate(
        model,
        val_loader,
        device=device,
        plot_dir=str(pr_dir),
        plot_pr=True,
        pr_iou=0.5,
    )
    if train_summary:
        metrics.update({k: v for k, v in train_summary.items() if v is not None})

    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    ap_plot = output_dir / "per_class_metrics.png"
    plotted = plot_per_class_ap_ar(
        metrics.get("per_class_AP", {}),
        metrics.get("per_class_AR", {}),
        ap_plot,
        title="Per-class AP/AR (val)",
    )
    if plotted:
        print(f"Saved per-class plot: {ap_plot}")
    else:
        print("Skipped per-class plot (missing data or matplotlib).")

    loss_plot = output_dir / "loss_curves.png"
    if plot_loss_curves(output_dir / "metrics_history.jsonl", loss_plot):
        print(f"Saved loss curves: {loss_plot}")
    else:
        print("Skipped loss curves (missing history or matplotlib).")


if __name__ == "__main__":
    main()
