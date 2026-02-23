#!/usr/bin/env python
"""舌象分类模型训练脚本"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
import subprocess

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
import math
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.lr_scheduler import _LRScheduler
from PIL import Image, ImageDraw, ImageFont, ImageFile

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except Exception:
    HAS_MATPLOTLIB = False

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "Tongue_segment"))

from tcm_tongue.data.classification_dataset import (
    MaskAugmentation,
    create_classification_dataloaders,
)
from tcm_tongue.engine.cls_trainer import ClassificationTrainer
from tcm_tongue.losses import (
    BCEDiceLoss,
    ClassBalancedFocalLoss,
    DiceLoss,
    FocalLoss,
    MaskBBoxLoss,
    SeesawLoss,
)
from tcm_tongue.models.classifier import build_classifier


def parse_args():
    parser = argparse.ArgumentParser(description="Train tongue classification model")

    # 数据参数
    parser.add_argument("--data-root", type=str, default="datasets/shezhenv3-coco",
                        help="Dataset root directory")
    parser.add_argument("--image-size", type=int, default=640,
                        help="Input image size")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Batch size")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="Number of data loading workers")

    # 模型参数
    parser.add_argument("--model-type", type=str, default="seg_attention",
                        choices=["baseline", "seg_attention", "seg_attention_v2",
                                 "seg_attention_multiscale", "dual_stream"],
                        help="Model type")
    parser.add_argument("--backbone", type=str, default="resnet50",
                        choices=["resnet34", "resnet50", "resnet101", "efficientnet_b0", "efficientnet_b2", "efficientnet_b3"],
                        help="Backbone network")
    parser.add_argument("--num-classes", type=int, default=13,
                        help="Number of classes (default 13 for basic tongue types)")
    parser.add_argument("--seg-weights", type=str,
                        default="Tongue_segment/weights/tongue05.pt",
                        help="Segmentation model weights path")
    parser.add_argument("--dropout", type=float, default=0.5,
                        help="Dropout rate")

    # 注意力机制参数 (v2/multiscale)
    parser.add_argument("--soft-floor", type=float, default=0.1,
                        help="Soft attention floor value (0-1), prevents complete suppression")
    parser.add_argument("--no-channel-attention", action="store_true",
                        help="Disable channel attention in seg_attention_v2")
    parser.add_argument("--init-alpha", type=float, default=0.0,
                        help="Initial alpha value for residual weight (0=balanced, negative=more attention)")
    parser.add_argument("--attention-mode", type=str, default="add",
                        choices=["add", "gate"],
                        help="Attention mode: add (interpolate) or gate (additive)")
    parser.add_argument("--train-seg", action="store_true",
                        help="Unfreeze segmentation encoder for training")

    # 训练参数
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--scheduler", type=str, default="cosine_warmup",
                        choices=["cosine_warmup", "cosine", "none"],
                        help="Learning rate scheduler")
    parser.add_argument("--warmup-epochs", type=int, default=5,
                        help="Warmup epochs for cosine_warmup")
    parser.add_argument("--min-lr", type=float, default=1e-6,
                        help="Minimum learning rate for cosine schedules")
    parser.add_argument("--weight-decay", type=float, default=1e-4,
                        help="Weight decay")
    parser.add_argument("--grad-clip", type=float, default=1.0,
                        help="Gradient clipping value")

    # 损失函数
    parser.add_argument("--loss", type=str, default="focal",
                        choices=["ce", "focal", "cb_focal", "seesaw"],
                        help="Loss function")
    parser.add_argument("--focal-gamma", type=float, default=2.0,
                        help="Focal loss gamma")

    # 其他
    parser.add_argument("--output-dir", type=str, default="runs/classification",
                        help="Output directory")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use")
    parser.add_argument("--amp", action="store_true", default=True,
                        help="Use automatic mixed precision")
    parser.add_argument("--no-amp", action="store_false", dest="amp",
                        help="Disable automatic mixed precision")
    parser.add_argument("--weighted-sampler", action="store_true",
                        help="Use weighted random sampler")
    parser.add_argument("--sampler-strategy", type=str, default="sqrt",
                        choices=["sqrt", "inverse"],
                        help="Sampler weighting strategy (sqrt=gentler, inverse=aggressive)")
    parser.add_argument("--strong-aug", action="store_true",
                        help="Use strong data augmentation")
    parser.add_argument("--aug-scale", action="store_true",
                        help="Use random scale augmentation")
    parser.add_argument("--aug-scale-limit", type=float, default=0.2,
                        help="Random scale limit (e.g. 0.1 => 0.9~1.1)")
    parser.add_argument("--aug-scale-prob", type=float, default=0.3,
                        help="Random scale probability")
    parser.add_argument("--aug-affine", action="store_true",
                        help="Use random affine augmentation (translate, scale, rotate)")
    parser.add_argument("--aug-rotate", action="store_true",
                        help="Use random rotation augmentation (+/-15 degrees)")
    parser.add_argument("--aug-rotate-limit", type=float, default=15.0,
                        help="Rotation limit in degrees (e.g. 5, 8, 15)")
    parser.add_argument("--aug-rotate-prob", type=float, default=0.3,
                        help="Rotation probability")
    parser.add_argument("--aug-rotate-fill", type=str, default="gray",
                        choices=["gray", "noise"],
                        help="Rotation fill mode for background")
    parser.add_argument("--aug-gauss-noise", action="store_true",
                        help="Use global Gaussian noise augmentation")
    parser.add_argument("--early-stop", type=int, default=10,
                        help="Early stopping patience (0 to disable)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")

    # BBox损失参数
    parser.add_argument("--use-bbox-loss", action="store_true",
                        help="Use bbox supervision for mask quality")
    parser.add_argument("--bbox-loss-weight", type=float, default=0.1,
                        help="Weight for bbox loss")
    parser.add_argument("--bbox-iou-weight", type=float, default=1.0,
                        help="IoU loss weight in bbox loss")
    parser.add_argument("--bbox-coverage-weight", type=float, default=1.0,
                        help="Coverage loss weight in bbox loss")
    parser.add_argument("--bbox-boundary-weight", type=float, default=0.5,
                        help="Boundary loss weight in bbox loss")
    parser.add_argument("--use-distance-penalty", action="store_true", default=True,
                        help="Use distance-based soft penalty for boundary loss")
    parser.add_argument("--no-distance-penalty", action="store_false", dest="use_distance_penalty",
                        help="Disable distance-based penalty")
    parser.add_argument("--distance-scale", type=float, default=5.0,
                        help="Scale factor for distance penalty")
    parser.add_argument("--use-mask-refiner", action="store_true",
                        help="Use learnable mask refiner")
    parser.add_argument("--visualize-samples", action="store_true",
                        help="Visualize mask-bbox alignment during evaluation")

    # Mask增强参数
    parser.add_argument("--mask-aug", action="store_true",
                        help="Use segmentation mask for data augmentation (crop/background)")
    parser.add_argument("--mask-aug-mode", type=str, default="both",
                        choices=["crop", "background", "both"],
                        help="Mask augmentation mode")
    parser.add_argument("--mask-aug-prob", type=float, default=0.5,
                        help="Probability of applying mask augmentation")
    parser.add_argument("--mask-aug-crop-padding", type=float, default=0.1,
                        help="Padding ratio for mask crop")
    parser.add_argument("--mask-aug-bg-mode", type=str, default="blur",
                        choices=["solid", "blur", "noise", "random"],
                        help="Background replacement mode (random: randomly choose)")
    parser.add_argument("--mask-aug-bg-color", type=str, default="gray",
                        choices=["gray", "random"],
                        help="Background color for solid mode (gray: 114,114,114; random: random color)")
    parser.add_argument("--mask-aug-bg-blur", type=float, default=12.0,
                        help="Blur radius for background replacement (blur mode)")
    parser.add_argument("--mask-aug-dilate", type=int, default=15,
                        help="Mask dilation pixels to protect tongue edges")
    parser.add_argument("--mask-source", type=str, default="dataset",
                        choices=["dataset", "seg_model"],
                        help="Mask source for mask-aug/seg-loss")
    parser.add_argument("--mask-seg-weights", type=str, default=None,
                        help="Segmentation weights for mask-source=seg_model (default: --seg-weights)")
    parser.add_argument("--mask-seg-device", type=str, default="cpu",
                        help="Device for mask segmentation model (default: cpu)")
    parser.add_argument("--mask-seg-threshold", type=float, default=0.5,
                        help="Threshold for mask segmentation output")

    # 多任务分割损失
    parser.add_argument("--seg-loss", type=str, default="none",
                        choices=["none", "bce", "dice", "bce_dice"],
                        help="Segmentation loss for multitask training")
    parser.add_argument("--seg-loss-weight", type=float, default=0.2,
                        help="Weight for segmentation loss")

    return parser.parse_args()


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _plot_training_curves(records: list[dict], output_dir: str) -> None:
    if not HAS_MATPLOTLIB:
        print("Warning: matplotlib not available, skipping curve plots")
        return

    epochs = [r.get("epoch") for r in records if r.get("epoch") is not None]
    train_loss = [r.get("train_loss") if r.get("train_loss") is not None else float("nan") for r in records]
    val_loss = [r.get("val_loss") if r.get("val_loss") is not None else float("nan") for r in records]
    train_acc = [r.get("train_accuracy") if r.get("train_accuracy") is not None else float("nan") for r in records]
    val_acc = [r.get("val_accuracy") if r.get("val_accuracy") is not None else float("nan") for r in records]

    if epochs:
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, train_loss, label="train_loss")
        if np.isfinite(val_loss).any():
            plt.plot(epochs, val_loss, label="val_loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Loss Curve")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "loss_curve.png"), dpi=150)
        plt.close()

        plt.figure(figsize=(8, 5))
        plt.plot(epochs, train_acc, label="train_acc")
        if np.isfinite(val_acc).any():
            plt.plot(epochs, val_acc, label="val_acc")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Accuracy Curve")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "acc_curve.png"), dpi=150)
        plt.close()


def _unwrap_dataset(dataset):
    if hasattr(dataset, "valid_images"):
        return dataset
    if hasattr(dataset, "dataset"):
        return _unwrap_dataset(dataset.dataset)
    return None


def _save_original_grid(
    dataset,
    class_names: dict[int, str],
    output_dir: str,
    seed: int,
    rows: int = 4,
    cols: int = 4,
    gap: int = 1,
) -> None:
    dataset = _unwrap_dataset(dataset)
    if dataset is None or not hasattr(dataset, "valid_images"):
        print("Warning: could not access dataset for original grid.")
        return

    total = rows * cols
    rng = random.Random(seed)
    indices = list(range(len(dataset)))
    if len(indices) >= total:
        pick = rng.sample(indices, total)
    else:
        pick = [rng.choice(indices) for _ in range(total)]

    target_h, target_w = dataset.image_size
    grid_w = cols * target_w + (cols - 1) * gap
    grid_h = rows * target_h + (rows - 1) * gap
    grid = Image.new("RGB", (grid_w, grid_h), color=(0, 0, 0))
    font = ImageFont.load_default()

    print("Original grid samples:")
    for i, idx in enumerate(pick):
        info = dataset.valid_images[idx]
        image_id = info["id"]
        image_path = os.path.join(dataset.image_dir, info["file_name"])
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        image = Image.open(image_path).convert("RGB")
        orig_w, orig_h = image.size
        anns = dataset._annotations_by_image_id.get(image_id, [])
        label_id = dataset._get_label(anns)
        label_name = class_names.get(label_id, str(label_id))
        x1, y1, x2, y2 = dataset._get_bbox_pixels(anns, orig_h, orig_w)

        sx = target_w / max(orig_w, 1)
        sy = target_h / max(orig_h, 1)
        image = image.resize((target_w, target_h), Image.BILINEAR)
        draw = ImageDraw.Draw(image)
        draw.rectangle(
            [x1 * sx, y1 * sy, x2 * sx, y2 * sy],
            outline=(255, 0, 0),
            width=2,
        )
        text = f"{label_name}"
        text_w, text_h = draw.textsize(text, font=font)
        draw.rectangle(
            [2, 2, 2 + text_w + 4, 2 + text_h + 4],
            fill=(0, 0, 0),
        )
        draw.text((4, 4), text, fill=(255, 255, 255), font=font)

        row = i // cols
        col = i % cols
        x = col * (target_w + gap)
        y = row * (target_h + gap)
        grid.paste(image, (x, y))

        print(f"- idx={idx}, file={info['file_name']}, label={label_name}, bbox=({x1},{y1},{x2},{y2})")

    out_path = os.path.join(output_dir, "orig_grid_epoch1.png")
    grid.save(out_path)
    print(f"Saved original grid to: {out_path}")


def _run_aug_grid(args, output_dir: str, seed: int) -> None:
    script_path = PROJECT_ROOT / "scripts" / "visualize_aug_grid.py"
    output_path = os.path.join(output_dir, "aug_grid_epoch1.png")
    cmd = [
        sys.executable,
        str(script_path),
        "--data-root", args.data_root,
        "--split", "train",
        "--image-size", str(args.image_size),
        "--rows", "4",
        "--cols", "4",
        "--seed", str(seed),
        "--output", output_path,
        "--random",
    ]
    if args.mask_aug:
        cmd += [
            "--mask-aug",
            "--mask-aug-mode", args.mask_aug_mode,
            "--mask-aug-prob", str(args.mask_aug_prob),
            "--mask-aug-crop-padding", str(args.mask_aug_crop_padding),
            "--mask-aug-bg-mode", args.mask_aug_bg_mode,
            "--mask-aug-bg-color", args.mask_aug_bg_color,
            "--mask-aug-bg-blur", str(args.mask_aug_bg_blur),
            "--mask-aug-dilate", str(args.mask_aug_dilate),
        ]
        if args.mask_source == "seg_model":
            seg_weights = args.mask_seg_weights or args.seg_weights
            seg_weights_path = Path(seg_weights)
            if not seg_weights_path.is_absolute():
                seg_weights_path = PROJECT_ROOT / seg_weights_path
            if seg_weights_path.exists():
                cmd += [
                    "--mask-source", "seg_model",
                    "--seg-weights", str(seg_weights_path),
                    "--device", args.mask_seg_device,
                ]
            else:
                print(f"Warning: seg weights not found at {seg_weights_path}, using dataset masks for aug grid.")
    if args.strong_aug:
        cmd.append("--strong-aug")
    if args.aug_scale:
        cmd += [
            "--aug-scale",
            "--aug-scale-limit", str(args.aug_scale_limit),
            "--aug-scale-prob", str(args.aug_scale_prob),
        ]
    if args.aug_affine:
        cmd.append("--aug-affine")
    if args.aug_rotate:
        cmd += [
            "--aug-rotate",
            "--aug-rotate-limit", str(args.aug_rotate_limit),
            "--aug-rotate-prob", str(args.aug_rotate_prob),
            "--aug-rotate-fill", args.aug_rotate_fill,
        ]
    if args.aug_gauss_noise:
        cmd.append("--aug-gauss-noise")

    subprocess.run(cmd, check=True)
    print(f"Saved augmentation grid to: {output_path}")


def _binary_roc_curve(y_true: np.ndarray, y_score: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    desc = np.argsort(-y_score)
    y_true = y_true[desc]
    y_score = y_score[desc]
    distinct = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct, y_true.size - 1]
    tps = np.cumsum(y_true)[threshold_idxs]
    fps = 1 + threshold_idxs - tps
    tps = np.r_[0, tps]
    fps = np.r_[0, fps]
    if tps[-1] == 0:
        tpr = np.zeros_like(tps, dtype=np.float32)
    else:
        tpr = tps / tps[-1]
    if fps[-1] == 0:
        fpr = np.zeros_like(fps, dtype=np.float32)
    else:
        fpr = fps / fps[-1]
    return fpr, tpr


def _plot_roc_curves(
    probs: np.ndarray,
    labels: np.ndarray,
    class_names: dict[int, str],
    output_dir: str,
) -> None:
    if not HAS_MATPLOTLIB:
        print("Warning: matplotlib not available, skipping ROC plot")
        return

    num_classes = probs.shape[1]
    fpr_grid = np.linspace(0.0, 1.0, 101)
    tprs = []

    plt.figure(figsize=(8, 6))
    for i in range(num_classes):
        y_true = (labels == i).astype(np.int32)
        if y_true.sum() == 0 or y_true.sum() == len(y_true):
            continue
        fpr, tpr = _binary_roc_curve(y_true, probs[:, i])
        tprs.append(np.interp(fpr_grid, fpr, tpr))
        name = class_names.get(i, str(i))
        plt.plot(fpr, tpr, alpha=0.5, linewidth=1, label=name)

    if tprs:
        mean_tpr = np.mean(np.stack(tprs, axis=0), axis=0)
        plt.plot(fpr_grid, mean_tpr, color="black", linewidth=2, label="macro-avg")

    plt.plot([0, 1], [0, 1], "k--", linewidth=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    plt.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "roc_curve.png"), dpi=150)
    plt.close()


def _evaluate_final_metrics(
    model: nn.Module,
    dataloader,
    device: str,
    class_names: dict[int, str],
    output_dir: str,
) -> None:
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in dataloader:
            images, labels = batch[:2]
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, preds = outputs.max(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    labels_np = np.array(all_labels)
    preds_np = np.array(all_preds)
    probs_np = np.array(all_probs)
    num_classes = probs_np.shape[1]
    total = len(labels_np)

    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    for p, l in zip(preds_np, labels_np):
        confusion[l, p] += 1

    print("\nPer-class metrics (final):")
    print(f"{'Class':<20} {'Acc':<10} {'Recall':<10} {'F1':<10} {'Support':<8}")
    print("-" * 64)
    for i in range(num_classes):
        tp = confusion[i, i]
        fn = confusion[i, :].sum() - tp
        fp = confusion[:, i].sum() - tp
        tn = total - tp - fp - fn
        acc = (tp + tn) / total if total > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        support = confusion[i, :].sum()
        name = class_names.get(i, str(i))
        print(f"{name:<20} {acc:<10.4f} {recall:<10.4f} {f1:<10.4f} {support:<8}")

    _plot_roc_curves(probs_np, labels_np, class_names, output_dir)


def create_loss_function(
    loss_type: str,
    num_classes: int,
    class_counts: dict,
    gamma: float = 2.0,
) -> nn.Module:
    """创建损失函数"""
    if loss_type == "ce":
        return nn.CrossEntropyLoss()
    elif loss_type == "focal":
        return FocalLoss(gamma=gamma)
    elif loss_type == "cb_focal":
        counts = [class_counts.get(i, 1) for i in range(num_classes)]
        return ClassBalancedFocalLoss(class_counts=counts, gamma=gamma)
    elif loss_type == "seesaw":
        return SeesawLoss(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


class WarmupCosineLR(_LRScheduler):
    def __init__(self, optimizer, warmup_epochs: int, total_epochs: int, min_lr: float, last_epoch: int = -1):
        self.warmup_epochs = max(int(warmup_epochs), 0)
        self.total_epochs = max(int(total_epochs), 1)
        if self.warmup_epochs >= self.total_epochs:
            self.warmup_epochs = max(self.total_epochs - 1, 0)
        self.min_lr = float(min_lr)
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            scale = float(self.last_epoch + 1) / float(max(self.warmup_epochs, 1))
            return [base_lr * scale for base_lr in self.base_lrs]

        progress = (self.last_epoch - self.warmup_epochs) / float(max(self.total_epochs - self.warmup_epochs, 1))
        cos = 0.5 * (1.0 + math.cos(math.pi * progress))
        return [self.min_lr + (base_lr - self.min_lr) * cos for base_lr in self.base_lrs]


def main():
    args = parse_args()
    _set_seed(args.seed)

    if args.seg_loss != "none" and args.model_type not in [
        "seg_attention",
        "seg_attention_v2",
        "seg_attention_multiscale",
    ]:
        print(f"Segmentation loss requires seg_attention models, got: {args.model_type}")
        sys.exit(1)

    # 设置设备
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = "cpu"

    print(f"Using device: {args.device}")

    # 创建输出目录
    output_dir = os.path.join(args.output_dir, args.model_type)
    os.makedirs(output_dir, exist_ok=True)

    # 创建数据加载器
    print("\nLoading dataset...")
    mask_aug = None
    if args.mask_aug:
        # 解析背景颜色参数
        bg_color = (114, 114, 114) if args.mask_aug_bg_color == "gray" else "random"
        mask_aug = MaskAugmentation(
            mode=args.mask_aug_mode,
            prob=args.mask_aug_prob,
            crop_padding=args.mask_aug_crop_padding,
            bg_mode=args.mask_aug_bg_mode,
            bg_color=bg_color,
            bg_blur_radius=args.mask_aug_bg_blur,
            mask_dilate=args.mask_aug_dilate,
        )

    mask_seg_weights = args.mask_seg_weights or args.seg_weights
    mask_seg_weights_path = Path(mask_seg_weights)
    if not mask_seg_weights_path.is_absolute():
        mask_seg_weights_path = PROJECT_ROOT / mask_seg_weights_path

    if args.mask_aug and args.mask_source == "seg_model":
        if not mask_seg_weights_path.exists():
            raise FileNotFoundError(f"Seg weights not found for mask_source=seg_model: {mask_seg_weights_path}")
        if args.mask_seg_device != "cpu" and args.num_workers > 0:
            print("Warning: mask seg model on CUDA with num_workers>0 can be unstable. Setting num_workers=0.")
            args.num_workers = 0

    use_seg_loss = args.seg_loss != "none"
    train_loader, val_loader, dataset_info = create_classification_dataloaders(
        root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=(args.image_size, args.image_size),
        use_weighted_sampler=args.weighted_sampler,
        sampler_strategy=args.sampler_strategy,
        strong_aug=args.strong_aug,
        aug_scale=args.aug_scale,
        aug_scale_limit=args.aug_scale_limit,
        aug_scale_prob=args.aug_scale_prob,
        aug_affine=args.aug_affine,
        aug_rotate=args.aug_rotate,
        aug_rotate_limit=args.aug_rotate_limit,
        aug_rotate_prob=args.aug_rotate_prob,
        aug_rotate_fill=args.aug_rotate_fill,
        aug_gauss_noise=args.aug_gauss_noise,
        return_bbox=args.use_bbox_loss,
        return_mask=use_seg_loss,
        mask_aug=mask_aug,
        mask_source=args.mask_source,
        mask_seg_weights=str(mask_seg_weights_path),
        mask_seg_device=args.mask_seg_device,
        mask_seg_threshold=args.mask_seg_threshold,
    )

    num_classes = dataset_info["num_classes"]
    class_names = dataset_info["class_names"]
    class_counts = dataset_info["class_counts"]

    print(f"Number of classes: {num_classes}")
    print(f"Training samples: {dataset_info['train_size']}")
    print(f"Validation samples: {dataset_info['val_size']}")
    print(f"Class distribution: {class_counts}")

    # 创建模型
    print(f"\nBuilding model: {args.model_type}")
    seg_weights_path = None
    if args.model_type in ["seg_attention", "seg_attention_v2", "seg_attention_multiscale", "dual_stream"]:
        seg_weights_path = str(PROJECT_ROOT / args.seg_weights)
        if not os.path.exists(seg_weights_path):
            print(f"Warning: Segmentation weights not found at {seg_weights_path}")
            seg_weights_path = None

    freeze_seg = not args.train_seg
    if use_seg_loss:
        freeze_seg = False

    model = build_classifier(
        model_type=args.model_type,
        num_classes=num_classes,
        backbone=args.backbone,
        pretrained=True,
        seg_weights_path=seg_weights_path,
        dropout=args.dropout,
        use_mask_refiner=args.use_mask_refiner,
        soft_floor=args.soft_floor,
        use_channel_attention=not args.no_channel_attention,
        init_alpha=args.init_alpha,
        attention_mode=args.attention_mode,
        freeze_seg=freeze_seg,
    )

    # 打印模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # 创建损失函数
    criterion = create_loss_function(
        args.loss, num_classes, class_counts, args.focal_gamma
    )
    print(f"Loss function: {args.loss}")

    # 创建bbox损失（如果启用）
    mask_bbox_loss = None
    if args.use_bbox_loss:
        mask_bbox_loss = MaskBBoxLoss(
            iou_weight=args.bbox_iou_weight,
            coverage_weight=args.bbox_coverage_weight,
            boundary_weight=args.bbox_boundary_weight,
            use_distance_penalty=args.use_distance_penalty,
            distance_scale=args.distance_scale,
        )
        print(f"BBox loss enabled: weight={args.bbox_loss_weight}, distance_penalty={args.use_distance_penalty}")

    mask_seg_loss = None
    if use_seg_loss:
        if args.seg_loss == "bce":
            mask_seg_loss = nn.BCELoss()
        elif args.seg_loss == "dice":
            mask_seg_loss = DiceLoss()
        else:
            mask_seg_loss = BCEDiceLoss()
        print(f"Seg loss enabled: {args.seg_loss} (weight={args.seg_loss_weight})")

    # 创建优化器和调度器
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    if args.scheduler == "none":
        scheduler = None
    elif args.scheduler == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)
    else:
        scheduler = WarmupCosineLR(
            optimizer,
            warmup_epochs=args.warmup_epochs,
            total_epochs=args.epochs,
            min_lr=args.min_lr,
        )

    def _epoch_end_callback(epoch: int, train_metrics, val_metrics) -> None:
        if epoch != 1:
            return
        print("\nGenerating epoch-1 visual samples...")
        try:
            _run_aug_grid(args, output_dir, seed=args.seed)
        except Exception as exc:
            print(f"Warning: failed to generate aug grid ({exc}).")
        try:
            _save_original_grid(
                train_loader.dataset,
                class_names,
                output_dir,
                seed=args.seed,
            )
        except Exception as exc:
            print(f"Warning: failed to generate original grid ({exc}).")

    # 创建训练器
    trainer = ClassificationTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        device=args.device,
        output_dir=output_dir,
        max_epochs=args.epochs,
        grad_clip=args.grad_clip,
        amp=args.amp,
        early_stop_patience=args.early_stop,
        class_names=class_names,
        mask_bbox_loss=mask_bbox_loss,
        mask_loss_weight=args.bbox_loss_weight,
        use_bbox=args.use_bbox_loss,
        mask_seg_loss=mask_seg_loss,
        seg_loss_weight=args.seg_loss_weight,
        use_seg=use_seg_loss,
        visualize_samples=args.visualize_samples,
        print_per_class_metrics=False,
        epoch_end_callback=_epoch_end_callback,
    )

    # 恢复训练
    if args.resume:
        print(f"\nResuming from {args.resume}")
        trainer.load_checkpoint(args.resume)

    # 开始训练
    print("\nStarting training...")
    print("=" * 60)
    results = trainer.train()

    print("\n" + "=" * 60)
    print(f"Training completed!")
    print(f"Best macro_f1: {results['best_macro_f1']:.4f}")
    print(f"Results saved to: {output_dir}")

    history_path = os.path.join(output_dir, "metrics_history.jsonl")
    records = []
    if os.path.exists(history_path):
        with open(history_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    if records:
        def _get_metric(record, key):
            value = record.get(key)
            return value if isinstance(value, (int, float)) else 0.0

        best_record = max(records, key=lambda r: _get_metric(r, "val_macro_f1"))
        metrics = {
            "best_macro_f1": _get_metric(best_record, "val_macro_f1"),
            "best_accuracy": _get_metric(best_record, "val_accuracy"),
            "best_epoch": best_record.get("epoch", 0),
            "final_epoch": records[-1].get("epoch", 0),
        }
        metrics_path = os.path.join(output_dir, "metrics.json")
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        print(f"Metrics saved to: {metrics_path}")

        _plot_training_curves(records, output_dir)

    best_path = os.path.join(output_dir, "best.pth")
    if os.path.exists(best_path):
        try:
            trainer.load_checkpoint(best_path)
        except Exception as exc:
            print(f"Warning: failed to load best checkpoint ({exc}). Using current model.")
    if val_loader is not None:
        _evaluate_final_metrics(
            trainer.model,
            val_loader,
            args.device,
            class_names,
            output_dir,
        )


if __name__ == "__main__":
    main()
