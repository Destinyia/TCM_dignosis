#!/usr/bin/env python
"""舌象分类模型训练脚本"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

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
                        choices=["resnet50", "resnet101", "efficientnet_b0", "efficientnet_b3"],
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
    parser.add_argument("--early-stop", type=int, default=10,
                        help="Early stopping patience (0 to disable)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint")

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
                        choices=["solid", "blur", "noise"],
                        help="Background replacement mode")
    parser.add_argument("--mask-aug-bg-blur", type=float, default=12.0,
                        help="Blur radius for background replacement (blur mode)")
    parser.add_argument("--mask-aug-dilate", type=int, default=15,
                        help="Mask dilation pixels to protect tongue edges")

    # 多任务分割损失
    parser.add_argument("--seg-loss", type=str, default="none",
                        choices=["none", "bce", "dice", "bce_dice"],
                        help="Segmentation loss for multitask training")
    parser.add_argument("--seg-loss-weight", type=float, default=0.2,
                        help="Weight for segmentation loss")

    return parser.parse_args()


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


def main():
    args = parse_args()

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
        mask_aug = MaskAugmentation(
            mode=args.mask_aug_mode,
            prob=args.mask_aug_prob,
            crop_padding=args.mask_aug_crop_padding,
            bg_mode=args.mask_aug_bg_mode,
            bg_blur_radius=args.mask_aug_bg_blur,
            mask_dilate=args.mask_aug_dilate,
        )

    use_seg_loss = args.seg_loss != "none"
    train_loader, val_loader, dataset_info = create_classification_dataloaders(
        root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=(args.image_size, args.image_size),
        use_weighted_sampler=args.weighted_sampler,
        sampler_strategy=args.sampler_strategy,
        strong_aug=args.strong_aug,
        return_bbox=args.use_bbox_loss,
        return_mask=use_seg_loss,
        mask_aug=mask_aug,
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
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

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
    if os.path.exists(history_path):
        records = []
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


if __name__ == "__main__":
    main()
