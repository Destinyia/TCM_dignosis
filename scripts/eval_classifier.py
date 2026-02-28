#!/usr/bin/env python
"""舌象分类模型评估脚本"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
import numpy as np

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "Tongue_segment"))

from tcm_tongue.data.classification_dataset import (
    TongueClassificationDataset,
    ClassificationTransform,
)
from tcm_tongue.engine.cls_trainer import ClassificationEvaluator
from tcm_tongue.models.classifier import build_classifier


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate tongue classification model")

    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--data-root", type=str, default="datasets/shezhenv3-coco",
                        help="Dataset root directory")
    parser.add_argument("--split", type=str, default="val",
                        choices=["train", "val", "test"],
                        help="Dataset split to evaluate")
    parser.add_argument("--image-size", type=int, default=640,
                        help="Input image size")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Batch size")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="Number of data loading workers")

    # 模型参数
    parser.add_argument("--model-type", type=str, default="seg_attention",
                        choices=["baseline", "seg_attention", "dual_stream"],
                        help="Model type")
    parser.add_argument("--backbone", type=str, default="resnet50",
                        help="Backbone network")
    parser.add_argument("--num-classes", type=int, default=8,
                        help="Number of classes")
    parser.add_argument("--seg-weights", type=str,
                        default="Tongue_segment/weights/tongue05.pt",
                        help="Segmentation model weights path")

    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use")
    parser.add_argument("--save-predictions", action="store_true",
                        help="Save predictions to file")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for predictions")
    parser.add_argument("--weights-only", action="store_true",
                        help="Load checkpoint with weights_only=True (safer, may fail for some checkpoints)")
    parser.add_argument("--plot-confusion", action="store_true",
                        help="Save confusion matrix plots")
    parser.add_argument("--plot-calibration", action="store_true",
                        help="Save calibration plots (reliability + histogram)")
    parser.add_argument("--calib-bins", type=int, default=10,
                        help="Number of bins for calibration plot")

    return parser.parse_args()


def _plot_confusion_matrices(confusion: np.ndarray, class_names: dict, output_dir: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Warning: matplotlib not available, skipping confusion plots: {exc}")
        return

    labels = [class_names.get(i, str(i)) for i in range(confusion.shape[0])]
    cm = confusion.astype(float)
    cm_norm = cm / cm.sum(axis=1, keepdims=True)
    cm_norm = np.nan_to_num(cm_norm)

    def _plot(cm_data: np.ndarray, title: str, filename: str) -> None:
        fig, ax = plt.subplots(figsize=(8, 7))
        im = ax.imshow(cm_data, interpolation="nearest", cmap="Blues")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_yticklabels(labels)
        ax.set_ylabel("True")
        ax.set_xlabel("Pred")
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, filename), dpi=150)
        plt.close(fig)

    _plot(cm, "Confusion Matrix (Raw)", "confusion_raw.png")
    _plot(cm_norm, "Confusion Matrix (Normalized)", "confusion_norm.png")


def _plot_calibration(probs: np.ndarray, labels: np.ndarray, output_dir: str, bins: int = 10) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Warning: matplotlib not available, skipping calibration plot: {exc}")
        return

    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    correct = (predictions == labels).astype(float)

    bin_edges = np.linspace(0.0, 1.0, bins + 1)
    bin_ids = np.digitize(confidences, bin_edges, right=True) - 1
    bin_ids = np.clip(bin_ids, 0, bins - 1)

    bin_acc = np.zeros(bins, dtype=float)
    bin_conf = np.zeros(bins, dtype=float)
    bin_counts = np.zeros(bins, dtype=float)

    for b in range(bins):
        mask = bin_ids == b
        if not np.any(mask):
            continue
        bin_counts[b] = mask.sum()
        bin_acc[b] = correct[mask].mean()
        bin_conf[b] = confidences[mask].mean()

    total = bin_counts.sum() if bin_counts.sum() > 0 else 1.0
    ece = np.sum(np.abs(bin_acc - bin_conf) * (bin_counts / total))

    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(7, 8), gridspec_kw={"height_ratios": [2, 1]}
    )
    ax1.plot([0, 1], [0, 1], "k--", label="Ideal")
    ax1.bar(bin_centers, bin_acc, width=1.0 / bins, edgecolor="black", alpha=0.7, label="Accuracy")
    ax1.plot(bin_centers, bin_conf, "o-", label="Confidence")
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.set_ylabel("Accuracy")
    ax1.set_title(f"Calibration (ECE={ece:.4f})")
    ax1.legend()

    ax2.hist(confidences, bins=bin_edges, edgecolor="black")
    ax2.set_xlim(0, 1)
    ax2.set_xlabel("Confidence")
    ax2.set_ylabel("Count")

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "calibration.png"), dpi=150)
    plt.close(fig)


def main():
    args = parse_args()

    # 设置设备
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = "cpu"

    print(f"Using device: {args.device}")

    # 创建数据集
    print(f"\nLoading {args.split} dataset...")
    transform = ClassificationTransform(
        image_size=(args.image_size, args.image_size),
        is_train=False,
    )
    dataset = TongueClassificationDataset(
        root=args.data_root,
        split=args.split,
        transform=transform,
        image_size=(args.image_size, args.image_size),
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    print(f"Dataset size: {len(dataset)}")
    print(f"Number of classes: {dataset.num_classes}")

    # 创建模型
    print(f"\nBuilding model: {args.model_type}")
    seg_weights_path = None
    if args.model_type in ["seg_attention", "dual_stream"]:
        seg_weights_path = str(PROJECT_ROOT / args.seg_weights)
        if not os.path.exists(seg_weights_path):
            seg_weights_path = None

    model = build_classifier(
        model_type=args.model_type,
        num_classes=args.num_classes,
        backbone=args.backbone,
        pretrained=False,
        seg_weights_path=seg_weights_path,
    )

    # 加载检查点
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(
        args.checkpoint,
        map_location=args.device,
        weights_only=args.weights_only,
    )
    if "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
    else:
        model.load_state_dict(checkpoint)

    # 评估
    evaluator = ClassificationEvaluator(
        model=model,
        dataloader=dataloader,
        device=args.device,
        class_names=dataset.get_class_names(),
    )

    print("\nEvaluating...")
    need_probs = args.plot_confusion or args.plot_calibration
    metrics = evaluator.evaluate(return_probs=need_probs)
    evaluator.print_report(metrics)

    # 保存预测结果
    if args.save_predictions:
        output_dir = args.output_dir or os.path.dirname(args.checkpoint)
        os.makedirs(output_dir, exist_ok=True)

        output_path = os.path.join(output_dir, f"predictions_{args.split}.npz")
        np.savez(
            output_path,
            confusion_matrix=np.array(metrics["confusion_matrix"]),
            accuracy=metrics["accuracy"],
        )
        print(f"\nPredictions saved to: {output_path}")

    # 可视化输出
    output_dir = args.output_dir or os.path.dirname(args.checkpoint)
    if (args.plot_confusion or args.plot_calibration) and output_dir:
        os.makedirs(output_dir, exist_ok=True)

    if args.plot_confusion:
        _plot_confusion_matrices(
            np.array(metrics["confusion_matrix"]),
            dataset.get_class_names(),
            output_dir,
        )

    if args.plot_calibration:
        if "probs" not in metrics or "labels" not in metrics:
            print("Warning: calibration plot requested but probabilities are unavailable.")
        else:
            _plot_calibration(
                metrics["probs"],
                metrics["labels"],
                output_dir,
                bins=args.calib_bins,
            )


if __name__ == "__main__":
    main()
