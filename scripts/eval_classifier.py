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

    return parser.parse_args()


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
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
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
    metrics = evaluator.evaluate()
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


if __name__ == "__main__":
    main()
