#!/usr/bin/env python
"""批量运行分类实验脚本"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from datetime import datetime


EXPERIMENTS = {
    "baseline": {
        "model_type": "baseline",
        "description": "ResNet50 直接分类（基线）",
    },
    "seg_attention": {
        "model_type": "seg_attention",
        "description": "分割注意力分类",
    },
    "seg_attention_cb": {
        "model_type": "seg_attention",
        "loss": "cb_focal",
        "weighted_sampler": True,
        "strong_aug": True,
        "description": "分割注意力 + 类别平衡",
    },
    "dual_stream": {
        "model_type": "dual_stream",
        "description": "双流融合分类",
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description="Run classification experiments")

    parser.add_argument("--experiments", type=str, nargs="+",
                        default=list(EXPERIMENTS.keys()),
                        help="Experiments to run")
    parser.add_argument("--data-root", type=str, default="datasets/shezhenv3-coco",
                        help="Dataset root directory")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--output-dir", type=str, default="runs/classification",
                        help="Output directory")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without running")
    parser.add_argument("--conda-env", type=str, default="cv",
                        help="Conda environment name")

    return parser.parse_args()


def build_command(exp_name: str, exp_config: dict, args) -> list:
    """构建训练命令"""
    cmd = [
        "python", "scripts/train_classifier.py",
        "--data-root", args.data_root,
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--lr", str(args.lr),
        "--device", args.device,
        "--output-dir", os.path.join(args.output_dir, exp_name),
        "--model-type", exp_config.get("model_type", "seg_attention"),
    ]

    if exp_config.get("loss"):
        cmd.extend(["--loss", exp_config["loss"]])

    if exp_config.get("weighted_sampler"):
        cmd.append("--weighted-sampler")

    if exp_config.get("strong_aug"):
        cmd.append("--strong-aug")

    if exp_config.get("backbone"):
        cmd.extend(["--backbone", exp_config["backbone"]])

    return cmd


def main():
    args = parse_args()

    print("=" * 60)
    print("舌象分类实验批量运行")
    print("=" * 60)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"实验列表: {args.experiments}")
    print(f"输出目录: {args.output_dir}")
    print()

    results = {}

    for exp_name in args.experiments:
        if exp_name not in EXPERIMENTS:
            print(f"Warning: Unknown experiment '{exp_name}', skipping")
            continue

        exp_config = EXPERIMENTS[exp_name]
        print("-" * 60)
        print(f"实验: {exp_name}")
        print(f"描述: {exp_config['description']}")
        print("-" * 60)

        cmd = build_command(exp_name, exp_config, args)

        if args.conda_env:
            cmd = ["conda", "run", "-n", args.conda_env] + cmd

        print(f"命令: {' '.join(cmd)}")

        if args.dry_run:
            print("(Dry run, skipping execution)")
            continue

        try:
            result = subprocess.run(
                cmd,
                cwd=str(Path(__file__).parent.parent),
                check=True,
            )
            results[exp_name] = "SUCCESS"
        except subprocess.CalledProcessError as e:
            print(f"Error running experiment {exp_name}: {e}")
            results[exp_name] = "FAILED"
        except KeyboardInterrupt:
            print("\nInterrupted by user")
            results[exp_name] = "INTERRUPTED"
            break

        print()

    # 打印总结
    print("=" * 60)
    print("实验总结")
    print("=" * 60)
    for exp_name, status in results.items():
        print(f"  {exp_name}: {status}")
    print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
