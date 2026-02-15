#!/usr/bin/env python
"""Mask增强与多任务学习对比实验 - 手动执行脚本

使用方法:
    conda activate cv
    python scripts/run_ablation_mask_v2.py

可选参数:
    --skip-completed  跳过已完成的实验
    --only A1 M1      只运行指定实验
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

OUTPUT_BASE = "runs/ablation_mask_v2"
DATA_ROOT = "datasets/shezhenv3-coco"

COMMON_ARGS = [
    "--data-root", DATA_ROOT,
    "--image-size", "640",
    "--batch-size", "16",
    "--backbone", "resnet50",
    "--epochs", "50",
    "--lr", "0.0001",
    "--loss", "focal",
    "--early-stop", "10",
    "--amp",
]

EXPERIMENTS = {
    "B0": {
        "name": "B0_baseline",
        "desc": "Baseline 无mask增强",
        "args": ["--model-type", "baseline"],
    },
    "A1": {
        "name": "A1_mask_crop",
        "desc": "Mask裁剪增强",
        "args": [
            "--model-type", "baseline",
            "--mask-aug",
            "--mask-aug-mode", "crop",
        ],
    },
    "A2": {
        "name": "A2_mask_bg_blur",
        "desc": "Mask背景模糊 (dilate=25)",
        "args": [
            "--model-type", "baseline",
            "--mask-aug",
            "--mask-aug-mode", "background",
            "--mask-aug-bg-mode", "blur",
            "--mask-aug-dilate", "25",
        ],
    },
    "M1": {
        "name": "M1_multitask_seg_v2",
        "desc": "分类+分割联合训练",
        "args": [
            "--model-type", "seg_attention_v2",
            "--seg-loss", "bce_dice",
            "--seg-loss-weight", "0.2",
            "--train-seg",
            "--soft-floor", "0.1",
        ],
    },
}


def is_completed(exp_key: str) -> bool:
    """检查实验是否已完成"""
    exp = EXPERIMENTS[exp_key]
    model_type = "baseline"
    for i, arg in enumerate(exp["args"]):
        if arg == "--model-type" and i + 1 < len(exp["args"]):
            model_type = exp["args"][i + 1]
            break

    metrics_path = Path(OUTPUT_BASE) / exp["name"] / model_type / "metrics.json"
    return metrics_path.exists()


def run_experiment(exp_key: str) -> bool:
    """运行单个实验"""
    exp = EXPERIMENTS[exp_key]

    # 确定 model_type 用于输出目录
    model_type = "baseline"
    for i, arg in enumerate(exp["args"]):
        if arg == "--model-type" and i + 1 < len(exp["args"]):
            model_type = exp["args"][i + 1]
            break

    output_dir = f"{OUTPUT_BASE}/{exp['name']}"

    cmd = [
        "python", "scripts/train_classifier.py",
        *COMMON_ARGS,
        *exp["args"],
        "--output-dir", output_dir,
    ]

    print(f"\n{'='*60}")
    print(f"实验: {exp['name']}")
    print(f"描述: {exp['desc']}")
    print(f"命令: {' '.join(cmd)}")
    print("=" * 60)

    try:
        subprocess.run(cmd, check=True)
        print(f"\n[SUCCESS] {exp['name']} 完成")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n[FAILED] {exp['name']} 失败: {e}")
        return False
    except KeyboardInterrupt:
        print(f"\n[INTERRUPTED] {exp['name']} 被中断")
        return False


def analyze_results():
    """分析实验结果"""
    print("\n" + "=" * 60)
    print("分析实验结果...")
    print("=" * 60)

    try:
        subprocess.run([
            "python", "scripts/run_mask_aug_multitask_compare.py",
            "--analyze",
            "--output-base", OUTPUT_BASE,
        ], check=True)
    except subprocess.CalledProcessError:
        print("分析脚本执行失败")


def main():
    parser = argparse.ArgumentParser(description="运行Mask增强与多任务对比实验")
    parser.add_argument("--skip-completed", action="store_true",
                        help="跳过已完成的实验")
    parser.add_argument("--only", nargs="+", choices=list(EXPERIMENTS.keys()),
                        help="只运行指定的实验 (B0, A1, A2, M1)")
    parser.add_argument("--list", action="store_true",
                        help="列出所有实验及状态")
    parser.add_argument("--analyze", action="store_true",
                        help="只分析结果，不运行实验")
    args = parser.parse_args()

    if args.list:
        print("实验列表:")
        for key, exp in EXPERIMENTS.items():
            status = "已完成" if is_completed(key) else "未完成"
            print(f"  [{key}] {exp['name']}: {exp['desc']} ({status})")
        return

    if args.analyze:
        analyze_results()
        return

    # 确定要运行的实验
    exp_keys = args.only if args.only else list(EXPERIMENTS.keys())

    print("=" * 60)
    print("Mask增强与多任务学习对比实验")
    print("=" * 60)

    results = {}
    for i, key in enumerate(exp_keys, 1):
        exp = EXPERIMENTS[key]

        if args.skip_completed and is_completed(key):
            print(f"\n[{i}/{len(exp_keys)}] {exp['name']} - 已完成，跳过")
            results[key] = "跳过"
            continue

        print(f"\n[{i}/{len(exp_keys)}] 运行 {exp['name']}...")
        success = run_experiment(key)
        results[key] = "成功" if success else "失败"

        if not success:
            print(f"\n实验 {key} 失败，是否继续? (y/n)")
            try:
                if input().strip().lower() != 'y':
                    break
            except EOFError:
                break

    # 打印结果汇总
    print("\n" + "=" * 60)
    print("运行结果汇总:")
    print("=" * 60)
    for key, status in results.items():
        print(f"  {EXPERIMENTS[key]['name']}: {status}")

    # 分析结果
    analyze_results()

    print("\nDone!")


if __name__ == "__main__":
    main()
