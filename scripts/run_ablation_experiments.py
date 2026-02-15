#!/usr/bin/env python
"""Mask Refiner 有效性对比实验 + 注意力机制对比

运行方式: conda run -n cv python scripts/run_ablation_experiments.py
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


# 实验配置
EXPERIMENTS = {
    "A": {
        "name": "基线 - 无注意力机制 (纯ResNet50)",
        "args": [
            "--model-type", "baseline",
            "--output-dir", "runs/ablation/exp_A_no_attention",
        ],
    },
    "B": {
        "name": "seg_attention - 无bbox监督",
        "args": [
            "--model-type", "seg_attention",
            "--output-dir", "runs/ablation/exp_B_seg_attention_baseline",
        ],
    },
    "C": {
        "name": "seg_attention + bbox监督 (无refiner)",
        "args": [
            "--model-type", "seg_attention",
            "--use-bbox-loss", "--bbox-loss-weight", "0.05",
            "--visualize-samples",
            "--output-dir", "runs/ablation/exp_C_bbox_only",
        ],
    },
    "D": {
        "name": "seg_attention + bbox监督 + mask_refiner",
        "args": [
            "--model-type", "seg_attention",
            "--use-bbox-loss", "--bbox-loss-weight", "0.05",
            "--use-mask-refiner",
            "--visualize-samples",
            "--output-dir", "runs/ablation/exp_D_with_refiner",
        ],
    },
}

# 公共参数
COMMON_ARGS = [
    "--epochs", "30",
    "--early-stop", "10",
    "--weighted-sampler",
    "--strong-aug",
]


def run_experiment(exp_id: str, extra_args: list = None) -> int:
    """运行单个实验"""
    if exp_id not in EXPERIMENTS:
        print(f"未知实验ID: {exp_id}")
        return 1

    exp = EXPERIMENTS[exp_id]
    print("=" * 50)
    print(f"实验 {exp_id}: {exp['name']}")
    print("=" * 50)

    script_path = Path(__file__).parent / "train_classifier.py"
    cmd = [sys.executable, str(script_path)]
    cmd.extend(exp["args"])
    cmd.extend(COMMON_ARGS)

    if extra_args:
        cmd.extend(extra_args)

    print(f"命令: {' '.join(cmd)}\n")

    result = subprocess.run(cmd)
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description="运行消融实验")
    parser.add_argument(
        "--exp", type=str, default="all",
        help="实验ID (A/B/C/D) 或 'all' 运行全部"
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="覆盖默认epoch数"
    )
    parser.add_argument(
        "--early-stop", type=int, default=None,
        help="覆盖默认早停patience"
    )
    args = parser.parse_args()

    # 构建额外参数
    extra_args = []
    if args.epochs:
        extra_args.extend(["--epochs", str(args.epochs)])
    if args.early_stop:
        extra_args.extend(["--early-stop", str(args.early_stop)])

    # 运行实验
    if args.exp.lower() == "all":
        exp_ids = list(EXPERIMENTS.keys())
    else:
        exp_ids = [e.strip().upper() for e in args.exp.split(",")]

    failed = []
    for exp_id in exp_ids:
        ret = run_experiment(exp_id, extra_args)
        if ret != 0:
            failed.append(exp_id)
        print()

    # 总结
    print("=" * 50)
    print("实验完成！")
    print(f"成功: {len(exp_ids) - len(failed)}/{len(exp_ids)}")
    if failed:
        print(f"失败: {', '.join(failed)}")
    print("结果保存在 runs/ablation/ 目录")
    print("=" * 50)

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
