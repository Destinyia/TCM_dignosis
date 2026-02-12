#!/usr/bin/env python
"""采样策略和损失函数对比实验

基于 aug_all 配置，对比不同采样策略和损失函数的效果。
"""
from __future__ import annotations

import argparse
import datetime
import json
import subprocess
from pathlib import Path

# 实验分组
SAMPLER_EXPERIMENTS = [
    "baseline_aug",      # baseline: default sampler + ce
    "samp_oversample",   # oversample sampler
    "samp_stratified",   # stratified sampler
    "samp_class_aware",  # class-aware sampler
]

LOSS_EXPERIMENTS = [
    "baseline_aug",      # baseline: default sampler + ce
    "loss_focal",        # focal loss
    "loss_weighted_ce",  # weighted ce
    "loss_cb_focal",     # class-balanced focal
    "loss_seesaw",       # seesaw loss
]

ALL_EXPERIMENTS = [
    "baseline_aug",
    "samp_oversample",
    "samp_stratified",
    "samp_class_aware",
    "loss_focal",
    "loss_weighted_ce",
    "loss_cb_focal",
    "loss_seesaw",
    "best_combined",
]

CONFIG_DIR = "configs/experiments/sampler_loss_compare"


def run_experiment(name: str, output_base: str, extra_args: list[str]) -> Path:
    """运行单个实验，返回输出目录"""
    config_path = f"{CONFIG_DIR}/{name}.yaml"
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(output_base) / f"{name}_{timestamp}"

    cmd = [
        "python", "scripts/train.py",
        "--config", config_path,
        "--output-dir", str(output_dir),
    ] + extra_args

    print(f"\n{'='*60}")
    print(f"Running: {name}")
    print(f"Config: {config_path}")
    print(f"Output: {output_dir}")
    print(f"{'='*60}\n")

    subprocess.run(cmd, check=False)
    return output_dir


def load_metrics(output_dir: Path) -> dict | None:
    """加载实验指标"""
    metrics_file = output_dir / "metrics.json"
    if not metrics_file.exists():
        return None
    with open(metrics_file, "r", encoding="utf-8") as f:
        return json.load(f)


def generate_report(results: dict[str, dict], output_path: Path) -> None:
    """生成对比报告"""
    lines = [
        "# 采样策略与损失函数对比实验报告",
        "",
        f"生成时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## 实验配置",
        "",
        "基于 aug_all 数据增强配置（水平翻转 + 颜色增强 + 噪声 + 仿射变换）",
        "",
        "### 采样策略实验",
        "",
        "| 实验 | 采样策略 | 损失函数 |",
        "|------|---------|---------|",
        "| baseline_aug | default | ce |",
        "| samp_oversample | oversample | ce |",
        "| samp_stratified | stratified | ce |",
        "| samp_class_aware | class_aware | ce |",
        "",
        "### 损失函数实验",
        "",
        "| 实验 | 采样策略 | 损失函数 |",
        "|------|---------|---------|",
        "| baseline_aug | default | ce |",
        "| loss_focal | default | focal |",
        "| loss_weighted_ce | default | weighted_ce |",
        "| loss_cb_focal | default | class_balanced_focal |",
        "| loss_seesaw | default | seesaw |",
        "",
    ]

    # 采样策略结果
    lines.extend([
        "## 采样策略对比结果",
        "",
        "| 实验 | mAP | mAP@50 | mAP@75 | 训练时间 |",
        "|------|-----|--------|--------|---------|",
    ])

    for name in SAMPLER_EXPERIMENTS:
        if name not in results or results[name] is None:
            lines.append(f"| {name} | - | - | - | - |")
            continue
        m = results[name]
        mAP = m.get("mAP", 0) * 100
        mAP50 = m.get("mAP_50", 0) * 100
        mAP75 = m.get("mAP_75", 0) * 100
        duration = m.get("train_duration_min", "-")
        if isinstance(duration, (int, float)):
            duration = f"{duration:.1f}min"
        lines.append(f"| {name} | {mAP:.2f} | {mAP50:.2f} | {mAP75:.2f} | {duration} |")

    # 损失函数结果
    lines.extend([
        "",
        "## 损失函数对比结果",
        "",
        "| 实验 | mAP | mAP@50 | mAP@75 | 训练时间 |",
        "|------|-----|--------|--------|---------|",
    ])

    for name in LOSS_EXPERIMENTS:
        if name not in results or results[name] is None:
            lines.append(f"| {name} | - | - | - | - |")
            continue
        m = results[name]
        mAP = m.get("mAP", 0) * 100
        mAP50 = m.get("mAP_50", 0) * 100
        mAP75 = m.get("mAP_75", 0) * 100
        duration = m.get("train_duration_min", "-")
        if isinstance(duration, (int, float)):
            duration = f"{duration:.1f}min"
        lines.append(f"| {name} | {mAP:.2f} | {mAP50:.2f} | {mAP75:.2f} | {duration} |")

    # best_combined 结果
    if "best_combined" in results and results["best_combined"] is not None:
        m = results["best_combined"]
        mAP = m.get("mAP", 0) * 100
        mAP50 = m.get("mAP_50", 0) * 100
        mAP75 = m.get("mAP_75", 0) * 100
        duration = m.get("train_duration_min", "-")
        if isinstance(duration, (int, float)):
            duration = f"{duration:.1f}min"
        lines.extend([
            "",
            "## 最佳组合结果",
            "",
            "| 实验 | mAP | mAP@50 | mAP@75 | 训练时间 |",
            "|------|-----|--------|--------|---------|",
            f"| best_combined | {mAP:.2f} | {mAP50:.2f} | {mAP75:.2f} | {duration} |",
        ])

    # 每类AP对比
    lines.extend([
        "",
        "## 每类 AP@50 对比",
        "",
    ])

    # 收集所有类别
    all_classes = set()
    for m in results.values():
        if m and "per_class_AP" in m:
            all_classes.update(m["per_class_AP"].keys())
    all_classes = sorted(all_classes)

    if all_classes:
        exp_names = [n for n in ALL_EXPERIMENTS if n in results and results[n] is not None]
        header = "| 类别 | " + " | ".join(exp_names) + " |"
        sep = "|------|" + "|".join(["------"] * len(exp_names)) + "|"
        lines.extend([header, sep])

        for cls in all_classes:
            row = [cls]
            for name in exp_names:
                m = results.get(name)
                if m and "per_class_AP" in m:
                    ap = m["per_class_AP"].get(cls, 0) * 100
                    row.append(f"{ap:.1f}")
                else:
                    row.append("-")
            lines.append("| " + " | ".join(row) + " |")

    # 写入文件
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n报告已保存: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="采样策略和损失函数对比实验")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--train-size", type=int, default=4000)
    parser.add_argument("--val-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-base", default="runs/sampler_loss_compare")
    parser.add_argument("--only", help="只运行指定实验，逗号分隔")
    parser.add_argument("--skip-combined", action="store_true",
                        help="跳过 best_combined 实验")
    args = parser.parse_args()

    # 构建额外参数
    extra_args = [
        "--epochs", str(args.epochs),
        "--device", args.device,
        "--batch-size", str(args.batch_size),
        "--num-workers", str(args.num_workers),
        "--train-size", str(args.train_size),
        "--val-size", str(args.val_size),
        "--seed", str(args.seed),
    ]

    # 选择要运行的实验
    if args.only:
        selected = [x.strip() for x in args.only.split(",")]
    else:
        selected = ALL_EXPERIMENTS.copy()
        if args.skip_combined:
            selected.remove("best_combined")

    # 创建输出目录
    output_base = Path(args.output_base)
    output_base.mkdir(parents=True, exist_ok=True)

    # 运行实验并收集结果
    output_dirs = {}
    for name in selected:
        if name not in ALL_EXPERIMENTS:
            print(f"警告: 未知实验 {name}，跳过")
            continue
        output_dirs[name] = run_experiment(name, str(output_base), extra_args)

    # 加载所有结果
    results = {}
    for name, out_dir in output_dirs.items():
        results[name] = load_metrics(out_dir)

    # 生成报告
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = output_base / f"report_{timestamp}.md"
    generate_report(results, report_path)

    print("\n" + "=" * 60)
    print("所有实验完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
