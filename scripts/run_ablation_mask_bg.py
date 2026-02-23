#!/usr/bin/env python
"""Mask背景增强消融实验 - 使用CE损失作为baseline

使用方法:
    conda activate cv
    python scripts/run_ablation_mask_bg.py

可选参数:
    --skip-completed  跳过已完成的实验
    --only A1 A2      只运行指定实验
    --list            列出所有实验及状态
    --analyze         只分析结果
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

OUTPUT_BASE = "runs/ablation_mask_bg"
DATA_ROOT = "datasets/shezhenv3-coco"

COMMON_ARGS = [
    "--data-root", DATA_ROOT,
    "--image-size", "640",
    "--batch-size", "16",
    "--backbone", "resnet50",
    "--epochs", "50",
    "--lr", "0.0001",
    "--loss", "ce",  # 使用普通交叉熵损失
    "--early-stop", "10",
    "--amp",
]

EXPERIMENTS = {
    "B0": {
        "name": "B0_baseline_ce",
        "desc": "Baseline CE损失，无mask增强",
        "args": ["--model-type", "baseline"],
    },
    "A1": {
        "name": "A1_bg_blur",
        "desc": "背景高斯模糊",
        "args": [
            "--model-type", "baseline",
            "--mask-aug",
            "--mask-aug-mode", "background",
            "--mask-aug-bg-mode", "blur",
            "--mask-aug-dilate", "25",
        ],
    },
    "A2": {
        "name": "A2_bg_solid_gray",
        "desc": "固定灰色背景 (114,114,114)",
        "args": [
            "--model-type", "baseline",
            "--mask-aug",
            "--mask-aug-mode", "background",
            "--mask-aug-bg-mode", "solid",
            "--mask-aug-bg-color", "gray",
            "--mask-aug-dilate", "25",
        ],
    },
    "A3": {
        "name": "A3_bg_solid_random",
        "desc": "随机纯色背景",
        "args": [
            "--model-type", "baseline",
            "--mask-aug",
            "--mask-aug-mode", "background",
            "--mask-aug-bg-mode", "solid",
            "--mask-aug-bg-color", "random",
            "--mask-aug-dilate", "25",
        ],
    },
    "A4": {
        "name": "A4_bg_noise",
        "desc": "随机噪声背景",
        "args": [
            "--model-type", "baseline",
            "--mask-aug",
            "--mask-aug-mode", "background",
            "--mask-aug-bg-mode", "noise",
            "--mask-aug-dilate", "25",
        ],
    },
    "A5": {
        "name": "A5_bg_random_mode",
        "desc": "随机背景模式 (blur/solid/noise随机)",
        "args": [
            "--model-type", "baseline",
            "--mask-aug",
            "--mask-aug-mode", "background",
            "--mask-aug-bg-mode", "random",
            "--mask-aug-bg-color", "random",
            "--mask-aug-dilate", "25",
        ],
    },
}


def is_completed(exp_key: str) -> bool:
    """检查实验是否已完成"""
    exp = EXPERIMENTS[exp_key]
    metrics_path = Path(OUTPUT_BASE) / exp["name"] / "baseline" / "metrics.json"
    return metrics_path.exists()


def run_experiment(exp_key: str) -> bool:
    """运行单个实验"""
    exp = EXPERIMENTS[exp_key]
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
    """分析实验结果并生成报告"""
    print("\n" + "=" * 60)
    print("分析实验结果...")
    print("=" * 60)

    results = []
    for key, exp in EXPERIMENTS.items():
        metrics_path = Path(OUTPUT_BASE) / exp["name"] / "baseline" / "metrics.json"
        history_path = Path(OUTPUT_BASE) / exp["name"] / "baseline" / "metrics_history.jsonl"

        if not metrics_path.exists():
            results.append({
                "key": key,
                "name": exp["name"],
                "desc": exp["desc"],
                "status": "未运行",
                "best_macro_f1": None,
                "accuracy": None,
                "best_epoch": None,
            })
            continue

        # 从 history 中找最佳 epoch
        best_f1 = 0.0
        best_acc = 0.0
        best_epoch = 0

        if history_path.exists():
            with open(history_path) as f:
                for line in f:
                    record = json.loads(line)
                    if record.get("val_macro_f1", 0) > best_f1:
                        best_f1 = record["val_macro_f1"]
                        best_acc = record.get("val_accuracy", 0)
                        best_epoch = record.get("epoch", 0)

        results.append({
            "key": key,
            "name": exp["name"],
            "desc": exp["desc"],
            "status": "已完成",
            "best_macro_f1": best_f1,
            "accuracy": best_acc,
            "best_epoch": best_epoch,
        })

    # 打印结果表格
    print("\n实验结果对比:")
    print("-" * 80)
    print(f"{'实验':<25} {'描述':<25} {'Best F1':>10} {'Acc':>10} {'Epoch':>6}")
    print("-" * 80)

    for r in results:
        if r["status"] == "未运行":
            print(f"{r['name']:<25} {r['desc']:<25} {'未运行':>10} {'-':>10} {'-':>6}")
        else:
            print(f"{r['name']:<25} {r['desc']:<25} {r['best_macro_f1']:>10.4f} {r['accuracy']:>10.4f} {r['best_epoch']:>6}")

    print("-" * 80)

    # 生成 Markdown 报告
    report_path = Path("docs/ablation_mask_bg_compare.md")
    report_path.parent.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d")
    report = f"""# Mask 背景增强消融实验报告（{timestamp}）

## 实验说明
- 损失函数：普通交叉熵 (CE)
- 数据来源：`{OUTPUT_BASE}/*/*/metrics_history.jsonl`
- 指标口径：按 `val_macro_f1` 选最佳 epoch

## 对比结果
| 实验名称 | 描述 | Best Macro F1 | Accuracy | Best Epoch |
| --- | --- | --- | --- | --- |
"""
    for r in results:
        if r["status"] == "未运行":
            report += f"| {r['name']} | {r['desc']} | 未运行 | - | - |\n"
        else:
            report += f"| {r['name']} | {r['desc']} | {r['best_macro_f1']:.4f} | {r['accuracy']:.4f} | {r['best_epoch']} |\n"

    # 找出最佳实验
    completed = [r for r in results if r["status"] == "已完成"]
    if completed:
        best = max(completed, key=lambda x: x["best_macro_f1"] or 0)
        report += f"""
## 结论摘要
- 最佳实验：`{best['name']}`（Best Macro F1={best['best_macro_f1']:.4f}）
"""

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"\n报告已保存至: {report_path}")


def main():
    parser = argparse.ArgumentParser(description="运行Mask背景增强消融实验")
    parser.add_argument("--skip-completed", action="store_true",
                        help="跳过已完成的实验")
    parser.add_argument("--only", nargs="+", choices=list(EXPERIMENTS.keys()),
                        help="只运行指定的实验 (B0, A1, A2, A3, A4, A5)")
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
    print("Mask背景增强消融实验 (CE损失)")
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
