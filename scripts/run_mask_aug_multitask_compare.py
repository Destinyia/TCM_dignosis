#!/usr/bin/env python
"""Mask增强与多任务学习对比实验脚本"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

EXPERIMENTS = [
    {
        "name": "B0_baseline",
        "desc": "Baseline 无mask增强",
        "args": {
            "model-type": "baseline",
        },
    },
    {
        "name": "A1_mask_crop",
        "desc": "Mask裁剪增强（真实分割mask）",
        "args": {
            "model-type": "baseline",
            "mask-aug": True,
            "mask-aug-mode": "crop",
        },
    },
    {
        "name": "A2_mask_bg_blur",
        "desc": "Mask背景模糊（dilate=25）",
        "args": {
            "model-type": "baseline",
            "mask-aug": True,
            "mask-aug-mode": "background",
            "mask-aug-bg-mode": "blur",
            "mask-aug-dilate": 25,
        },
    },
    {
        "name": "M1_multitask_seg_v2",
        "desc": "分类+分割联合训练",
        "args": {
            "model-type": "seg_attention_v2",
            "seg-loss": "bce_dice",
            "seg-loss-weight": 0.2,
            "train-seg": True,
            "soft-floor": 0.1,
        },
    },
]

DEFAULT_ARGS = {
    "data-root": "datasets/shezhenv3-coco",
    "image-size": 640,
    "batch-size": 16,
    "backbone": "resnet50",
    "epochs": 50,
    "lr": 1e-4,
    "loss": "focal",
    "early-stop": 10,
    "amp": True,
}


def _latest_file(paths):
    if not paths:
        return None
    return max(paths, key=lambda p: p.stat().st_mtime)


def _load_metrics_history(history_path: Path) -> dict | None:
    if not history_path or not history_path.exists():
        return None

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

    if not records:
        return None

    def _get_metric(record, key):
        value = record.get(key)
        return value if isinstance(value, (int, float)) else 0.0

    best_record = max(records, key=lambda r: _get_metric(r, "val_macro_f1"))
    return {
        "best_macro_f1": _get_metric(best_record, "val_macro_f1"),
        "best_accuracy": _get_metric(best_record, "val_accuracy"),
        "best_epoch": best_record.get("epoch", 0),
        "final_epoch": records[-1].get("epoch", 0),
    }


def build_command(exp_config: dict, output_base: str) -> list:
    cmd = ["python", "scripts/train_classifier.py"]

    for key, value in DEFAULT_ARGS.items():
        if isinstance(value, bool):
            if value:
                cmd.append(f"--{key}")
        else:
            cmd.extend([f"--{key}", str(value)])

    for key, value in exp_config["args"].items():
        if isinstance(value, bool):
            if value:
                cmd.append(f"--{key}")
        else:
            cmd.extend([f"--{key}", str(value)])

    output_dir = f"{output_base}/{exp_config['name']}"
    cmd.extend(["--output-dir", output_dir])

    return cmd


def run_experiment(exp_config: dict, output_base: str, dry_run: bool = False) -> bool:
    cmd = build_command(exp_config, output_base)

    print(f"\n{'='*60}")
    print(f"实验: {exp_config['name']}")
    print(f"描述: {exp_config['desc']}")
    print(f"命令: {' '.join(cmd)}")
    print("="*60)

    if dry_run:
        print("[DRY RUN] 跳过执行")
        return True

    try:
        subprocess.run(cmd, check=True)
        print(f"\n[SUCCESS] {exp_config['name']} 完成")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n[FAILED] {exp_config['name']} 失败: {e}")
        return False
    except KeyboardInterrupt:
        print(f"\n[INTERRUPTED] {exp_config['name']} 被中断")
        return False


def analyze_results(output_base: str):
    print("\n" + "="*80)
    print("实验结果分析")
    print("="*80)

    results = []
    output_path = Path(output_base)

    for exp in EXPERIMENTS:
        model_type = exp["args"].get("model-type", "baseline")
        exp_dir = output_path / exp["name"] / model_type

        metrics_file = exp_dir / "metrics.json"
        if not metrics_file.exists():
            metrics_file = _latest_file(list(exp_dir.glob("**/metrics.json")))

        if metrics_file and metrics_file.exists():
            with open(metrics_file) as f:
                metrics = json.load(f)

            result = {
                "name": exp["name"],
                "desc": exp["desc"],
                "best_macro_f1": metrics.get("best_macro_f1", 0),
                "best_accuracy": metrics.get("best_accuracy", 0),
                "best_epoch": metrics.get("best_epoch", 0),
                "final_epoch": metrics.get("final_epoch", 0),
            }
        else:
            history_file = _latest_file(list(exp_dir.glob("**/metrics_history.jsonl")))
            history_metrics = _load_metrics_history(history_file) if history_file else None
            if history_metrics:
                result = {
                    "name": exp["name"],
                    "desc": exp["desc"],
                    **history_metrics,
                }
            else:
                result = {"name": exp["name"], "desc": exp["desc"], "status": "未运行"}

        results.append(result)

    print(f"\n{'实验名称':<25} {'描述':<35} {'Best F1':>10} {'Accuracy':>10} {'Epoch':>8}")
    print("-" * 95)

    for r in results:
        if "status" in r:
            print(f"{r['name']:<25} {r['desc']:<35} {r['status']:>10}")
        else:
            f1 = r["best_macro_f1"]
            acc = r["best_accuracy"]
            epoch = r["best_epoch"]
            f1_str = f"{f1:.4f}" if f1 > 0 else "-"
            acc_str = f"{acc:.4f}" if acc > 0 else "-"
            print(f"{r['name']:<25} {r['desc']:<35} {f1_str:>10} {acc_str:>10} {epoch:>8}")

    valid_results = [r for r in results if "best_macro_f1" in r and r["best_macro_f1"] > 0]
    if valid_results:
        best = max(valid_results, key=lambda x: x["best_macro_f1"])
        print("\n" + "-" * 95)
        print(f"最佳实验: {best['name']} (F1={best['best_macro_f1']:.4f})")

    _write_compare_report(results)
    _plot_compare_results(results)


def _write_compare_report(results: list):
    report_path = Path("docs/ablation_mask_multitask_compare.md")
    report_path.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append(f"# Mask 增强与多任务对比实验报告（{datetime.now().strftime('%Y-%m-%d')}）")
    lines.append("")
    lines.append("## 实验说明")
    lines.append("- 数据来源：`runs/ablation_mask_multitask/*/*/metrics_history.jsonl`")
    lines.append("- 指标口径：按 `val_macro_f1` 选最佳 epoch，并记录对应的 `val_accuracy`")
    lines.append("")
    lines.append("## 对比结果")
    lines.append("| 实验名称 | 描述 | Best Macro F1 | Accuracy | Best Epoch |")
    lines.append("| --- | --- | --- | --- | --- |")
    for r in results:
        if "status" in r:
            lines.append(f"| {r['name']} | {r['desc']} | 未运行 | 未运行 | - |")
        else:
            lines.append(
                f"| {r['name']} | {r['desc']} | "
                f"{r['best_macro_f1']:.4f} | {r['best_accuracy']:.4f} | {r['best_epoch']} |"
            )

    valid_results = [r for r in results if "best_macro_f1" in r and r["best_macro_f1"] > 0]
    lines.append("")
    lines.append("## 结论摘要")
    if valid_results:
        best = max(valid_results, key=lambda x: x["best_macro_f1"])
        lines.append(f"- 最佳实验：`{best['name']}`（Best Macro F1={best['best_macro_f1']:.4f}）。")
    else:
        lines.append("- 当前没有可用实验结果。")

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"对比文档已写入: {report_path}")


def _plot_compare_results(results: list):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("Warning: matplotlib not available, skipping plot.")
        return

    names = []
    values = []
    for r in results:
        names.append(r["name"])
        values.append(r.get("best_macro_f1", 0.0))

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(range(len(values)), values, color="#54A24B")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Best Macro F1")
    ax.set_title("Mask Augmentation & Multitask - Best Macro F1")
    fig.tight_layout()

    out_path = Path("docs/ablation_mask_multitask_compare.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"可视化已写入: {out_path}")

def main():
    parser = argparse.ArgumentParser(description="运行Mask增强与多任务对比实验")
    parser.add_argument("--output-base", type=str,
                        default="runs/ablation_mask_multitask",
                        help="输出目录基础路径")
    parser.add_argument("--experiments", type=str, nargs="+",
                        help="指定要运行的实验名称 (默认全部)")
    parser.add_argument("--dry-run", action="store_true",
                        help="只打印命令，不执行")
    parser.add_argument("--list", action="store_true",
                        help="列出所有可用实验")
    parser.add_argument("--analyze", action="store_true",
                        help="只分析已有结果，不运行实验")
    args = parser.parse_args()

    if args.list:
        print("可用实验:")
        for exp in EXPERIMENTS:
            print(f"  - {exp['name']}: {exp['desc']}")
        return

    if args.analyze:
        analyze_results(args.output_base)
        return

    if args.experiments:
        exp_names = set(args.experiments)
        experiments = [e for e in EXPERIMENTS if e["name"] in exp_names]
        if not experiments:
            print(f"错误: 未找到指定的实验 {args.experiments}")
            sys.exit(1)
    else:
        experiments = EXPERIMENTS

    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"共 {len(experiments)} 个实验")

    run_results = {}
    for i, exp in enumerate(experiments, 1):
        print(f"\n[{i}/{len(experiments)}] 运行实验 {exp['name']}...")
        success = run_experiment(exp, args.output_base, args.dry_run)
        run_results[exp["name"]] = "成功" if success else "失败"

    print("\n" + "="*60)
    print("运行结果汇总:")
    print("="*60)
    for name, status in run_results.items():
        print(f"  {name}: {status}")
    print(f"\n结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    analyze_results(args.output_base)

if __name__ == "__main__":
     main()
