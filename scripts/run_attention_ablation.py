#!/usr/bin/env python
"""注意力机制改进消融实验脚本"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# 实验配置
EXPERIMENTS = [
    {
        "name": "E1a_soft_0.3",
        "desc": "更高的 soft_floor (保留更多背景)",
        "args": {
            "model-type": "seg_attention_v2",
            "soft-floor": 0.3,
            "init-alpha": 0.0,
        },
    },
    {
        "name": "E1b_alpha_neg1",
        "desc": "负 init_alpha (初始更依赖注意力)",
        "args": {
            "model-type": "seg_attention_v2",
            "soft-floor": 0.1,
            "init-alpha": -1.0,
        },
    },
    {
        "name": "E1c_no_channel",
        "desc": "无通道注意力 (简化模型)",
        "args": {
            "model-type": "seg_attention_v2",
            "soft-floor": 0.2,
            "no-channel-attention": True,
        },
    },
    {
        "name": "E1d_gate_mode",
        "desc": "gate 模式 (加性增强)",
        "args": {
            "model-type": "seg_attention_v2",
            "soft-floor": 0.1,
            "attention-mode": "gate",
            "init-alpha": -2.0,
        },
    },
    {
        "name": "E1e_weighted",
        "desc": "soft attention + weighted sampler",
        "args": {
            "model-type": "seg_attention_v2",
            "soft-floor": 0.2,
            "weighted-sampler": True,
            "sampler-strategy": "sqrt",
        },
    },
    {
        "name": "E2_multiscale",
        "desc": "多尺度注意力",
        "args": {
            "model-type": "seg_attention_multiscale",
            "soft-floor": 0.1,
        },
    },
    {
        "name": "E2a_multiscale_weighted",
        "desc": "多尺度注意力 + weighted sampler",
        "args": {
            "model-type": "seg_attention_multiscale",
            "soft-floor": 0.2,
            "weighted-sampler": True,
            "sampler-strategy": "sqrt",
        },
    },
]

# 默认参数
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


def build_command(exp_config: dict, output_base: str) -> list:
    """构建训练命令"""
    cmd = ["python", "scripts/train_classifier.py"]

    # 添加默认参数
    for key, value in DEFAULT_ARGS.items():
        if isinstance(value, bool):
            if value:
                cmd.append(f"--{key}")
        else:
            cmd.extend([f"--{key}", str(value)])

    # 添加实验特定参数
    for key, value in exp_config["args"].items():
        if isinstance(value, bool):
            if value:
                cmd.append(f"--{key}")
        else:
            cmd.extend([f"--{key}", str(value)])

    # 输出目录：让 train_classifier.py 负责追加 model_type，避免重复层级
    output_dir = f"{output_base}/{exp_config['name']}"
    cmd.extend(["--output-dir", output_dir])

    return cmd


def run_experiment(exp_config: dict, output_base: str, dry_run: bool = False) -> bool:
    """运行单个实验"""
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
        result = subprocess.run(cmd, check=True)
        print(f"\n[SUCCESS] {exp_config['name']} 完成")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n[FAILED] {exp_config['name']} 失败: {e}")
        return False
    except KeyboardInterrupt:
        print(f"\n[INTERRUPTED] {exp_config['name']} 被中断")
        return False


def _latest_file(paths):
    if not paths:
        return None
    return max(paths, key=lambda p: p.stat().st_mtime)


def _load_metrics_history(history_path: Path) -> dict | None:
    if not history_path.exists():
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


def analyze_results(output_base: str):
    """分析所有实验结果"""
    print("\n" + "="*80)
    print("实验结果分析")
    print("="*80)

    results = []
    output_path = Path(output_base)

    for exp in EXPERIMENTS:
        model_type = exp["args"].get("model-type", "seg_attention_v2")
        exp_dir = output_path / exp["name"] / model_type

        # 查找 metrics.json
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
            # 尝试从 training_log.json 读取
            log_file = exp_dir / "training_log.json"
            if log_file.exists():
                with open(log_file) as f:
                    logs = json.load(f)
                if logs:
                    best_log = max(logs, key=lambda x: x.get("val_macro_f1", 0))
                    result = {
                        "name": exp["name"],
                        "desc": exp["desc"],
                        "best_macro_f1": best_log.get("val_macro_f1", 0),
                        "best_accuracy": best_log.get("val_accuracy", 0),
                        "best_epoch": best_log.get("epoch", 0),
                        "final_epoch": logs[-1].get("epoch", 0),
                    }
                else:
                    result = {"name": exp["name"], "desc": exp["desc"], "status": "无数据"}
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

    # 打印结果表格
    print(f"\n{'实验名称':<25} {'描述':<30} {'Best F1':>10} {'Accuracy':>10} {'Epoch':>8}")
    print("-" * 90)

    # Baseline 参考
    print(f"{'[Baseline]':<25} {'无注意力':<30} {'0.3402':>10} {'0.6391':>10} {'-':>8}")
    print("-" * 90)

    for r in results:
        if "status" in r:
            print(f"{r['name']:<25} {r['desc']:<30} {r['status']:>10}")
        else:
            f1 = r["best_macro_f1"]
            acc = r["best_accuracy"]
            epoch = r["best_epoch"]
            # 标记是否超过 baseline
            f1_str = f"{f1:.4f}" if f1 > 0 else "-"
            if f1 > 0.3402:
                f1_str = f"{f1:.4f}*"
            acc_str = f"{acc:.4f}" if acc > 0 else "-"
            print(f"{r['name']:<25} {r['desc']:<30} {f1_str:>10} {acc_str:>10} {epoch:>8}")

    # 找出最佳实验
    valid_results = [r for r in results if "best_macro_f1" in r and r["best_macro_f1"] > 0]
    if valid_results:
        best = max(valid_results, key=lambda x: x["best_macro_f1"])
        print("\n" + "-" * 90)
        print(f"最佳实验: {best['name']} (F1={best['best_macro_f1']:.4f})")

        if best["best_macro_f1"] > 0.3402:
            improvement = (best["best_macro_f1"] - 0.3402) / 0.3402 * 100
            print(f"相比 Baseline 提升: +{improvement:.2f}%")
        else:
            decline = (0.3402 - best["best_macro_f1"]) / 0.3402 * 100
            print(f"相比 Baseline 下降: -{decline:.2f}%")

    # 分析建议
    print("\n" + "="*80)
    print("分析与建议")
    print("="*80)

    if valid_results:
        best_f1 = max(r["best_macro_f1"] for r in valid_results)
        if best_f1 < 0.3402:
            print("""
问题诊断:
1. 注意力机制仍未能超越 baseline，可能原因：
   - 分割 mask 虽然准确，但对分类任务不是最优的空间先验
   - 舌象分类可能更依赖全局纹理/颜色特征，而非局部空间特征
   - 类别极度不平衡问题主导了性能

建议下一步:
1. 尝试完全不同的方向：
   - 使用分割 mask 做数据增强（裁剪/背景替换）而非特征加权
   - 多任务学习：分类 + 分割联合训练
   - 对比学习预训练

2. 解决类别不平衡：
   - 过采样少数类
   - 使用更激进的 class-balanced loss
   - 考虑合并相似类别
""")
        else:
            print(f"""
成功! 最佳配置 {best['name']} 超越了 baseline。

建议:
1. 使用该配置进行更多 epoch 训练
2. 尝试微调超参数
3. 结合其他技术（如 mixup, cutmix）进一步提升
""")

    _write_compare_report(results)
    _plot_compare_results(results)


def _write_compare_report(results: list):
    report_path = Path("docs/ablation_attention_compare.md")
    report_path.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append(f"# 注意力机制消融对比实验报告（{datetime.now().strftime('%Y-%m-%d')}）")
    lines.append("")
    lines.append("## 实验说明")
    lines.append("- 数据来源：`runs/ablation_attention/*/*/metrics_history.jsonl`")
    lines.append("- 指标口径：按 `val_macro_f1` 选最佳 epoch，并记录对应的 `val_accuracy`")
    lines.append("- 基线：无注意力模型（Best Macro F1=0.3402，Accuracy=0.6391）")
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
        diff = (best["best_macro_f1"] - 0.3402) / 0.3402 * 100
        lines.append(f"- 最佳实验：`{best['name']}`（Best Macro F1={best['best_macro_f1']:.4f}）。")
        if diff >= 0:
            lines.append(f"- 与基线相比：提升 {diff:.2f}%（{best['best_macro_f1']:.4f} vs 0.3402）。")
        else:
            lines.append(f"- 与基线相比：下降 {abs(diff):.2f}%（{best['best_macro_f1']:.4f} vs 0.3402）。")
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
    ax.bar(range(len(values)), values, color="#4C78A8")
    ax.axhline(0.3402, color="#F58518", linestyle="--", linewidth=1, label="Baseline 0.3402")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Best Macro F1")
    ax.set_title("Attention Ablation - Best Macro F1")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()

    out_path = Path("docs/ablation_attention_compare.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"可视化已写入: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="运行注意力机制消融实验")
    parser.add_argument("--output-base", type=str,
                        default="runs/ablation_attention",
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

    # 列出实验
    if args.list:
        print("可用实验:")
        for exp in EXPERIMENTS:
            print(f"  - {exp['name']}: {exp['desc']}")
        return

    # 只分析
    if args.analyze:
        analyze_results(args.output_base)
        return

    # 筛选实验
    if args.experiments:
        exp_names = set(args.experiments)
        experiments = [e for e in EXPERIMENTS if e["name"] in exp_names]
        if not experiments:
            print(f"错误: 未找到指定的实验 {args.experiments}")
            sys.exit(1)
    else:
        experiments = EXPERIMENTS

    # 运行实验
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"共 {len(experiments)} 个实验")

    run_results = {}
    for i, exp in enumerate(experiments, 1):
        print(f"\n[{i}/{len(experiments)}] 运行实验 {exp['name']}...")
        success = run_experiment(exp, args.output_base, args.dry_run)
        run_results[exp["name"]] = "成功" if success else "失败"

    # 汇总运行结果
    print("\n" + "="*60)
    print("运行结果汇总:")
    print("="*60)
    for name, status in run_results.items():
        print(f"  {name}: {status}")
    print(f"\n结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 分析结果（始终输出对比文档和可视化）
    analyze_results(args.output_base)


if __name__ == "__main__":
    main()
