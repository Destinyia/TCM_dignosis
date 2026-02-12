#!/usr/bin/env python
"""采样策略与损失函数对比可视化

对比 runs/sampler_loss_compare 中所有实验的 AP 指标。
每个子图显示一个实验的各类别 AP。
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 实验目录
RUNS_DIR = Path("runs/sampler_loss_compare")

# 8个目标类别
TARGET_CLASSES = [
    "baitaishe", "hongdianshe", "liewenshe", "chihenshe",
    "hongshe", "huangtaishe", "pangdashe", "botaishe"
]

# 实验名称映射（简短显示名）
EXP_NAMES = {
    "baseline_aug": "Baseline",
    "samp_oversample": "Oversample",
    "samp_stratified": "Stratified",
    "samp_class_aware": "ClassAware",
    "loss_focal": "Focal",
    "loss_weighted_ce": "WeightedCE",
    "loss_cb_focal": "CB-Focal",
    "loss_seesaw": "Seesaw",
    "best_combined": "Best Combined",
}


def load_all_metrics() -> dict:
    """加载所有实验的 metrics.json"""
    results = {}
    for exp_dir in RUNS_DIR.iterdir():
        if not exp_dir.is_dir():
            continue
        metrics_file = exp_dir / "metrics.json"
        if not metrics_file.exists():
            continue
        # 提取实验名（去掉时间戳）
        exp_name = "_".join(exp_dir.name.split("_")[:-2])
        with open(metrics_file, "r", encoding="utf-8") as f:
            results[exp_name] = json.load(f)
    return results


def plot_per_class_ap_subplots(results: dict):
    """每个实验一个子图，显示各类别AP"""
    exp_names = sorted(results.keys())
    n_exps = len(exp_names)

    # 计算子图布局
    ncols = 3
    nrows = (n_exps + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4 * nrows))
    axes = axes.flatten() if n_exps > 1 else [axes]

    colors = plt.cm.Set2(np.linspace(0, 1, len(TARGET_CLASSES) + 1))

    for idx, exp_name in enumerate(exp_names):
        ax = axes[idx]
        metrics = results[exp_name]

        # 获取各类别AP50
        per_class_ap = metrics.get("per_class_AP50", {})
        class_aps = [per_class_ap.get(cls, 0) * 100 for cls in TARGET_CLASSES]
        mean_ap = metrics.get("mAP_50", 0) * 100

        # 绘制柱状图
        x = np.arange(len(TARGET_CLASSES) + 1)
        values = class_aps + [mean_ap]
        labels = TARGET_CLASSES + ["Mean"]

        bars = ax.bar(x, values, color=colors)
        bars[-1].set_color('#E53935')  # Mean用红色

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
        ax.set_ylabel('AP@50 (%)')
        ax.set_title(EXP_NAMES.get(exp_name, exp_name), fontsize=11)
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3, axis='y')

        # 添加数值标签
        for bar, val in zip(bars, values):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width()/2, val + 1,
                       f'{val:.1f}', ha='center', va='bottom', fontsize=7)

    # 隐藏多余子图
    for idx in range(n_exps, len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle('Per-Class AP@50 Comparison', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(RUNS_DIR / 'per_class_ap_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    print(f"图表已保存: {RUNS_DIR / 'per_class_ap_comparison.png'}")


def plot_heatmap(results: dict):
    """热力图对比所有实验的各类别AP"""
    exp_names = sorted(results.keys())

    # 构建数据矩阵
    data = []
    for exp_name in exp_names:
        per_class_ap = results[exp_name].get("per_class_AP50", {})
        row = [per_class_ap.get(cls, 0) * 100 for cls in TARGET_CLASSES]
        row.append(results[exp_name].get("mAP_50", 0) * 100)
        data.append(row)

    data = np.array(data)
    labels = TARGET_CLASSES + ["Mean"]
    display_names = [EXP_NAMES.get(n, n) for n in exp_names]

    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(data, cmap='YlOrRd', aspect='auto')

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(exp_names)))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_yticklabels(display_names)

    # 添加数值
    for i in range(len(exp_names)):
        for j in range(len(labels)):
            color = 'white' if data[i, j] > 50 else 'black'
            ax.text(j, i, f'{data[i, j]:.1f}', ha='center', va='center',
                   color=color, fontsize=9)

    plt.colorbar(im, ax=ax, label='AP@50 (%)')
    ax.set_title('AP@50 Heatmap: Experiments vs Classes', fontsize=14)
    plt.tight_layout()
    plt.savefig(RUNS_DIR / 'ap_heatmap.png', dpi=150)
    plt.show()
    print(f"图表已保存: {RUNS_DIR / 'ap_heatmap.png'}")


def plot_radar(results: dict):
    """雷达图对比"""
    exp_names = sorted(results.keys())
    angles = np.linspace(0, 2 * np.pi, len(TARGET_CLASSES), endpoint=False)
    angles = np.concatenate([angles, [angles[0]]])

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    colors = plt.cm.tab10(np.linspace(0, 1, len(exp_names)))

    for idx, exp_name in enumerate(exp_names):
        per_class_ap = results[exp_name].get("per_class_AP50", {})
        values = [per_class_ap.get(cls, 0) * 100 for cls in TARGET_CLASSES]
        values = values + [values[0]]
        ax.plot(angles, values, 'o-', label=EXP_NAMES.get(exp_name, exp_name),
               color=colors[idx], linewidth=1.5, markersize=4)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(TARGET_CLASSES, fontsize=9)
    ax.set_ylim(0, 100)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=9)
    ax.set_title('Per-Class AP@50 Radar Chart', fontsize=14, y=1.08)

    plt.tight_layout()
    plt.savefig(RUNS_DIR / 'ap_radar.png', dpi=150, bbox_inches='tight')
    plt.show()
    print(f"图表已保存: {RUNS_DIR / 'ap_radar.png'}")


def plot_overall_comparison(results: dict):
    """总体指标对比柱状图"""
    exp_names = sorted(results.keys())
    display_names = [EXP_NAMES.get(n, n) for n in exp_names]

    mAP = [results[n].get("mAP", 0) * 100 for n in exp_names]
    mAP_50 = [results[n].get("mAP_50", 0) * 100 for n in exp_names]
    mAP_75 = [results[n].get("mAP_75", 0) * 100 for n in exp_names]

    x = np.arange(len(exp_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(x - width, mAP, width, label='mAP', color='#2196F3')
    ax.bar(x, mAP_50, width, label='mAP@50', color='#4CAF50')
    ax.bar(x + width, mAP_75, width, label='mAP@75', color='#FF9800')

    ax.set_xticks(x)
    ax.set_xticklabels(display_names, rotation=45, ha='right')
    ax.set_ylabel('AP (%)')
    ax.set_title('Overall AP Comparison', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(RUNS_DIR / 'overall_ap_comparison.png', dpi=150)
    plt.show()
    print(f"图表已保存: {RUNS_DIR / 'overall_ap_comparison.png'}")


def print_summary(results: dict):
    """打印结果汇总"""
    exp_names = sorted(results.keys())
    print("\n" + "=" * 70)
    print("采样策略与损失函数对比结果汇总")
    print("=" * 70)
    print(f"{'Experiment':<15} {'mAP':<8} {'mAP@50':<8} {'mAP@75':<8}")
    print("-" * 50)
    for n in exp_names:
        m = results[n]
        print(f"{EXP_NAMES.get(n, n):<15} "
              f"{m.get('mAP', 0)*100:<8.2f} "
              f"{m.get('mAP_50', 0)*100:<8.2f} "
              f"{m.get('mAP_75', 0)*100:<8.2f}")
    print("=" * 70)


def main():
    results = load_all_metrics()
    if not results:
        print("未找到实验结果")
        return

    print_summary(results)
    plot_per_class_ap_subplots(results)
    plot_heatmap(results)
    plot_radar(results)
    plot_overall_comparison(results)


if __name__ == "__main__":
    main()