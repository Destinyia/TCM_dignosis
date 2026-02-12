#!/usr/bin/env python
"""过采样倍率对比可视化

对比不同 oversample_factor 的 AP/AR 指标并生成可视化图表。
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 实验结果数据
RESULTS = {
    1.5: {"mAP": 11.73, "mAP_50": 16.04, "mAP_75": 12.05},
    2.0: {"mAP": 11.71, "mAP_50": 15.65, "mAP_75": 12.11},
    3.0: {"mAP": 12.05, "mAP_50": 16.02, "mAP_75": 12.22},
    4.0: {"mAP": 12.19, "mAP_50": 16.32, "mAP_75": 12.60},
}

# 如果有新实验结果，在这里添加
# 5.0: {"mAP": ?, "mAP_50": ?, "mAP_75": ?},
# 6.0: {"mAP": ?, "mAP_50": ?, "mAP_75": ?},
# 8.0: {"mAP": ?, "mAP_50": ?, "mAP_75": ?},
# 10.0: {"mAP": ?, "mAP_50": ?, "mAP_75": ?},


def plot_comparison():
    """生成对比图表"""
    factors = sorted(RESULTS.keys())
    mAP = [RESULTS[f]["mAP"] for f in factors]
    mAP_50 = [RESULTS[f]["mAP_50"] for f in factors]
    mAP_75 = [RESULTS[f]["mAP_75"] for f in factors]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # mAP
    axes[0].plot(factors, mAP, 'o-', color='#2196F3', linewidth=2, markersize=8)
    axes[0].set_xlabel('Oversample Factor')
    axes[0].set_ylabel('mAP (%)')
    axes[0].set_title('mAP vs Oversample Factor')
    axes[0].grid(True, alpha=0.3)
    for i, (x, y) in enumerate(zip(factors, mAP)):
        axes[0].annotate(f'{y:.2f}', (x, y), textcoords="offset points",
                        xytext=(0, 8), ha='center', fontsize=9)

    # mAP@50
    axes[1].plot(factors, mAP_50, 's-', color='#4CAF50', linewidth=2, markersize=8)
    axes[1].set_xlabel('Oversample Factor')
    axes[1].set_ylabel('mAP@50 (%)')
    axes[1].set_title('mAP@50 vs Oversample Factor')
    axes[1].grid(True, alpha=0.3)
    for i, (x, y) in enumerate(zip(factors, mAP_50)):
        axes[1].annotate(f'{y:.2f}', (x, y), textcoords="offset points",
                        xytext=(0, 8), ha='center', fontsize=9)

    # mAP@75
    axes[2].plot(factors, mAP_75, '^-', color='#FF9800', linewidth=2, markersize=8)
    axes[2].set_xlabel('Oversample Factor')
    axes[2].set_ylabel('mAP@75 (%)')
    axes[2].set_title('mAP@75 vs Oversample Factor')
    axes[2].grid(True, alpha=0.3)
    for i, (x, y) in enumerate(zip(factors, mAP_75)):
        axes[2].annotate(f'{y:.2f}', (x, y), textcoords="offset points",
                        xytext=(0, 8), ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig('runs/oversample_factor_compare/ap_comparison.png', dpi=150)
    plt.show()
    print("图表已保存: runs/oversample_factor_compare/ap_comparison.png")


def plot_combined():
    """生成综合对比图"""
    factors = sorted(RESULTS.keys())
    mAP = [RESULTS[f]["mAP"] for f in factors]
    mAP_50 = [RESULTS[f]["mAP_50"] for f in factors]
    mAP_75 = [RESULTS[f]["mAP_75"] for f in factors]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(factors, mAP, 'o-', label='mAP', linewidth=2, markersize=8)
    ax.plot(factors, mAP_50, 's-', label='mAP@50', linewidth=2, markersize=8)
    ax.plot(factors, mAP_75, '^-', label='mAP@75', linewidth=2, markersize=8)

    ax.set_xlabel('Oversample Factor', fontsize=12)
    ax.set_ylabel('AP (%)', fontsize=12)
    ax.set_title('AP Metrics vs Oversample Factor', fontsize=14)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('runs/oversample_factor_compare/ap_combined.png', dpi=150)
    plt.show()
    print("图表已保存: runs/oversample_factor_compare/ap_combined.png")


def plot_bar_comparison():
    """生成柱状图对比"""
    factors = sorted(RESULTS.keys())
    mAP = [RESULTS[f]["mAP"] for f in factors]
    mAP_50 = [RESULTS[f]["mAP_50"] for f in factors]
    mAP_75 = [RESULTS[f]["mAP_75"] for f in factors]

    x = np.arange(len(factors))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width, mAP, width, label='mAP', color='#2196F3')
    bars2 = ax.bar(x, mAP_50, width, label='mAP@50', color='#4CAF50')
    bars3 = ax.bar(x + width, mAP_75, width, label='mAP@75', color='#FF9800')

    ax.set_xlabel('Oversample Factor', fontsize=12)
    ax.set_ylabel('AP (%)', fontsize=12)
    ax.set_title('AP Metrics Comparison by Oversample Factor', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([str(f) for f in factors])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # 添加数值标签
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig('runs/oversample_factor_compare/ap_bar.png', dpi=150)
    plt.show()
    print("图表已保存: runs/oversample_factor_compare/ap_bar.png")


def print_summary():
    """打印结果汇总表"""
    factors = sorted(RESULTS.keys())
    print("\n" + "=" * 60)
    print("过采样倍率对比结果汇总")
    print("=" * 60)
    print(f"{'Factor':<10} {'mAP':<10} {'mAP@50':<10} {'mAP@75':<10}")
    print("-" * 40)
    for f in factors:
        r = RESULTS[f]
        print(f"{f:<10} {r['mAP']:<10.2f} {r['mAP_50']:<10.2f} {r['mAP_75']:<10.2f}")

    # 找出最佳
    best_factor = max(factors, key=lambda x: RESULTS[x]["mAP"])
    print("-" * 40)
    print(f"最佳倍率: {best_factor} (mAP={RESULTS[best_factor]['mAP']:.2f}%)")
    print("=" * 60)


def main():
    import os
    os.makedirs('runs/oversample_factor_compare', exist_ok=True)

    print_summary()
    plot_comparison()
    plot_combined()
    plot_bar_comparison()


if __name__ == "__main__":
    main()
