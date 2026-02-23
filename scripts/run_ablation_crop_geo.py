#!/usr/bin/env python
"""Crop裁剪与几何变换消融实验

以 noise 背景 0.5 作为 baseline，探索 crop 裁剪与几何变换的组合效果。

使用方法:
    conda activate cv
    python scripts/run_ablation_crop_geo.py

可选参数:
    --skip-completed  跳过已完成的实验
    --only C1 C2      只运行指定实验
    --list            列出所有实验及状态
    --analyze         只分析结果
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
from datetime import datetime
from pathlib import Path

OUTPUT_BASE = "runs/ablation_crop_geo1"
DATA_ROOT = "datasets/shezhenv3-coco"
RUN_SEED = 42
NUM_WORKERS = 4

COMMON_ARGS = [
    "--data-root", DATA_ROOT,
    "--image-size", "640",
    "--batch-size", "16",
    "--backbone", "efficientnet_b3",
    "--epochs", "50",
    "--lr", "0.0005",
    "--loss", "ce",
    "--early-stop", "10",
    "--scheduler", "cosine_warmup",
    "--warmup-epochs", "5",
    "--min-lr", "0.000001",
    "--amp",
]

EXPERIMENTS = {
    "B0": {
        "name": "B0_effnetb3_baseline_noaug",
        "desc": "EffNet-B3 基线 (no aug, lr=5e-4)",
        "args": [
            "--model-type", "baseline",
        ],
    },
    "A0": {
        "name": "A0_baseline_noaug",
        "desc": "A0 基线无mask-aug",
        "args": [
            "--model-type", "baseline",
        ],
    },
    "A1": {
        "name": "A1_bg_blur",
        "desc": "A1 背景blur (prob=0.5, dilate=15)",
        "args": [
            "--model-type", "baseline",
            "--mask-aug",
            "--mask-aug-mode", "background",
            "--mask-aug-bg-mode", "blur",
            "--mask-aug-prob", "0.5",
            "--mask-aug-dilate", "15",
        ],
    },
    "A2": {
        "name": "A2_bg_noise",
        "desc": "A2 背景noise (prob=0.5, dilate=15)",
        "args": [
            "--model-type", "baseline",
            "--mask-aug",
            "--mask-aug-mode", "background",
            "--mask-aug-bg-mode", "noise",
            "--mask-aug-prob", "0.5",
            "--mask-aug-dilate", "15",
        ],
    },
    "A3": {
        "name": "A3_bg_solid_gray",
        "desc": "A3 背景纯色gray (prob=0.5, dilate=15)",
        "args": [
            "--model-type", "baseline",
            "--mask-aug",
            "--mask-aug-mode", "background",
            "--mask-aug-bg-mode", "solid",
            "--mask-aug-bg-color", "gray",
            "--mask-aug-prob", "0.5",
            "--mask-aug-dilate", "15",
        ],
    },
    "B1": {
        "name": "B1_noise_baseline",
        "desc": "noise背景0.5_dilate15",
        "args": [
            "--model-type", "baseline",
            "--mask-aug",
            "--mask-aug-mode", "background",
            "--mask-aug-bg-mode", "noise",
            "--mask-aug-prob", "0.5",
            "--mask-aug-dilate", "15",
        ],
    },
    "B2": {
        "name": "B2_noise_baseline",
        "desc": "noise背景0.5_dilate8",
        "args": [
            "--model-type", "baseline",
            "--mask-aug",
            "--mask-aug-mode", "background",
            "--mask-aug-bg-mode", "noise",
            "--mask-aug-prob", "0.5",
            "--mask-aug-dilate", "8",
        ],
    },
    "B3": {
        "name": "B2_noise_baseline_noaug",
        "desc": "noise背景0.5_dilate8",
        "args": [
            "--model-type", "baseline",
        ],
    },
    "S1": {
        "name": "S1_EfficientNet_B3",
        "desc": "EffB3 mask + noise0.5 (lr=1e-3)",
        "args": [
            "--model-type", "baseline",
            "--lr", "0.001",
            "--backbone", "efficientnet_b3",
            "--mask-aug",
            "--mask-aug-mode", "background",
            "--mask-aug-bg-mode", "noise",
            "--mask-aug-prob", "0.5",
            "--mask-aug-dilate", "25",
        ],
    },
    "S2": {
        "name": "S2_EfficientNet_B3",
        "desc": "EffB3 mask + noise0.5 (lr=5e-4)",
        "args": [
            "--model-type", "baseline",
            "--lr", "0.0005",
            "--backbone", "efficientnet_b3",
            "--mask-aug",
            "--mask-aug-mode", "background",
            "--mask-aug-bg-mode", "noise",
            "--mask-aug-prob", "0.5",
            "--mask-aug-dilate", "25",
        ],
    },
    "S3": {
        "name": "S3_EfficientNet_B3_lr3e4",
        "desc": "EffB3 mask + noise0.5 (lr=3e-4)",
        "args": [
            "--model-type", "baseline",
            "--lr", "0.0003",
            "--backbone", "efficientnet_b3",
            "--mask-aug",
            "--mask-aug-mode", "background",
            "--mask-aug-bg-mode", "noise",
            "--mask-aug-prob", "0.5",
            "--mask-aug-dilate", "25",
        ],
    },
    "S5": {
        "name": "S5_EfficientNet_B3_lr7e4",
        "desc": "EffB3 mask + noise0.5 (lr=7e-4)",
        "args": [
            "--model-type", "baseline",
            "--lr", "0.0007",
            "--backbone", "efficientnet_b3",
            "--mask-aug",
            "--mask-aug-mode", "background",
            "--mask-aug-bg-mode", "noise",
            "--mask-aug-prob", "0.5",
            "--mask-aug-dilate", "25",
        ],
    },
    "M01": {
        "name": "M01_resnet34_noaug",
        "desc": "ResNet34 (no aug, lr=5e-4)",
        "args": [
            "--model-type", "baseline",
            "--backbone", "resnet34",
            "--lr", "0.0005",
        ],
    },
    "M02": {
        "name": "M02_resnet50_noaug",
        "desc": "ResNet50 (no aug, lr=5e-4)",
        "args": [
            "--model-type", "baseline",
            "--backbone", "resnet50",
            "--lr", "0.0005",
        ],
    },
    "M03": {
        "name": "M03_effnet_b0_noaug",
        "desc": "EffNet-B0 (no aug, lr=5e-4)",
        "args": [
            "--model-type", "baseline",
            "--backbone", "efficientnet_b0",
            "--lr", "0.0005",
        ],
    },
    "M04": {
        "name": "M04_effnet_b2_noaug",
        "desc": "EffNet-B2 (no aug, lr=5e-4)",
        "args": [
            "--model-type", "baseline",
            "--backbone", "efficientnet_b2",
            "--lr", "0.0005",
        ],
    },
    "M05": {
        "name": "M05_effnet_b3_noaug",
        "desc": "EffNet-B3 (no aug, lr=5e-4)",
        "args": [
            "--model-type", "baseline",
            "--backbone", "efficientnet_b3",
            "--lr", "0.0005",
        ],
    },
    "C0": {
        "name": "C0_rotate3_gray",
        "desc": "旋转3度p=0.2",
        "args": [
            "--model-type", "baseline",
            "--aug-rotate",
            "--aug-rotate-limit", "3",
            "--aug-rotate-prob", "0.2",
            "--aug-rotate-fill", "gray",
        ],
    },
    "C1": {
        "name": "C1_rotate5_gray",
        "desc": "旋转5度p=0.2",
        "args": [
            "--model-type", "baseline",
            "--aug-rotate",
            "--aug-rotate-limit", "5",
            "--aug-rotate-prob", "0.2",
            "--aug-rotate-fill", "gray",
        ],
    },
    "C01": {
        "name": "C01_bg_solid_gray_rotate3_gray",
        "desc": "背景gray (p=0.5)旋转1度p=0.2",
        "args": [
            "--model-type", "baseline",
            "--mask-aug",
            "--mask-aug-mode", "background",
            "--mask-aug-bg-mode", "solid",
            "--mask-aug-bg-color", "gray",
            "--mask-aug-prob", "0.5",
            "--mask-aug-dilate", "15",
            "--aug-rotate",
            "--aug-rotate-limit", "3",
            "--aug-rotate-prob", "0.2",
            "--aug-rotate-fill", "gray",
        ],
    },
    "C02": {
        "name": "C02_bg_solid_gray_rotate5_gray",
        "desc": "背景gray (p=0.5)旋转5度p=0.2",
        "args": [
            "--model-type", "baseline",
            "--mask-aug",
            "--mask-aug-mode", "background",
            "--mask-aug-bg-mode", "solid",
            "--mask-aug-bg-color", "gray",
            "--mask-aug-prob", "0.5",
            "--mask-aug-dilate", "15",
            "--aug-rotate",
            "--aug-rotate-limit", "5",
            "--aug-rotate-prob", "0.2",
            "--aug-rotate-fill", "gray",
        ],
    },
    "C03": {
        "name": "C03_scale_005",
        "desc": "缩放±0.05 p=0.2",
        "args": [
            "--aug-scale",
            "--aug-scale-limit", "0.05",
            "--aug-scale-prob", "0.2",
        ],
    },
    "C04": {
        "name": "C04_scale_rotate5_gray",
        "desc": "缩放±0.05 p=0.2 旋转5度p=0.2",
        "args": [
            "--aug-scale",
            "--aug-scale-limit", "0.05",
            "--aug-scale-prob", "0.2",
            "--aug-rotate",
            "--aug-rotate-limit", "3",
            "--aug-rotate-prob", "0.2",
            "--aug-rotate-fill", "gray",
        ],
    },
    "C05": {
        "name": "C05_bg_solid_gray_scale",
        "desc": "背景纯色gray (prob=0.5, dilate=15)缩放0.9~1.1 p=0.2",
        "args": [
            "--model-type", "baseline",
            "--mask-aug",
            "--mask-aug-mode", "background",
            "--mask-aug-bg-mode", "solid",
            "--mask-aug-bg-color", "gray",
            "--mask-aug-prob", "0.5",
            "--mask-aug-dilate", "15",
            "--aug-scale",
            "--aug-scale-limit", "0.1",
            "--aug-scale-prob", "0.2",
        ],
    },
}


def is_completed(exp_key: str) -> bool:
    """检查实验是否已完成"""
    exp = EXPERIMENTS[exp_key]
    exp_dir = Path(OUTPUT_BASE) / exp["name"]
    seed_dirs = sorted(exp_dir.glob("seed_*"))
    has_root = (exp_dir / "baseline" / "metrics.json").exists()
    if seed_dirs:
        return has_root or any((seed_dir / "baseline" / "metrics.json").exists() for seed_dir in seed_dirs)
    return has_root


def is_completed_seed(exp_key: str, seed: int) -> bool:
    """检查某个seed是否已完成"""
    exp = EXPERIMENTS[exp_key]
    metrics_path = Path(OUTPUT_BASE) / exp["name"] / f"seed_{seed}" / "baseline" / "metrics.json"
    return metrics_path.exists()


def run_experiment(exp_key: str, seed: int) -> bool:
    """运行单个实验"""
    exp = EXPERIMENTS[exp_key]
    output_dir = f"{OUTPUT_BASE}/{exp['name']}/seed_{seed}"

    cmd = [
        "python", "scripts/train_classifier.py",
        *COMMON_ARGS,
        *exp["args"],
        "--seed", str(seed),
        "--num-workers", str(NUM_WORKERS),
        "--output-dir", output_dir,
    ]

    print(f"\n{'='*60}")
    print(f"实验: {exp['name']} (seed={seed})")
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


def _mean_std(values):
    if not values:
        return None, None
    if len(values) == 1:
        return values[0], 0.0
    return statistics.mean(values), statistics.pstdev(values)


def analyze_results():
    """分析实验结果并生成报告"""
    print("\n" + "=" * 60)
    print("分析实验结果...")
    print("=" * 60)

    results = []
    for key, exp in EXPERIMENTS.items():
        exp_dir = Path(OUTPUT_BASE) / exp["name"]
        seed_dirs = sorted(exp_dir.glob("seed_*"))
        if (exp_dir / "baseline" / "metrics_history.jsonl").exists():
            seed_dirs = [exp_dir] + seed_dirs
        if not seed_dirs:
            seed_dirs = [exp_dir]

        per_seed = []
        for seed_dir in seed_dirs:
            history_path = seed_dir / "baseline" / "metrics_history.jsonl"
            if not history_path.exists():
                continue
            best_f1 = None
            best_train_acc = None
            best_val_acc = None
            best_epoch = None
            with open(history_path) as f:
                for line in f:
                    record = json.loads(line)
                    val_f1 = record.get("val_macro_f1")
                    if val_f1 is None:
                        continue
                    if best_f1 is None or val_f1 > best_f1:
                        best_f1 = val_f1
                        best_val_acc = record.get("val_accuracy")
                        best_train_acc = record.get("train_accuracy")
                        best_epoch = record.get("epoch")
            if best_f1 is not None:
                per_seed.append({
                    "best_macro_f1": best_f1,
                    "train_accuracy": best_train_acc,
                    "val_accuracy": best_val_acc,
                    "best_epoch": best_epoch,
                })

        if not per_seed:
            results.append({
                "key": key,
                "name": exp["name"],
                "desc": exp["desc"],
                "status": "未运行",
                "best_macro_f1": None,
                "train_accuracy": None,
                "val_accuracy": None,
                "gap": None,
                "best_epoch": None,
                "seed_count": 0,
                "std_macro_f1": None,
                "std_train_accuracy": None,
                "std_val_accuracy": None,
                "std_gap": None,
            })
            continue

        f1_values = [r["best_macro_f1"] for r in per_seed if r["best_macro_f1"] is not None]
        train_values = [r["train_accuracy"] for r in per_seed if r["train_accuracy"] is not None]
        val_values = [r["val_accuracy"] for r in per_seed if r["val_accuracy"] is not None]
        gap_values = []
        for r in per_seed:
            if r["train_accuracy"] is not None and r["val_accuracy"] is not None:
                gap_values.append(r["train_accuracy"] - r["val_accuracy"])

        f1_mean, f1_std = _mean_std(f1_values)
        train_mean, train_std = _mean_std(train_values)
        val_mean, val_std = _mean_std(val_values)
        gap_mean, gap_std = _mean_std(gap_values)
        epoch_values = [r["best_epoch"] for r in per_seed if r["best_epoch"] is not None]
        epoch_mean = statistics.mean(epoch_values) if epoch_values else None

        results.append({
            "key": key,
            "name": exp["name"],
            "desc": exp["desc"],
            "status": "已完成",
            "best_macro_f1": f1_mean,
            "train_accuracy": train_mean,
            "val_accuracy": val_mean,
            "gap": gap_mean,
            "best_epoch": epoch_mean,
            "seed_count": len(per_seed),
            "std_macro_f1": f1_std,
            "std_train_accuracy": train_std,
            "std_val_accuracy": val_std,
            "std_gap": gap_std,
        })

    # 打印结果表格
    print("\n实验结果对比:")
    print("-" * 130)
    print(f"{'实验':<25} {'描述':<28} {'Best F1':>16} {'Train Acc':>17} {'Val Acc':>15} {'Gap':>12} {'Epoch':>6}")
    print("-" * 130)

    for r in results:
        if r["status"] == "未运行":
            print(f"{r['name']:<25} {r['desc']:<28} {'未运行':>16} {'-':>17} {'-':>15} {'-':>12} {'-':>6}")
        else:
            best_f1 = "-" if r["best_macro_f1"] is None else f"{r['best_macro_f1']:.4f}±{r['std_macro_f1']:.4f}"
            train_acc = "-" if r["train_accuracy"] is None else f"{r['train_accuracy']:.4f}±{r['std_train_accuracy']:.4f}"
            val_acc = "-" if r["val_accuracy"] is None else f"{r['val_accuracy']:.4f}±{r['std_val_accuracy']:.4f}"
            gap = "-" if r["gap"] is None else f"{r['gap']:.4f}±{r['std_gap']:.4f}"
            epoch = "-" if r["best_epoch"] is None else f"{r['best_epoch']:.1f}"
            print(f"{r['name']:<25} {r['desc']:<28} {best_f1:>16} {train_acc:>17} {val_acc:>15} {gap:>12} {epoch:>6}")

    print("-" * 130)

    # 生成 Markdown 报告
    report_path = Path("docs/ablation_crop_geo_compare.md")
    report_path.parent.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d")
    report = f"""# Crop 裁剪与几何变换消融实验报告（{timestamp}）

## 实验说明
- Baseline：noise 背景 0.5
- 探索：crop 裁剪与几何变换的组合效果（包含不同损失/骨干的对照）

## 对比结果
| 实验名称 | 描述 | Best Macro F1 | Train Acc | Val Acc | Gap | Best Epoch |
| --- | --- | --- | --- | --- | --- | --- |
"""
    for r in results:
        if r["status"] == "未运行":
            report += f"| {r['name']} | {r['desc']} | 未运行 | - | - | - | - |\n"
        else:
            best_f1 = "-" if r["best_macro_f1"] is None else f"{r['best_macro_f1']:.4f}±{r['std_macro_f1']:.4f}"
            train_acc = "-" if r["train_accuracy"] is None else f"{r['train_accuracy']:.4f}±{r['std_train_accuracy']:.4f}"
            val_acc = "-" if r["val_accuracy"] is None else f"{r['val_accuracy']:.4f}±{r['std_val_accuracy']:.4f}"
            gap = "-" if r["gap"] is None else f"{r['gap']:.4f}±{r['std_gap']:.4f}"
            epoch = "-" if r["best_epoch"] is None else f"{r['best_epoch']:.1f}"
            report += f"| {r['name']} | {r['desc']} | {best_f1} | {train_acc} | {val_acc} | {gap} | {epoch} |\n"

    # 找出最佳实验
    completed = [r for r in results if r["status"] == "已完成"]
    if completed:
        best = max(completed, key=lambda x: x["best_macro_f1"] or 0)
        baseline = next((r for r in completed if r["key"] == "B0"), None)

        best_f1 = best.get("best_macro_f1")
        best_f1_text = "-" if best_f1 is None else f"{best_f1:.4f}"
        report += f"""
## 结论摘要
- 最佳实验：`{best['name']}`（Best Macro F1={best_f1_text}）
"""
        baseline_f1 = baseline.get("best_macro_f1") if baseline else None
        if baseline_f1 is not None and best_f1 is not None:
            improvement = best_f1 - baseline_f1
            report += f"- 相比 noise baseline 提升：{improvement:+.4f}\n"

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"\n报告已保存至: {report_path}")


def main():
    global NUM_WORKERS
    parser = argparse.ArgumentParser(description="运行Crop裁剪与几何变换消融实验")
    parser.add_argument("--skip-completed", action="store_true",
                        help="跳过已完成的实验")
    parser.add_argument("--only", nargs="+", choices=list(EXPERIMENTS.keys()),
                        help="只运行指定的实验 (B0, A0, A1, A2, A3, B1, B2, S1, S2, S3, S5, M01, M02, M03, M04, M05, C01, C02, C03, C04, C05)")
    parser.add_argument("--list", action="store_true",
                        help="列出所有实验及状态")
    parser.add_argument("--analyze", action="store_true",
                        help="只分析结果，不运行实验")
    parser.add_argument("--seeds", type=str, default=str(RUN_SEED),
                        help="随机种子列表，用逗号分隔 (如 42,43,44)")
    parser.add_argument("--seed", type=int, default=None,
                        help="单个随机种子（优先于 --seeds）")
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS,
                        help="DataLoader workers")
    args = parser.parse_args()
    NUM_WORKERS = args.num_workers

    if args.seed is not None:
        seeds = [args.seed]
    else:
        seeds = []
        for item in args.seeds.replace(",", " ").split():
            try:
                seeds.append(int(item))
            except ValueError:
                continue
        if not seeds:
            seeds = [RUN_SEED]

    if args.list:
        print("实验列表:")
        for key, exp in EXPERIMENTS.items():
            status = "已完成" if is_completed(key) else "未完成"
            print(f"  [{key}] {exp['name']}: {exp['desc']} ({status})")
        return

    if args.analyze:
        analyze_results()
        return

    exp_keys = args.only if args.only else list(EXPERIMENTS.keys())

    print("=" * 60)
    print("Crop裁剪与几何变换消融实验")
    print("=" * 60)

    results = {}
    for i, key in enumerate(exp_keys, 1):
        exp = EXPERIMENTS[key]
        results[key] = []

        for seed in seeds:
            if args.skip_completed and is_completed_seed(key, seed):
                print(f"\n[{i}/{len(exp_keys)}] {exp['name']} (seed={seed}) - 已完成，跳过")
                results[key].append(f"seed {seed}: 跳过")
                continue

            print(f"\n[{i}/{len(exp_keys)}] 运行 {exp['name']} (seed={seed})...")
            success = run_experiment(key, seed)
            results[key].append(f"seed {seed}: {'成功' if success else '失败'}")

            if not success:
                print(f"\n实验 {key} (seed={seed}) 失败，是否继续? (y/n)")
                try:
                    if input().strip().lower() != 'y':
                        break
                except EOFError:
                    break

    print("\n" + "=" * 60)
    print("运行结果汇总:")
    print("=" * 60)
    for key, status in results.items():
        joined = ", ".join(status) if status else "-"
        print(f"  {EXPERIMENTS[key]['name']}: {joined}")

    analyze_results()
    print("\nDone!")


if __name__ == "__main__":
    main()
