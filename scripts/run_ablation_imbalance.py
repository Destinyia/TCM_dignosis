#!/usr/bin/env python
"""Imbalance ablation experiments (B0 baseline + loss/sampler variants).

Usage:
  python scripts/run_ablation_imbalance.py --phase phase1 --seeds 45,46,47
  python scripts/run_ablation_imbalance.py --phase phase2 --seeds 45,46,47
  python scripts/run_ablation_imbalance.py --phase phase3 --seeds 45,46,47
  python scripts/run_ablation_imbalance.py --phase phase4 --seeds 45,46,47
  python scripts/run_ablation_imbalance.py --combo-losses L1 L2 --phase combo
  python scripts/run_ablation_imbalance.py --analyze
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
from datetime import datetime
from pathlib import Path

OUTPUT_BASE = "runs/ablation_imbalance"
DATA_ROOT = "datasets/shezhenv3-coco"
RUN_SEED = 45
NUM_WORKERS = 4

COMMON_ARGS = [
    "--data-root", DATA_ROOT,
    "--image-size", "640",
    "--batch-size", "16",
    "--backbone", "efficientnet_b3",
    "--epochs", "50",
    "--lr", "0.0005",
    "--loss", "ce",
    "--early-stop", "6",
    "--scheduler", "cosine_warmup",
    "--warmup-epochs", "5",
    "--min-lr", "0.000001",
    "--amp",
    "--model-type", "baseline",
]

LOSS_EXPERIMENTS = {
    "L1": {
        "name": "L1_focal",
        "desc": "Focal loss (gamma=2.0)",
        "args": ["--loss", "focal", "--focal-gamma", "2.0"],
    },
    "L2": {
        "name": "L2_cb_focal",
        "desc": "Class-balanced focal (gamma=2.0)",
        "args": ["--loss", "cb_focal", "--focal-gamma", "2.0"],
    },
    "L3": {
        "name": "L3_seesaw",
        "desc": "Seesaw loss",
        "args": ["--loss", "seesaw"],
    },
}

SAMPLER_EXPERIMENTS = {
    "S1": {
        "name": "S1_weighted_sqrt",
        "desc": "Weighted sampler (sqrt)",
        "args": ["--weighted-sampler", "--sampler-strategy", "sqrt"],
    },
    "S2": {
        "name": "S2_weighted_inverse",
        "desc": "Weighted sampler (inverse)",
        "args": ["--weighted-sampler", "--sampler-strategy", "inverse"],
    },
}

PHASE2_SAMPLERS = {
    "S1": {
        "suffix": "wsqrt",
        "name": "S1_weighted_sqrt",
        "desc": "Weighted sampler (sqrt)",
        "args": ["--weighted-sampler", "--sampler-strategy", "sqrt"],
    },
    "S2": {
        "suffix": "winv",
        "name": "S2_weighted_inverse",
        "desc": "Weighted sampler (inverse)",
        "args": ["--weighted-sampler", "--sampler-strategy", "inverse"],
    },
    "U05": {
        "suffix": "under05",
        "desc": "undersample (ratio=0.5)",
        "args": ["--sampler-type", "undersample", "--undersample-ratio", "0.5"],
    },
    "O2": {
        "suffix": "over2",
        "desc": "oversample (x2)",
        "args": ["--sampler-type", "oversample", "--oversample-factor", "2"],
    },
    "O3": {
        "suffix": "over3",
        "desc": "oversample (x3)",
        "args": ["--sampler-type", "oversample", "--oversample-factor", "3"],
    },
    "O4": {
        "suffix": "over4",
        "desc": "oversample (x4)",
        "args": ["--sampler-type", "oversample", "--oversample-factor", "4"],
    },
}

# Phase3: top configs for confirmation / stability checks.
PHASE3_KEYS = [
    "B0",
    "L2",
    "S2",
    "P2_L1_S1",
    "P2_L2_O2",
    "P2_B0_S1",
]

PHASE4_BASE_KEYS = [
    "P2_L1_S1",
    "L2",
    "P2_L2_O2",
]

PHASE4_AUGS = {
    "gray": {
        "suffix": "gray",
        "desc": "mask bg solid gray (p=0.5, dilate=15)",
        "args": [
            "--mask-aug",
            "--mask-aug-mode", "background",
            "--mask-aug-bg-mode", "solid",
            "--mask-aug-bg-color", "gray",
            "--mask-aug-prob", "0.5",
            "--mask-aug-dilate", "15",
        ],
    },
    "rot5": {
        "suffix": "rot5",
        "desc": "rotate 5deg (p=0.2, fill=gray)",
        "args": [
            "--aug-rotate",
            "--aug-rotate-limit", "5",
            "--aug-rotate-prob", "0.2",
            "--aug-rotate-fill", "gray",
        ],
    },
    "d02": {
        "suffix": "d02",
        "desc": "one-of mask0.3 + scale0.05 + rotate8 (p=0.1 each)",
        "args": [
            "--aug-oneof",
            "--mask-aug",
            "--mask-aug-mode", "background",
            "--mask-aug-bg-mode", "solid",
            "--mask-aug-bg-color", "gray",
            "--mask-aug-prob", "0.3",
            "--mask-aug-dilate", "15",
            "--aug-scale",
            "--aug-scale-limit", "0.05",
            "--aug-scale-prob", "0.1",
            "--aug-rotate",
            "--aug-rotate-limit", "8",
            "--aug-rotate-prob", "0.1",
            "--aug-rotate-fill", "gray",
        ],
    },
}

EXPERIMENTS = {
    "B0": {
        "name": "B0_effnetb3_baseline_noaug",
        "desc": "EffNet-B3 baseline (CE, no sampler)",
        "args": [],
    },
    **LOSS_EXPERIMENTS,
    **SAMPLER_EXPERIMENTS,
}


def _build_combo_experiments(loss_keys: list[str]) -> dict:
    combos = {}
    if len(loss_keys) != 2:
        return combos
    k1, k2 = loss_keys
    for loss_key, loss_rank in [(k1, "best"), (k2, "second")]:
        if loss_key not in LOSS_EXPERIMENTS:
            continue
        for sampler_key, sampler_name in [("S1", "sqrt"), ("S2", "inverse")]:
            exp_key = f"C{len(combos)+1}"
            loss_exp = LOSS_EXPERIMENTS[loss_key]
            sampler_exp = SAMPLER_EXPERIMENTS[sampler_key]
            combos[exp_key] = {
                "name": f"{exp_key}_{loss_exp['name']}_{sampler_name}",
                "desc": f"{loss_rank} loss + sampler({sampler_name})",
                "args": [*loss_exp["args"], *sampler_exp["args"]],
            }
    return combos


def _build_phase2_experiments() -> dict:
    phase2 = {}
    base_losses = {
        "B0": EXPERIMENTS["B0"],
        "L1": LOSS_EXPERIMENTS["L1"],
        "L2": LOSS_EXPERIMENTS["L2"],
    }
    for base_key, base_exp in base_losses.items():
        for sampler_key, sampler_exp in PHASE2_SAMPLERS.items():
            exp_key = f"P2_{base_key}_{sampler_key}"
            phase2[exp_key] = {
                "name": f"P2_{base_key}_{sampler_exp['suffix']}",
                "desc": f"{base_exp['desc']} + {sampler_exp['desc']}",
                "args": [*base_exp["args"], *sampler_exp["args"]],
            }
    return phase2


def _build_phase4_experiments(experiments: dict) -> dict:
    phase4 = {}
    for base_key in PHASE4_BASE_KEYS:
        if base_key not in experiments:
            continue
        base_exp = experiments[base_key]
        for aug_key, aug in PHASE4_AUGS.items():
            exp_key = f"P4_{base_key}_{aug_key}"
            phase4[exp_key] = {
                "name": f"P4_{base_exp['name']}_{aug['suffix']}",
                "desc": f"{base_exp['desc']} + {aug['desc']}",
                "args": [*base_exp["args"], *aug["args"]],
            }
    return phase4


def is_completed_seed(exp_key: str, seed: int, experiments: dict) -> bool:
    exp = experiments[exp_key]
    metrics_path = Path(OUTPUT_BASE) / exp["name"] / f"seed_{seed}" / "baseline" / "metrics.json"
    return metrics_path.exists()


def run_experiment(exp_key: str, seed: int, experiments: dict) -> bool:
    exp = experiments[exp_key]
    output_dir = f"{OUTPUT_BASE}/{exp['name']}/seed_{seed}"
    cmd = [
        "python", "scripts/train_classifier.py",
        *COMMON_ARGS,
        *exp["args"],
        "--seed", str(seed),
        "--num-workers", str(NUM_WORKERS),
        "--output-dir", output_dir,
    ]

    print("\n" + "=" * 60)
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


def analyze_results(experiments: dict) -> None:
    print("\n" + "=" * 60)
    print("分析实验结果...")
    print("=" * 60)

    results = []
    for key, exp in experiments.items():
        exp_dir = Path(OUTPUT_BASE) / exp["name"]
        seed_dirs = sorted(exp_dir.glob("seed_*"))
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
            with open(history_path, encoding="utf-8") as f:
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

    report_path = Path("docs/ablation_imbalance_compare.md")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d")
    report = f"""# 不均衡消融实验报告（{timestamp}）

## 对比结果
| 实验名称 | 描述 | Best Macro F1 | Train Acc | Val Acc | Gap | Best Epoch |
| --- | --- | --- | --- | --- | --- | --- |
"""
    for r in results:
        if r["status"] == "未运行":
            report += f"| {r['name']} | {r['desc']} | 未运行 | - | - | - | - |\\n"
        else:
            best_f1 = "-" if r["best_macro_f1"] is None else f"{r['best_macro_f1']:.4f}±{r['std_macro_f1']:.4f}"
            train_acc = "-" if r["train_accuracy"] is None else f"{r['train_accuracy']:.4f}±{r['std_train_accuracy']:.4f}"
            val_acc = "-" if r["val_accuracy"] is None else f"{r['val_accuracy']:.4f}±{r['std_val_accuracy']:.4f}"
            gap = "-" if r["gap"] is None else f"{r['gap']:.4f}±{r['std_gap']:.4f}"
            epoch = "-" if r["best_epoch"] is None else f"{r['best_epoch']:.1f}"
            report += f"| {r['name']} | {r['desc']} | {best_f1} | {train_acc} | {val_acc} | {gap} | {epoch} |\\n"

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\\n报告已保存至: {report_path}")


def _parse_seed_list(seed_str: str) -> list[int]:
    seeds = []
    for item in seed_str.replace(",", " ").split():
        try:
            seeds.append(int(item))
        except ValueError:
            continue
    return seeds


def main() -> None:
    global NUM_WORKERS
    parser = argparse.ArgumentParser(description="运行不均衡消融实验")
    parser.add_argument("--skip-completed", action="store_true")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--analyze", action="store_true")
    parser.add_argument("--only", nargs="+")
    parser.add_argument(
        "--phase",
        type=str,
        default="phase1",
        choices=["baseline", "loss", "sampler", "phase1", "phase2", "phase3", "phase4", "combo", "all"],
    )
    parser.add_argument(
        "--combo-losses",
        nargs=2,
        default=None,
        help="Two loss keys for combo phase (e.g. L1 L2)",
    )
    parser.add_argument("--seeds", type=str, default=str(RUN_SEED))
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    args = parser.parse_args()
    NUM_WORKERS = args.num_workers

    experiments = dict(EXPERIMENTS)
    experiments.update(_build_phase2_experiments())
    experiments.update(_build_phase4_experiments(experiments))
    if args.combo_losses:
        experiments.update(_build_combo_experiments(args.combo_losses))

    phases = {
        "baseline": ["B0"],
        "loss": list(LOSS_EXPERIMENTS.keys()),
        "sampler": list(SAMPLER_EXPERIMENTS.keys()),
        "phase1": ["B0", *LOSS_EXPERIMENTS.keys(), *SAMPLER_EXPERIMENTS.keys()],
        "phase2": list(_build_phase2_experiments().keys()),
        "phase3": PHASE3_KEYS,
        "phase4": [k for k in experiments.keys() if k.startswith("P4_")],
        "combo": [k for k in experiments.keys() if k.startswith("C")],
        "all": list(experiments.keys()),
    }

    if args.list:
        print("实验列表:")
        for key, exp in experiments.items():
            print(f"  [{key}] {exp['name']}: {exp['desc']}")
        return

    if args.analyze:
        analyze_results(experiments)
        return

    if args.seed is not None:
        seeds = [args.seed]
    else:
        seeds = _parse_seed_list(args.seeds) or [RUN_SEED]

    exp_keys = args.only if args.only else phases[args.phase]

    print("=" * 60)
    print("不均衡消融实验")
    print("=" * 60)

    for i, key in enumerate(exp_keys, 1):
        if key not in experiments:
            print(f"Skip unknown exp key: {key}")
            continue
        exp = experiments[key]
        for seed in seeds:
            if args.skip_completed and is_completed_seed(key, seed, experiments):
                print(f"\n[{i}/{len(exp_keys)}] {exp['name']} (seed={seed}) - 已完成，跳过")
                continue
            print(f"\n[{i}/{len(exp_keys)}] 运行 {exp['name']} (seed={seed})...")
            run_experiment(key, seed, experiments)

    analyze_results(experiments)
    print("\nDone!")


if __name__ == "__main__":
    main()
