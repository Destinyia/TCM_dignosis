#!/usr/bin/env python
"""Run a fixed subset of experiments and report LR-stability stats.

Targets: B0, S1, C1, C5 (as defined in run_ablation_crop_geo.py)
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import run_ablation_crop_geo as ablation

TARGET_KEYS = ["B0", "B1", "B2",  "B3", "S1", "S2", "S3", "S5", "M01", "M02", "M03", "M04", "M05"]


def _parse_seeds(seed_arg: Optional[int], seeds_arg: str) -> List[int]:
    if seed_arg is not None:
        return [seed_arg]
    seeds: List[int] = []
    for item in seeds_arg.replace(",", " ").split():
        try:
            seeds.append(int(item))
        except ValueError:
            continue
    return seeds or [ablation.RUN_SEED]


def _extract_lr(exp_args: List[str]) -> Optional[float]:
    def _find_lr(args: List[str]) -> Optional[float]:
        for i, arg in enumerate(args):
            if arg == "--lr" and i + 1 < len(args):
                try:
                    return float(args[i + 1])
                except ValueError:
                    return None
        return None

    lr = _find_lr(exp_args)
    if lr is not None:
        return lr
    return _find_lr(ablation.COMMON_ARGS)


def _mean_std(values: List[float]) -> Tuple[Optional[float], Optional[float]]:
    if not values:
        return None, None
    if len(values) == 1:
        return values[0], 0.0
    return statistics.mean(values), statistics.pstdev(values)


def _best_from_history(history_path: Path) -> Optional[Dict[str, float]]:
    if not history_path.exists():
        return None
    best = None
    with history_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            val_f1 = record.get("val_macro_f1")
            if val_f1 is None:
                continue
            if best is None or val_f1 > best["best_macro_f1"]:
                best = {
                    "best_macro_f1": val_f1,
                    "train_accuracy": record.get("train_accuracy"),
                    "val_accuracy": record.get("val_accuracy"),
                    "best_epoch": record.get("epoch"),
                }
    return best


def analyze_subset(target_keys: List[str]) -> None:
    results = []
    for key in target_keys:
        exp = ablation.EXPERIMENTS[key]
        exp_dir = Path(ablation.OUTPUT_BASE) / exp["name"]
        seed_dirs = sorted(exp_dir.glob("seed_*"))
        if (exp_dir / "baseline" / "metrics_history.jsonl").exists():
            seed_dirs = [exp_dir] + seed_dirs
        if not seed_dirs:
            seed_dirs = [exp_dir]

        per_seed = []
        for seed_dir in seed_dirs:
            history_path = seed_dir / "baseline" / "metrics_history.jsonl"
            best = _best_from_history(history_path)
            if best is not None:
                per_seed.append(best)

        lr = _extract_lr(exp["args"])
        if not per_seed:
            results.append({
                "name": exp["name"],
                "desc": exp["desc"],
                "lr": lr,
                "status": "未运行",
            })
            continue

        f1_values = [r["best_macro_f1"] for r in per_seed if r.get("best_macro_f1") is not None]
        train_values = [r["train_accuracy"] for r in per_seed if r.get("train_accuracy") is not None]
        val_values = [r["val_accuracy"] for r in per_seed if r.get("val_accuracy") is not None]
        gap_values = []
        for r in per_seed:
            if r.get("train_accuracy") is not None and r.get("val_accuracy") is not None:
                gap_values.append(r["train_accuracy"] - r["val_accuracy"])

        f1_mean, f1_std = _mean_std(f1_values)
        train_mean, train_std = _mean_std(train_values)
        val_mean, val_std = _mean_std(val_values)
        gap_mean, gap_std = _mean_std(gap_values)
        epoch_values = [r.get("best_epoch") for r in per_seed if r.get("best_epoch") is not None]
        epoch_mean = statistics.mean(epoch_values) if epoch_values else None

        results.append({
            "name": exp["name"],
            "desc": exp["desc"],
            "lr": lr,
            "status": "已完成",
            "best_macro_f1": f1_mean,
            "std_macro_f1": f1_std,
            "train_accuracy": train_mean,
            "std_train_accuracy": train_std,
            "val_accuracy": val_mean,
            "std_val_accuracy": val_std,
            "gap": gap_mean,
            "std_gap": gap_std,
            "best_epoch": epoch_mean,
            "seed_count": len(per_seed),
        })

    target_label = ", ".join(target_keys)
    print(f"\nLR对比结果 ({target_label}):")
    print("-" * 140)
    print(
        f"{'实验':<25} {'描述':<28} {'LR':>8} {'Best F1':>16} "
        f"{'Train Acc':>17} {'Val Acc':>15} {'Gap':>12} {'Epoch':>6} {'Seeds':>6}"
    )
    print("-" * 140)
    for r in results:
        lr_text = "-" if r.get("lr") is None else f"{r['lr']:.1e}"
        if r["status"] == "未运行":
            print(
                f"{r['name']:<25} {r['desc']:<28} {lr_text:>8} "
                f"{'未运行':>16} {'-':>17} {'-':>15} {'-':>12} {'-':>6} {'-':>6}"
            )
            continue
        best_f1 = f"{r['best_macro_f1']:.4f}±{r['std_macro_f1']:.4f}"
        train_acc = f"{r['train_accuracy']:.4f}±{r['std_train_accuracy']:.4f}"
        val_acc = f"{r['val_accuracy']:.4f}±{r['std_val_accuracy']:.4f}"
        gap = f"{r['gap']:.4f}±{r['std_gap']:.4f}"
        epoch = "-" if r["best_epoch"] is None else f"{r['best_epoch']:.1f}"
        seeds = r.get("seed_count", 0)
        print(
            f"{r['name']:<25} {r['desc']:<28} {lr_text:>8} {best_f1:>16} "
            f"{train_acc:>17} {val_acc:>15} {gap:>12} {epoch:>6} {seeds:>6}"
        )
    print("-" * 140)


def _resolve_targets(only_keys: Optional[List[str]]) -> List[str]:
    if not only_keys:
        return TARGET_KEYS
    unknown = [k for k in only_keys if k not in ablation.EXPERIMENTS]
    if unknown:
        raise ValueError(f"Unknown experiment keys: {', '.join(unknown)}")
    return only_keys


def main() -> None:
    parser = argparse.ArgumentParser(description="Run subset experiments and compare LR stability")
    parser.add_argument("--skip-completed", action="store_true", help="跳过已完成的实验")
    parser.add_argument("--analyze", action="store_true", help="只分析结果，不运行实验")
    parser.add_argument("--only", nargs="+", choices=list(ablation.EXPERIMENTS.keys()),
                        help="只运行指定的实验")
    parser.add_argument("--seeds", type=str, default=str(ablation.RUN_SEED),
                        help="随机种子列表，用逗号分隔 (如 42,43,44)")
    parser.add_argument("--seed", type=int, default=None, help="单个随机种子（优先于 --seeds）")
    args = parser.parse_args()

    seeds = _parse_seeds(args.seed, args.seeds)
    target_keys = _resolve_targets(args.only)

    if args.analyze:
        analyze_subset(target_keys)
        return

    results: Dict[str, List[str]] = {k: [] for k in target_keys}
    for i, key in enumerate(target_keys, 1):
        exp = ablation.EXPERIMENTS[key]
        for seed in seeds:
            if args.skip_completed and ablation.is_completed_seed(key, seed):
                print(f"\n[{i}/{len(TARGET_KEYS)}] {exp['name']} (seed={seed}) - 已完成，跳过")
                results[key].append(f"seed {seed}: 跳过")
                continue
            print(f"\n[{i}/{len(TARGET_KEYS)}] 运行 {exp['name']} (seed={seed})...")
            success = ablation.run_experiment(key, seed)
            results[key].append(f"seed {seed}: {'成功' if success else '失败'}")
            if not success:
                print(f"\n实验 {key} (seed={seed}) 失败，是否继续? (y/n)")
                try:
                    if input().strip().lower() != "y":
                        break
                except EOFError:
                    break

    print("\n" + "=" * 60)
    print("运行结果汇总:")
    print("=" * 60)
    for key, status in results.items():
        joined = ", ".join(status) if status else "-"
        print(f"  {ablation.EXPERIMENTS[key]['name']}: {joined}")

    analyze_subset(target_keys)


if __name__ == "__main__":
    main()
