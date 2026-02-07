from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import pandas as pd


def load_experiment_results(exp_dir: Path) -> dict:
    metrics_path = exp_dir / "metrics.json"
    if not metrics_path.is_file():
        return {}
    with metrics_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def generate_comparison_table(results: dict) -> pd.DataFrame:
    rows = []
    for exp_name, metrics in results.items():
        rows.append(
            {
                "Experiment": exp_name,
                "mAP": metrics.get("mAP"),
                "mAP@50": metrics.get("mAP_50"),
                "mAP@75": metrics.get("mAP_75"),
                "Head mAP": metrics.get("head_mAP", "-"),
                "Medium mAP": metrics.get("medium_mAP", "-"),
                "Tail mAP": metrics.get("tail_mAP", "-"),
                "Head-Tail Gap": metrics.get("head_tail_gap", "-"),
            }
        )
    return pd.DataFrame(rows)


def plot_class_ap_comparison(results: dict, output_path: Path):
    if not results:
        return
    exp_names = list(results.keys())
    classes = None
    data = {}
    for exp_name, metrics in results.items():
        per_class = metrics.get("per_class_AP", {})
        if not per_class:
            continue
        if classes is None:
            classes = list(per_class.keys())
        data[exp_name] = [per_class.get(cls, 0.0) for cls in classes]

    if not data or classes is None:
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    x = range(len(classes))
    width = 0.8 / max(len(data), 1)

    for idx, (exp_name, values) in enumerate(data.items()):
        ax.bar([i + idx * width for i in x], values, width=width, label=exp_name)

    ax.set_xticks([i + width * (len(data) - 1) / 2 for i in x])
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_ylabel("AP")
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)


def generate_experiment_report(results: dict, output_path: Path):
    table = generate_comparison_table(results)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        f.write("# Experiment Report\n\n")
        f.write(table.to_markdown(index=False))
        f.write("\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="runs/experiments")
    parser.add_argument("--report-path", default="runs/experiments/report.md")
    parser.add_argument("--plot-path", default="runs/experiments/class_ap.png")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    results: Dict[str, dict] = {}
    for exp_dir in output_dir.glob("*"):
        if not exp_dir.is_dir():
            continue
        metrics = load_experiment_results(exp_dir)
        if metrics:
            results[exp_dir.name] = metrics

    generate_experiment_report(results, Path(args.report_path))
    plot_class_ap_comparison(results, Path(args.plot_path))


if __name__ == "__main__":
    main()
