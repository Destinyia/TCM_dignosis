from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt


def _load_history(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    if not path.is_file():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    rows.sort(key=lambda r: r.get("epoch", 0))
    return rows


def _plot_curves(rows: List[Dict], out_path: Path) -> None:
    epochs = [r.get("epoch") for r in rows]
    train_loss = [r.get("train_train_loss") for r in rows]
    train_map50 = [r.get("train_mAP_50") for r in rows]
    val_map = [r.get("val_mAP") for r in rows]
    val_map50 = [r.get("val_mAP_50") for r in rows]

    fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

    axes[0].plot(epochs, train_loss, label="train_loss")
    axes[0].set_ylabel("loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(epochs, train_map50, label="train_mAP50")
    axes[1].plot(epochs, val_map50, label="val_mAP50")
    axes[1].plot(epochs, val_map, label="val_mAP")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("mAP")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _collect_per_class(metrics: Dict) -> List[Tuple[str, float, float, float, float]]:
    ap = metrics.get("per_class_AP", {}) or {}
    ap50 = metrics.get("per_class_AP50", {}) or {}
    ar = metrics.get("per_class_AR", {}) or {}
    ar50 = metrics.get("per_class_AR50", {}) or {}
    names = sorted(set(ap.keys()) | set(ap50.keys()) | set(ar.keys()) | set(ar50.keys()))
    rows: List[Tuple[str, float, float, float, float]] = []
    for name in names:
        rows.append(
            (
                name,
                float(ap.get(name, 0.0)),
                float(ap50.get(name, 0.0)),
                float(ar.get(name, 0.0)),
                float(ar50.get(name, 0.0)),
            )
        )
    return rows


def _save_per_class(rows: List[Tuple[str, float, float, float, float]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        f.write("class,AP,AP50,AR,AR50\n")
        for name, ap, ap50, ar, ar50 in rows:
            f.write(f"{name},{ap:.6f},{ap50:.6f},{ar:.6f},{ar50:.6f}\n")


def _print_top_bottom(rows: List[Tuple[str, float, float, float, float]], top_k: int) -> None:
    rows_by_ap = sorted(rows, key=lambda r: r[1], reverse=True)
    rows_by_ar = sorted(rows, key=lambda r: r[3], reverse=True)
    rows_by_ap50 = sorted(rows, key=lambda r: r[2], reverse=True)
    rows_by_ar50 = sorted(rows, key=lambda r: r[4], reverse=True)

    print("Top AP:")
    for name, ap, ap50, ar, ar50 in rows_by_ap[:top_k]:
        print(f"  {name:20s} AP={ap:.4f} AP50={ap50:.4f} AR={ar:.4f} AR50={ar50:.4f}")

    print("Bottom AP:")
    for name, ap, ap50, ar, ar50 in rows_by_ap[-top_k:]:
        print(f"  {name:20s} AP={ap:.4f} AP50={ap50:.4f} AR={ar:.4f} AR50={ar50:.4f}")

    print("Top AR:")
    for name, ap, ap50, ar, ar50 in rows_by_ar[:top_k]:
        print(f"  {name:20s} AP={ap:.4f} AP50={ap50:.4f} AR={ar:.4f} AR50={ar50:.4f}")

    print("Bottom AR:")
    for name, ap, ap50, ar, ar50 in rows_by_ar[-top_k:]:
        print(f"  {name:20s} AP={ap:.4f} AP50={ap50:.4f} AR={ar:.4f} AR50={ar50:.4f}")

    print("Top AP50:")
    for name, ap, ap50, ar, ar50 in rows_by_ap50[:top_k]:
        print(f"  {name:20s} AP={ap:.4f} AP50={ap50:.4f} AR={ar:.4f} AR50={ar50:.4f}")

    print("Top AR50:")
    for name, ap, ap50, ar, ar50 in rows_by_ar50[:top_k]:
        print(f"  {name:20s} AP={ap:.4f} AP50={ap50:.4f} AR={ar:.4f} AR50={ar50:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--curve-out",
        default=None,
        help="Output path for curve PNG (default: <run-dir>/summary_curves.png)",
    )
    parser.add_argument(
        "--per-class-out",
        default=None,
        help="Output CSV path (default: <run-dir>/per_class_metrics.csv)",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    history_path = run_dir / "metrics_history.jsonl"
    metrics_path = run_dir / "metrics.json"

    rows = _load_history(history_path)
    if rows:
        curve_out = Path(args.curve_out) if args.curve_out else run_dir / "summary_curves.png"
        _plot_curves(rows, curve_out)
        print(f"Saved curves: {curve_out}")
    else:
        print(f"No history found: {history_path}")

    if metrics_path.is_file():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        per_class = _collect_per_class(metrics)
        if per_class:
            per_class_out = (
                Path(args.per_class_out) if args.per_class_out else run_dir / "per_class_metrics.csv"
            )
            _save_per_class(per_class, per_class_out)
            print(f"Saved per-class metrics: {per_class_out}")
            _print_top_bottom(per_class, args.top_k)
        else:
            print("No per-class metrics found in metrics.json.")
    else:
        print(f"No metrics.json found: {metrics_path}")


if __name__ == "__main__":
    main()
