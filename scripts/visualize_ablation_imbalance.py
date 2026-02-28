#!/usr/bin/env python
"""Generate eval visualizations for all ablation_imbalance runs."""
from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path


def _find_checkpoint(run_dir: Path) -> Path | None:
    best_path = run_dir / "best.pth"
    if best_path.exists():
        return best_path

    epoch_paths = sorted(run_dir.glob("epoch_*.pth"))
    if not epoch_paths:
        return None

    def _epoch_num(p: Path) -> int:
        match = re.search(r"epoch_(\d+)\.pth", p.name)
        return int(match.group(1)) if match else -1

    return max(epoch_paths, key=_epoch_num)


def _already_done(run_dir: Path) -> bool:
    return (run_dir / "confusion_raw.png").exists() and (run_dir / "calibration.png").exists()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate visualizations for ablation_imbalance runs")
    parser.add_argument("--root", type=str, default="runs/ablation_imbalance",
                        help="Root directory containing experiment runs")
    parser.add_argument("--data-root", type=str, default="datasets/shezhenv3-coco",
                        help="Dataset root directory")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"],
                        help="Dataset split to evaluate")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of data loader workers")
    parser.add_argument("--num-classes", type=int, default=13, help="Number of classes")
    parser.add_argument("--backbone", type=str, default="efficientnet_b3", help="Backbone")
    parser.add_argument("--model-type", type=str, default="baseline", help="Model type")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip runs that already have confusion & calibration plots")
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        raise SystemExit(f"Root not found: {root}")

    run_dirs = sorted(root.glob("*/seed_*/baseline"))
    if not run_dirs:
        raise SystemExit(f"No runs found under {root}")

    print(f"Found {len(run_dirs)} runs under {root}")
    for run_dir in run_dirs:
        if args.skip_existing and _already_done(run_dir):
            print(f"[skip] {run_dir}")
            continue

        ckpt = _find_checkpoint(run_dir)
        if ckpt is None:
            print(f"[warn] no checkpoint in {run_dir}")
            continue

        cmd = [
            "python", "scripts/eval_classifier.py",
            "--checkpoint", str(ckpt),
            "--data-root", args.data_root,
            "--split", args.split,
            "--batch-size", str(args.batch_size),
            "--num-workers", str(args.num_workers),
            "--device", args.device,
            "--model-type", args.model_type,
            "--backbone", args.backbone,
            "--num-classes", str(args.num_classes),
            "--plot-confusion",
            "--plot-calibration",
        ]

        print(f"\n[run] {run_dir}")
        subprocess.run(cmd, check=False)


if __name__ == "__main__":
    main()
