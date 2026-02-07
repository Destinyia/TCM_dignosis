from __future__ import annotations

import argparse
import datetime
import subprocess

EXPERIMENTS = [
    "baseline",
    "focal_loss",
    "weighted_ce",
    "weighted_ce_sqrt",
    "oversampling",
    "oversample_focal",
    "stratified",
    "class_balanced_focal",
    "strong_augment",
    "tcm_prior_augment",
    "combined",
]


def run_experiment(config_name: str, extra_args: list[str]):
    config_path = f"configs/experiments/{config_name}.yaml"
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"runs/experiments/{config_name}_{timestamp}"

    cmd = [
        "python",
        "scripts/train.py",
        "--config",
        config_path,
        "--output-dir",
        output_dir,
    ] + extra_args

    subprocess.run(cmd, check=False)


def _parse_list(value: str | None) -> list[str]:
    if not value:
        return []
    parts = [item.strip() for item in value.split(",")]
    return [item for item in parts if item]


def _build_extra_args(args: argparse.Namespace) -> list[str]:
    extra: list[str] = []
    for name, key in [
        ("--epochs", "epochs"),
        ("--device", "device"),
        ("--batch-size", "batch_size"),
        ("--num-workers", "num_workers"),
        ("--train-size", "train_size"),
        ("--val-size", "val_size"),
        ("--seed", "seed"),
        ("--image-size", "image_size"),
    ]:
        value = getattr(args, key)
        if value is not None:
            if key == "image_size" and isinstance(value, (list, tuple)):
                extra.extend([name] + [str(v) for v in value])
            else:
                extra.extend([name, str(value)])
    return extra


def run_all_experiments(only: list[str], skip: list[str], extra_args: list[str]):
    if only:
        selected = [name for name in EXPERIMENTS if name in set(only)]
        missing = sorted(set(only) - set(EXPERIMENTS))
        if missing:
            raise ValueError(f"Unknown experiments: {', '.join(missing)}")
    else:
        selected = EXPERIMENTS[:]

    if skip:
        selected = [name for name in selected if name not in set(skip)]

    for exp_name in selected:
        print(f"Running experiment: {exp_name}")
        run_experiment(exp_name, extra_args)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--only",
        default="baseline",
        help="Comma-separated experiment names to run (default: baseline)",
    )
    parser.add_argument("--skip", help="Comma-separated experiment names to skip")
    parser.add_argument("--list", action="store_true", help="List available experiments")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--train-size", type=int, default=4000)
    parser.add_argument("--val-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-size", nargs=2, type=int, default=[1024, 1024])
    args = parser.parse_args()

    if args.list:
        print("\n".join(EXPERIMENTS))
        return

    only = _parse_list(args.only)
    skip = _parse_list(args.skip)
    extra_args = _build_extra_args(args)
    run_all_experiments(only=only, skip=skip, extra_args=extra_args)


if __name__ == "__main__":
    main()
