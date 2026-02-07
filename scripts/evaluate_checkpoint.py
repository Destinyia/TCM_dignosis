from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from tcm_tongue.config import Config
from tcm_tongue.data import ValTransform, TongueCocoDataset
from tcm_tongue.engine import COCOEvaluator
from tcm_tongue.models import build_detector


def _subset_dataset(dataset, size: int | None):
    if size is None:
        return dataset
    size = min(size, len(dataset))
    indices = list(range(size))
    return Subset(dataset, indices)


def build_val_loader(cfg: Config, subset_size: int | None):
    normalize = getattr(cfg.data, "normalize", True)
    resize = getattr(cfg.data, "resize_in_dataset", True)
    image_size = getattr(cfg.data, "image_size", (800, 800))
    if isinstance(image_size, list):
        image_size = tuple(image_size)

    transform = ValTransform(image_size=image_size, normalize=normalize, resize=resize)
    dataset = TongueCocoDataset(
        root=cfg.data.root,
        split=cfg.data.val_split,
        transforms=transform,
        label_offset=getattr(cfg.data, "label_offset", 0),
    )
    dataset = _subset_dataset(dataset, subset_size)

    loader = DataLoader(
        dataset,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        collate_fn=dataset.dataset.collate_fn if isinstance(dataset, Subset) else dataset.collate_fn,
    )
    return loader


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--val-size", type=int, default=None)
    parser.add_argument("--image-size", nargs=2, type=int, default=None)
    parser.add_argument(
        "--output",
        default=None,
        help="Output metrics path (default: <run-dir>/metrics_best.json).",
    )
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config)
    if args.batch_size is not None:
        cfg.data.batch_size = args.batch_size
    if args.num_workers is not None:
        cfg.data.num_workers = args.num_workers
    if args.image_size is not None:
        cfg.data.image_size = list(args.image_size)

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    val_loader = build_val_loader(cfg, subset_size=args.val_size)
    model = build_detector(cfg)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.to(device)

    evaluator = COCOEvaluator(val_loader.dataset)
    metrics = evaluator.evaluate(model, val_loader, device=device)
    metrics["checkpoint"] = str(args.checkpoint)
    metrics["config"] = str(args.config)

    output_path = Path(args.output) if args.output else Path(args.checkpoint).parent / "metrics_best.json"
    output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
