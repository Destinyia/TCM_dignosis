#!/usr/bin/env python
"""Compare pre-generated masks with segmentation model masks."""
from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

try:
    import cv2
except Exception as exc:  # pragma: no cover
    raise SystemExit(f"opencv-python is required: {exc}")

try:
    import matplotlib.pyplot as plt
except Exception as exc:  # pragma: no cover
    raise SystemExit(f"matplotlib is required: {exc}")

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tcm_tongue.data.classification_dataset import TongueClassificationDataset  # noqa: E402
from tcm_tongue.models.seg_attention import SegmentationEncoder  # noqa: E402


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _predict_mask(
    image_np: np.ndarray,
    seg_encoder: SegmentationEncoder,
    device: str,
    threshold: float,
) -> np.ndarray:
    h, w = image_np.shape[:2]
    pad_h = (32 - (h % 32)) % 32
    pad_w = (32 - (w % 32)) % 32
    if pad_h > 0 or pad_w > 0:
        image_pad = cv2.copyMakeBorder(
            image_np,
            0,
            pad_h,
            0,
            pad_w,
            borderType=cv2.BORDER_CONSTANT,
            value=(0, 0, 0),
        )
    else:
        image_pad = image_np

    img_t = torch.from_numpy(image_pad).permute(2, 0, 1).float() / 255.0
    img_t = img_t.unsqueeze(0).to(device)
    with torch.no_grad():
        pred = seg_encoder(img_t)
    pred = pred.squeeze(0).squeeze(0).cpu().numpy()
    pred = np.clip(pred, 0.0, 1.0)
    if pred.shape[0] != h + pad_h or pred.shape[1] != w + pad_w:
        pred = cv2.resize(pred, (w + pad_w, h + pad_h), interpolation=cv2.INTER_LINEAR)
    if pad_h > 0 or pad_w > 0:
        pred = pred[:h, :w]
    return (pred > threshold).astype(np.uint8)


def _mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(bool)
    b = b.astype(bool)
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 0.0
    return inter / union


def _diff_overlay(ds_mask: np.ndarray, seg_mask: np.ndarray) -> np.ndarray:
    h, w = ds_mask.shape[:2]
    overlay = np.zeros((h, w, 3), dtype=np.uint8)
    ds_only = np.logical_and(ds_mask == 1, seg_mask == 0)
    seg_only = np.logical_and(seg_mask == 1, ds_mask == 0)
    both = np.logical_and(seg_mask == 1, ds_mask == 1)
    overlay[ds_only] = (255, 0, 0)    # red: dataset only
    overlay[seg_only] = (0, 255, 0)   # green: seg only
    overlay[both] = (255, 255, 255)   # white: overlap
    return overlay


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare dataset masks with seg-model masks")
    parser.add_argument("--data-root", type=str, default="datasets/shezhenv3-coco")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test"])
    parser.add_argument("--rows", type=int, default=4)
    parser.add_argument("--cols", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="docs/mask_compare_grid.png")
    parser.add_argument("--seg-weights", type=str, default="Tongue_segment/weights/tongue05.pt")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--random", action="store_true", default=True, help="Pick random images")
    parser.add_argument("--index", type=int, default=0, help="Start index if not random")
    args = parser.parse_args()

    _set_seed(args.seed)

    dataset = TongueClassificationDataset(
        root=args.data_root,
        split=args.split,
        transform=None,
        image_size=(640, 640),
    )

    seg_weights_path = Path(args.seg_weights)
    if not seg_weights_path.is_absolute():
        seg_weights_path = PROJECT_ROOT / seg_weights_path
    if not seg_weights_path.exists():
        raise FileNotFoundError(f"Seg weights not found: {seg_weights_path}")

    seg_encoder = SegmentationEncoder(weights_path=str(seg_weights_path), freeze=True)
    seg_encoder = seg_encoder.to(args.device)
    seg_encoder.eval()

    total = args.rows
    start = args.index
    if args.random:
        indices = random.sample(range(len(dataset)), k=total)
    else:
        indices = [(start + i) % len(dataset) for i in range(total)]

    fig, axes = plt.subplots(args.rows, args.cols, figsize=(args.cols * 3.0, args.rows * 3.0))
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0.01, hspace=0.01)

    for row, idx in enumerate(indices):
        info = dataset.valid_images[idx]
        image_id = info["id"]
        image_path = os.path.join(dataset.image_dir, info["file_name"])
        image = Image.open(image_path).convert("RGB")
        image_np = np.array(image)
        orig_h, orig_w = image_np.shape[:2]
        anns = dataset._annotations_by_image_id.get(image_id, [])
        ds_mask = dataset._get_segmentation_mask(anns, orig_h, orig_w, info["file_name"])
        seg_mask = _predict_mask(image_np, seg_encoder, args.device, args.threshold)

        iou = _mask_iou(ds_mask, seg_mask)
        overlay = _diff_overlay(ds_mask, seg_mask)

        axes[row][0].imshow(image_np)
        axes[row][0].axis("off")
        axes[row][0].set_title(f"{info['file_name']}", fontsize=7)

        axes[row][1].imshow(ds_mask, cmap="gray", vmin=0, vmax=1)
        axes[row][1].axis("off")
        axes[row][1].set_title("dataset mask", fontsize=7)

        axes[row][2].imshow(seg_mask, cmap="gray", vmin=0, vmax=1)
        axes[row][2].axis("off")
        axes[row][2].set_title("seg mask", fontsize=7)

        axes[row][3].imshow(overlay)
        axes[row][3].axis("off")
        axes[row][3].set_title(f"diff IoU={iou:.2f}", fontsize=7)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved grid to: {output_path}")


if __name__ == "__main__":
    main()
