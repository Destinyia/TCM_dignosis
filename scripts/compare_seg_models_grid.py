#!/usr/bin/env python
"""Compare masks from two segmentation models on a 4x4 sample grid."""
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
sys.path.insert(0, str(PROJECT_ROOT / "Tongue_segment"))

from tcm_tongue.data.classification_dataset import TongueClassificationDataset  # noqa: E402
from tcm_tongue.models.seg_attention import SegmentationEncoder  # noqa: E402
from models.unet import ResUNet1, ResUNet2, ResUNet2Dist  # noqa: E402


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _predict_mask(
    image_np: np.ndarray,
    model,
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
        pred = model(img_t)
        if isinstance(pred, (tuple, list)):
            pred = pred[0]
    pred = pred.squeeze(0).squeeze(0).cpu().numpy()
    pred = np.clip(pred, 0.0, 1.0)
    if pred.shape[0] != h + pad_h or pred.shape[1] != w + pad_w:
        pred = cv2.resize(pred, (w + pad_w, h + pad_h), interpolation=cv2.INTER_LINEAR)
    if pad_h > 0 or pad_w > 0:
        pred = pred[:h, :w]
    return (pred > threshold).astype(np.uint8)


def _build_new_model(state_dict: dict, override: str | None):
    if override:
        name = override.lower()
        if name == "resunet2dist":
            return ResUNet2Dist(num_classes=33)
        if name == "resunet2":
            return ResUNet2(num_classes=33)
        if name == "resunet1":
            return ResUNet1(num_classes=6)
        raise ValueError(f"Unknown new model: {override}")

    if "dist_head.weight" in state_dict:
        return ResUNet2Dist(num_classes=33)
    fc3 = state_dict.get("fc.3.weight")
    if fc3 is not None:
        if fc3.shape[0] == 32:
            return ResUNet1(num_classes=6)
        if fc3.shape[0] == 64:
            return ResUNet2(num_classes=33)
    return ResUNet1(num_classes=6)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two segmentation models")
    parser.add_argument("--data-root", type=str, default="datasets/shezhenv3-coco")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="docs/seg_model_compare_grid.png")
    parser.add_argument("--rows", type=int, default=4)
    parser.add_argument("--cols", type=int, default=4)
    parser.add_argument("--random", action="store_true")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--old-weights", type=str, default="Tongue_segment/weights/tongue05.pt")
    parser.add_argument("--new-weights", type=str, required=True)
    parser.add_argument("--new-model", type=str, default=None,
                        choices=["resunet1", "resunet2", "resunet2dist"],
                        help="Override new model type (auto-detect by default)")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    _set_seed(args.seed)

    dataset = TongueClassificationDataset(
        root=args.data_root,
        split=args.split,
        transform=None,
        image_size=(640, 640),
    )

    def _resolve(path_str: str) -> Path:
        p = Path(path_str)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        return p

    old_path = _resolve(args.old_weights)
    new_path = _resolve(args.new_weights)
    if not old_path.exists():
        raise FileNotFoundError(f"Old weights not found: {old_path}")
    if not new_path.exists():
        raise FileNotFoundError(f"New weights not found: {new_path}")

    old_encoder = SegmentationEncoder(weights_path=str(old_path), freeze=True).to(args.device)
    old_encoder.eval()

    checkpoint = torch.load(str(new_path), map_location="cpu")
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint
    new_model = _build_new_model(state_dict, args.new_model).to(args.device)
    new_model.load_state_dict(state_dict, strict=False)
    new_model.eval()

    total = args.rows * args.cols
    if args.random:
        indices = random.sample(range(len(dataset)), k=total)
    else:
        indices = [(args.index + i) % len(dataset) for i in range(total)]

    fig, axes = plt.subplots(args.rows, args.cols * 3, figsize=(args.cols * 4.5, args.rows * 3.0))
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0.01, hspace=0.01)

    def _ax_at(r, c):
        if args.rows == 1:
            return axes[c]
        return axes[r][c]

    for i, idx in enumerate(indices):
        info = dataset.valid_images[idx]
        image_path = os.path.join(dataset.image_dir, info["file_name"])
        image = Image.open(image_path).convert("RGB")
        image_np = np.array(image)

        old_mask = _predict_mask(image_np, old_encoder, args.device, args.threshold)
        new_mask = _predict_mask(image_np, new_model, args.device, args.threshold)

        row = i // args.cols
        col = i % args.cols
        ax_img = _ax_at(row, col)
        ax_old = _ax_at(row, col + args.cols)
        ax_new = _ax_at(row, col + args.cols * 2)

        ax_img.imshow(image_np)
        ax_img.axis("off")
        ax_img.set_title(f"{info['file_name']}", fontsize=7)

        ax_old.imshow(old_mask, cmap="gray", vmin=0, vmax=1)
        ax_old.axis("off")
        ax_old.set_title("old mask", fontsize=7)

        ax_new.imshow(new_mask, cmap="gray", vmin=0, vmax=1)
        ax_new.axis("off")
        ax_new.set_title("new mask", fontsize=7)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved grid to: {output_path}")


if __name__ == "__main__":
    main()
