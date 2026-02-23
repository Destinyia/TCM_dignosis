#!/usr/bin/env python
"""Save a grid of augmented samples for visual inspection."""
from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from PIL import Image

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
except Exception as exc:  # pragma: no cover
    raise SystemExit(f"albumentations is required: {exc}")

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

from tcm_tongue.data.classification_dataset import (  # noqa: E402
    MaskAugmentation,
    TongueClassificationDataset,
)
from tcm_tongue.models.seg_attention import SegmentationEncoder  # noqa: E402


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _load_image_and_mask(
    dataset: TongueClassificationDataset,
    idx: int,
    use_mask: bool,
    mask_source: str,
    seg_encoder: Optional[SegmentationEncoder],
    device: str,
) -> Tuple[np.ndarray, Optional[np.ndarray], str]:
    image_info = dataset.valid_images[idx]
    image_id = image_info["id"]
    image_path = os.path.join(dataset.image_dir, image_info["file_name"])
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)

    mask = None
    if use_mask:
        h, w = image_np.shape[:2]
        if mask_source == "seg_model":
            if seg_encoder is None:
                raise RuntimeError("seg_encoder is required when mask_source=seg_model")
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
            mask = (pred > 0.5).astype(np.uint8)
        else:
            anns = dataset._annotations_by_image_id.get(image_id, [])
            mask = dataset._get_segmentation_mask(anns, h, w, image_info["file_name"])
    return image_np, mask, image_info["file_name"]


def _denorm(img: np.ndarray) -> np.ndarray:
    img = (img * IMAGENET_STD) + IMAGENET_MEAN
    return np.clip(img, 0.0, 1.0)


def _rotate_with_noise(
    image: np.ndarray,
    mask: Optional[np.ndarray],
    angle: float,
    fill_mode: str,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    h, w = image.shape[:2]
    center = (w / 2.0, h / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    rotated = cv2.warpAffine(
        image,
        matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    rotated_mask = None
    if mask is not None:
        rotated_mask = cv2.warpAffine(
            mask,
            matrix,
            (w, h),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

    valid = cv2.warpAffine(
        np.ones((h, w), dtype=np.uint8),
        matrix,
        (w, h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    if fill_mode == "noise":
        noise = np.random.randint(0, 256, image.shape, dtype=np.uint8)
        rotated[valid == 0] = noise[valid == 0]
    else:
        rotated[valid == 0] = 114

    return rotated, rotated_mask


def _build_replay_transform(args) -> A.ReplayCompose:
    ops = []

    if args.aug_scale and not args.aug_affine:
        ops.append(A.RandomScale(scale_limit=args.aug_scale_limit, p=args.aug_scale_prob))

    ops.extend([
        A.LongestMaxSize(max_size=args.image_size),
        A.PadIfNeeded(
            min_height=args.image_size,
            min_width=args.image_size,
            border_mode=0,
            fill=(114, 114, 114),
        ),
    ])

    ops.append(A.HorizontalFlip(p=0.5))
    ops.append(A.RandomBrightnessContrast(p=0.3))
    ops.append(A.HueSaturationValue(
        hue_shift_limit=10,
        sat_shift_limit=20,
        val_shift_limit=20,
        p=0.3,
    ))

    if args.aug_affine:
        ops.append(A.Affine(
            translate_percent=0.1,
            scale=(0.8, 1.2),
            rotate=(-15, 15),
            border_mode=0,
            fill=(114, 114, 114),
            p=0.3,
        ))

    if args.aug_gauss_noise:
        ops.append(A.GaussNoise(p=0.3))

    if args.strong_aug:
        ops.extend([
            A.Affine(
                translate_percent=0.1,
                scale=(0.8, 1.2),
                rotate=(-15, 15),
                border_mode=0,
                fill=(114, 114, 114),
                p=0.3,
            ),
            A.GaussNoise(p=0.1),
            A.RandomGamma(p=0.2),
            A.CoarseDropout(
                max_holes=8,
                max_height=32,
                max_width=32,
                fill_value=114,
                p=0.2,
            ),
        ])

    ops.append(A.Normalize(mean=IMAGENET_MEAN.tolist(), std=IMAGENET_STD.tolist()))
    ops.append(ToTensorV2())

    return A.ReplayCompose(ops)


def _extract_scale_from_replay(replay: dict) -> Optional[float]:
    for t in replay.get("transforms", []):
        if "RandomScale" in t.get("__class_fullname__", ""):
            if not t.get("applied", False):
                return None
            params = t.get("params", {})
            if "scale" in params:
                return float(params["scale"])
            if "scale_factor" in params:
                return float(params["scale_factor"])
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize augmentation grid")
    parser.add_argument("--data-root", type=str, default="datasets/shezhenv3-coco")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test"])
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--index", type=int, default=0, help="Image index in dataset")
    parser.add_argument("--random", action="store_true", help="Pick a random image index")
    parser.add_argument("--rows", type=int, default=4)
    parser.add_argument("--cols", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="docs/aug_grid.png")
    parser.add_argument("--output-stem", type=str, default=None,
                        help="If set, write outputs to docs/{stem}_img.png and docs/{stem}_mask.png")
    parser.add_argument("--mask-source", type=str, default="dataset",
                        choices=["dataset", "seg_model"],
                        help="Mask source: dataset mask or segmentation model")
    parser.add_argument("--seg-weights", type=str, default="Tongue_segment/weights/tongue05.pt",
                        help="Segmentation model weights (for mask-source=seg_model)")
    parser.add_argument("--device", type=str, default="cpu", help="Device for segmentation model")

    # mask-aug options
    parser.add_argument("--mask-aug", action="store_true")
    parser.add_argument("--mask-aug-mode", type=str, default="background", choices=["crop", "background", "both"])
    parser.add_argument("--mask-aug-prob", type=float, default=0.5)
    parser.add_argument("--mask-aug-crop-padding", type=float, default=0.1)
    parser.add_argument("--mask-aug-bg-mode", type=str, default="blur", choices=["solid", "blur", "noise", "random"])
    parser.add_argument("--mask-aug-bg-color", type=str, default="gray", choices=["gray", "random"])
    parser.add_argument("--mask-aug-bg-blur", type=float, default=12.0)
    parser.add_argument("--mask-aug-dilate", type=int, default=15)

    # transform options
    parser.add_argument("--strong-aug", action="store_true")
    parser.add_argument("--aug-scale", action="store_true")
    parser.add_argument("--aug-scale-limit", type=float, default=0.2)
    parser.add_argument("--aug-scale-prob", type=float, default=0.3)
    parser.add_argument("--aug-affine", action="store_true")
    parser.add_argument("--aug-rotate", action="store_true")
    parser.add_argument("--aug-rotate-limit", type=float, default=15.0)
    parser.add_argument("--aug-rotate-prob", type=float, default=0.3)
    parser.add_argument("--aug-rotate-fill", type=str, default="gray", choices=["gray", "noise"])
    parser.add_argument("--aug-gauss-noise", action="store_true")
    args = parser.parse_args()

    _set_seed(args.seed)

    dataset = TongueClassificationDataset(
        root=args.data_root,
        split=args.split,
        transform=None,
        image_size=(args.image_size, args.image_size),
    )

    idx = random.randrange(len(dataset)) if args.random else args.index
    use_mask = args.mask_aug
    seg_encoder = None
    if use_mask and args.mask_source == "seg_model":
        seg_encoder = SegmentationEncoder(weights_path=args.seg_weights, freeze=True)
        seg_encoder = seg_encoder.to(args.device)
        seg_encoder.eval()

    image_np, mask, filename = _load_image_and_mask(
        dataset, idx, use_mask, args.mask_source, seg_encoder, args.device
    )
    print(f"Using image index={idx}, file={filename}")

    mask_aug = None
    if args.mask_aug:
        bg_color = (114, 114, 114) if args.mask_aug_bg_color == "gray" else "random"
        mask_aug = MaskAugmentation(
            mode=args.mask_aug_mode,
            prob=1.0,
            crop_padding=args.mask_aug_crop_padding,
            bg_mode=args.mask_aug_bg_mode,
            bg_color=bg_color,
            bg_blur_radius=args.mask_aug_bg_blur,
            mask_dilate=args.mask_aug_dilate,
        )

    transform = _build_replay_transform(args)

    total = args.rows * args.cols
    fig, axes = plt.subplots(
        args.rows,
        args.cols * 2,
        figsize=(args.cols * 5.0, args.rows * 2.5),
    )
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0.002, hspace=0.002)

    def _ax_at(r: int, c: int):
        if args.rows == 1:
            return axes[c]
        return axes[r][c]

    samples = []

    for i in range(total):
        img = image_np.copy()
        mask_i = mask.copy() if mask is not None else None
        mask_applied = False
        if mask_aug is not None and mask_i is not None:
            if random.random() < args.mask_aug_prob:
                img, mask_i = mask_aug(img, mask_i)
                mask_applied = True

        angle = None
        if args.aug_rotate and random.random() < args.aug_rotate_prob:
            angle = random.uniform(-args.aug_rotate_limit, args.aug_rotate_limit)
            img, mask_i = _rotate_with_noise(img, mask_i, angle, args.aug_rotate_fill)

        if mask_i is not None:
            transformed = transform(image=img, mask=mask_i)
        else:
            transformed = transform(image=img)
        scale = _extract_scale_from_replay(transformed.get("replay", {})) if args.aug_scale else None
        img_t = transformed["image"].permute(1, 2, 0).cpu().numpy()
        img_vis = _denorm(img_t)

        lines = []
        if args.mask_aug:
            lines.append(f"mask={'on' if mask_applied else 'off'}")
            lines.append(f"mode={args.mask_aug_mode}")
            lines.append(f"bg={args.mask_aug_bg_mode}")
            lines.append(f"dil={args.mask_aug_dilate}")
        if args.aug_rotate:
            if angle is None:
                lines.append("rot=0")
            else:
                lines.append(f"rot={angle:.2f}")
        if args.aug_scale:
            if scale is None:
                lines.append("scale=1.000")
            else:
                lines.append(f"scale={scale:.3f}")

        mask_bin = None
        if mask_applied and mask_i is not None:
            mask_t = transformed.get("mask")
            if mask_t is not None:
                mask_np = mask_t.cpu().numpy()
                if mask_np.ndim == 3:
                    mask_np = mask_np[0]
                mask_bin = (mask_np > 0.5).astype(np.float32)

        samples.append({
            "img": img_vis,
            "mask": mask_bin,
            "lines": lines,
        })

        row = i // args.cols
        col = i % args.cols
        ax_img = _ax_at(row, col)
        ax_img.imshow(img_vis)
        if lines:
            ax_img.text(
                0.02, 0.02,
                "\n".join(lines),
                transform=ax_img.transAxes,
                fontsize=7,
                color="white",
                va="bottom",
                ha="left",
                bbox=dict(facecolor="black", alpha=0.5, pad=1),
            )
        ax_img.axis("off")

        ax_mask = _ax_at(row, col + args.cols)
        if mask_bin is not None:
            ax_mask.imshow(mask_bin, cmap="gray", vmin=0, vmax=1)
        ax_mask.axis("off")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(pad=0)
    if args.output_stem:
        base = Path("docs")
        base.mkdir(parents=True, exist_ok=True)
        img_path = base / f"{args.output_stem}_img.png"
        mask_path = base / f"{args.output_stem}_mask.png"

        fig_img, axes_img = plt.subplots(args.rows, args.cols, figsize=(args.cols * 2.5, args.rows * 2.5))
        fig_mask, axes_mask = plt.subplots(args.rows, args.cols, figsize=(args.cols * 2.5, args.rows * 2.5))
        fig_img.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0.002, hspace=0.002)
        fig_mask.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0.002, hspace=0.002)
        for i, sample in enumerate(samples):
            row = i // args.cols
            col = i % args.cols
            ax_img = axes_img[row][col] if args.rows > 1 else axes_img[col]
            ax_mask = axes_mask[row][col] if args.rows > 1 else axes_mask[col]
            ax_img.imshow(sample["img"])
            ax_img.axis("off")
            if sample["mask"] is not None:
                ax_mask.imshow(sample["mask"], cmap="gray", vmin=0, vmax=1)
            ax_mask.axis("off")
        fig_img.tight_layout(pad=0)
        fig_mask.tight_layout(pad=0)
        fig_img.savefig(img_path, dpi=300)
        fig_mask.savefig(mask_path, dpi=300)
        plt.close(fig_img)
        plt.close(fig_mask)
        print(f"Saved image grid to: {img_path}")
        print(f"Saved mask grid to: {mask_path}")
    else:
        plt.savefig(output_path, dpi=300)
        print(f"Saved grid to: {output_path}")
    plt.close()


if __name__ == "__main__":
    main()
