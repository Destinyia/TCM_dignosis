#!/usr/bin/env python
"""Regenerate dataset masks using segmentation model."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image, ImageFile

try:
    import cv2
except Exception as exc:  # pragma: no cover
    raise SystemExit(f"opencv-python is required: {exc}")

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "Tongue_segment"))

from tcm_tongue.data.classification_dataset import TongueClassificationDataset  # noqa: E402
from tcm_tongue.models.seg_attention import SegmentationEncoder  # noqa: E402
from util import apply_dense_crf  # noqa: E402
from models.unet import ResUNet1, ResUNet2, ResUNet2Dist  # noqa: E402

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None


def _iter_splits(arg: str) -> Iterable[str]:
    for item in arg.split(","):
        name = item.strip()
        if name:
            yield name


def _predict_mask(
    image_np: np.ndarray,
    model,
    device: str,
    threshold: float,
    use_crf: bool,
    crf_params: dict,
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
    prob = pred
    if use_crf:
        prob = apply_dense_crf(image_np, prob, **crf_params)
    return (prob > threshold).astype(np.uint8)


def _infer_model(state_dict: dict) -> str:
    if "dist_head.weight" in state_dict:
        return "resunet2dist"
    fc3 = state_dict.get("fc.3.weight")
    if fc3 is not None:
        if fc3.shape[0] == 32:
            return "resunet1"
        if fc3.shape[0] == 64:
            return "resunet2"
    return "resunet1"


def _load_state_dict_safely(model, state_dict: dict) -> None:
    model_state = model.state_dict()
    cleaned = {}
    for key, value in state_dict.items():
        k = key
        if k.startswith("module."):
            k = k[7:]
        if k in model_state and model_state[k].shape == value.shape:
            cleaned[k] = value
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if missing:
        print(f"Warning: missing keys when loading: {missing}")
    if unexpected:
        print(f"Warning: unexpected keys when loading: {unexpected}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate masks using seg model")
    parser.add_argument("--data-root", type=str, default="datasets/shezhenv3-coco")
    parser.add_argument("--splits", type=str, default="train,val",
                        help="Comma-separated splits to process (train,val,test)")
    parser.add_argument("--seg-weights", type=str, default="Tongue_segment/weights/tongue05.pt")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--model", type=str, default="auto",
                        choices=["auto", "seg_encoder", "resunet1", "resunet2", "resunet2dist"],
                        help="Model type for new weights (auto to infer)")
    parser.add_argument("--use-crf", action="store_true",
                        help="Apply DenseCRF post-processing")
    parser.add_argument("--crf-iter", type=int, default=5)
    parser.add_argument("--crf-sxy-gaussian", type=int, default=3)
    parser.add_argument("--crf-compat-gaussian", type=int, default=3)
    parser.add_argument("--crf-sxy-bilateral", type=int, default=80)
    parser.add_argument("--crf-srgb-bilateral", type=int, default=13)
    parser.add_argument("--crf-compat-bilateral", type=int, default=10)
    parser.add_argument("--output-root", type=str, default=None,
                        help="If set, write masks to {output_root}/{split}/masks")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing masks")
    parser.add_argument("--limit", type=int, default=0,
                        help="Limit number of images per split (0=all)")
    args = parser.parse_args()

    seg_weights_path = Path(args.seg_weights)
    if not seg_weights_path.is_absolute():
        seg_weights_path = PROJECT_ROOT / seg_weights_path
    if not seg_weights_path.exists():
        raise FileNotFoundError(f"Seg weights not found: {seg_weights_path}")

    model = None
    if args.model == "seg_encoder":
        model = SegmentationEncoder(weights_path=str(seg_weights_path), freeze=True)
        model = model.to(args.device)
        model.eval()
    else:
        checkpoint = torch.load(str(seg_weights_path), map_location="cpu")
        if isinstance(checkpoint, dict) and "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint

        model_name = args.model
        if model_name == "auto":
            model_name = _infer_model(state_dict)

        if model_name == "resunet2dist":
            model = ResUNet2Dist(num_classes=33)
        elif model_name == "resunet2":
            model = ResUNet2(num_classes=33)
        else:
            model = ResUNet1(num_classes=6)

        _load_state_dict_safely(model, state_dict)
        model = model.to(args.device)
        model.eval()

    for split in _iter_splits(args.splits):
        dataset = TongueClassificationDataset(
            root=args.data_root,
            split=split,
            transform=None,
            image_size=(640, 640),
        )
        if args.output_root:
            out_dir = Path(args.output_root) / split / "masks"
        else:
            out_dir = Path(args.data_root) / split / "masks"
        out_dir.mkdir(parents=True, exist_ok=True)

        indices = list(range(len(dataset)))
        if args.limit and args.limit > 0:
            indices = indices[: args.limit]

        iterator = tqdm(indices, desc=f"{split} masks") if tqdm else indices
        crf_params = {
            "iterations": args.crf_iter,
            "sxy_gaussian": args.crf_sxy_gaussian,
            "compat_gaussian": args.crf_compat_gaussian,
            "sxy_bilateral": args.crf_sxy_bilateral,
            "srgb_bilateral": args.crf_srgb_bilateral,
            "compat_bilateral": args.crf_compat_bilateral,
        }

        for idx in iterator:
            info = dataset.valid_images[idx]
            image_path = os.path.join(dataset.image_dir, info["file_name"])
            try:
                ImageFile.LOAD_TRUNCATED_IMAGES = True
                image = Image.open(image_path).convert("RGB")
                image_np = np.array(image)
            except Exception as exc:
                print(f"Warning: failed to load image {image_path} ({exc}), skipping.")
                continue

            mask = _predict_mask(
                image_np,
                model,
                args.device,
                args.threshold,
                args.use_crf,
                crf_params,
            )

            stem = Path(info["file_name"]).stem
            out_path = out_dir / f"{stem}.png"
            if out_path.exists() and not args.overwrite:
                continue
            Image.fromarray((mask * 255).astype(np.uint8)).save(out_path)

        print(f"Saved masks to: {out_dir}")


if __name__ == "__main__":
    main()
