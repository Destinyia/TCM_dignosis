#!/usr/bin/env python
"""使用 Tongue_segment 分割模型为分类数据集生成舌头 mask"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "Tongue_segment"))

from Tongue_segment.models.unet import UNet, ResUNet, CBAMUNet, ResUNet1, ResUNet2


def load_segmentation_model(
    model_type: str = "resunet1",
    weights_path: str = None,
    device: str = "cuda",
) -> torch.nn.Module:
    """加载分割模型

    Args:
        model_type: 模型类型 (unet, resunet, resunet1, resunet2, cbamunet)
        weights_path: 权重文件路径
        device: 设备
    """
    if model_type == "unet":
        model = UNet()
    elif model_type == "resunet":
        model = ResUNet()
    elif model_type == "resunet1":
        model = ResUNet1(layers=[2, 2, 2])
    elif model_type == "resunet2":
        model = ResUNet2(layers=[2, 2, 2])
    elif model_type == "cbamunet":
        model = CBAMUNet()
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    if weights_path and os.path.exists(weights_path):
        state_dict = torch.load(weights_path, map_location=device)
        # 处理可能的 key 不匹配
        result = model.load_state_dict(state_dict, strict=False)
        if result.missing_keys:
            print(f"Warning: Missing keys: {result.missing_keys[:5]}...")
        if result.unexpected_keys:
            print(f"Warning: Unexpected keys: {result.unexpected_keys[:5]}...")
        print(f"Loaded weights from {weights_path}")
    else:
        print(f"Warning: No weights loaded, using random initialization")

    model = model.to(device)
    model.eval()
    return model


def preprocess_image(image: Image.Image, target_size: tuple = (320, 320)) -> torch.Tensor:
    """预处理图像"""
    # 转换为 numpy
    img_np = np.array(image.convert("RGB"))

    # 转换为 tensor [C, H, W]
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0

    # 添加 batch 维度
    img_tensor = img_tensor.unsqueeze(0)

    # 下采样到目标尺寸
    img_tensor = F.interpolate(img_tensor, size=target_size, mode="bilinear", align_corners=False)

    return img_tensor


def postprocess_mask(
    mask_tensor: torch.Tensor,
    original_size: tuple,
    threshold: float = 0.5,
) -> np.ndarray:
    """后处理 mask"""
    # 上采样到原始尺寸
    mask_tensor = F.interpolate(
        mask_tensor, size=original_size, mode="bilinear", align_corners=False
    )

    # 转换为 numpy
    mask_np = mask_tensor.squeeze().cpu().numpy()

    # 二值化
    mask_binary = (mask_np > threshold).astype(np.uint8)

    return mask_binary


def generate_masks_for_dataset(
    data_root: str,
    split: str,
    model: torch.nn.Module,
    device: str = "cuda",
    input_size: tuple = (320, 320),
    threshold: float = 0.5,
    output_dir: str = None,
    batch_size: int = 1,
):
    """为数据集生成 mask

    Args:
        data_root: 数据集根目录
        split: 数据集划分 (train/val/test)
        model: 分割模型
        device: 设备
        input_size: 模型输入尺寸
        threshold: 二值化阈值
        output_dir: mask 输出目录，默认为 {data_root}/{split}/masks
    """
    from PIL import ImageFile
    ImageFile.LOAD_TRUNCATED_IMAGES = True  # 允许加载截断的图像

    # 路径设置
    image_dir = Path(data_root) / split / "images"
    if output_dir is None:
        mask_dir = Path(data_root) / split / "masks"
    else:
        mask_dir = Path(output_dir)

    mask_dir.mkdir(parents=True, exist_ok=True)

    # 获取所有图像
    image_files = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))
    print(f"Found {len(image_files)} images in {image_dir}")

    # 生成 mask
    model.eval()
    success_count = 0
    error_count = 0

    with torch.no_grad():
        for img_path in tqdm(image_files, desc=f"Generating masks for {split}"):
            try:
                # 加载图像
                image = Image.open(img_path).convert("RGB")
                original_size = (image.height, image.width)

                # 预处理
                img_tensor = preprocess_image(image, input_size).to(device)

                # 推理
                output = model(img_tensor)
                if isinstance(output, tuple):
                    output = output[0]  # 有些模型返回 (mask, pred)

                # 后处理
                mask = postprocess_mask(output, original_size, threshold)

                # 保存 mask
                mask_path = mask_dir / f"{img_path.stem}.png"
                mask_image = Image.fromarray(mask * 255).convert("L")
                mask_image.save(mask_path)
                success_count += 1
            except Exception as e:
                error_count += 1
                if error_count <= 5:  # 只打印前5个错误
                    print(f"\nWarning: Failed to process {img_path.name}: {e}")

    print(f"Saved {success_count} masks to {mask_dir} ({error_count} errors)")
    return mask_dir


def visualize_samples(
    data_root: str,
    split: str,
    mask_dir: str,
    num_samples: int = 5,
    output_path: str = None,
):
    """可视化生成的 mask 样本"""
    import matplotlib.pyplot as plt

    image_dir = Path(data_root) / split / "images"
    mask_dir = Path(mask_dir)

    image_files = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))
    samples = image_files[:num_samples]

    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)

    for i, img_path in enumerate(samples):
        # 加载图像和 mask
        image = Image.open(img_path).convert("RGB")
        mask_path = mask_dir / f"{img_path.stem}.png"

        if mask_path.exists():
            mask = np.array(Image.open(mask_path).convert("L"))
        else:
            mask = np.zeros((image.height, image.width), dtype=np.uint8)

        # 创建叠加图
        image_np = np.array(image)
        overlay = image_np.copy()
        mask_bool = mask > 127
        overlay[~mask_bool] = (overlay[~mask_bool] * 0.3).astype(np.uint8)

        # 绘制
        axes[i, 0].imshow(image_np)
        axes[i, 0].set_title(f"Original: {img_path.name}")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(mask, cmap="gray")
        axes[i, 1].set_title("Generated Mask")
        axes[i, 1].axis("off")

        axes[i, 2].imshow(overlay)
        axes[i, 2].set_title("Overlay")
        axes[i, 2].axis("off")

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150)
        print(f"Saved visualization to {output_path}")
    else:
        plt.show()

    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Generate tongue segmentation masks")
    parser.add_argument(
        "--data-root",
        type=str,
        default="datasets/shezhenv3-coco",
        help="Dataset root directory",
    )
    parser.add_argument(
        "--splits",
        type=str,
        nargs="+",
        default=["train", "val"],
        help="Dataset splits to process",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="resunet1",
        choices=["unet", "resunet", "resunet1", "resunet2", "cbamunet"],
        help="Segmentation model type",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="Tongue_segment/weights/unet222.pt",
        help="Path to model weights",
    )
    parser.add_argument(
        "--input-size",
        type=int,
        default=320,
        help="Model input size",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Binarization threshold",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Visualize sample results",
    )
    parser.add_argument(
        "--num-vis-samples",
        type=int,
        default=5,
        help="Number of samples to visualize",
    )
    args = parser.parse_args()

    print(f"Device: {args.device}")
    print(f"Model: {args.model_type}")
    print(f"Weights: {args.weights}")

    # 加载模型
    model = load_segmentation_model(
        model_type=args.model_type,
        weights_path=args.weights,
        device=args.device,
    )

    # 为每个 split 生成 mask
    for split in args.splits:
        print(f"\nProcessing {split}...")
        mask_dir = generate_masks_for_dataset(
            data_root=args.data_root,
            split=split,
            model=model,
            device=args.device,
            input_size=(args.input_size, args.input_size),
            threshold=args.threshold,
        )

        # 可视化
        if args.visualize:
            vis_path = f"docs/mask_generation_{split}_samples.png"
            visualize_samples(
                data_root=args.data_root,
                split=split,
                mask_dir=mask_dir,
                num_samples=args.num_vis_samples,
                output_path=vis_path,
            )

    print("\nDone!")


if __name__ == "__main__":
    main()
