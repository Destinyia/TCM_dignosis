#!/usr/bin/env python
"""可视化 mask 增强效果"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tcm_tongue.data.classification_dataset import (
    TongueClassificationDataset,
    ClassificationTransform,
    MaskAugmentation,
)


def visualize_mask_augmentation(
    data_root: str,
    num_samples: int = 5,
    output_path: str = None,
):
    """可视化不同 mask 增强模式的效果"""

    # 创建不同增强模式的数据集
    transform = ClassificationTransform(image_size=(640, 640), is_train=False, normalize=False)

    # 无增强（原图 + mask）
    dataset_no_aug = TongueClassificationDataset(
        root=data_root,
        split="train",
        transform=transform,
        return_mask=True,
        mask_aug=None,
    )

    # Crop 增强
    mask_aug_crop = MaskAugmentation(mode="crop", prob=1.0, crop_padding=0.1)
    dataset_crop = TongueClassificationDataset(
        root=data_root,
        split="train",
        transform=transform,
        return_mask=True,
        mask_aug=mask_aug_crop,
    )

    # Background 增强（无膨胀）
    mask_aug_bg_no_dilate = MaskAugmentation(mode="background", prob=1.0, bg_mode="blur", bg_blur_radius=15, mask_dilate=0)
    dataset_bg_no_dilate = TongueClassificationDataset(
        root=data_root,
        split="train",
        transform=transform,
        return_mask=True,
        mask_aug=mask_aug_bg_no_dilate,
    )

    # Background 增强（膨胀 15 像素）
    mask_aug_bg = MaskAugmentation(mode="background", prob=1.0, bg_mode="blur", bg_blur_radius=15, mask_dilate=15)
    dataset_bg = TongueClassificationDataset(
        root=data_root,
        split="train",
        transform=transform,
        return_mask=True,
        mask_aug=mask_aug_bg,
    )

    # Background 增强（膨胀 25 像素）
    mask_aug_bg_more = MaskAugmentation(mode="background", prob=1.0, bg_mode="blur", bg_blur_radius=15, mask_dilate=25)
    dataset_bg_more = TongueClassificationDataset(
        root=data_root,
        split="train",
        transform=transform,
        return_mask=True,
        mask_aug=mask_aug_bg_more,
    )

    # 获取类别名称
    class_names = dataset_no_aug.get_class_names()

    # 可视化
    fig, axes = plt.subplots(num_samples, 6, figsize=(24, 4 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)

    # 随机选择样本
    np.random.seed(42)
    indices = np.random.choice(len(dataset_no_aug), num_samples, replace=False)

    for row, idx in enumerate(indices):
        # 原图 + mask
        img_no_aug, label, mask_no_aug = dataset_no_aug[idx]
        img_no_aug = img_no_aug.permute(1, 2, 0).numpy()
        mask_no_aug = mask_no_aug.squeeze().numpy()

        # Crop 增强
        img_crop, _, mask_crop = dataset_crop[idx]
        img_crop = img_crop.permute(1, 2, 0).numpy()

        # Background 增强（无膨胀）
        img_bg_no_dilate, _, _ = dataset_bg_no_dilate[idx]
        img_bg_no_dilate = img_bg_no_dilate.permute(1, 2, 0).numpy()

        # Background 增强（膨胀 15）
        img_bg, _, _ = dataset_bg[idx]
        img_bg = img_bg.permute(1, 2, 0).numpy()

        # Background 增强（膨胀 25）
        img_bg_more, _, _ = dataset_bg_more[idx]
        img_bg_more = img_bg_more.permute(1, 2, 0).numpy()

        # 绘制
        class_name = class_names.get(label, f"class_{label}")

        # 原图
        axes[row, 0].imshow(np.clip(img_no_aug, 0, 1))
        axes[row, 0].set_title(f"Original\n{class_name}")
        axes[row, 0].axis("off")

        # Mask
        axes[row, 1].imshow(mask_no_aug, cmap="gray")
        axes[row, 1].set_title("Generated Mask")
        axes[row, 1].axis("off")

        # Crop 增强
        axes[row, 2].imshow(np.clip(img_crop, 0, 1))
        axes[row, 2].set_title("Crop")
        axes[row, 2].axis("off")

        # Background 无膨胀
        axes[row, 3].imshow(np.clip(img_bg_no_dilate, 0, 1))
        axes[row, 3].set_title("BG Blur (dilate=0)")
        axes[row, 3].axis("off")

        # Background 膨胀 15
        axes[row, 4].imshow(np.clip(img_bg, 0, 1))
        axes[row, 4].set_title("BG Blur (dilate=15)")
        axes[row, 4].axis("off")

        # Background 膨胀 25
        axes[row, 5].imshow(np.clip(img_bg_more, 0, 1))
        axes[row, 5].set_title("BG Blur (dilate=25)")
        axes[row, 5].axis("off")

    plt.suptitle("Mask Augmentation Comparison - Dilate Effect on Edge Preservation", fontsize=14, y=1.02)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved visualization to {output_path}")
    else:
        plt.show()

    plt.close()


def visualize_mask_quality_comparison(
    data_root: str,
    num_samples: int = 5,
    output_path: str = None,
):
    """对比真实 mask 和 bbox mask 的质量"""

    transform = ClassificationTransform(image_size=(640, 640), is_train=False, normalize=False)

    # 使用真实 mask
    dataset_real = TongueClassificationDataset(
        root=data_root,
        split="train",
        transform=transform,
        return_mask=True,
        mask_aug=None,
    )

    class_names = dataset_real.get_class_names()

    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)

    np.random.seed(42)
    indices = np.random.choice(len(dataset_real), num_samples, replace=False)

    for row, idx in enumerate(indices):
        img, label, mask = dataset_real[idx]
        img = img.permute(1, 2, 0).numpy()
        mask = mask.squeeze().numpy()

        class_name = class_names.get(label, f"class_{label}")

        # 原图
        axes[row, 0].imshow(np.clip(img, 0, 1))
        axes[row, 0].set_title(f"Original\n{class_name}")
        axes[row, 0].axis("off")

        # Mask
        axes[row, 1].imshow(mask, cmap="gray")
        axes[row, 1].set_title("Segmentation Mask")
        axes[row, 1].axis("off")

        # Overlay
        overlay = img.copy()
        mask_bool = mask > 0.5
        overlay_rgb = np.clip(img, 0, 1).copy()
        # 舌头区域保持原色，背景变暗
        for c in range(3):
            overlay_rgb[:, :, c] = np.where(mask_bool, overlay_rgb[:, :, c], overlay_rgb[:, :, c] * 0.3)
        axes[row, 2].imshow(overlay_rgb)
        axes[row, 2].set_title("Mask Overlay")
        axes[row, 2].axis("off")

        # 边缘检测显示 mask 轮廓
        from scipy import ndimage
        edges = ndimage.sobel(mask.astype(float))
        edges = (np.abs(edges) > 0.1).astype(float)

        contour_img = np.clip(img, 0, 1).copy()
        contour_img[edges > 0] = [1, 0, 0]  # 红色轮廓
        axes[row, 3].imshow(contour_img)
        axes[row, 3].set_title("Mask Contour")
        axes[row, 3].axis("off")

    plt.suptitle("Real Segmentation Mask Quality", fontsize=14, y=1.02)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved visualization to {output_path}")
    else:
        plt.show()

    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Visualize mask augmentation effects")
    parser.add_argument(
        "--data-root",
        type=str,
        default="datasets/shezhenv3-coco",
        help="Dataset root directory",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=5,
        help="Number of samples to visualize",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="docs",
        help="Output directory for visualizations",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="both",
        choices=["augmentation", "quality", "both"],
        help="Visualization mode",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.mode in ("augmentation", "both"):
        print("Visualizing mask augmentation effects...")
        visualize_mask_augmentation(
            data_root=args.data_root,
            num_samples=args.num_samples,
            output_path=str(output_dir / "mask_augmentation_comparison.png"),
        )

    if args.mode in ("quality", "both"):
        print("Visualizing mask quality...")
        visualize_mask_quality_comparison(
            data_root=args.data_root,
            num_samples=args.num_samples,
            output_path=str(output_dir / "mask_quality_visualization.png"),
        )

    print("Done!")


if __name__ == "__main__":
    main()
