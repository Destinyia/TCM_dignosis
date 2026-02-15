"""分类数据集 - 用于舌象图像分类任务"""
from __future__ import annotations

import json
import os
import random
from collections import defaultdict
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image, ImageFile, ImageFilter
from torch.utils.data import Dataset

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
except ImportError:
    A = None
    ToTensorV2 = None

try:
    from pycocotools import mask as mask_utils
except ImportError:
    mask_utils = None


# 舌头基本类型（13类）
TONGUE_BASIC_TYPES = [
    "jiankangshe",   # 健康舌
    "botaishe",      # 薄苔舌
    "hongshe",       # 红舌
    "zishe",         # 紫舌
    "pangdashe",     # 胖大舌
    "shoushe",       # 瘦舌
    "hongdianshe",   # 红点舌
    "liewenshe",     # 裂纹舌
    "chihenshe",     # 齿痕舌
    "baitaishe",     # 白苔舌
    "huangtaishe",   # 黄苔舌
    "heitaishe",     # 黑苔舌
    "huataishe",     # 滑苔舌
]


class TongueClassificationDataset(Dataset):
    """舌象分类数据集

    从COCO格式的检测数据集转换为分类数据集。
    每张图像对应一个类别标签（取该图像中所有标注的主要类别）。
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        image_size: Tuple[int, int] = (640, 640),
        class_filter: Optional[List[str]] = None,
        label_strategy: str = "first",  # first, majority, all
        use_basic_types: bool = True,  # 默认使用舌头基本类型
        return_bbox: bool = False,  # 是否返回bbox
        return_mask: bool = False,  # 是否返回mask
        mask_aug: Optional[Callable] = None,  # 分割mask引导的数据增强
    ):
        """
        Args:
            root: 数据集根目录
            split: 数据集划分 (train/val/test)
            transform: 数据增强变换
            image_size: 图像尺寸
            class_filter: 类别过滤列表（优先级高于use_basic_types）
            label_strategy: 标签策略
                - first: 使用第一个标注的类别
                - majority: 使用出现次数最多的类别
                - all: 返回所有类别（多标签）
            use_basic_types: 是否使用舌头基本类型（13类），默认True
        """
        self.root = root
        self.split = split
        self.transform = transform
        self.image_size = image_size
        self.label_strategy = label_strategy
        self.return_bbox = return_bbox
        self.return_mask = return_mask
        self.mask_aug = mask_aug
        self._warned_no_mask = False

        # 如果未指定class_filter且use_basic_types为True，使用舌头基本类型
        if class_filter is None and use_basic_types:
            class_filter = TONGUE_BASIC_TYPES.copy()

        ann_path = os.path.join(root, split, "annotations", f"{split}.json")
        img_dir = os.path.join(root, split, "images")

        if not os.path.isfile(ann_path):
            raise FileNotFoundError(f"Annotation not found: {ann_path}")
        if not os.path.isdir(img_dir):
            raise FileNotFoundError(f"Image dir not found: {img_dir}")

        with open(ann_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.image_dir = img_dir
        self.images = list(data.get("images", []))
        self._image_id_to_info = {img["id"]: img for img in self.images}

        # 处理类别
        categories = sorted(data.get("categories", []), key=lambda c: c["id"])
        if class_filter:
            allowed_ids = self._resolve_class_filter(categories, class_filter)
            categories = [c for c in categories if c["id"] in allowed_ids]

        self.category_ids = [c["id"] for c in categories]
        self.category_names = {c["id"]: c["name"] for c in categories}
        self._cat_id_to_contiguous = {
            cat_id: idx for idx, cat_id in enumerate(self.category_ids)
        }
        self.num_classes = len(self.category_ids)

        # 构建图像到标签的映射
        self._annotations_by_image_id = defaultdict(list)
        annotations = data.get("annotations", [])
        if class_filter:
            annotations = [
                ann for ann in annotations
                if ann.get("category_id") in set(self.category_ids)
            ]
        for ann in annotations:
            self._annotations_by_image_id[ann["image_id"]].append(ann)

        # 过滤没有标注的图像
        self.valid_images = [
            img for img in self.images
            if img["id"] in self._annotations_by_image_id
        ]

        print(f"Loaded {len(self.valid_images)} images with {self.num_classes} classes")

    def _resolve_class_filter(
        self, categories: List[Dict], class_filter: List[str]
    ) -> set:
        """解析类别过滤器"""
        id_set = {int(c["id"]) for c in categories}
        name_to_id = {c["name"]: int(c["id"]) for c in categories}
        allowed = set()

        for item in class_filter:
            if isinstance(item, str):
                if item in name_to_id:
                    allowed.add(name_to_id[item])
                else:
                    try:
                        val = int(item)
                        if val in id_set:
                            allowed.add(val)
                    except ValueError:
                        pass
            elif isinstance(item, int) and item in id_set:
                allowed.add(item)

        return allowed

    def __len__(self) -> int:
        return len(self.valid_images)

    def __getitem__(
        self, idx: int
    ) -> (
        Tuple[torch.Tensor, int]
        | Tuple[torch.Tensor, int, torch.Tensor]
        | Tuple[torch.Tensor, int, torch.Tensor, torch.Tensor]
    ):
        """获取样本

        Returns:
            image: 图像张量 (C, H, W)
            label: 类别标签
            bbox: (可选) 归一化bbox [x1, y1, x2, y2]
        """
        image_info = self.valid_images[idx]
        image_id = image_info["id"]

        # 加载图像
        image_path = os.path.join(self.image_dir, image_info["file_name"])
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        image = Image.open(image_path).convert("RGB")
        image_np = np.array(image)
        orig_h, orig_w = image_np.shape[:2]

        # 获取标签和bbox
        anns = self._annotations_by_image_id.get(image_id, [])
        label = self._get_label(anns)
        bbox = self._get_bbox(anns, orig_h, orig_w)
        mask = None
        if self.return_mask or self.mask_aug is not None:
            mask = self._get_segmentation_mask(anns, orig_h, orig_w, image_info["file_name"])
            if self.mask_aug is not None:
                image_np, mask = self.mask_aug(image_np, mask)

        # 应用变换
        if self.transform is not None:
            if mask is not None:
                transformed = self.transform(image=image_np, mask=mask)
                image_tensor = transformed["image"]
                mask_tensor = transformed.get("mask")
            else:
                transformed = self.transform(image=image_np)
                image_tensor = transformed["image"]
                mask_tensor = None
        else:
            image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0
            mask_tensor = torch.from_numpy(mask).unsqueeze(0).float() if mask is not None else None

        if self.return_mask and mask_tensor is not None:
            if mask_tensor.ndim == 2:
                mask_tensor = mask_tensor.unsqueeze(0)
            mask_tensor = mask_tensor.float()

        if self.return_bbox and self.return_mask:
            return image_tensor, label, bbox, mask_tensor
        if self.return_bbox:
            return image_tensor, label, bbox
        if self.return_mask:
            return image_tensor, label, mask_tensor
        return image_tensor, label

    def _get_bbox(self, anns: List[Dict], orig_h: int, orig_w: int) -> torch.Tensor:
        """获取归一化的bbox坐标

        Args:
            anns: 标注列表
            orig_h: 原始图像高度
            orig_w: 原始图像宽度

        Returns:
            bbox: [x1, y1, x2, y2] 归一化坐标
        """
        if not anns:
            return torch.tensor([0.0, 0.0, 1.0, 1.0], dtype=torch.float32)

        # 取第一个有效bbox
        for ann in anns:
            if "bbox" in ann and len(ann["bbox"]) == 4:
                x, y, bw, bh = ann["bbox"]
                # 归一化到[0,1]
                x1 = x / orig_w
                y1 = y / orig_h
                x2 = (x + bw) / orig_w
                y2 = (y + bh) / orig_h
                # 裁剪到[0,1]范围
                x1 = max(0.0, min(1.0, x1))
                y1 = max(0.0, min(1.0, y1))
                x2 = max(0.0, min(1.0, x2))
                y2 = max(0.0, min(1.0, y2))
                return torch.tensor([x1, y1, x2, y2], dtype=torch.float32)

        return torch.tensor([0.0, 0.0, 1.0, 1.0], dtype=torch.float32)

    def _get_bbox_pixels(self, anns: List[Dict], orig_h: int, orig_w: int) -> Tuple[int, int, int, int]:
        """获取像素级bbox坐标 (x1, y1, x2, y2)"""
        if not anns:
            return 0, 0, orig_w, orig_h
        for ann in anns:
            if "bbox" in ann and len(ann["bbox"]) == 4:
                x, y, bw, bh = ann["bbox"]
                x1 = max(0, int(x))
                y1 = max(0, int(y))
                x2 = min(orig_w, int(x + bw))
                y2 = min(orig_h, int(y + bh))
                if x2 > x1 and y2 > y1:
                    return x1, y1, x2, y2
        return 0, 0, orig_w, orig_h

    def _get_segmentation_mask(self, anns: List[Dict], orig_h: int, orig_w: int, image_filename: str = None) -> np.ndarray:
        """获取分割mask，优先使用预生成的mask文件。

        查找顺序：
        1. 预生成的 mask 文件 ({root}/{split}/masks/{image_stem}.png)
        2. COCO segmentation 标注（如果是真实轮廓而非bbox矩形）
        3. 回退到 bbox 矩形 mask
        """
        # 1. 尝试加载预生成的 mask 文件
        if image_filename:
            image_stem = os.path.splitext(image_filename)[0]
            mask_path = os.path.join(self.root, self.split, "masks", f"{image_stem}.png")
            if os.path.exists(mask_path):
                mask_img = Image.open(mask_path).convert("L")
                mask_np = np.array(mask_img)
                # 确保尺寸匹配
                if mask_np.shape[0] != orig_h or mask_np.shape[1] != orig_w:
                    mask_img = mask_img.resize((orig_w, orig_h), Image.NEAREST)
                    mask_np = np.array(mask_img)
                # 二值化 (mask 文件是 0-255)
                return (mask_np > 127).astype(np.uint8)

        # 2. 尝试使用 COCO segmentation（检查是否为真实轮廓）
        masks = []
        if mask_utils is not None:
            for ann in anns:
                seg = ann.get("segmentation")
                if not seg:
                    continue
                rle = None
                if isinstance(seg, list):
                    # 检查是否为真实轮廓（超过4个点）
                    if len(seg) > 0 and len(seg[0]) > 8:  # 超过4个点
                        rles = mask_utils.frPyObjects(seg, orig_h, orig_w)
                        rle = mask_utils.merge(rles)
                elif isinstance(seg, dict) and "counts" in seg:
                    rle = seg
                if rle is None:
                    continue
                m = mask_utils.decode(rle)
                if m is None:
                    continue
                if m.ndim == 3:
                    m = m[:, :, 0]
                masks.append(m.astype(np.uint8))
        elif not self._warned_no_mask:
            print("Warning: pycocotools not available, falling back to bbox masks.")
            self._warned_no_mask = True

        if masks:
            combined = np.clip(np.sum(masks, axis=0), 0, 1).astype(np.uint8)
            return combined

        # 3. 回退到 bbox 矩形 mask
        x1, y1, x2, y2 = self._get_bbox_pixels(anns, orig_h, orig_w)
        mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
        mask[y1:y2, x1:x2] = 1
        return mask

    def _get_label(self, anns: List[Dict]) -> int:
        """根据策略获取标签"""
        if not anns:
            return 0

        cat_ids = [
            self._cat_id_to_contiguous.get(ann.get("category_id"))
            for ann in anns
            if ann.get("category_id") in self._cat_id_to_contiguous
        ]
        cat_ids = [c for c in cat_ids if c is not None]

        if not cat_ids:
            return 0

        if self.label_strategy == "first":
            return cat_ids[0]
        elif self.label_strategy == "majority":
            from collections import Counter
            return Counter(cat_ids).most_common(1)[0][0]
        else:
            return cat_ids[0]

    def get_class_counts(self) -> Dict[int, int]:
        """获取各类别样本数"""
        counts = {i: 0 for i in range(self.num_classes)}
        for img in self.valid_images:
            anns = self._annotations_by_image_id.get(img["id"], [])
            label = self._get_label(anns)
            counts[label] += 1
        return counts

    def get_class_names(self) -> Dict[int, str]:
        """获取类别名称映射"""
        return {
            self._cat_id_to_contiguous[k]: v
            for k, v in self.category_names.items()
        }

    def get_sample_weights(self, strategy: str = "sqrt") -> torch.Tensor:
        """获取样本权重（用于加权采样）

        Args:
            strategy: 采样策略
                - "inverse": 逆频率加权 (原始方式)
                - "sqrt": 平方根加权 (更温和，推荐)
        """
        counts = self.get_class_counts()
        total = sum(counts.values())

        if strategy == "sqrt":
            # 平方根加权：更温和的重采样，避免过度采样尾部类别
            class_weights = {
                k: np.sqrt(total / (len(counts) * v))
                for k, v in counts.items() if v > 0
            }
        else:
            # 逆频率加权
            class_weights = {
                k: total / (len(counts) * v)
                for k, v in counts.items() if v > 0
            }

        # 归一化到 [0, 1] 范围
        max_w = max(class_weights.values()) if class_weights else 1.0
        class_weights = {k: v / max_w for k, v in class_weights.items()}

        weights = []
        for img in self.valid_images:
            anns = self._annotations_by_image_id.get(img["id"], [])
            label = self._get_label(anns)
            weights.append(class_weights.get(label, 1.0))

        return torch.tensor(weights, dtype=torch.float32)


class ClassificationTransform:
    """分类任务的数据增强"""

    def __init__(
        self,
        image_size: Tuple[int, int] = (640, 640),
        is_train: bool = True,
        normalize: bool = True,
        strong_aug: bool = False,
    ):
        if A is None:
            raise ImportError("albumentations is required for transforms")

        ops = []

        # 调整尺寸
        ops.extend([
            A.LongestMaxSize(max_size=max(image_size)),
            A.PadIfNeeded(
                min_height=image_size[0],
                min_width=image_size[1],
                border_mode=0,
                fill=(114, 114, 114),
            ),
        ])

        if is_train:
            ops.append(A.HorizontalFlip(p=0.5))
            ops.append(A.RandomBrightnessContrast(p=0.3))
            ops.append(A.HueSaturationValue(
                hue_shift_limit=10,
                sat_shift_limit=20,
                val_shift_limit=20,
                p=0.3,
            ))

            if strong_aug:
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

        if normalize:
            ops.append(A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ))
        else:
            ops.append(A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
            ))

        ops.append(ToTensorV2())

        self.transform = A.Compose(ops)

    def __call__(self, image: np.ndarray, mask: Optional[np.ndarray] = None) -> Dict[str, torch.Tensor]:
        if mask is None:
            return self.transform(image=image)
        result = self.transform(image=image, mask=mask)
        if "mask" in result:
            mask_tensor = result["mask"]
            if hasattr(mask_tensor, "ndim") and mask_tensor.ndim == 2:
                mask_tensor = mask_tensor.unsqueeze(0)
            result["mask"] = mask_tensor.float()
        return result


class MaskAugmentation:
    """使用分割mask的裁剪/背景替换增强"""

    def __init__(
        self,
        mode: str = "both",  # crop, background, both
        prob: float = 0.5,
        crop_padding: float = 0.1,
        bg_mode: str = "blur",  # solid, blur, noise
        bg_color: Tuple[int, int, int] = (114, 114, 114),
        bg_blur_radius: float = 12.0,
        mask_threshold: float = 0.5,
        mask_dilate: int = 15,  # mask 膨胀像素数，防止边缘模糊
    ):
        self.mode = mode
        self.prob = prob
        self.crop_padding = crop_padding
        self.bg_mode = bg_mode
        self.bg_color = bg_color
        self.bg_blur_radius = bg_blur_radius
        self.mask_threshold = mask_threshold
        self.mask_dilate = mask_dilate

    def _dilate_mask(self, mask: np.ndarray) -> np.ndarray:
        """膨胀 mask 以保护舌头边缘"""
        if self.mask_dilate <= 0:
            return mask
        from scipy import ndimage
        # 二值化
        binary_mask = (mask > self.mask_threshold).astype(np.uint8)
        # 膨胀操作
        struct = ndimage.generate_binary_structure(2, 1)
        dilated = ndimage.binary_dilation(
            binary_mask, structure=struct, iterations=self.mask_dilate
        )
        return dilated.astype(np.uint8)

    def __call__(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if mask is None or random.random() > self.prob:
            return image, mask

        if self.mode in ("crop", "both"):
            image, mask = self._crop_to_mask(image, mask)

        if self.mode in ("background", "both"):
            image = self._replace_background(image, mask)

        return image, mask

    def _crop_to_mask(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # 保存原始图像以便在裁剪失败时返回
        orig_image, orig_mask = image, mask

        ys, xs = np.where(mask > self.mask_threshold)
        if len(xs) == 0 or len(ys) == 0:
            return orig_image, orig_mask

        y1, y2 = ys.min(), ys.max()
        x1, x2 = xs.min(), xs.max()
        h = y2 - y1 + 1
        w = x2 - x1 + 1
        pad_y = int(h * self.crop_padding)
        pad_x = int(w * self.crop_padding)

        y1 = max(0, y1 - pad_y)
        x1 = max(0, x1 - pad_x)
        y2 = min(image.shape[0], y2 + pad_y)
        x2 = min(image.shape[1], x2 + pad_x)

        # 确保裁剪后尺寸有效（至少 1x1）
        if y2 <= y1 or x2 <= x1:
            return orig_image, orig_mask

        cropped_image = image[y1:y2, x1:x2]
        cropped_mask = mask[y1:y2, x1:x2]

        # 确保裁剪后图像非空
        if cropped_image.shape[0] == 0 or cropped_image.shape[1] == 0:
            return orig_image, orig_mask

        return cropped_image, cropped_mask

    def _replace_background(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        if self.bg_mode == "solid":
            bg = np.full_like(image, self.bg_color, dtype=np.uint8)
        elif self.bg_mode == "noise":
            bg = np.random.randint(0, 256, image.shape, dtype=np.uint8)
        else:
            bg = np.array(Image.fromarray(image).filter(ImageFilter.GaussianBlur(radius=self.bg_blur_radius)))

        # 膨胀 mask 以保护舌头边缘
        dilated_mask = self._dilate_mask(mask)

        mask_bool = dilated_mask > self.mask_threshold
        if mask_bool.ndim == 2:
            mask_bool = np.repeat(mask_bool[:, :, None], 3, axis=2)
        elif mask_bool.ndim == 3 and mask_bool.shape[2] == 1:
            mask_bool = np.repeat(mask_bool, 3, axis=2)
        composite = image.copy()
        composite[~mask_bool] = bg[~mask_bool]
        return composite


def create_classification_dataloaders(
    root: str,
    batch_size: int = 16,
    num_workers: int = 4,
    image_size: Tuple[int, int] = (640, 640),
    class_filter: Optional[List[str]] = None,
    use_weighted_sampler: bool = False,
    sampler_strategy: str = "sqrt",
    strong_aug: bool = False,
    use_basic_types: bool = True,
    return_bbox: bool = False,
    return_mask: bool = False,
    mask_aug: Optional[Callable] = None,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, Dict]:
    """创建分类数据加载器

    Args:
        root: 数据集根目录
        batch_size: 批次大小
        num_workers: 数据加载线程数
        image_size: 图像尺寸
        class_filter: 类别过滤列表（优先级高于use_basic_types）
        use_weighted_sampler: 是否使用加权采样
        sampler_strategy: 采样策略 ("sqrt" 或 "inverse")
        strong_aug: 是否使用强数据增强
        use_basic_types: 是否使用舌头基本类型（13类），默认True
        return_bbox: 是否返回bbox

    Returns:
        train_loader, val_loader, dataset_info
    """
    train_transform = ClassificationTransform(
        image_size=image_size,
        is_train=True,
        strong_aug=strong_aug,
    )
    val_transform = ClassificationTransform(
        image_size=image_size,
        is_train=False,
    )

    train_dataset = TongueClassificationDataset(
        root=root,
        split="train",
        transform=train_transform,
        image_size=image_size,
        class_filter=class_filter,
        use_basic_types=use_basic_types,
        return_bbox=return_bbox,
        return_mask=return_mask,
        mask_aug=mask_aug,
    )
    val_dataset = TongueClassificationDataset(
        root=root,
        split="val",
        transform=val_transform,
        image_size=image_size,
        class_filter=class_filter,
        use_basic_types=use_basic_types,
        return_bbox=return_bbox,
        return_mask=return_mask,
        mask_aug=None,
    )

    # 采样器
    if use_weighted_sampler:
        weights = train_dataset.get_sample_weights(strategy=sampler_strategy)
        sampler = torch.utils.data.WeightedRandomSampler(
            weights, len(weights), replacement=True
        )
        shuffle = False
    else:
        sampler = None
        shuffle = True

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    dataset_info = {
        "num_classes": train_dataset.num_classes,
        "class_names": train_dataset.get_class_names(),
        "class_counts": train_dataset.get_class_counts(),
        "train_size": len(train_dataset),
        "val_size": len(val_dataset),
    }

    return train_loader, val_loader, dataset_info
