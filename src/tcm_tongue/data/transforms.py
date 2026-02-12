from __future__ import annotations

from typing import Tuple

import numpy as np
import torch

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
except Exception as exc:  # pragma: no cover - optional dependency
    A = None
    ToTensorV2 = None

try:
    import cv2
except Exception:  # pragma: no cover - optional dependency
    cv2 = None


class BaseTransform:
    """Base transform wrapper."""

    def __call__(
        self,
        image: np.ndarray,
        bboxes: np.ndarray,
        labels: np.ndarray,
    ) -> Tuple[torch.Tensor, np.ndarray, np.ndarray]:
        if self.transform is None:
            raise RuntimeError("Albumentations is required for transforms")

        if bboxes is None:
            bboxes = np.zeros((0, 4), dtype=np.float32)
        if labels is None:
            labels = np.zeros((0,), dtype=np.int64)

        result = self.transform(image=image, bboxes=bboxes.tolist(), labels=labels.tolist())
        return result["image"], np.array(result["bboxes"], dtype=np.float32), np.array(
            result["labels"], dtype=np.int64
        )


class TrainTransform(BaseTransform):
    """Training-time augmentation."""

    def __init__(
        self,
        image_size: Tuple[int, int] = (800, 800),
        use_mosaic: bool = False,
        use_mixup: bool = False,
        normalize: bool = True,
        resize: bool = True,
        horizontal_flip: bool = True,
        brightness_contrast: bool = True,
        hue_saturation: bool = True,
        gauss_noise: bool = True,
        strong: bool = True,
        tcm_prior: bool = False,
        tcm_prior_prob: float = 0.3,
    ):
        if A is None or ToTensorV2 is None:
            self.transform = None
            return

        _ = (use_mosaic, use_mixup)
        ops = []
        if resize:
            ops.extend(
                [
                    A.LongestMaxSize(max_size=max(image_size)),
                    A.PadIfNeeded(
                        min_height=image_size[0],
                        min_width=image_size[1],
                        border_mode=0,
                        fill=(114, 114, 114),
                    ),
                ]
            )
        if horizontal_flip:
            ops.append(A.HorizontalFlip(p=0.5))
        if brightness_contrast:
            ops.append(A.RandomBrightnessContrast(p=0.3))
        if hue_saturation:
            ops.append(
                A.HueSaturationValue(
                    hue_shift_limit=10,
                    sat_shift_limit=20,
                    val_shift_limit=20,
                    p=0.3,
                )
            )
        if gauss_noise:
            ops.append(A.GaussNoise(p=0.1))
        if strong:
            ops.extend(
                [
                    A.Affine(
                        translate_percent=0.1,
                        scale=(0.66, 1.5),
                        rotate=(-15, 15),
                        border_mode=0,
                        fill=(114, 114, 114),
                        p=0.3,
                    ),
                    A.RandomGamma(p=0.2),
                ]
            )
        if tcm_prior:
            prior = TonguePriorAugment()

            def _apply_prior(img, **kwargs):
                shift = np.random.choice(list(prior.color_shifts.keys()))
                return prior.apply_constitution_shift(img, shift)

            ops.append(A.Lambda(image=_apply_prior, p=float(tcm_prior_prob)))
        if normalize:
            ops.append(
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                )
            )
        else:
            ops.append(
                A.Normalize(
                    mean=[0.0, 0.0, 0.0],
                    std=[1.0, 1.0, 1.0],
                )
            )
        ops.append(ToTensorV2())

        self.transform = A.Compose(
            ops,
            bbox_params=A.BboxParams(
                format="coco",
                label_fields=["labels"],
                min_area=1,
                min_visibility=0.1,
            ),
        )


class ValTransform(BaseTransform):
    """Validation/test-time transform."""

    def __init__(
        self,
        image_size: Tuple[int, int] = (800, 800),
        normalize: bool = True,
        resize: bool = True,
    ):
        if A is None or ToTensorV2 is None:
            self.transform = None
            return

        ops = []
        if resize:
            ops.extend(
                [
                    A.LongestMaxSize(max_size=max(image_size)),
                    A.PadIfNeeded(
                        min_height=image_size[0],
                        min_width=image_size[1],
                        border_mode=0,
                        fill=(114, 114, 114),
                    ),
                ]
            )
        else:
            # Avoid Albumentations warning when bboxes are provided but no bbox-aware ops exist.
            ops.append(A.HorizontalFlip(p=0.0))
        if normalize:
            ops.append(
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                )
            )
        else:
            ops.append(
                A.Normalize(
                    mean=[0.0, 0.0, 0.0],
                    std=[1.0, 1.0, 1.0],
                )
            )
        ops.append(ToTensorV2())

        self.transform = A.Compose(
            ops,
            bbox_params=A.BboxParams(
                format="coco",
                label_fields=["labels"],
                min_area=1,
                min_visibility=0.1,
            ),
        )


class TonguePriorAugment:
    """Augmentations based on tongue-prior knowledge."""

    def __init__(self):
        self.color_shifts = {
            "cold": {"hue": -10, "sat": -20},
            "heat": {"hue": 10, "sat": 20},
            "damp": {"hue": 5, "sat": -10},
        }

    def apply_constitution_shift(self, image: np.ndarray, constitution: str) -> np.ndarray:
        """Apply constitution-specific color shift."""
        shift = self.color_shifts.get(constitution)
        if shift is None:
            return image
        if cv2 is None:
            return image

        img = image
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)

        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(hsv)
        h = (h.astype(np.int16) + int(shift["hue"])) % 180
        s = np.clip(s.astype(np.int16) + int(shift["sat"]), 0, 255)
        hsv = cv2.merge([h.astype(np.uint8), s.astype(np.uint8), v])
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
