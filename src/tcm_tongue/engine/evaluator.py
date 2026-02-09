from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from contextlib import contextmanager, redirect_stdout
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Subset

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


class COCOEvaluator:
    """COCO-style evaluator."""

    def __init__(self, dataset, silent: bool = True):
        base_dataset, subset_image_ids = _unwrap_dataset(dataset)
        self.dataset = base_dataset
        self.subset_image_ids = subset_image_ids
        self.silent = silent
        self.coco_gt = self._build_coco_gt()
        self._contiguous_to_cat = {
            v: k for k, v in getattr(base_dataset, "_cat_id_to_contiguous", {}).items()
        }
        self.label_offset = getattr(base_dataset, "label_offset", 0)

    def _build_coco_gt(self) -> COCO:
        ann_path = _resolve_annotation_path(self.dataset)
        if ann_path and os.path.isfile(ann_path) and not self.subset_image_ids:
            with _suppress_stdout(self.silent):
                return COCO(ann_path)

        if ann_path and os.path.isfile(ann_path) and self.subset_image_ids:
            with _suppress_stdout(self.silent):
                full = COCO(ann_path)
            coco_dict = _filter_coco(full, self.subset_image_ids)
            with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as fp:
                json.dump(coco_dict, fp)
                temp_path = fp.name
            with _suppress_stdout(self.silent):
                return COCO(temp_path)

        coco_dict = {
            "images": getattr(self.dataset, "images", []),
            "annotations": _flatten_annotations(self.dataset),
            "categories": _build_categories(self.dataset),
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as fp:
            json.dump(coco_dict, fp)
            temp_path = fp.name
        with _suppress_stdout(self.silent):
            coco = COCO(temp_path)
        return coco

    @torch.no_grad()
    def evaluate(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        device: str = "cuda",
        plot_dir: str | None = None,
        plot_pr: bool = False,
        pr_iou: float = 0.5,
    ) -> Dict[str, float | Dict[str, float]]:
        model.eval()
        predictions: List[Dict] = []

        valid_image_ids = set(self.subset_image_ids or [])

        for images, targets in dataloader:
            images = [img.to(device) for img in images]
            _, detections = model(images)

            for target, detection in zip(targets, detections):
                image_id = int(target["image_id"].item())
                if valid_image_ids and image_id not in valid_image_ids:
                    continue
                boxes = detection.get("boxes", torch.empty((0, 4)))
                scores = detection.get("scores", torch.empty((0,)))
                labels = detection.get("labels", torch.empty((0,), dtype=torch.int64))

                boxes = boxes.cpu()
                scores = scores.cpu()
                labels = labels.cpu()

                for box, score, label in zip(boxes, scores, labels):
                    x1, y1, x2, y2 = box.tolist()
                    label_idx = int(label) - int(self.label_offset)
                    if label_idx < 0:
                        continue
                    cat_id = self._contiguous_to_cat.get(label_idx, label_idx)
                    predictions.append(
                        {
                            "image_id": image_id,
                            "category_id": cat_id,
                            "bbox": [x1, y1, x2 - x1, y2 - y1],
                            "score": float(score.item()),
                        }
                    )

        if not predictions:
            empty_metrics = {
                "mAP": 0.0,
                "mAP_50": 0.0,
                "mAP_75": 0.0,
                "mAP_small": 0.0,
                "mAP_medium": 0.0,
                "mAP_large": 0.0,
                "AR_1": 0.0,
                "AR_10": 0.0,
                "AR_100": 0.0,
                "AR_small": 0.0,
                "AR_medium": 0.0,
                "AR_large": 0.0,
                "per_class_AP": {},
            }
            return empty_metrics

        with _suppress_stdout(self.silent):
            coco_dt = self.coco_gt.loadRes(predictions)
            coco_eval = COCOeval(self.coco_gt, coco_dt, "bbox")
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()

        stats = getattr(coco_eval, "stats", None)
        stats = list(stats) if stats is not None else []
        if len(stats) < 12:
            stats += [0.0] * (12 - len(stats))

        metrics = {
            "mAP": float(stats[0]),
            "mAP_50": float(stats[1]),
            "mAP_75": float(stats[2]),
            "mAP_small": float(stats[3]),
            "mAP_medium": float(stats[4]),
            "mAP_large": float(stats[5]),
            "AR_1": float(stats[6]),
            "AR_10": float(stats[7]),
            "AR_100": float(stats[8]),
            "AR_small": float(stats[9]),
            "AR_medium": float(stats[10]),
            "AR_large": float(stats[11]),
        }

        metrics["per_class_AP"] = self._compute_per_class_ap(coco_eval)
        metrics["per_class_AP50"] = self._compute_per_class_ap_at_iou(coco_eval, iou_index=0)
        metrics["per_class_AR"] = self._compute_per_class_ar(coco_eval)
        metrics["per_class_AR50"] = self._compute_per_class_ar_at_iou(coco_eval, iou_index=0)

        if plot_dir and plot_pr:
            _plot_per_class_pr_curves(coco_eval, self.dataset, plot_dir, iou=pr_iou)
        return metrics

    def _compute_per_class_ap(self, coco_eval: COCOeval) -> Dict[str, float]:
        precision = coco_eval.eval.get("precision")
        if precision is None:
            return {}

        # precision dims: [TxRxKxAxM]
        per_class_ap: Dict[str, float] = {}
        cat_ids = coco_eval.params.catIds
        cat_id_to_name = _category_id_to_name(self.dataset)

        for idx, cat_id in enumerate(cat_ids):
            pr = precision[:, :, idx, 0, -1]
            valid = pr[pr > -1]
            ap = float(np.mean(valid)) if valid.size > 0 else 0.0
            name = cat_id_to_name.get(cat_id, str(cat_id))
            per_class_ap[name] = ap

        return per_class_ap

    def _compute_per_class_ap_at_iou(self, coco_eval: COCOeval, iou_index: int) -> Dict[str, float]:
        precision = coco_eval.eval.get("precision")
        if precision is None:
            return {}

        per_class_ap: Dict[str, float] = {}
        cat_ids = coco_eval.params.catIds
        cat_id_to_name = _category_id_to_name(self.dataset)

        for idx, cat_id in enumerate(cat_ids):
            pr = precision[iou_index, :, idx, 0, -1]
            valid = pr[pr > -1]
            ap = float(np.mean(valid)) if valid.size > 0 else 0.0
            name = cat_id_to_name.get(cat_id, str(cat_id))
            per_class_ap[name] = ap

        return per_class_ap

    def _compute_per_class_ar(self, coco_eval: COCOeval) -> Dict[str, float]:
        recall = coco_eval.eval.get("recall")
        if recall is None:
            return {}

        # recall dims: [TxKxAxM]
        per_class_ar: Dict[str, float] = {}
        cat_ids = coco_eval.params.catIds
        cat_id_to_name = _category_id_to_name(self.dataset)

        for idx, cat_id in enumerate(cat_ids):
            rc = recall[:, idx, 0, -1]
            valid = rc[rc > -1]
            ar = float(np.mean(valid)) if valid.size > 0 else 0.0
            name = cat_id_to_name.get(cat_id, str(cat_id))
            per_class_ar[name] = ar

        return per_class_ar

    def _compute_per_class_ar_at_iou(self, coco_eval: COCOeval, iou_index: int) -> Dict[str, float]:
        recall = coco_eval.eval.get("recall")
        if recall is None:
            return {}

        per_class_ar: Dict[str, float] = {}
        cat_ids = coco_eval.params.catIds
        cat_id_to_name = _category_id_to_name(self.dataset)

        for idx, cat_id in enumerate(cat_ids):
            rc = recall[iou_index, idx, 0, -1]
            if np.isscalar(rc):
                ar = float(rc) if rc > -1 else 0.0
            else:
                valid = rc[rc > -1]
                ar = float(np.mean(valid)) if valid.size > 0 else 0.0
            name = cat_id_to_name.get(cat_id, str(cat_id))
            per_class_ar[name] = ar

        return per_class_ar


class ClassImbalanceAnalyzer:
    """Analyze head/medium/tail class performance."""

    def __init__(self, evaluator: COCOEvaluator):
        self.evaluator = evaluator

    def analyze_head_tail_performance(
        self,
        per_class_ap: Dict[str, float],
        class_counts: Dict[str, int],
    ) -> Dict[str, float]:
        head = []
        medium = []
        tail = []

        for cls_name, count in class_counts.items():
            ap = per_class_ap.get(cls_name, 0.0)
            if count > 1000:
                head.append(ap)
            elif count >= 100:
                medium.append(ap)
            else:
                tail.append(ap)

        head_map = float(sum(head) / len(head)) if head else 0.0
        medium_map = float(sum(medium) / len(medium)) if medium else 0.0
        tail_map = float(sum(tail) / len(tail)) if tail else 0.0

        return {
            "head_mAP": head_map,
            "medium_mAP": medium_map,
            "tail_mAP": tail_map,
            "head_tail_gap": head_map - tail_map,
        }


def format_coco_metrics(metrics: Dict[str, float]) -> str:
    ap_headers = ["mAP", "mAP50", "mAP75", "small", "medium", "large"]
    ar_headers = ["AR1", "AR10", "AR100", "small", "medium", "large"]
    col_width = 7
    ap_line = " ".join(f"{name:>{col_width}}" for name in ap_headers)
    ar_line = " ".join(f"{name:>{col_width}}" for name in ar_headers)
    header1 = f"{'Average Precision':^{len(ap_line)}} | {'Average Recall':^{len(ar_line)}}"
    header2 = f"{ap_line} | {ar_line}"
    rule = "-" * len(header2)
    keys = [
        "mAP",
        "mAP_50",
        "mAP_75",
        "mAP_small",
        "mAP_medium",
        "mAP_large",
        "AR_1",
        "AR_10",
        "AR_100",
        "AR_small",
        "AR_medium",
        "AR_large",
    ]
    values = " ".join(f"{metrics.get(key, 0.0):{col_width}.3f}" for key in keys[:6])
    values += " | "
    values += " ".join(f"{metrics.get(key, 0.0):{col_width}.3f}" for key in keys[6:])
    return "\n".join([rule, header1, header2, rule, values, rule])


def format_per_class_metrics(
    per_class_ap: Dict[str, float],
    per_class_ap50: Dict[str, float],
    per_class_ar: Dict[str, float],
    per_class_ar50: Dict[str, float],
) -> str:
    if not per_class_ap and not per_class_ar and not per_class_ap50 and not per_class_ar50:
        return "Per-class AP/AR: (no data)"

    names = sorted(
        set(per_class_ap.keys())
        | set(per_class_ap50.keys())
        | set(per_class_ar.keys())
        | set(per_class_ar50.keys())
    )
    name_width = max(6, min(24, max(len(n) for n in names)))
    header = (
        f"{'Class':<{name_width}} | {'AP':>8} | {'AP50':>8} | {'AR':>8} | {'AR50':>8}"
    )
    rule = "-" * len(header)
    lines = [rule, header, rule]
    for name in names:
        ap = float(per_class_ap.get(name, 0.0))
        ap50 = float(per_class_ap50.get(name, 0.0))
        ar = float(per_class_ar.get(name, 0.0))
        ar50 = float(per_class_ar50.get(name, 0.0))
        lines.append(
            f"{name:<{name_width}} | {ap:8.3f} | {ap50:8.3f} | {ar:8.3f} | {ar50:8.3f}"
        )
    lines.append(rule)
    return "\n".join(lines)


def _plot_per_class_pr_curves(
    coco_eval: COCOeval, dataset, plot_dir: str, iou: float = 0.5
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    precision = coco_eval.eval.get("precision")
    if precision is None:
        return

    iou_thrs = coco_eval.params.iouThrs
    if iou_thrs is None or len(iou_thrs) == 0:
        return
    iou_index = int((abs(iou_thrs - iou)).argmin())

    rec_thrs = coco_eval.params.recThrs
    if rec_thrs is None or len(rec_thrs) == 0:
        return

    cat_ids = coco_eval.params.catIds
    cat_id_to_name = _category_id_to_name(dataset)
    out_dir = Path(plot_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for idx, cat_id in enumerate(cat_ids):
        pr = precision[iou_index, :, idx, 0, -1]
        if pr is None:
            continue
        valid = pr > -1
        if not valid.any():
            continue
        recalls = rec_thrs[valid]
        precisions = pr[valid]
        auc = float(np.trapz(precisions, recalls)) if recalls.size > 0 else 0.0
        name = cat_id_to_name.get(cat_id, str(cat_id))
        safe_name = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in name)

        plt.figure(figsize=(5, 4))
        plt.plot(recalls, precisions, label=f"AUC={auc:.3f}")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"{name} (IoU={iou:.2f})")
        plt.grid(True, alpha=0.3)
        plt.legend(loc="best")
        plt.tight_layout()
        plt.savefig(out_dir / f"pr_{safe_name}.png", dpi=150)
        plt.close()


def _resolve_annotation_path(dataset) -> str | None:
    root = getattr(dataset, "root", None)
    split = getattr(dataset, "split", None)
    if root and split:
        return os.path.join(root, split, "annotations", f"{split}.json")
    return None


@contextmanager
def _suppress_stdout(enabled: bool):
    if not enabled:
        yield
        return
    with redirect_stdout(_NullWriter()):
        yield


class _NullWriter:
    def write(self, _: str) -> int:
        return 0

    def flush(self) -> None:
        return None


def _flatten_annotations(dataset) -> List[Dict]:
    anns = []
    ann_map = getattr(dataset, "_annotations_by_image_id", {})
    for image_id, items in ann_map.items():
        for ann in items:
            if "id" not in ann:
                ann = dict(ann)
                ann["id"] = len(anns) + 1
            anns.append(ann)
    return anns


def _build_categories(dataset) -> List[Dict]:
    cat_ids = getattr(dataset, "category_ids", [])
    cat_names = getattr(dataset, "category_names", {})
    categories = []
    for cat_id in cat_ids:
        categories.append({"id": cat_id, "name": cat_names.get(cat_id, str(cat_id))})
    return categories


def _category_id_to_name(dataset) -> Dict[int, str]:
    cat_names = getattr(dataset, "category_names", {})
    if isinstance(cat_names, dict):
        return {int(k): v for k, v in cat_names.items()}
    return {}


def _unwrap_dataset(dataset) -> Tuple[object, Optional[set[int]]]:
    if isinstance(dataset, Subset):
        base = dataset.dataset
        indices = dataset.indices
        image_ids = None
        if hasattr(base, "images"):
            image_ids = {int(base.images[i]["id"]) for i in indices}
        return base, image_ids
    return dataset, None


def _filter_coco(coco: COCO, image_ids: set[int]) -> Dict:
    images = [img for img in coco.dataset.get("images", []) if img["id"] in image_ids]
    anns = [ann for ann in coco.dataset.get("annotations", []) if ann["image_id"] in image_ids]
    cats = coco.dataset.get("categories", [])
    return {"images": images, "annotations": anns, "categories": cats}
