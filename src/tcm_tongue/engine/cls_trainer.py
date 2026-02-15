"""分类训练器 - 用于舌象分类模型的训练"""
from __future__ import annotations

import json
import logging
import os
from contextlib import nullcontext
from datetime import datetime
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


class ClassificationTrainer:
    """分类模型训练器"""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        optimizer: Optimizer,
        criterion: nn.Module,
        scheduler: Optional[_LRScheduler] = None,
        device: str = "cuda",
        output_dir: str = "runs/classification",
        log_interval: int = 20,
        eval_interval: int = 1,
        save_interval: int = 5,
        max_epochs: int = 50,
        grad_clip: Optional[float] = None,
        amp: bool = True,
        early_stop_patience: int = 10,
        early_stop_min_delta: float = 0.0,
        class_names: Optional[Dict[int, str]] = None,
        mask_bbox_loss: Optional[nn.Module] = None,
        mask_loss_weight: float = 0.1,
        use_bbox: bool = False,
        mask_seg_loss: Optional[nn.Module] = None,
        seg_loss_weight: float = 0.2,
        use_seg: bool = False,
        visualize_samples: bool = False,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.device = device
        self.output_dir = output_dir
        self.log_interval = log_interval
        self.eval_interval = eval_interval
        self.save_interval = save_interval
        self.max_epochs = max_epochs
        self.grad_clip = grad_clip
        self.amp = amp and device.startswith("cuda") and torch.cuda.is_available()
        self.early_stop_patience = max(int(early_stop_patience), 0)
        self.early_stop_min_delta = float(early_stop_min_delta)
        self.class_names = class_names or {}

        # bbox损失相关
        self.mask_bbox_loss = mask_bbox_loss
        self.mask_loss_weight = mask_loss_weight
        self.use_bbox = use_bbox
        self.mask_seg_loss = mask_seg_loss
        self.seg_loss_weight = seg_loss_weight
        self.use_seg = use_seg
        self.visualize_samples = visualize_samples

        self.scaler = self._create_grad_scaler() if self.amp else None
        self.logger = logging.getLogger(__name__)
        self.best_metric = 0.0
        self.current_epoch = 0
        self._early_stop_counter = 0

    def _create_grad_scaler(self):
        try:
            return torch.amp.GradScaler("cuda")
        except Exception:
            return torch.cuda.amp.GradScaler()

    def _autocast(self):
        try:
            return torch.amp.autocast("cuda")
        except Exception:
            return torch.cuda.amp.autocast()

    def train(self) -> Dict[str, float]:
        """运行完整训练循环"""
        os.makedirs(self.output_dir, exist_ok=True)

        for epoch in range(self.current_epoch, self.max_epochs):
            self.current_epoch = epoch

            # 训练一个epoch
            train_metrics = self._train_epoch()
            history = {"epoch": epoch + 1}
            history.update(train_metrics)

            # 验证
            val_metrics = None
            if self.val_loader is not None and (epoch + 1) % self.eval_interval == 0:
                val_metrics = self._evaluate()
                history.update({f"val_{k}": v for k, v in val_metrics.items()})

                # 检查是否是最佳模型（使用macro_f1作为早停指标）
                current_metric = val_metrics.get("macro_f1", 0.0)
                if current_metric > self.best_metric + self.early_stop_min_delta:
                    self.best_metric = current_metric
                    self._save_checkpoint("best.pth")
                    self._early_stop_counter = 0
                    print(f"  New best macro_f1: {self.best_metric:.4f}")
                else:
                    self._early_stop_counter += 1

            # 定期保存
            if (epoch + 1) % self.save_interval == 0:
                self._save_checkpoint(f"epoch_{epoch + 1}.pth")

            # 学习率调度
            if self.scheduler is not None:
                self.scheduler.step()

            # 记录历史
            history["best_macro_f1"] = self.best_metric
            history["timestamp"] = datetime.utcnow().isoformat(timespec="seconds") + "Z"
            self._append_metrics_history(history)

            # 早停
            if self.early_stop_patience > 0 and self._early_stop_counter >= self.early_stop_patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

        return {"best_macro_f1": self.best_metric}

    def _train_epoch(self) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        total_cls_loss = 0.0
        total_mask_loss = 0.0
        total_seg_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1}")
        for batch_idx, batch in enumerate(pbar):
            # 解包数据（支持有/无bbox两种情况）
            seg_masks = None
            if self.use_bbox and self.use_seg and len(batch) >= 4:
                images, labels, bboxes, seg_masks = batch[:4]
                bboxes = bboxes.to(self.device)
            elif self.use_bbox and len(batch) >= 3:
                images, labels, bboxes = batch[:3]
                bboxes = bboxes.to(self.device)
            elif self.use_seg and len(batch) >= 3:
                images, labels, seg_masks = batch[:3]
            else:
                images, labels = batch[:2]
                bboxes = None

            images = images.to(self.device)
            labels = labels.to(self.device)
            if seg_masks is not None:
                seg_masks = seg_masks.to(self.device).float()

            self.optimizer.zero_grad()

            autocast_ctx = self._autocast() if self.amp else nullcontext()
            with autocast_ctx:
                # 前向传播
                need_mask = (
                    (self.mask_bbox_loss is not None and bboxes is not None)
                    or (self.mask_seg_loss is not None and seg_masks is not None)
                )
                if need_mask:
                    outputs, mask = self.model(images, return_mask=True)
                else:
                    outputs = self.model(images)
                    mask = None

                cls_loss = self.criterion(outputs, labels)
                loss = cls_loss

                if self.mask_bbox_loss is not None and bboxes is not None and mask is not None:
                    mask_loss = self.mask_bbox_loss(mask, bboxes)
                    loss = loss + self.mask_loss_weight * mask_loss
                    total_mask_loss += mask_loss.item()

                if self.mask_seg_loss is not None and seg_masks is not None and mask is not None:
                    if seg_masks.ndim == 3:
                        seg_masks = seg_masks.unsqueeze(1)
                    if mask.shape[-2:] != seg_masks.shape[-2:]:
                        seg_masks_resized = F.interpolate(
                            seg_masks, size=mask.shape[-2:], mode="nearest"
                        )
                    else:
                        seg_masks_resized = seg_masks
                    seg_loss = self.mask_seg_loss(mask, seg_masks_resized)
                    loss = loss + self.seg_loss_weight * seg_loss
                    total_seg_loss += seg_loss.item()

                total_cls_loss += cls_loss.item()

            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                if self.grad_clip:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                if self.grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            if (batch_idx + 1) % self.log_interval == 0:
                acc = 100.0 * correct / total
                postfix = {"loss": f"{loss.item():.3f}", "acc": f"{acc:.1f}%"}
                if self.mask_bbox_loss is not None and bboxes is not None:
                    postfix["mask"] = f"{mask_loss.item():.3f}"
                if self.mask_seg_loss is not None and seg_masks is not None:
                    postfix["seg"] = f"{seg_loss.item():.3f}"
                pbar.set_postfix(postfix)

        avg_loss = total_loss / len(self.train_loader)
        avg_cls_loss = total_cls_loss / len(self.train_loader)
        accuracy = correct / total

        metrics = {"train_loss": avg_loss, "train_cls_loss": avg_cls_loss, "train_accuracy": accuracy}
        if self.mask_bbox_loss is not None and self.use_bbox:
            metrics["train_mask_loss"] = total_mask_loss / len(self.train_loader)
        if self.mask_seg_loss is not None and self.use_seg:
            metrics["train_seg_loss"] = total_seg_loss / len(self.train_loader)

        return metrics

    @torch.no_grad()
    def _evaluate(self) -> Dict[str, float]:
        """评估模型"""
        self.model.eval()
        total_loss = 0.0
        total_mask_loss = 0.0
        total_seg_loss = 0.0
        correct = 0
        total = 0

        all_preds = []
        all_labels = []

        # 用于可视化的样本收集
        vis_samples = []
        vis_count = 0
        max_vis = 16  # 4x4网格

        for batch in tqdm(self.val_loader, desc="Evaluating"):
            # 解包数据
            seg_masks = None
            if self.use_bbox and self.use_seg and len(batch) >= 4:
                images, labels, bboxes, seg_masks = batch[:4]
                bboxes = bboxes.to(self.device)
            elif self.use_bbox and len(batch) >= 3:
                images, labels, bboxes = batch[:3]
                bboxes = bboxes.to(self.device)
            elif self.use_seg and len(batch) >= 3:
                images, labels, seg_masks = batch[:3]
            else:
                images, labels = batch[:2]
                bboxes = None

            images = images.to(self.device)
            labels = labels.to(self.device)
            if seg_masks is not None:
                seg_masks = seg_masks.to(self.device).float()

            # 前向传播
            need_mask = (
                (self.mask_bbox_loss is not None and bboxes is not None)
                or (self.mask_seg_loss is not None and seg_masks is not None)
            )
            if need_mask:
                outputs, mask = self.model(images, return_mask=True)
            else:
                outputs = self.model(images)
                mask = None

            if self.mask_bbox_loss is not None and bboxes is not None and mask is not None:
                mask_loss = self.mask_bbox_loss(mask, bboxes)
                total_mask_loss += mask_loss.item()

                # 收集可视化样本
                if self.visualize_samples and vis_count < max_vis:
                    for i in range(min(images.size(0), max_vis - vis_count)):
                        vis_samples.append({
                            "image": images[i].cpu(),
                            "mask": mask[i].cpu(),
                            "bbox": bboxes[i].cpu(),
                            "label": labels[i].item(),
                            "pred": outputs[i].argmax().item(),
                        })
                        vis_count += 1
                        if vis_count >= max_vis:
                            break

            if self.mask_seg_loss is not None and seg_masks is not None and mask is not None:
                if seg_masks.ndim == 3:
                    seg_masks = seg_masks.unsqueeze(1)
                if mask.shape[-2:] != seg_masks.shape[-2:]:
                    seg_masks_resized = F.interpolate(
                        seg_masks, size=mask.shape[-2:], mode="nearest"
                    )
                else:
                    seg_masks_resized = seg_masks
                seg_loss = self.mask_seg_loss(mask, seg_masks_resized)
                total_seg_loss += seg_loss.item()

            loss = self.criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / len(self.val_loader)
        accuracy = correct / total

        # 计算每类指标
        per_class_metrics = self._compute_per_class_metrics(all_preds, all_labels)

        print(f"\nVal Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
        if self.mask_bbox_loss is not None and self.use_bbox:
            avg_mask_loss = total_mask_loss / len(self.val_loader)
            print(f"Mask Loss: {avg_mask_loss:.4f}")
        if self.mask_seg_loss is not None and self.use_seg:
            avg_seg_loss = total_seg_loss / len(self.val_loader)
            print(f"Seg Loss: {avg_seg_loss:.4f}")
        self._print_per_class_metrics(per_class_metrics)

        # 可视化
        if self.visualize_samples and vis_samples:
            self._visualize_mask_bbox(vis_samples)

        metrics = {
            "loss": avg_loss,
            "accuracy": accuracy,
            **per_class_metrics,
        }
        if self.mask_bbox_loss is not None and self.use_bbox:
            metrics["mask_loss"] = total_mask_loss / len(self.val_loader)
        if self.mask_seg_loss is not None and self.use_seg:
            metrics["seg_loss"] = total_seg_loss / len(self.val_loader)

        return metrics

    def _visualize_mask_bbox(self, samples: List[Dict]) -> None:
        """可视化掩码和bbox的对齐情况

        左侧4x4: 原始图像 + bbox
        右侧4x4: 掩码 + bbox
        """
        if not HAS_MATPLOTLIB:
            print("Warning: matplotlib not available, skipping visualization")
            return

        n = min(len(samples), 16)
        rows = 4
        cols = 8  # 左4列图像，右4列掩码
        fig, axes = plt.subplots(rows, cols, figsize=(24, 12))

        # ImageNet反归一化参数
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

        for idx in range(16):
            row = idx // 4
            col_img = idx % 4        # 左侧图像列 (0-3)
            col_mask = idx % 4 + 4   # 右侧掩码列 (4-7)

            ax_img = axes[row, col_img]
            ax_mask = axes[row, col_mask]

            if idx >= n:
                ax_img.axis("off")
                ax_mask.axis("off")
                continue

            sample = samples[idx]
            img = sample["image"]
            mask = sample["mask"]
            bbox = sample["bbox"]
            label = sample["label"]
            pred = sample["pred"]

            # 反归一化图像
            img = img * std + mean
            img = img.clamp(0, 1).permute(1, 2, 0).numpy()

            # 获取掩码
            mask_np = mask.squeeze().numpy()
            H, W = img.shape[:2]

            # 调整掩码尺寸
            if mask_np.shape[0] != H or mask_np.shape[1] != W:
                mask_resized = F.interpolate(
                    mask.unsqueeze(0), size=(H, W), mode="bilinear", align_corners=False
                )
                mask_np = mask_resized.squeeze().numpy()

            # bbox坐标
            x1, y1, x2, y2 = bbox.numpy()

            # 左侧：原始图像 + bbox
            ax_img.imshow(img)
            rect_img = patches.Rectangle(
                (x1 * W, y1 * H), (x2 - x1) * W, (y2 - y1) * H,
                linewidth=2, edgecolor="lime", facecolor="none"
            )
            ax_img.add_patch(rect_img)

            # 标题（预测结果）
            label_name = self.class_names.get(label, str(label))
            pred_name = self.class_names.get(pred, str(pred))
            color = "green" if label == pred else "red"
            ax_img.set_title(f"GT: {label_name}\nPred: {pred_name}", fontsize=8, color=color)
            ax_img.axis("off")

            # 右侧：掩码 + bbox
            ax_mask.imshow(mask_np, cmap="hot", vmin=0, vmax=1)
            rect_mask = patches.Rectangle(
                (x1 * W, y1 * H), (x2 - x1) * W, (y2 - y1) * H,
                linewidth=2, edgecolor="lime", facecolor="none"
            )
            ax_mask.add_patch(rect_mask)
            ax_mask.set_title(f"Mask #{idx+1}", fontsize=8)
            ax_mask.axis("off")

        # 添加列标题
        fig.text(0.25, 0.98, "Original Images + BBox", ha="center", fontsize=14, fontweight="bold")
        fig.text(0.75, 0.98, "Masks + BBox", ha="center", fontsize=14, fontweight="bold")

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        save_path = os.path.join(self.output_dir, f"mask_bbox_vis_epoch{self.current_epoch + 1}.png")
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Visualization saved to: {save_path}")

    def _compute_per_class_metrics(
        self, preds: List[int], labels: List[int]
    ) -> Dict[str, float]:
        """计算每类的精确率、召回率、F1"""
        import numpy as np
        from collections import defaultdict

        preds = np.array(preds)
        labels = np.array(labels)

        classes = np.unique(np.concatenate([preds, labels]))
        metrics = {}

        precisions = []
        recalls = []
        f1s = []

        for cls in classes:
            tp = np.sum((preds == cls) & (labels == cls))
            fp = np.sum((preds == cls) & (labels != cls))
            fn = np.sum((preds != cls) & (labels == cls))

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

            cls_name = self.class_names.get(cls, str(cls))
            metrics[f"precision_{cls_name}"] = precision
            metrics[f"recall_{cls_name}"] = recall
            metrics[f"f1_{cls_name}"] = f1

            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)

        # 宏平均
        metrics["macro_precision"] = np.mean(precisions)
        metrics["macro_recall"] = np.mean(recalls)
        metrics["macro_f1"] = np.mean(f1s)

        return metrics

    def _print_per_class_metrics(self, metrics: Dict[str, float]) -> None:
        """打印每类指标"""
        print("\nPer-class metrics:")
        print(f"{'Class':<20} {'Precision':<12} {'Recall':<12} {'F1':<12}")
        print("-" * 56)

        # 提取类别
        classes = set()
        for key in metrics:
            if key.startswith("precision_"):
                cls_name = key.replace("precision_", "")
                classes.add(cls_name)

        for cls_name in sorted(classes):
            p = metrics.get(f"precision_{cls_name}", 0.0)
            r = metrics.get(f"recall_{cls_name}", 0.0)
            f1 = metrics.get(f"f1_{cls_name}", 0.0)
            print(f"{cls_name:<20} {p:<12.4f} {r:<12.4f} {f1:<12.4f}")

        print("-" * 56)
        print(f"{'Macro Average':<20} {metrics.get('macro_precision', 0):<12.4f} "
              f"{metrics.get('macro_recall', 0):<12.4f} {metrics.get('macro_f1', 0):<12.4f}")

    def _save_checkpoint(self, filename: str) -> None:
        """保存检查点"""
        os.makedirs(self.output_dir, exist_ok=True)
        path = os.path.join(self.output_dir, filename)
        payload = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.scheduler else None,
            "epoch": self.current_epoch,
            "best_metric": self.best_metric,
        }
        torch.save(payload, path)

    def load_checkpoint(self, path: str) -> Dict[str, float]:
        """加载检查点"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model"])
        if checkpoint.get("optimizer"):
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        if self.scheduler is not None and checkpoint.get("scheduler"):
            self.scheduler.load_state_dict(checkpoint["scheduler"])
        self.current_epoch = checkpoint.get("epoch", 0)
        self.best_metric = checkpoint.get("best_metric", 0.0)
        return {"epoch": self.current_epoch, "best_metric": self.best_metric}

    def _append_metrics_history(self, record: Dict) -> None:
        """追加指标历史"""
        path = os.path.join(self.output_dir, "metrics_history.jsonl")
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


class ClassificationEvaluator:
    """分类模型评估器"""

    def __init__(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        device: str = "cuda",
        class_names: Optional[Dict[int, str]] = None,
    ):
        self.model = model.to(device)
        self.dataloader = dataloader
        self.device = device
        self.class_names = class_names or {}

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """评估模型"""
        self.model.eval()
        all_preds = []
        all_labels = []
        all_probs = []

        for images, labels in tqdm(self.dataloader, desc="Evaluating"):
            images = images.to(self.device)
            outputs = self.model(images)
            probs = torch.softmax(outputs, dim=1)

            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

        import numpy as np
        preds = np.array(all_preds)
        labels = np.array(all_labels)
        probs = np.array(all_probs)

        # 基本指标
        accuracy = np.mean(preds == labels)

        # 混淆矩阵
        num_classes = probs.shape[1]
        confusion = np.zeros((num_classes, num_classes), dtype=np.int32)
        for p, l in zip(preds, labels):
            confusion[l, p] += 1

        # 每类指标
        per_class = {}
        for i in range(num_classes):
            tp = confusion[i, i]
            fp = confusion[:, i].sum() - tp
            fn = confusion[i, :].sum() - tp

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

            cls_name = self.class_names.get(i, str(i))
            per_class[cls_name] = {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": int(confusion[i, :].sum()),
            }

        return {
            "accuracy": accuracy,
            "confusion_matrix": confusion.tolist(),
            "per_class": per_class,
        }

    def print_report(self, metrics: Dict) -> None:
        """打印评估报告"""
        print(f"\nAccuracy: {metrics['accuracy']:.4f}")
        print("\nClassification Report:")
        print(f"{'Class':<20} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Support':<10}")
        print("-" * 66)

        per_class = metrics.get("per_class", {})
        for cls_name, m in sorted(per_class.items()):
            print(f"{cls_name:<20} {m['precision']:<12.4f} {m['recall']:<12.4f} "
                  f"{m['f1']:<12.4f} {m['support']:<10}")

        print("\nConfusion Matrix:")
        confusion = metrics.get("confusion_matrix", [])
        if confusion:
            import numpy as np
            cm = np.array(confusion)
            print(cm)
