from __future__ import annotations

import logging
import os
from contextlib import nullcontext
import json
from datetime import datetime
from typing import Callable, Dict, Optional

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm


class Trainer:
    """Model trainer."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        optimizer: Optimizer,
        scheduler: Optional[_LRScheduler] = None,
        device: str = "cuda",
        output_dir: str = "runs/train",
        log_interval: int = 50,
        eval_interval: int = 1,
        save_interval: int = 5,
        max_epochs: int = 50,
        grad_clip: Optional[float] = None,
        amp: bool = True,
        train_metric_samples: int = 0,
        early_stop_patience: int = 0,
        early_stop_min_delta: float = 0.0,
        early_stop_metric: str = "mAP",
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.output_dir = output_dir
        self.log_interval = log_interval
        self.eval_interval = eval_interval
        self.save_interval = save_interval
        self.max_epochs = max_epochs
        self.grad_clip = grad_clip
        self.amp = amp and device.startswith("cuda") and torch.cuda.is_available()
        self.train_metric_samples = train_metric_samples
        self._train_metric_loader: Optional[DataLoader] = None
        self._train_metric_evaluator = None
        self._last_train_map50: Optional[float] = None
        self.early_stop_patience = max(int(early_stop_patience), 0)
        self.early_stop_min_delta = float(early_stop_min_delta)
        self.early_stop_metric = str(early_stop_metric)
        self._early_stop_counter = 0

        self.scaler = _create_grad_scaler() if self.amp else None
        self.logger = logging.getLogger(__name__)
        self.best_metric = 0.0
        self.current_epoch = 0

        self.callbacks: Dict[str, Callable] = {}

    def train(self) -> Dict[str, float]:
        """Run full training loop."""
        for epoch in range(self.current_epoch, self.max_epochs):
            self.current_epoch = epoch

            train_metrics = self._train_epoch()
            history: Dict[str, float | int | str] = {"epoch": self.current_epoch + 1}
            history.update(_filter_scalar_metrics(train_metrics, prefix="train_"))
            if self.train_metric_samples > 0:
                self._last_train_map50 = self._compute_train_map50()
                if self._last_train_map50 is not None:
                    print(f"Train mAP50 (subset): {self._last_train_map50:.3f}")
                    train_metrics["train_mAP_50"] = self._last_train_map50
                    history["train_mAP_50"] = self._last_train_map50

            val_metrics = None
            if self.val_loader is not None and (epoch + 1) % self.eval_interval == 0:
                val_metrics = self._evaluate()
                current_metric = float(val_metrics.get(self.early_stop_metric, 0.0))
                if current_metric > self.best_metric + self.early_stop_min_delta:
                    self.best_metric = current_metric
                    self._save_checkpoint("best.pth")
                    self._early_stop_counter = 0
                else:
                    if self.early_stop_patience > 0:
                        self._early_stop_counter += 1
                history.update(_filter_scalar_metrics(val_metrics, prefix="val_"))
                history["per_class_AP"] = val_metrics.get("per_class_AP", {})
                history["per_class_AP50"] = val_metrics.get("per_class_AP50", {})
                history["per_class_AR"] = val_metrics.get("per_class_AR", {})
                history["per_class_AR50"] = val_metrics.get("per_class_AR50", {})

            if (epoch + 1) % self.save_interval == 0:
                self._save_checkpoint(f"epoch_{epoch + 1}.pth")

            if self.scheduler is not None:
                self.scheduler.step()

            history["best_mAP"] = self.best_metric
            history["timestamp"] = datetime.utcnow().isoformat(timespec="seconds") + "Z"
            self._append_metrics_history(history)

            if self.early_stop_patience > 0 and self._early_stop_counter >= self.early_stop_patience:
                print(
                    f"Early stopping at epoch {self.current_epoch + 1} "
                    f"(metric={self.early_stop_metric}, "
                    f"patience={self.early_stop_patience})."
                )
                break

        summary = {"best_mAP": self.best_metric}
        if self._last_train_map50 is not None:
            summary["last_train_mAP_50"] = self._last_train_map50
        return summary

    def _train_epoch(self) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        total_steps = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1}")
        for batch_idx, (images, targets) in enumerate(pbar):
            images = [img.to(self.device) for img in images]
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

            self.optimizer.zero_grad()

            autocast_ctx = _autocast() if self.amp else nullcontext()
            with autocast_ctx:
                losses, _ = self.model(images, targets)
                loss = sum(losses.values())

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
            total_steps += 1

            if (batch_idx + 1) % self.log_interval == 0:
                postfix = {"loss": f"{loss.item():.3f}"}
                mem = _format_cuda_mem(self.device)
                if mem:
                    postfix["mem"] = mem
                pbar.set_postfix(postfix)

        avg_loss = total_loss / max(total_steps, 1)
        return {"train_loss": avg_loss}

    def _evaluate(self) -> Dict[str, float]:
        from .evaluator import COCOEvaluator, format_coco_metrics, format_per_class_metrics

        evaluator = COCOEvaluator(self.val_loader.dataset, silent=True)
        metrics = evaluator.evaluate(self.model, self.val_loader, self.device)
        print("Val bbox")
        print(format_coco_metrics(metrics))
        print("Per-class AP/AR")
        print(
            format_per_class_metrics(
                metrics.get("per_class_AP", {}),
                metrics.get("per_class_AP50", {}),
                metrics.get("per_class_AR", {}),
                metrics.get("per_class_AR50", {}),
            )
        )
        return metrics

    def _compute_train_map50(self) -> Optional[float]:
        if self.train_metric_samples <= 0:
            return None
        if self._train_metric_loader is None:
            dataset = self.train_loader.dataset
            base_dataset = dataset
            base_indices = None
            if isinstance(dataset, Subset):
                base_dataset = dataset.dataset
                base_indices = list(dataset.indices)
            total = len(dataset)
            if total == 0:
                return None
            size = min(self.train_metric_samples, total)
            if base_indices is None:
                indices = list(range(size))
                subset = Subset(base_dataset, indices)
            else:
                indices = base_indices[:size]
                subset = Subset(base_dataset, indices)
            self._train_metric_loader = DataLoader(
                subset,
                batch_size=self.train_loader.batch_size,
                shuffle=False,
                num_workers=self.train_loader.num_workers,
                collate_fn=getattr(base_dataset, "collate_fn", None),
            )

        from .evaluator import COCOEvaluator

        if self._train_metric_evaluator is None:
            self._train_metric_evaluator = COCOEvaluator(
                self._train_metric_loader.dataset, silent=True
            )

        was_training = self.model.training
        self.model.eval()
        metrics = self._train_metric_evaluator.evaluate(
            self.model, self._train_metric_loader, self.device
        )
        if was_training:
            self.model.train()
        return float(metrics.get("mAP_50", 0.0))

    def _save_checkpoint(self, filename: str) -> None:
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
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model"])
        if checkpoint.get("optimizer"):
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        if self.scheduler is not None and checkpoint.get("scheduler") is not None:
            self.scheduler.load_state_dict(checkpoint["scheduler"])
        self.current_epoch = checkpoint.get("epoch", 0)
        self.best_metric = checkpoint.get("best_metric", 0.0)
        return {"epoch": self.current_epoch, "best_metric": self.best_metric}

    def register_callback(self, event: str, callback: Callable) -> None:
        self.callbacks[event] = callback

    def _append_metrics_history(self, record: Dict[str, float | int | str]) -> None:
        os.makedirs(self.output_dir, exist_ok=True)
        path = os.path.join(self.output_dir, "metrics_history.jsonl")
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _create_grad_scaler() -> torch.amp.GradScaler:
    try:
        return torch.amp.GradScaler("cuda")
    except Exception:
        return torch.cuda.amp.GradScaler()


def _autocast() -> torch.amp.autocast:
    try:
        return torch.amp.autocast("cuda")
    except Exception:
        return torch.cuda.amp.autocast()


def _format_cuda_mem(device: str) -> str | None:
    if not device.startswith("cuda") or not torch.cuda.is_available():
        return None
    allocated = torch.cuda.memory_allocated() / (1024**3)
    reserved = torch.cuda.memory_reserved() / (1024**3)
    free_total = ""
    try:
        free, total = torch.cuda.mem_get_info()
        used = total - free
        free_total = f" {used / (1024**3):.2f}G/{total / (1024**3):.2f}G"
    except Exception:
        pass
    return f"{allocated:.2f}G/{reserved:.2f}G{free_total}"


def _filter_scalar_metrics(metrics: Dict[str, object], prefix: str = "") -> Dict[str, float]:
    filtered: Dict[str, float] = {}
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            filtered[f"{prefix}{key}"] = float(value)
    return filtered
