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
        val_loss = self._compute_val_loss()
        if val_loss is not None:
            metrics["loss"] = val_loss
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

    @torch.no_grad()
    def _compute_val_loss(self) -> Optional[float]:
        if self.val_loader is None:
            return None
        was_training = self.model.training
        self.model.train()
        total_loss = 0.0
        total_steps = 0
        autocast_ctx = _autocast() if self.amp else nullcontext()
        for images, targets in self.val_loader:
            images = [img.to(self.device) for img in images]
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
            with autocast_ctx:
                losses, _ = self.model(images, targets)
                loss = sum(losses.values())
            total_loss += float(loss.item())
            total_steps += 1
        if was_training:
            self.model.train()
        else:
            self.model.eval()
        if total_steps == 0:
            return None
        return total_loss / total_steps

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


class DecoupledTrainer(Trainer):
    """Decoupled training for long-tailed recognition.

    Two-stage training:
    1. Stage 1: Train full model with instance-balanced sampling
    2. Stage 2: Freeze backbone, retrain classifier with class-balanced sampling

    Reference: ICLR 2020 "Decoupling Representation and Classifier for
    Long-Tailed Recognition"
    """

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
        stage1_epochs: int = 30,
        stage2_epochs: int = 10,
        stage2_loader: Optional[DataLoader] = None,
        reinit_classifier: bool = False,
        classifier_lr_mult: float = 1.0,
    ):
        """Initialize DecoupledTrainer.

        Args:
            stage1_epochs: Number of epochs for stage 1 (representation learning).
            stage2_epochs: Number of epochs for stage 2 (classifier retraining).
            stage2_loader: DataLoader for stage 2 (class-balanced sampling).
                          If None, uses train_loader.
            reinit_classifier: Whether to reinitialize classifier before stage 2.
            classifier_lr_mult: Learning rate multiplier for classifier in stage 2.
        """
        total_epochs = stage1_epochs + stage2_epochs
        super().__init__(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            output_dir=output_dir,
            log_interval=log_interval,
            eval_interval=eval_interval,
            save_interval=save_interval,
            max_epochs=total_epochs,
            grad_clip=grad_clip,
            amp=amp,
            train_metric_samples=train_metric_samples,
            early_stop_patience=early_stop_patience,
            early_stop_min_delta=early_stop_min_delta,
            early_stop_metric=early_stop_metric,
        )
        self.stage1_epochs = stage1_epochs
        self.stage2_epochs = stage2_epochs
        self.stage2_loader = stage2_loader
        self.reinit_classifier = reinit_classifier
        self.classifier_lr_mult = classifier_lr_mult
        self._stage = 1
        self._original_train_loader = train_loader

    def train(self) -> Dict[str, float]:
        """Run decoupled training."""
        # Stage 1: Representation learning
        self.logger.info("Stage 1: Representation learning")
        print(f"=== Stage 1: Representation learning ({self.stage1_epochs} epochs) ===")

        for epoch in range(self.current_epoch, self.stage1_epochs):
            self.current_epoch = epoch
            self._stage = 1
            self._run_epoch()

        # Transition to Stage 2
        self._transition_to_stage2()

        # Stage 2: Classifier retraining
        self.logger.info("Stage 2: Classifier retraining")
        print(f"=== Stage 2: Classifier retraining ({self.stage2_epochs} epochs) ===")

        for epoch in range(self.stage1_epochs, self.stage1_epochs + self.stage2_epochs):
            self.current_epoch = epoch
            self._stage = 2
            self._run_epoch()

        summary = {"best_mAP": self.best_metric}
        if self._last_train_map50 is not None:
            summary["last_train_mAP_50"] = self._last_train_map50
        return summary

    def _run_epoch(self) -> None:
        """Run a single epoch with logging and evaluation."""
        train_metrics = self._train_epoch()
        history: Dict[str, float | int | str] = {
            "epoch": self.current_epoch + 1,
            "stage": self._stage,
        }
        history.update(_filter_scalar_metrics(train_metrics, prefix="train_"))

        if self.train_metric_samples > 0:
            self._last_train_map50 = self._compute_train_map50()
            if self._last_train_map50 is not None:
                print(f"Train mAP50 (subset): {self._last_train_map50:.3f}")
                train_metrics["train_mAP_50"] = self._last_train_map50
                history["train_mAP_50"] = self._last_train_map50

        if self.val_loader is not None and (self.current_epoch + 1) % self.eval_interval == 0:
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

        if (self.current_epoch + 1) % self.save_interval == 0:
            self._save_checkpoint(f"epoch_{self.current_epoch + 1}.pth")

        if self.scheduler is not None:
            self.scheduler.step()

        history["best_mAP"] = self.best_metric
        history["timestamp"] = datetime.utcnow().isoformat(timespec="seconds") + "Z"
        self._append_metrics_history(history)

    def _transition_to_stage2(self) -> None:
        """Transition from stage 1 to stage 2."""
        print("Transitioning to Stage 2...")

        # Freeze backbone
        self._freeze_backbone()

        # Switch to stage 2 data loader if provided
        if self.stage2_loader is not None:
            self.train_loader = self.stage2_loader

        # Reinitialize classifier if requested
        if self.reinit_classifier:
            self._reinit_classifier_weights()

        # Adjust optimizer for stage 2
        self._adjust_optimizer_for_stage2()

        # Save stage 1 checkpoint
        self._save_checkpoint("stage1_final.pth")

    def _freeze_backbone(self) -> None:
        """Freeze backbone parameters."""
        frozen_count = 0
        trainable_count = 0
        for name, param in self.model.named_parameters():
            # Freeze everything except classifier/head layers
            if not self._is_classifier_param(name):
                param.requires_grad = False
                frozen_count += 1
            else:
                trainable_count += 1

        # If no classifier params found, keep all params trainable
        if trainable_count == 0:
            print("Warning: No classifier parameters found, keeping all params trainable")
            for param in self.model.parameters():
                param.requires_grad = True
            return

        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        print(f"Frozen {frozen_count} parameter groups")
        print(f"Trainable parameters: {trainable:,} / {total:,}")

    def _is_classifier_param(self, name: str) -> bool:
        """Check if parameter belongs to classifier."""
        classifier_keywords = [
            "head", "classifier", "fc", "cls", "box_predictor",
            "rpn_head", "roi_heads", "class_logits", "bbox_pred"
        ]
        name_lower = name.lower()
        return any(kw in name_lower for kw in classifier_keywords)

    def _reinit_classifier_weights(self) -> None:
        """Reinitialize classifier weights."""
        for name, module in self.model.named_modules():
            if self._is_classifier_param(name):
                if hasattr(module, "reset_parameters"):
                    module.reset_parameters()
                    print(f"Reinitialized: {name}")

    def _adjust_optimizer_for_stage2(self) -> None:
        """Adjust optimizer learning rates for stage 2."""
        if self.classifier_lr_mult == 1.0:
            return

        for param_group in self.optimizer.param_groups:
            param_group["lr"] *= self.classifier_lr_mult
