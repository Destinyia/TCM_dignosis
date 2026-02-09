from __future__ import annotations

from pathlib import Path
from typing import Dict
import json


def plot_per_class_ap_ar(
    per_class_ap: Dict[str, float],
    per_class_ar: Dict[str, float],
    out_path: str | Path,
    title: str = "Per-class AP/AR",
) -> bool:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return False

    if not per_class_ap and not per_class_ar:
        return False

    names = sorted(
        set(per_class_ap.keys()) | set(per_class_ar.keys()),
        key=lambda n: per_class_ap.get(n, 0.0),
        reverse=True,
    )
    ap_values = [float(per_class_ap.get(n, 0.0)) for n in names]
    ar_values = [float(per_class_ar.get(n, 0.0)) for n in names]

    height = max(4.0, 0.35 * len(names))
    fig, axes = plt.subplots(1, 2, figsize=(12, height), sharey=True)
    fig.suptitle(title)

    y_pos = list(range(len(names)))
    axes[0].barh(y_pos, ap_values, color="#4C72B0")
    axes[0].set_xlabel("AP")
    axes[0].invert_yaxis()
    axes[0].grid(True, axis="x", alpha=0.3)
    axes[0].set_yticks(y_pos, names)

    axes[1].barh(y_pos, ar_values, color="#55A868")
    axes[1].set_xlabel("AR")
    axes[1].invert_yaxis()
    axes[1].grid(True, axis="x", alpha=0.3)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return True


def plot_loss_curves(history_path: str | Path, out_path: str | Path) -> bool:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return False

    history_path = Path(history_path)
    if not history_path.is_file():
        return False

    epochs = []
    train_loss = []
    val_loss = []
    for line in history_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if "epoch" in record:
            epochs.append(record["epoch"])
            train_loss.append(record.get("train_train_loss"))
            val_loss.append(record.get("val_loss"))

    if not epochs:
        return False

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(epochs, train_loss, label="train_loss")
    if any(v is not None for v in val_loss):
        ax.plot(epochs, val_loss, label="val_loss")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return True
