from __future__ import annotations

from pathlib import Path
from typing import Dict


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
