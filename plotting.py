"""Shared training curve plots."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _label_hline(ax, x: float, y: float, text: str, color: str, va: str = "center") -> None:
    ax.text(
        x,
        y,
        text,
        color=color,
        fontsize=8,
        va=va,
        ha="left",
        bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "edgecolor": color, "alpha": 0.85},
    )


def plot_loss_accuracy(
    history: Mapping[str, Sequence[float]],
    *,
    title_prefix: str,
    out_path: Path,
    dpi: int = 150,
) -> None:
    """Plot train/val loss and accuracy with dotted extrema reference lines."""
    epochs = np.asarray(history["epoch"])
    train_loss = np.asarray(history["train_loss"])
    val_loss = np.asarray(history["val_loss"])
    train_acc = np.asarray(history["train_acc"])
    val_acc = np.asarray(history["val_acc"])

    train_color = "C0"
    val_color = "C1"

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, train_loss, marker="o", color=train_color, label="Train")
    axes[0].plot(epochs, val_loss, marker="o", color=val_color, label="Validation")

    min_train_loss = float(train_loss.min())
    min_val_loss = float(val_loss.min())
    min_train_loss_ep = int(epochs[train_loss.argmin()])
    min_val_loss_ep = int(epochs[val_loss.argmin()])
    axes[0].axhline(
        min_train_loss,
        linestyle=":",
        color=train_color,
        linewidth=1.5,
        label="Train min",
    )
    axes[0].axhline(
        min_val_loss,
        linestyle=":",
        color=val_color,
        linewidth=1.5,
        label="Val min",
    )
    x_label = float(epochs.max())
    _label_hline(
        axes[0],
        x_label,
        min_train_loss,
        f" {min_train_loss:.4f} (ep {min_train_loss_ep})",
        train_color,
        va="bottom" if min_train_loss >= min_val_loss else "top",
    )
    _label_hline(
        axes[0],
        x_label,
        min_val_loss,
        f" {min_val_loss:.4f} (ep {min_val_loss_ep})",
        val_color,
        va="top" if min_train_loss >= min_val_loss else "bottom",
    )
    axes[0].set_title(f"Loss ({title_prefix})")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-entropy")
    axes[0].legend(fontsize=8)
    axes[0].margins(x=0.08)

    axes[1].plot(epochs, train_acc, marker="o", color=train_color, label="Train")
    axes[1].plot(epochs, val_acc, marker="o", color=val_color, label="Validation")

    max_train_acc = float(train_acc.max())
    max_val_acc = float(val_acc.max())
    max_train_acc_ep = int(epochs[train_acc.argmax()])
    max_val_acc_ep = int(epochs[val_acc.argmax()])
    axes[1].axhline(
        max_train_acc,
        linestyle=":",
        color=train_color,
        linewidth=1.5,
        label="Train max",
    )
    axes[1].axhline(
        max_val_acc,
        linestyle=":",
        color=val_color,
        linewidth=1.5,
        label="Val max",
    )
    _label_hline(
        axes[1],
        x_label,
        max_train_acc,
        f" {max_train_acc:.4f} (ep {max_train_acc_ep})",
        train_color,
        va="bottom" if max_train_acc >= max_val_acc else "top",
    )
    _label_hline(
        axes[1],
        x_label,
        max_val_acc,
        f" {max_val_acc:.4f} (ep {max_val_acc_ep})",
        val_color,
        va="top" if max_train_acc >= max_val_acc else "bottom",
    )
    axes[1].set_title(f"Accuracy ({title_prefix})")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0, 1.05)
    axes[1].legend(fontsize=8)
    axes[1].margins(x=0.08)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
