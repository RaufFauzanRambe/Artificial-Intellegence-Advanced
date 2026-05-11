"""
Visualization module for the Artificial Intelligence Advanced project.

Provides publication-quality plotting functions built on Matplotlib for
confusion matrices, ROC curves, precision-recall curves, training histories,
feature importance, and data distributions.  Every function can save to file
and/or return a Matplotlib figure object.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless environments

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Global style defaults
# ------------------------------------------------------------------

DEFAULT_FIGSIZE = (8, 6)
DEFAULT_DPI = 150
DEFAULT_FONTSIZE = 12


def _apply_defaults() -> None:
    """Set sensible Matplotlib rc defaults."""
    plt.rcParams.update({
        "figure.dpi": DEFAULT_DPI,
        "font.size": DEFAULT_FONTSIZE,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.figsize": DEFAULT_FIGSIZE,
    })


_apply_defaults()


# ------------------------------------------------------------------
# Confusion matrix
# ------------------------------------------------------------------

def plot_confusion_matrix(
    cm: Union[np.ndarray, List[List[int]]],
    class_names: Optional[List[str]] = None,
    normalize: bool = False,
    title: str = "Confusion Matrix",
    cmap: str = "Blues",
    save_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = DEFAULT_FIGSIZE,
    show: bool = False,
) -> plt.Figure:
    """Plot a confusion matrix as a heatmap.

    Parameters
    ----------
    cm : array-like of shape (n_classes, n_classes)
        Confusion matrix.
    class_names : list[str], optional
        Display names for each class.
    normalize : bool
        If *True*, normalize each row so values sum to 1.
    title : str
        Plot title.
    cmap : str
        Matplotlib colormap name.
    save_path : str or Path, optional
        If provided, the figure is saved to this path.
    figsize : tuple
        Figure size in inches.
    show : bool
        If *True*, call ``plt.show()``.

    Returns
    -------
    matplotlib.figure.Figure
    """
    cm = np.asarray(cm, dtype=np.float64)
    n_classes = cm.shape[0]

    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        cm = cm / np.maximum(row_sums, 1)
        fmt = ".2f"
    else:
        fmt = "d"

    if class_names is None:
        class_names = [str(i) for i in range(n_classes)]

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm, interpolation="nearest", cmap=cmap)
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(n_classes),
        yticks=np.arange(n_classes),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
        title=title,
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    thresh = cm.max() / 2.0
    for i in range(n_classes):
        for j in range(n_classes):
            if normalize:
                label = f"{cm[i, j]:{fmt}}"
            else:
                label = f"{int(cm[i, j]):{fmt}}"
            ax.text(j, i, label, ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=DEFAULT_DPI, bbox_inches="tight")
        logger.info("Saved confusion matrix to %s", save_path)

    if show:
        plt.show()

    return fig


# ------------------------------------------------------------------
# ROC curve
# ------------------------------------------------------------------

def plot_roc_curve(
    fpr: np.ndarray,
    tpr: np.ndarray,
    auc: Optional[float] = None,
    label: Optional[str] = None,
    title: str = "Receiver Operating Characteristic",
    save_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = DEFAULT_FIGSIZE,
    show: bool = False,
) -> plt.Figure:
    """Plot an ROC curve.

    Parameters
    ----------
    fpr : array-like
        False-positive rates.
    tpr : array-like
        True-positive rates.
    auc : float, optional
        AUC value to display in the legend.
    label : str, optional
        Curve label.  AUC is appended automatically if provided.
    title : str
    save_path, figsize, show : standard plotting options.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    if label is None:
        label = "Model"
    if auc is not None:
        label = f"{label} (AUC = {auc:.4f})"

    ax.plot(fpr, tpr, lw=2, label=label)
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Chance")
    ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate", title=title)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])

    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=DEFAULT_DPI, bbox_inches="tight")
        logger.info("Saved ROC curve to %s", save_path)

    if show:
        plt.show()

    return fig


# ------------------------------------------------------------------
# Precision-Recall curve
# ------------------------------------------------------------------

def plot_precision_recall_curve(
    precision: np.ndarray,
    recall: np.ndarray,
    ap: Optional[float] = None,
    label: Optional[str] = None,
    title: str = "Precision-Recall Curve",
    save_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = DEFAULT_FIGSIZE,
    show: bool = False,
) -> plt.Figure:
    """Plot a Precision-Recall curve.

    Parameters
    ----------
    precision : array-like
    recall : array-like
    ap : float, optional
        Average precision (displayed in legend).
    label : str, optional
    title : str
    save_path, figsize, show : standard plotting options.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    if label is None:
        label = "Model"
    if ap is not None:
        label = f"{label} (AP = {ap:.4f})"

    ax.plot(recall, precision, lw=2, label=label)
    ax.set(xlabel="Recall", ylabel="Precision", title=title)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])

    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=DEFAULT_DPI, bbox_inches="tight")
        logger.info("Saved PR curve to %s", save_path)

    if show:
        plt.show()

    return fig


# ------------------------------------------------------------------
# Training history
# ------------------------------------------------------------------

def plot_training_history(
    history: Dict[str, List[float]],
    title: str = "Training History",
    save_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (12, 5),
    show: bool = False,
) -> plt.Figure:
    """Plot training and validation loss / metric curves.

    Parameters
    ----------
    history : dict
        Keys should include ``"train_loss"``, ``"val_loss"`` etc.
        Values are lists of per-epoch scalars.
    title : str
    save_path, figsize, show : standard plotting options.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # -- Loss subplot -------------------------------------------------
    loss_keys = [k for k in history if "loss" in k.lower()]
    if loss_keys:
        for key in loss_keys:
            label = key.replace("_", " ").title()
            axes[0].plot(history[key], label=label)
        axes[0].set(xlabel="Epoch", ylabel="Loss", title="Loss")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

    # -- Metric subplot -----------------------------------------------
    metric_keys = [k for k in history if "loss" not in k.lower()]
    if metric_keys:
        for key in metric_keys:
            label = key.replace("_", " ").title()
            axes[1].plot(history[key], label=label)
        axes[1].set(xlabel="Epoch", ylabel="Metric", title="Metrics")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=16, y=1.02)
    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=DEFAULT_DPI, bbox_inches="tight")
        logger.info("Saved training history to %s", save_path)

    if show:
        plt.show()

    return fig


# ------------------------------------------------------------------
# Feature importance
# ------------------------------------------------------------------

def plot_feature_importance(
    importance: np.ndarray,
    feature_names: Optional[List[str]] = None,
    top_k: int = 20,
    title: str = "Feature Importance",
    save_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (10, 8),
    show: bool = False,
) -> plt.Figure:
    """Horizontal bar chart of feature importance scores.

    Parameters
    ----------
    importance : array-like
        One value per feature.
    feature_names : list[str], optional
        Descriptive names.  Falls back to ``"feature_{i}"``.
    top_k : int
        Number of top features to display.
    title : str
    save_path, figsize, show : standard plotting options.

    Returns
    -------
    matplotlib.figure.Figure
    """
    importance = np.asarray(importance, dtype=np.float64).ravel()
    n = len(importance)

    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(n)]

    # Sort and select top_k
    sorted_idx = np.argsort(importance)[::-1][:top_k]
    names = [feature_names[i] for i in sorted_idx]
    values = importance[sorted_idx]

    fig, ax = plt.subplots(figsize=figsize)
    y_pos = np.arange(len(names))
    bars = ax.barh(y_pos, values, color=plt.cm.viridis(np.linspace(0.2, 0.8, len(names))))
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set(xlabel="Importance", title=title)
    ax.grid(True, axis="x", alpha=0.3)

    # Annotate bars with numeric values
    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 0.01 * values.max(), bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=9)

    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=DEFAULT_DPI, bbox_inches="tight")
        logger.info("Saved feature importance plot to %s", save_path)

    if show:
        plt.show()

    return fig


# ------------------------------------------------------------------
# Data distribution
# ------------------------------------------------------------------

def plot_data_distribution(
    data: Union[np.ndarray, Dict[str, np.ndarray]],
    title: str = "Data Distribution",
    kind: str = "histogram",
    bins: int = 30,
    kde: bool = False,
    class_names: Optional[List[str]] = None,
    save_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = DEFAULT_FIGSIZE,
    show: bool = False,
) -> plt.Figure:
    """Visualize the distribution of data.

    Parameters
    ----------
    data : np.ndarray or dict
        If a dict, keys become series labels and values are arrays.
    title : str
    kind : str
        ``"histogram"``, ``"bar"`` (for discrete label counts), or ``"box"``.
    bins : int
        Number of histogram bins.
    kde : bool
        Overlay a kernel-density estimate (requires scipy).
    class_names : list[str], optional
        Labels for bar / count plots.
    save_path, figsize, show : standard plotting options.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    if isinstance(data, dict):
        # Multiple series
        if kind == "histogram":
            for label, arr in data.items():
                ax.hist(arr, bins=bins, alpha=0.6, label=label, density=True)
                if kde:
                    _plot_kde(ax, arr)
            ax.legend()
        elif kind == "box":
            box_data = list(data.values())
            bp = ax.boxplot(box_data, labels=list(data.keys()), patch_artist=True)
            colors = plt.cm.Set2(np.linspace(0, 1, len(bp["boxes"])))
            for patch, color in zip(bp["boxes"], colors):
                patch.set_facecolor(color)
        elif kind == "bar":
            for label, arr in data.items():
                values, counts = np.unique(arr, return_counts=True)
                names = class_names if class_names else [str(v) for v in values]
                ax.bar(range(len(counts)), counts, alpha=0.7, label=label)
                ax.set_xticks(range(len(counts)))
                ax.set_xticklabels(names)
            ax.legend()
        else:
            raise ValueError(f"Unknown kind={kind!r}")
    else:
        # Single array
        data_arr = np.asarray(data).ravel()
        if kind == "histogram":
            ax.hist(data_arr, bins=bins, alpha=0.7, edgecolor="black", density=True)
            if kde:
                _plot_kde(ax, data_arr)
        elif kind == "bar":
            values, counts = np.unique(data_arr, return_counts=True)
            names = class_names if class_names else [str(v) for v in values]
            ax.bar(range(len(counts)), counts, alpha=0.7, edgecolor="black")
            ax.set_xticks(range(len(counts)))
            ax.set_xticklabels(names)
        elif kind == "box":
            ax.boxplot(data_arr, patch_artist=True)
        else:
            raise ValueError(f"Unknown kind={kind!r}")

    ax.set_title(title)
    ax.set_xlabel("Value")
    ax.set_ylabel("Density" if kind == "histogram" else "Count" if kind == "bar" else "Value")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=DEFAULT_DPI, bbox_inches="tight")
        logger.info("Saved distribution plot to %s", save_path)

    if show:
        plt.show()

    return fig


# ------------------------------------------------------------------
# KDE helper (scipy optional)
# ------------------------------------------------------------------

def _plot_kde(ax: plt.Axes, data: np.ndarray, n_points: int = 200) -> None:
    """Overlay a kernel-density estimate curve on *ax* if scipy is available."""
    try:
        from scipy.stats import gaussian_kde

        kde = gaussian_kde(data)
        x_min, x_max = data.min(), data.max()
        xs = np.linspace(x_min, x_max, n_points)
        ax.plot(xs, kde(xs), lw=2, label="KDE")
    except ImportError:
        logger.debug("scipy not available — skipping KDE overlay.")
