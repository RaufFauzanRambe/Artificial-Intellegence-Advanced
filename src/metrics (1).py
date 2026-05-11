"""
Evaluation metrics module for the Artificial Intelligence Advanced project.

All metrics are implemented using only NumPy — no scikit-learn dependency.
Supports binary and multi-class classification with macro / micro / weighted
averaging.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional, Tuple, Union


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _check_shapes(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """Validate that label arrays have compatible shapes."""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}"
        )


def _one_vs_rest(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Convert multi-class arrays to a list of binary (positive / not-positive) pairs."""
    pairs: List[Tuple[np.ndarray, np.ndarray]] = []
    for cls in range(n_classes):
        yt = (y_true == cls).astype(np.int64)
        yp = (y_pred == cls).astype(np.int64)
        pairs.append((yt, yp))
    return pairs


# ------------------------------------------------------------------
# Confusion matrix
# ------------------------------------------------------------------

def confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compute an ``n_classes × n_classes`` confusion matrix.

    Parameters
    ----------
    y_true : array-like
        Ground-truth labels.
    y_pred : array-like
        Predicted labels.
    labels : array-like, optional
        Explicit class ordering.  If *None*, inferred from the sorted union
        of *y_true* and *y_pred*.

    Returns
    -------
    np.ndarray of shape (n_classes, n_classes)
        ``cm[i, j]`` is the number of samples with true label *i* predicted as *j*.
    """
    _check_shapes(y_true, y_pred)
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    if labels is None:
        labels = np.sort(np.unique(np.concatenate([y_true, y_pred])))
    else:
        labels = np.asarray(labels)

    n_classes = len(labels)
    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)

    for t, p in zip(y_true, y_pred):
        if t in label_to_idx and p in label_to_idx:
            cm[label_to_idx[t], label_to_idx[p]] += 1

    return cm


# ------------------------------------------------------------------
# Per-class TP / FP / FN / TN
# ------------------------------------------------------------------

def _per_class_counts(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_classes: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (TP, FP, FN, TN) arrays of length *n_classes*."""
    cm = confusion_matrix(y_true, y_pred)
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    tn = cm.sum() - tp - fp - fn
    return tp, fp, fn, tn


# ------------------------------------------------------------------
# Accuracy
# ------------------------------------------------------------------

def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Fraction of correct predictions.

    Parameters
    ----------
    y_true : array-like
    y_pred : array-like

    Returns
    -------
    float
    """
    _check_shapes(y_true, y_pred)
    return float(np.mean(np.asarray(y_true).ravel() == np.asarray(y_pred).ravel()))


# ------------------------------------------------------------------
# Precision, Recall, F1
# ------------------------------------------------------------------

def _averaged_metric(
    per_class: np.ndarray,
    average: str,
    y_true: Optional[np.ndarray] = None,
) -> float:
    """Apply macro / micro / weighted averaging to per-class values."""
    if average == "micro":
        # For precision/recall computed from counts, micro = global counts
        return float(per_class.sum()) / max(per_class.sum(), 1)
    elif average == "macro":
        return float(np.mean(per_class))
    elif average == "weighted":
        if y_true is None:
            raise ValueError("y_true is required for weighted averaging.")
        y_true = np.asarray(y_true).ravel()
        class_counts = np.bincount(y_true.astype(np.int64))
        weights = class_counts / class_counts.sum()
        return float(np.sum(per_class * weights))
    else:
        raise ValueError(f"Unknown average={average!r}. Use 'macro', 'micro', or 'weighted'.")


def precision(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    average: str = "macro",
) -> float:
    """Compute precision with the specified averaging strategy.

    Parameters
    ----------
    y_true, y_pred : array-like
    average : str
        One of ``"micro"``, ``"macro"``, ``"weighted"``.

    Returns
    -------
    float
    """
    _check_shapes(y_true, y_pred)
    y_true = np.asarray(y_true).ravel().astype(np.int64)
    y_pred = np.asarray(y_pred).ravel().astype(np.int64)
    n_classes = max(y_true.max(), y_pred.max()) + 1
    tp, fp, _, _ = _per_class_counts(y_true, y_pred, n_classes)

    if average == "micro":
        return float(tp.sum()) / max(tp.sum() + fp.sum(), 1)

    per_class = tp / np.maximum(tp + fp, 1)
    return _averaged_metric(per_class, average, y_true)


def recall(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    average: str = "macro",
) -> float:
    """Compute recall (sensitivity / true-positive rate).

    Parameters
    ----------
    y_true, y_pred : array-like
    average : str
        One of ``"micro"``, ``"macro"``, ``"weighted"``.

    Returns
    -------
    float
    """
    _check_shapes(y_true, y_pred)
    y_true = np.asarray(y_true).ravel().astype(np.int64)
    y_pred = np.asarray(y_pred).ravel().astype(np.int64)
    n_classes = max(y_true.max(), y_pred.max()) + 1
    tp, _, fn, _ = _per_class_counts(y_true, y_pred, n_classes)

    if average == "micro":
        return float(tp.sum()) / max(tp.sum() + fn.sum(), 1)

    per_class = tp / np.maximum(tp + fn, 1)
    return _averaged_metric(per_class, average, y_true)


def f1_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    average: str = "macro",
) -> float:
    """Compute the F1 score (harmonic mean of precision and recall).

    Parameters
    ----------
    y_true, y_pred : array-like
    average : str
        One of ``"micro"``, ``"macro"``, ``"weighted"``.

    Returns
    -------
    float
    """
    _check_shapes(y_true, y_pred)
    y_true = np.asarray(y_true).ravel().astype(np.int64)
    y_pred = np.asarray(y_pred).ravel().astype(np.int64)
    n_classes = max(y_true.max(), y_pred.max()) + 1
    tp, fp, fn, _ = _per_class_counts(y_true, y_pred, n_classes)

    per_class_p = tp / np.maximum(tp + fp, 1)
    per_class_r = tp / np.maximum(tp + fn, 1)
    per_class_f1 = 2 * per_class_p * per_class_r / np.maximum(per_class_p + per_class_r, 1e-12)

    if average == "micro":
        p_micro = tp.sum() / max(tp.sum() + fp.sum(), 1)
        r_micro = tp.sum() / max(tp.sum() + fn.sum(), 1)
        return float(2 * p_micro * r_micro / max(p_micro + r_micro, 1e-12))

    return _averaged_metric(per_class_f1, average, y_true)


# ------------------------------------------------------------------
# ROC AUC (binary only)
# ------------------------------------------------------------------

def roc_auc(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """Compute the Area Under the Receiver Operating Characteristic Curve.

    Only supports binary classification.

    Parameters
    ----------
    y_true : array-like
        Ground-truth binary labels (0 or 1).
    y_scores : array-like
        Predicted probability / score for the positive class.

    Returns
    -------
    float
        AUC value in [0, 1].
    """
    y_true = np.asarray(y_true).ravel()
    y_scores = np.asarray(y_scores).ravel()

    unique_classes = np.unique(y_true)
    if len(unique_classes) > 2:
        raise ValueError("roc_auc only supports binary classification.")
    if len(unique_classes) < 2:
        return 0.0  # Only one class present

    # Ensure consistent ordering: higher score = positive class
    if y_scores.max() <= 1.0 and y_scores.min() >= 0.0:
        # Scores look like probabilities — leave as-is
        pass

    # Sort by descending score
    desc_idx = np.argsort(-y_scores)
    y_true_sorted = y_true[desc_idx]
    y_scores_sorted = y_scores[desc_idx]

    n_pos = np.sum(y_true == 1)
    n_neg = len(y_true) - n_pos

    if n_pos == 0 or n_neg == 0:
        return 0.0

    # Compute TPR / FPR at each threshold
    tp = np.cumsum(y_true_sorted == 1)
    fp = np.cumsum(y_true_sorted == 0)

    tpr = tp / n_pos
    fpr = fp / n_neg

    # AUC via trapezoidal rule
    auc_val = np.trapz(tpr, fpr)
    return float(auc_val)


# ------------------------------------------------------------------
# Log loss (cross-entropy)
# ------------------------------------------------------------------

def log_loss(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    eps: float = 1e-15,
) -> float:
    """Compute the average logistic (cross-entropy) loss.

    Parameters
    ----------
    y_true : array-like
        Ground-truth labels (one-hot or integer class indices).
    y_prob : array-like
        Predicted class probabilities of shape ``(n_samples,)`` (binary) or
        ``(n_samples, n_classes)`` (multi-class).
    eps : float
        Clipping value to avoid ``log(0)``.

    Returns
    -------
    float
        Average log-loss.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_prob = np.asarray(y_prob, dtype=np.float64)

    # Auto-convert integer labels to one-hot
    if y_true.ndim == 1:
        n_classes = y_prob.shape[1] if y_prob.ndim == 2 else 2
        one_hot = np.zeros((len(y_true), n_classes), dtype=np.float64)
        for i, c in enumerate(y_true.astype(int)):
            if 0 <= c < n_classes:
                one_hot[i, c] = 1.0
        y_true = one_hot

    if y_prob.ndim == 1:
        y_prob = np.stack([1 - y_prob, y_prob], axis=1)

    # Clip and compute
    y_prob = np.clip(y_prob, eps, 1.0 - eps)
    loss = -np.sum(y_true * np.log(y_prob)) / len(y_true)
    return float(loss)


# ------------------------------------------------------------------
# Classification report
# ------------------------------------------------------------------

def classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[List[str]] = None,
) -> str:
    """Build a text classification report similar to scikit-learn's.

    Returns
    -------
    str
            precision    recall  f1-score   support
        ...
    """
    _check_shapes(y_true, y_pred)
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    classes = np.unique(np.concatenate([y_true, y_pred]))
    if labels is None:
        labels = [str(c) for c in classes]

    cm = confusion_matrix(y_true, y_pred, labels)
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    support = cm.sum(axis=1)

    per_p = tp / np.maximum(tp + fp, 1)
    per_r = tp / np.maximum(tp + fn, 1)
    per_f1 = 2 * per_p * per_r / np.maximum(per_p + per_r, 1e-12)

    header = f"{'':>14s}  {'precision':>9s}  {'recall':>9s}  {'f1-score':>9s}  {'support':>7s}"
    sep = "-" * len(header)
    lines = [header, sep]

    for i, name in enumerate(labels):
        lines.append(
            f"{name:>14s}  {per_p[i]:9.4f}  {per_r[i]:9.4f}  {per_f1[i]:9.4f}  {support[i]:7d}"
        )

    # Macro averages
    lines.append(sep)
    lines.append(
        f"{'macro avg':>14s}  {np.mean(per_p):9.4f}  {np.mean(per_r):9.4f}  "
        f"{np.mean(per_f1):9.4f}  {int(support.sum()):7d}"
    )

    acc_val = accuracy(y_true, y_pred)
    lines.append(
        f"{'accuracy':>14s}  {acc_val:9.4f}  {acc_val:9.4f}  {acc_val:9.4f}  {int(support.sum()):7d}"
    )

    return "\n".join(lines)
