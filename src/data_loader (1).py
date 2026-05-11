"""
Data loading module for the Artificial Intelligence Advanced project.

Provides a CustomDataset class (torch Dataset), stratified split utilities,
and a ``create_data_loaders`` factory function that returns PyTorch DataLoaders
for training, validation, and testing.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

# Lazy imports for optional torch dependency
_torch = None
_torch_utils = None


def _ensure_torch():
    """Import torch and torch.utils lazily."""
    global _torch, _torch_utils
    if _torch is None:
        import torch as _t
        import torch.utils.data as _tu
        _torch = _t
        _torch_utils = _tu


# ------------------------------------------------------------------
# CustomDataset
# ------------------------------------------------------------------

class CustomDataset:
    """A thin wrapper around NumPy arrays / lists that implements the PyTorch
    Dataset interface.

    Parameters
    ----------
    features : np.ndarray or list
        Input features of shape ``(n_samples, ...)``.
    labels : np.ndarray or list, optional
        Target values.  If *None*, the dataset is unlabelled.
    transform : callable, optional
        A function applied to each sample on ``__getitem__``.
    """

    def __init__(
        self,
        features: Union[np.ndarray, List],
        labels: Optional[Union[np.ndarray, List]] = None,
        transform: Optional[Any] = None,
    ) -> None:
        _ensure_torch()

        self.features = np.asarray(features, dtype=np.float32)
        self.labels = np.asarray(labels, dtype=np.float32) if labels is not None else None
        self.transform = transform

        if self.labels is not None and len(self.features) != len(self.labels):
            raise ValueError(
                f"Feature count ({len(self.features)}) != label count ({len(self.labels)})."
            )

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> Dict[str, _torch.Tensor]:  # type: ignore[valid-type]
        sample = self.features[idx]
        item: Dict[str, _torch.Tensor] = {"input": _torch.from_numpy(sample)}  # type: ignore[attr-defined]

        if self.labels is not None:
            label = self.labels[idx]
            item["label"] = _torch.from_numpy(  # type: ignore[attr-defined]
                np.atleast_1d(np.asarray(label, dtype=np.float32))
            ).squeeze()

        if self.transform is not None:
            item = self.transform(item)

        return item

    @property
    def num_classes(self) -> Optional[int]:
        """Return the number of unique classes if labels are discrete."""
        if self.labels is None:
            return None
        unique = np.unique(self.labels)
        if unique.dtype.kind in ("i", "u", "f"):
            # Heuristic: if all values are small non-negative ints, treat as class ids
            if np.all(unique == np.floor(unique)) and np.all(unique >= 0):
                return int(len(unique))
        return None

    @classmethod
    def from_csv(
        cls,
        filepath: Union[str, Path],
        label_column: Optional[str] = None,
        label_index: Optional[int] = None,
        delimiter: str = ",",
    ) -> "CustomDataset":
        """Create a dataset from a CSV file.

        Parameters
        ----------
        filepath : str or Path
            Path to the CSV file.
        label_column : str, optional
            Name of the column to use as labels.
        label_index : int, optional
            Index of the column to use as labels (alternative to *label_column*).
        delimiter : str
            Column separator.

        Returns
        -------
        CustomDataset
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"CSV file not found: {filepath}")

        with filepath.open("r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter=delimiter)
            header = next(reader)
            rows = [row for row in reader if row]

        data = np.array(rows)
        n_rows, n_cols = data.shape

        if label_column is not None:
            if label_column not in header:
                raise KeyError(f"Label column {label_column!r} not found in header: {header}")
            col_idx = header.index(label_column)
        elif label_index is not None:
            col_idx = label_index
        else:
            # Assume last column is the label
            col_idx = n_cols - 1

        features = np.delete(data, col_idx, axis=1).astype(np.float32)
        labels = data[:, col_idx].astype(np.float32)

        logger.info(
            "Loaded CSV %s: %d samples, %d features.",
            filepath, n_rows, features.shape[1],
        )
        return cls(features=features, labels=labels)


# ------------------------------------------------------------------
# Stratified split
# ------------------------------------------------------------------

def stratified_split(
    labels: np.ndarray,
    val_ratio: float = 0.2,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split sample indices into train / val / test sets preserving class proportions.

    Parameters
    ----------
    labels : np.ndarray
        Array of class labels (shape ``(n_samples,)``).
    val_ratio : float
        Fraction of data to allocate to validation.
    test_ratio : float
        Fraction of data to allocate to testing.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        (train_indices, val_indices, test_indices)
    """
    rng = np.random.RandomState(seed)
    classes = np.unique(labels)
    train_idx: List[int] = []
    val_idx: List[int] = []
    test_idx: List[int] = []

    for cls in classes:
        cls_indices = np.where(labels == cls)[0]
        rng.shuffle(cls_indices)
        n = len(cls_indices)
        n_test = max(1, int(n * test_ratio))
        n_val = max(1, int(n * val_ratio))

        test_idx.extend(cls_indices[:n_test].tolist())
        val_idx.extend(cls_indices[n_test : n_test + n_val].tolist())
        train_idx.extend(cls_indices[n_test + n_val :].tolist())

    return (
        np.array(train_idx),
        np.array(val_idx),
        np.array(test_idx),
    )


# ------------------------------------------------------------------
# Data loader factory
# ------------------------------------------------------------------

def create_data_loaders(
    features: np.ndarray,
    labels: Optional[np.ndarray] = None,
    batch_size: int = 64,
    val_ratio: float = 0.2,
    test_ratio: float = 0.1,
    num_workers: int = 4,
    pin_memory: bool = True,
    seed: int = 42,
    shuffle_train: bool = True,
) -> Dict[str, Any]:
    """Create PyTorch DataLoader dictionaries for train / val / test.

    Parameters
    ----------
    features : np.ndarray
        Feature matrix of shape ``(n_samples, n_features)``.
    labels : np.ndarray, optional
        Label array.  If provided, stratified splitting is used.
    batch_size : int
        Mini-batch size.
    val_ratio : float
        Validation set fraction.
    test_ratio : float
        Test set fraction.
    num_workers : int
        DataLoader worker count.
    pin_memory : bool
        Whether to pin memory for faster GPU transfer.
    seed : int
        Random seed.
    shuffle_train : bool
        Whether to shuffle the training set each epoch.

    Returns
    -------
    dict
        Keys ``"train"``, ``"val"``, ``"test"``, ``"datasets"``.
        ``"datasets"`` maps split names to their ``CustomDataset`` instances.
    """
    _ensure_torch()

    n_samples = len(features)

    if labels is not None:
        train_idx, val_idx, test_idx = stratified_split(labels, val_ratio, test_ratio, seed)
    else:
        # Random (non-stratified) split
        rng = np.random.RandomState(seed)
        indices = rng.permutation(n_samples)
        n_test = max(1, int(n_samples * test_ratio))
        n_val = max(1, int(n_samples * val_ratio))
        test_idx = indices[:n_test]
        val_idx = indices[n_test : n_test + n_val]
        train_idx = indices[n_test + n_val :]

    # Build per-split datasets
    datasets: Dict[str, CustomDataset] = {}
    for split_name, idx in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
        split_features = features[idx]
        split_labels = labels[idx] if labels is not None else None
        datasets[split_name] = CustomDataset(features=split_features, labels=split_labels)

    # Build DataLoaders
    loaders: Dict[str, Any] = {}
    for split_name, dataset in datasets.items():
        shuffle = shuffle_train if split_name == "train" else False
        loaders[split_name] = _torch.utils.data.DataLoader(  # type: ignore[attr-defined]
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=(split_name == "train"),
        )
        logger.info(
            "  %s loader: %d samples, %d batches.",
            split_name, len(dataset), len(loaders[split_name]),
        )

    loaders["datasets"] = datasets
    return loaders


# ------------------------------------------------------------------
# Numpy-only data splitting (no torch required)
# ------------------------------------------------------------------

def split_numpy_arrays(
    X: np.ndarray,
    y: Optional[np.ndarray] = None,
    val_ratio: float = 0.2,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> Dict[str, Tuple[np.ndarray, Optional[np.ndarray]]]:
    """Split NumPy arrays into train / val / test without requiring PyTorch.

    Returns
    -------
    dict
        Keys ``"train"``, ``"val"``, ``"test"`` with ``(X_split, y_split)``
        tuples.
    """
    if y is not None:
        train_idx, val_idx, test_idx = stratified_split(y, val_ratio, test_ratio, seed)
    else:
        rng = np.random.RandomState(seed)
        n = len(X)
        indices = rng.permutation(n)
        n_test = max(1, int(n * test_ratio))
        n_val = max(1, int(n * val_ratio))
        test_idx, val_idx, train_idx = (
            indices[:n_test],
            indices[n_test : n_test + n_val],
            indices[n_test + n_val :],
        )

    splits: Dict[str, Tuple[np.ndarray, Optional[np.ndarray]]] = {}
    for name, idx in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
        splits[name] = (X[idx], y[idx] if y is not None else None)

    logger.info(
        "Data split — train: %d, val: %d, test: %d.",
        len(train_idx), len(val_idx), len(test_idx),
    )
    return splits
