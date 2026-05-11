"""
Data preprocessing module for the Artificial Intelligence Advanced project.

Provides a DataPreprocessor class with methods for standardization, normalization,
missing value imputation, categorical encoding, and pipeline-based transformations.
Supports both NumPy arrays and pandas DataFrames.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union


class DataPreprocessor:
    """Composable data preprocessor supporting fit/transform/inverse_transform semantics.

    Each preprocessing step stores the statistics computed during ``fit`` so that
    the same transformation can be replayed on new data via ``transform``.

    Parameters
    ----------
    steps : list[str], optional
        Ordered list of preprocessing step names to apply.  Valid steps are
        ``"standardize"``, ``"normalize"``, ``"handle_missing"``, and
        ``"encode_categorical"``.  If *None*, no steps are applied automatically
        and the user must call individual methods explicitly.
    """

    VALID_STEPS = {"standardize", "normalize", "handle_missing", "encode_categorical"}

    def __init__(self, steps: Optional[List[str]] = None) -> None:
        self.steps: List[str] = steps or []
        for step in self.steps:
            if step not in self.VALID_STEPS:
                raise ValueError(f"Unknown preprocessing step: {step!r}")

        # Fitted state containers
        self._means: Optional[np.ndarray] = None
        self._stds: Optional[np.ndarray] = None
        self._min_vals: Optional[np.ndarray] = None
        self._range_vals: Optional[np.ndarray] = None
        self._fill_values: Optional[np.ndarray] = None
        self._category_maps: Dict[int, Dict[str, int]] = {}
        self._inverse_category_maps: Dict[int, Dict[int, str]] = {}
        self._is_fitted: bool = False

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: Optional[np.ndarray] = None) -> "DataPreprocessor":
        """Compute and store statistics from the training data.

        Parameters
        ----------
        X : np.ndarray or pd.DataFrame
            Training features of shape ``(n_samples, n_features)``.
        y : np.ndarray, optional
            Target values (unused in most transforms but kept for API consistency).

        Returns
        -------
        self
        """
        X = self._to_numpy(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        for step in self.steps:
            if step == "standardize":
                self._fit_standardize(X)
            elif step == "normalize":
                self._fit_normalize(X)
            elif step == "handle_missing":
                self._fit_handle_missing(X)
            elif step == "encode_categorical":
                self._fit_encode_categorical(X)

        self._is_fitted = True
        return self

    def transform(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Apply the learned preprocessing pipeline to *X*.

        Parameters
        ----------
        X : np.ndarray or pd.DataFrame
            Data to transform of shape ``(n_samples, n_features)``.

        Returns
        -------
        np.ndarray
            Transformed array.
        """
        if not self._is_fitted:
            raise RuntimeError("The preprocessor has not been fitted yet. Call fit() first.")

        X = self._to_numpy(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        for step in self.steps:
            if step == "standardize":
                X = self._transform_standardize(X)
            elif step == "normalize":
                X = self._transform_normalize(X)
            elif step == "handle_missing":
                X = self._transform_handle_missing(X)
            elif step == "encode_categorical":
                X = self._transform_encode_categorical(X)

        return X

    def fit_transform(
        self, X: Union[np.ndarray, pd.DataFrame], y: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Convenience method that calls ``fit`` followed by ``transform``."""
        return self.fit(X, y).transform(X)

    def inverse_transform(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Reverse the preprocessing pipeline (where possible).

        Note: ``handle_missing`` and ``encode_categorical`` are not invertible.

        Parameters
        ----------
        X : np.ndarray or pd.DataFrame
            Transformed data of shape ``(n_samples, n_features)``.

        Returns
        -------
        np.ndarray
            Approximate reconstruction of the original data.
        """
        if not self._is_fitted:
            raise RuntimeError("The preprocessor has not been fitted yet.")

        X = self._to_numpy(X).astype(np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        # Apply inverse steps in reverse order
        for step in reversed(self.steps):
            if step == "normalize":
                X = self._inverse_normalize(X)
            elif step == "standardize":
                X = self._inverse_standardize(X)

        return X

    # ------------------------------------------------------------------
    # Individual step methods (public for manual use)
    # ------------------------------------------------------------------

    def standardize(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Z-score standardization: ``(X - mean) / std``."""
        self._fit_standardize(self._to_numpy(X))
        return self._transform_standardize(self._to_numpy(X))

    def normalize(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Min-max normalization to the range [0, 1]."""
        self._fit_normalize(self._to_numpy(X))
        return self._transform_normalize(self._to_numpy(X))

    def handle_missing_values(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        strategy: str = "mean",
        fill_value: Optional[float] = None,
    ) -> np.ndarray:
        """Replace missing values (``NaN`` / ``None``) with a computed fill value.

        Parameters
        ----------
        X : np.ndarray or pd.DataFrame
            Input data.
        strategy : str
            One of ``"mean"``, ``"median"``, ``"most_frequent"``, or ``"constant"``.
        fill_value : float, optional
            Explicit fill value when *strategy* is ``"constant"``.

        Returns
        -------
        np.ndarray
            Array with missing values filled.
        """
        arr = self._to_numpy(X).astype(np.float64)
        self._fill_strategy = strategy
        if strategy == "mean":
            self._fill_values = np.nanmean(arr, axis=0)
        elif strategy == "median":
            self._fill_values = np.nanmedian(arr, axis=0)
        elif strategy == "most_frequent":
            self._fill_values = np.nan_to_num(
                np.array([float(pd.Series(arr[:, c]).mode()[0]) for c in range(arr.shape[1])])
            )
        elif strategy == "constant":
            self._fill_values = np.full(arr.shape[1], fill_value if fill_value is not None else 0.0)
        else:
            raise ValueError(f"Unknown missing-value strategy: {strategy!r}")

        mask = np.isnan(arr)
        for col_idx in range(arr.shape[1]):
            arr[mask[:, col_idx], col_idx] = self._fill_values[col_idx]
        return arr

    def encode_categorical(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        columns: Optional[List[int]] = None,
    ) -> np.ndarray:
        """Label-encode categorical columns.

        Parameters
        ----------
        X : np.ndarray or pd.DataFrame
            Input data.  String columns are automatically detected if *columns*
            is *None*.
        columns : list[int], optional
            Indices of columns to encode.

        Returns
        -------
        np.ndarray
            Encoded array (non-categorical columns left as-is, cast to float).
        """
        arr = self._to_numpy(X)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)

        # Auto-detect object / string columns
        if columns is None:
            columns = [c for c in range(arr.shape[1]) if arr[:, c].dtype.kind in ("O", "U", "S")]

        for col in columns:
            unique_vals: List[str] = sorted(set(str(v) for v in arr[:, col] if v is not None))
            mapping = {val: idx for idx, val in enumerate(unique_vals)}
            self._category_maps[col] = mapping
            self._inverse_category_maps[col] = {idx: val for val, idx in mapping.items()}
            arr[:, col] = np.array([mapping.get(str(v), -1) for v in arr[:, col]])

        return arr.astype(np.float64)

    # ------------------------------------------------------------------
    # Pipeline helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, steps: List[str]) -> "DataPreprocessor":
        """Create a preprocessor directly from a step list (useful with Config objects)."""
        return cls(steps=steps)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_numpy(X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            return X.values
        return np.asarray(X)

    # -- standardize ---------------------------------------------------

    def _fit_standardize(self, X: np.ndarray) -> None:
        self._means = np.nanmean(X, axis=0)
        self._stds = np.nanstd(X, axis=0)
        # Avoid division by zero for constant features
        self._stds[self._stds == 0] = 1.0

    def _transform_standardize(self, X: np.ndarray) -> np.ndarray:
        return (X - self._means) / self._stds

    def _inverse_standardize(self, X: np.ndarray) -> np.ndarray:
        return X * self._stds + self._means

    # -- normalize -----------------------------------------------------

    def _fit_normalize(self, X: np.ndarray) -> None:
        self._min_vals = np.nanmin(X, axis=0)
        self._range_vals = np.nanmax(X, axis=0) - self._min_vals
        self._range_vals[self._range_vals == 0] = 1.0

    def _transform_normalize(self, X: np.ndarray) -> np.ndarray:
        return (X - self._min_vals) / self._range_vals

    def _inverse_normalize(self, X: np.ndarray) -> np.ndarray:
        return X * self._range_vals + self._min_vals

    # -- handle_missing ------------------------------------------------

    def _fit_handle_missing(self, X: np.ndarray) -> None:
        self._fill_values = np.nanmean(X, axis=0)

    def _transform_handle_missing(self, X: np.ndarray) -> np.ndarray:
        arr = X.astype(np.float64).copy()
        mask = np.isnan(arr)
        for col_idx in range(arr.shape[1]):
            arr[mask[:, col_idx], col_idx] = self._fill_values[col_idx]
        return arr

    # -- encode_categorical --------------------------------------------

    def _fit_encode_categorical(self, X: np.ndarray) -> None:
        columns = [c for c in range(X.shape[1]) if X[:, c].dtype.kind in ("O", "U", "S")]
        for col in columns:
            unique_vals = sorted(set(str(v) for v in X[:, col] if v is not None))
            mapping = {val: idx for idx, val in enumerate(unique_vals)}
            self._category_maps[col] = mapping
            self._inverse_category_maps[col] = {idx: val for val, idx in mapping.items()}

    def _transform_encode_categorical(self, X: np.ndarray) -> np.ndarray:
        arr = X.copy()
        for col, mapping in self._category_maps.items():
            arr[:, col] = np.array([mapping.get(str(v), -1) for v in arr[:, col]])
        return arr.astype(np.float64)

    def __repr__(self) -> str:
        steps_str = ", ".join(self.steps) if self.steps else "(none)"
        return f"DataPreprocessor(steps=[{steps_str}], fitted={self._is_fitted})"
