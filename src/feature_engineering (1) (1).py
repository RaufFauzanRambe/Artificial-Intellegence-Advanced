"""
Feature engineering module for the Artificial Intelligence Advanced project.

Provides a FeatureEngineer class that can generate polynomial features,
interaction terms, binned features, and perform feature selection based on
variance thresholds and correlation analysis.  Also supports extracting
date/time features from timestamp columns.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from itertools import combinations
from typing import Dict, List, Optional, Tuple, Union


class FeatureEngineer:
    """Stateful feature engineer supporting fit / transform semantics.

    Parameters
    ----------
    degree : int
        Maximum polynomial degree (used by polynomial feature generation).
    interaction_only : bool
        If *True*, only interaction terms (cross-terms) are generated;
        powers of individual features are excluded.
    include_bias : bool
        If *True*, a constant column of ones is prepended.
    """

    def __init__(
        self,
        degree: int = 2,
        interaction_only: bool = False,
        include_bias: bool = False,
    ) -> None:
        if degree < 1:
            raise ValueError("degree must be >= 1")
        self.degree = degree
        self.interaction_only = interaction_only
        self.include_bias = include_bias

        self._n_features_in: Optional[int] = None
        self._powers: Optional[np.ndarray] = None  # shape (n_output_features, n_features_in)
        self._selected_indices: Optional[np.ndarray] = None
        self._variances: Optional[np.ndarray] = None
        self._is_fitted: bool = False

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Optional[np.ndarray] = None,
    ) -> "FeatureEngineer":
        """Learn parameters from training data (e.g., feature variances for selection).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like, optional
            Target vector (unused but kept for pipeline compatibility).

        Returns
        -------
        self
        """
        X = self._to_numpy(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        self._n_features_in = X.shape[1]
        self._compute_powers()
        self._variances = np.nanvar(X, axis=0)
        self._is_fitted = True
        return self

    def transform(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Apply learned feature transformations.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        np.ndarray of shape (n_samples, n_output_features)
        """
        if not self._is_fitted:
            raise RuntimeError("FeatureEngineer has not been fitted yet.")
        X = self._to_numpy(X).astype(np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        # Apply feature selection mask if set
        if self._selected_indices is not None:
            X = X[:, self._selected_indices]

        return X

    def fit_transform(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Fit and transform in one call."""
        return self.fit(X, y).transform(X)

    # ------------------------------------------------------------------
    # Feature generators
    # ------------------------------------------------------------------

    def polynomial_features(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Generate polynomial and interaction features up to the configured degree.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        np.ndarray of shape (n_samples, n_output_features)
        """
        X = self._to_numpy(X).astype(np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        n_samples, n_features = X.shape
        output_columns: List[np.ndarray] = []

        if self.include_bias:
            output_columns.append(np.ones((n_samples, 1)))

        # Generate all monomial combinations up to self.degree
        for d in range(1, self.degree + 1):
            if d == 1 and not self.interaction_only:
                output_columns.append(X)
            else:
                for combo in combinations_with_replacement_indices(n_features, d):
                    if self.interaction_only and len(set(combo)) == 1:
                        continue
                    col = np.ones(n_samples, dtype=np.float64)
                    for idx in combo:
                        col *= X[:, idx]
                    output_columns.append(col.reshape(-1, 1))

        return np.hstack(output_columns)

    def interaction_features(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Generate pairwise interaction features (product of every two columns).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        np.ndarray of shape (n_samples, n_features + C(n_features, 2))
        """
        X = self._to_numpy(X).astype(np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        n_samples, n_features = X.shape
        interactions: List[np.ndarray] = [X]

        for i in range(n_features):
            for j in range(i + 1, n_features):
                interactions.append((X[:, i] * X[:, j]).reshape(-1, 1))

        return np.hstack(interactions)

    def binning_features(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        n_bins: int = 5,
        strategy: str = "uniform",
    ) -> np.ndarray:
        """Discretize continuous features into equal-width or equal-frequency bins.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        n_bins : int
            Number of bins per feature.
        strategy : str
            ``"uniform"`` for equal-width bins, ``"quantile"`` for equal-frequency bins.

        Returns
        -------
        np.ndarray of shape (n_samples, n_features)
            Integer bin labels starting from 0.
        """
        X = self._to_numpy(X).astype(np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        n_samples, n_features = X.shape
        binned = np.zeros_like(X, dtype=np.float64)

        for col in range(n_features):
            col_data = X[:, col]
            if strategy == "uniform":
                col_min, col_max = col_data.min(), col_data.max()
                if col_max == col_min:
                    binned[:, col] = 0
                else:
                    binned[:, col] = np.clip(
                        np.floor((col_data - col_min) / (col_max - col_min) * n_bins).astype(int),
                        0,
                        n_bins - 1,
                    )
            elif strategy == "quantile":
                percentiles = np.linspace(0, 100, n_bins + 1)
                edges = np.percentile(col_data, percentiles)
                # Ensure edges are strictly increasing
                for k in range(1, len(edges)):
                    if edges[k] <= edges[k - 1]:
                        edges[k] = edges[k - 1] + 1e-9
                binned[:, col] = np.clip(
                    np.digitize(col_data, edges[1:-1]), 0, n_bins - 1
                ).astype(np.float64)
            else:
                raise ValueError(f"Unknown binning strategy: {strategy!r}")

        return binned

    # ------------------------------------------------------------------
    # Feature selection
    # ------------------------------------------------------------------

    def select_by_variance(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        threshold: float = 0.0,
    ) -> np.ndarray:
        """Remove features whose variance is below *threshold*.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        threshold : float
            Minimum variance a feature must have to be retained.

        Returns
        -------
        np.ndarray of shape (n_samples, n_selected_features)
        """
        X = self._to_numpy(X).astype(np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        variances = np.nanvar(X, axis=0)
        mask = variances >= threshold
        self._selected_indices = np.where(mask)[0]
        self._variances = variances

        if self._selected_indices.size == 0:
            raise ValueError("All features were removed by variance threshold.")

        return X[:, self._selected_indices]

    def select_by_correlation(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        threshold: float = 0.95,
    ) -> np.ndarray:
        """Remove one of each pair of features whose absolute correlation exceeds
        *threshold* (greedy removal).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        threshold : float
            Correlation threshold (default 0.95).

        Returns
        -------
        np.ndarray of shape (n_samples, n_selected_features)
        """
        X = self._to_numpy(X).astype(np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        corr = np.abs(np.corrcoef(X, rowvar=False))
        n_features = X.shape[1]

        # Handle single-feature edge case
        if n_features == 1:
            self._selected_indices = np.array([0])
            return X

        removed: set = set()
        for i in range(n_features):
            if i in removed:
                continue
            for j in range(i + 1, n_features):
                if j in removed:
                    continue
                if np.isnan(corr[i, j]):
                    continue
                if corr[i, j] > threshold:
                    # Remove the feature that has higher average correlation with others
                    avg_i = np.nanmean(np.delete(corr[i], i))
                    avg_j = np.nanmean(np.delete(corr[j], j))
                    removed.add(j if avg_j >= avg_i else i)

        keep = [i for i in range(n_features) if i not in removed]
        if not keep:
            # Fallback: keep at least one feature
            keep = [int(np.argmax(np.nanvar(X, axis=0)))]
        self._selected_indices = np.array(keep)
        return X[:, self._selected_indices]

    # ------------------------------------------------------------------
    # Date / time feature extraction
    # ------------------------------------------------------------------

    @staticmethod
    def extract_datetime_features(
        X: Union[np.ndarray, pd.DataFrame],
        columns: Optional[List[Union[int, str]]] = None,
    ) -> np.ndarray:
        """Extract date/time components from timestamp columns.

        For each datetime column the following components are extracted:
        year, month, day, dayofweek, hour, minute, second.

        Parameters
        ----------
        X : array-like or pd.DataFrame
            Input containing timestamp columns.
        columns : list, optional
            Column indices (int) or names (str) to treat as datetime.
            If *None*, auto-detects ``datetime64`` columns in a DataFrame.

        Returns
        -------
        np.ndarray
            Extended feature matrix with datetime components appended.
        """
        if isinstance(X, pd.DataFrame):
            return FeatureEngineer._extract_dt_dataframe(X, columns)

        # NumPy path: assume specified columns contain datetime strings
        X_arr = np.asarray(X)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)

        if columns is None:
            # Try to auto-detect object columns
            columns = [c for c in range(X_arr.shape[1]) if X_arr[:, c].dtype.kind in ("O", "U", "S")]

        new_features: List[np.ndarray] = [X_arr]
        for col in columns:
            dates = pd.to_datetime(X_arr[:, col], errors="coerce")
            new_features.append(dates.dt.year.values.reshape(-1, 1).astype(np.float64))
            new_features.append(dates.dt.month.values.reshape(-1, 1).astype(np.float64))
            new_features.append(dates.dt.day.values.reshape(-1, 1).astype(np.float64))
            new_features.append(dates.dt.dayofweek.values.reshape(-1, 1).astype(np.float64))
            new_features.append(dates.dt.hour.values.reshape(-1, 1).astype(np.float64))
            new_features.append(dates.dt.minute.values.reshape(-1, 1).astype(np.float64))
            new_features.append(dates.dt.second.values.reshape(-1, 1).astype(np.float64))

        return np.hstack(new_features)

    @staticmethod
    def _extract_dt_dataframe(
        df: pd.DataFrame,
        columns: Optional[List[Union[int, str]]],
    ) -> np.ndarray:
        """Helper to extract datetime features from a DataFrame."""
        if columns is None:
            # Auto-detect datetime columns
            columns = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]

        new_cols: Dict[str, np.ndarray] = {}
        for col in columns:
            series = pd.to_datetime(df[col], errors="coerce")
            prefix = str(col)
            new_cols[f"{prefix}_year"] = series.dt.year.values.astype(np.float64)
            new_cols[f"{prefix}_month"] = series.dt.month.values.astype(np.float64)
            new_cols[f"{prefix}_day"] = series.dt.day.values.astype(np.float64)
            new_cols[f"{prefix}_dayofweek"] = series.dt.dayofweek.values.astype(np.float64)
            new_cols[f"{prefix}_hour"] = series.dt.hour.values.astype(np.float64)
            new_cols[f"{prefix}_minute"] = series.dt.minute.values.astype(np.float64)
            new_cols[f"{prefix}_second"] = series.dt.second.values.astype(np.float64)

        extended_df = df.copy()
        for name, arr in new_cols.items():
            extended_df[name] = arr

        return extended_df.values

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_numpy(X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            return X.values
        return np.asarray(X)

    def _compute_powers(self) -> None:
        """Pre-compute the exponent matrix for polynomial feature generation."""
        n = self._n_features_in
        if n is None:
            return

        # Use a simple recursive approach to enumerate monomial exponent vectors
        from itertools import combinations_with_replacement as cwr

        power_list: List[Tuple[int, ...]] = []
        for d in range(0, self.degree + 1):
            for combo in cwr(range(n), d):
                if d == 0:
                    if self.include_bias:
                        power_list.append(tuple(0 for _ in range(n)))
                else:
                    exponents = [0] * n
                    for idx in combo:
                        exponents[idx] += 1
                    if self.interaction_only and any(e > 1 for e in exponents):
                        continue
                    if d == 1 and not self.interaction_only:
                        power_list.append(tuple(exponents))
                    elif d > 1:
                        power_list.append(tuple(exponents))
            if d == 1 and not self.interaction_only:
                break  # Original features already added

        if power_list:
            self._powers = np.array(power_list, dtype=np.int32)
        else:
            self._powers = np.empty((0, n), dtype=np.int32)

    def get_feature_names_out(self, input_features: Optional[List[str]] = None) -> List[str]:
        """Return output feature names for polynomial features.

        Parameters
        ----------
        input_features : list[str], optional
            Names of the input features.

        Returns
        -------
        list[str]
            Names of the output features.
        """
        if self._n_features_in is None:
            raise RuntimeError("Must fit before calling get_feature_names_out.")

        if input_features is None:
            input_features = [f"x{i}" for i in range(self._n_features_in)]

        if self._powers is None or len(self._powers) == 0:
            return list(input_features)

        names: List[str] = []
        for row in self._powers:
            nonzero = [(exp, input_features[i]) for i, exp in enumerate(row) if exp > 0]
            if not nonzero:
                names.append("1")
            elif len(nonzero) == 1:
                exp, feat = nonzero[0]
                if exp == 1:
                    names.append(feat)
                else:
                    names.append(f"{feat}^{exp}")
            else:
                parts = [f"{feat}^{exp}" if exp > 1 else feat for exp, feat in nonzero]
                names.append(" * ".join(parts))

        return names

    def __repr__(self) -> str:
        return (
            f"FeatureEngineer(degree={self.degree}, "
            f"interaction_only={self.interaction_only}, "
            f"include_bias={self.include_bias}, "
            f"fitted={self._is_fitted})"
        )


# ------------------------------------------------------------------
# Module-level helper
# ------------------------------------------------------------------

def combinations_with_replacement_indices(n: int, r: int):
    """Yield all *r*-length combinations of indices from ``range(n)`` with
    replacement.  Equivalent to ``itertools.combinations_with_replacement`` but
    operates on integer indices for speed."""
    if r == 0:
        yield ()
        return
    if n == 0:
        return
    indices = list(range(n))

    def _comb(start, depth):
        if depth == 0:
            yield ()
            return
        for i in range(start, n):
            for rest in _comb(i, depth - 1):
                yield (indices[i],) + rest

    yield from _comb(0, r)
