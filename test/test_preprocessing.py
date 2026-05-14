"""
Unit tests for the DataPreprocessor module.

Covers standardization, normalization, missing value handling, categorical
encoding, and the full pipeline (fit/transform/inverse_transform) workflow.
Run with:  pytest tests/test_preprocessing.py -v
"""

import os
import sys
import numpy as np
import pandas as pd
import pytest

# Ensure project root is importable
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.preprocessing import DataPreprocessor


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def simple_data():
    """A small 2-D numeric array for basic tests."""
    np.random.seed(42)
    return np.random.randn(50, 4)


@pytest.fixture
def data_with_missing():
    """Array with NaN values injected at known positions."""
    np.random.seed(42)
    data = np.random.randn(50, 4)
    data[0, 0] = np.nan
    data[1, 2] = np.nan
    data[5, 1] = np.nan
    data[10, 3] = np.nan
    return data


@pytest.fixture
def data_with_categories():
    """Mixed numeric and string data as a 2-D object array."""
    data = np.array(
        [
            ["cat", 1.0, "A"],
            ["dog", 2.0, "B"],
            ["cat", 3.0, "A"],
            ["bird", 4.0, "C"],
            ["dog", 5.0, "B"],
        ],
        dtype=object,
    )
    return data


# ===========================================================================
# Tests – Standardization
# ===========================================================================


class TestStandardization:
    """Tests for Z-score standardization."""

    def test_standardize_output_shape(self, simple_data):
        """Standardized output should have the same shape as input."""
        preprocessor = DataPreprocessor(steps=["standardize"])
        result = preprocessor.fit_transform(simple_data)
        assert result.shape == simple_data.shape

    def test_standardize_zero_mean(self, simple_data):
        """After standardization, each column should have approximately zero mean."""
        preprocessor = DataPreprocessor(steps=["standardize"])
        result = preprocessor.fit_transform(simple_data)
        means = np.nanmean(result, axis=0)
        np.testing.assert_allclose(means, np.zeros(simple_data.shape[1]), atol=1e-10)

    def test_standardize_unit_variance(self, simple_data):
        """After standardization, each column should have approximately unit std."""
        preprocessor = DataPreprocessor(steps=["standardize"])
        result = preprocessor.fit_transform(simple_data)
        stds = np.nanstd(result, axis=0)
        np.testing.assert_allclose(stds, np.ones(simple_data.shape[1]), atol=1e-10)

    def test_standardize_standalone_method(self, simple_data):
        """The standalone standardize() method should produce the same result."""
        preprocessor = DataPreprocessor()
        standalone = preprocessor.standardize(simple_data)
        pipeline = preprocessor.fit_transform(simple_data)
        # After calling standardize, the pipeline hasn't been "fitted" via steps
        # so we use a second preprocessor for comparison
        prep2 = DataPreprocessor(steps=["standardize"])
        pipeline2 = prep2.fit_transform(simple_data)
        np.testing.assert_allclose(standalone, pipeline2, atol=1e-12)

    def test_standardize_constant_column(self):
        """A column with zero variance should not cause division by zero."""
        data = np.array([[1.0, 5.0], [1.0, 6.0], [1.0, 7.0]])
        preprocessor = DataPreprocessor(steps=["standardize"])
        result = preprocessor.fit_transform(data)
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))


# ===========================================================================
# Tests – Normalization
# ===========================================================================


class TestNormalization:
    """Tests for min-max normalization to [0, 1]."""

    def test_normalize_range(self, simple_data):
        """Normalized data should fall within [0, 1]."""
        preprocessor = DataPreprocessor(steps=["normalize"])
        result = preprocessor.fit_transform(simple_data)
        assert np.all(result >= 0.0 - 1e-10)
        assert np.all(result <= 1.0 + 1e-10)

    def test_normalize_min_equals_zero(self, simple_data):
        """The minimum of each normalized column should be exactly 0.0."""
        preprocessor = DataPreprocessor(steps=["normalize"])
        result = preprocessor.fit_transform(simple_data)
        np.testing.assert_allclose(np.min(result, axis=0), np.zeros(simple_data.shape[1]), atol=1e-10)

    def test_normalize_max_equals_one(self, simple_data):
        """The maximum of each normalized column should be exactly 1.0."""
        preprocessor = DataPreprocessor(steps=["normalize"])
        result = preprocessor.fit_transform(simple_data)
        np.testing.assert_allclose(np.max(result, axis=0), np.ones(simple_data.shape[1]), atol=1e-10)

    def test_inverse_normalize_reconstruction(self, simple_data):
        """inverse_transform on normalized data should reconstruct the original."""
        preprocessor = DataPreprocessor(steps=["normalize"])
        transformed = preprocessor.fit_transform(simple_data)
        reconstructed = preprocessor.inverse_transform(transformed)
        np.testing.assert_allclose(reconstructed, simple_data, atol=1e-10)


# ===========================================================================
# Tests – Missing Value Handling
# ===========================================================================


class TestMissingValues:
    """Tests for missing value imputation."""

    def test_no_nans_after_handling(self, data_with_missing):
        """After handling missing values, no NaN should remain."""
        preprocessor = DataPreprocessor(steps=["handle_missing"])
        result = preprocessor.fit_transform(data_with_missing)
        assert not np.any(np.isnan(result))

    def test_handle_missing_mean_strategy(self, data_with_missing):
        """NaN values should be replaced with the column mean."""
        preprocessor = DataPreprocessor(steps=["handle_missing"])
        result = preprocessor.fit_transform(data_with_missing)
        # Column 0 had NaN at row 0; it should now equal the column mean
        col_mean = np.nanmean(data_with_missing[:, 0])
        assert abs(result[0, 0] - col_mean) < 1e-10

    def test_handle_missing_median_strategy(self, data_with_missing):
        """The 'median' strategy should fill NaNs with the column median."""
        preprocessor = DataPreprocessor()
        result = preprocessor.handle_missing_values(data_with_missing, strategy="median")
        assert not np.any(np.isnan(result))
        col_median = np.nanmedian(data_with_missing[:, 0])
        assert abs(result[0, 0] - col_median) < 1e-10

    def test_handle_missing_constant_strategy(self, data_with_missing):
        """The 'constant' strategy should fill NaNs with the given value."""
        preprocessor = DataPreprocessor()
        result = preprocessor.handle_missing_values(data_with_missing, strategy="constant", fill_value=-999.0)
        assert not np.any(np.isnan(result))
        assert result[0, 0] == -999.0
        assert result[1, 2] == -999.0

    def test_handle_missing_invalid_strategy_raises(self, data_with_missing):
        """An invalid strategy name should raise ValueError."""
        preprocessor = DataPreprocessor()
        with pytest.raises(ValueError, match="Unknown missing-value strategy"):
            preprocessor.handle_missing_values(data_with_missing, strategy="unknown")


# ===========================================================================
# Tests – Categorical Encoding
# ===========================================================================


class TestCategoricalEncoding:
    """Tests for label encoding of categorical columns."""

    def test_encode_returns_floats(self, data_with_categories):
        """Encoded output should be a float array."""
        preprocessor = DataPreprocessor()
        result = preprocessor.encode_categorical(data_with_categories)
        assert result.dtype in (np.float64, np.float32)

    def test_encode_preserves_numeric_columns(self, data_with_categories):
        """Numeric columns should pass through unchanged."""
        preprocessor = DataPreprocessor()
        result = preprocessor.encode_categorical(data_with_categories)
        np.testing.assert_allclose(result[:, 1], np.array([1.0, 2.0, 3.0, 4.0, 5.0]))

    def test_encode_categories_are_integers(self, data_with_categories):
        """Encoded categorical values should be non-negative integers stored as floats."""
        preprocessor = DataPreprocessor()
        result = preprocessor.encode_categorical(data_with_categories)
        cats_col0 = result[:, 0]
        unique_vals = np.unique(cats_col0)
        assert np.all(unique_vals == np.floor(unique_vals))  # all integers

    def test_encode_consistent_mapping(self, data_with_categories):
        """The same category value should map to the same code across rows."""
        preprocessor = DataPreprocessor()
        result = preprocessor.encode_categorical(data_with_categories)
        # "cat" appears in rows 0 and 2; they should have the same code
        assert result[0, 0] == result[2, 0]
        # "dog" appears in rows 1 and 4
        assert result[1, 0] == result[4, 0]

    def test_encode_with_explicit_columns(self, data_with_categories):
        """Encoding only the specified column should leave others unchanged."""
        preprocessor = DataPreprocessor()
        result = preprocessor.encode_categorical(data_with_categories, columns=[0])
        # Column 2 ("A"/"B"/"C") should remain as strings (object dtype)
        assert result[:, 2].dtype.kind in ("O", "U", "S")


# ===========================================================================
# Tests – Pipeline / fit/transform
# ===========================================================================


class TestPipeline:
    """Tests for the full pipeline workflow."""

    def test_fit_transform_roundtrip(self, simple_data):
        """Fit-transform followed by inverse_transform should approximately reconstruct input."""
        preprocessor = DataPreprocessor(steps=["standardize", "normalize"])
        transformed = preprocessor.fit_transform(simple_data)
        reconstructed = preprocessor.inverse_transform(transformed)
        np.testing.assert_allclose(reconstructed, simple_data, atol=1e-10)

    def test_transform_without_fit_raises(self, simple_data):
        """Calling transform before fit should raise RuntimeError."""
        preprocessor = DataPreprocessor(steps=["standardize"])
        with pytest.raises(RuntimeError, match="not been fitted"):
            preprocessor.transform(simple_data)

    def test_from_config(self):
        """from_config should create a preprocessor with the given steps."""
        prep = DataPreprocessor.from_config(steps=["standardize", "handle_missing"])
        assert prep.steps == ["standardize", "handle_missing"]
        assert not prep._is_fitted

    def test_invalid_step_raises(self):
        """Providing an invalid step name should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown preprocessing step"):
            DataPreprocessor(steps=["bogus_step"])

    def test_transform_with_dataframe(self, simple_data):
        """Passing a pandas DataFrame to fit_transform should work the same as numpy."""
        df = pd.DataFrame(simple_data, columns=[f"f{i}" for i in range(simple_data.shape[1])])
        preprocessor = DataPreprocessor(steps=["standardize"])
        result_df = preprocessor.fit_transform(df)
        result_arr = preprocessor.fit_transform(simple_data)
        np.testing.assert_allclose(result_df, result_arr, atol=1e-12)
