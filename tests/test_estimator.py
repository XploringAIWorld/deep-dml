"""
Tests for DeepDML estimator class and cross-fitting workflow.
"""

import math
import numpy as np
import pandas as pd
import pytest
from deepdml.datasets import make_synthetic_causal_data
from deepdml.estimator import DeepDML


@pytest.fixture
def sample_causal_data():
    """Fixture providing a small synthetic causal dataset for fast testing."""
    return make_synthetic_causal_data(
        n_samples=120,
        n_features=4,
        treatment_effect=2.0,
        binary_treatment=False,
        random_state=42,
    )


def test_estimator_fit_success(sample_causal_data):
    """Verify that DeepDML fits successfully and estimates a finite float ATE."""
    X, t, y, _ = sample_causal_data
    model = DeepDML(hidden_dim=16, epochs=5, n_splits=2, random_state=42)
    fitted_model = model.fit(X, t, y)

    assert fitted_model is model
    assert hasattr(model, "ate_")
    assert isinstance(model.ate_, float)
    assert not math.isnan(model.ate_)
    assert not math.isinf(model.ate_)


@pytest.mark.parametrize("binary_treatment", [False, True])
def test_recovers_known_effect_on_synthetic_data(binary_treatment):
    """Check treatment-effect recovery on a reproducible data-generating process."""
    X, treatment, outcome, true_effect = make_synthetic_causal_data(
        n_samples=1000,
        n_features=8,
        treatment_effect=2.0,
        binary_treatment=binary_treatment,
        random_state=42,
    )

    model = DeepDML(n_splits=3, random_state=42).fit(X, treatment, outcome)

    assert abs(model.ate_ - true_effect) < 0.2


def test_fitted_attributes(sample_causal_data):
    """Verify all required public attributes are properly populated upon fitting."""
    X, t, y, _ = sample_causal_data
    n_samples, n_features = X.shape
    model = DeepDML(hidden_dim=16, epochs=5, n_splits=3, random_state=42)
    model.fit(X, t, y)

    assert model.n_samples_ == n_samples
    assert model.n_features_ == n_features
    assert model.n_splits_ == 3
    assert isinstance(model.outcome_residuals_, np.ndarray)
    assert model.outcome_residuals_.shape == (n_samples,)
    assert isinstance(model.treatment_residuals_, np.ndarray)
    assert model.treatment_residuals_.shape == (n_samples,)
    assert isinstance(model.nuisance_predictions_, dict)
    assert "outcome" in model.nuisance_predictions_
    assert "treatment" in model.nuisance_predictions_
    assert model.nuisance_predictions_["outcome"].shape == (n_samples,)
    assert model.nuisance_predictions_["treatment"].shape == (n_samples,)


def test_reproducibility(sample_causal_data):
    """Verify that same random seed produces identical ATE estimates."""
    X, t, y, _ = sample_causal_data

    model1 = DeepDML(hidden_dim=16, epochs=8, n_splits=2, random_state=123)
    model1.fit(X, t, y)

    model2 = DeepDML(hidden_dim=16, epochs=8, n_splits=2, random_state=123)
    model2.fit(X, t, y)

    assert model1.ate_ == pytest.approx(model2.ate_, abs=1e-6)
    np.testing.assert_allclose(model1.treatment_residuals_, model2.treatment_residuals_, atol=1e-5)
    np.testing.assert_allclose(model1.outcome_residuals_, model2.outcome_residuals_, atol=1e-5)


def test_pandas_inputs():
    """Verify that pandas DataFrame and Series are supported as inputs."""
    X_raw, t_raw, y_raw, _ = make_synthetic_causal_data(
        n_samples=100, n_features=3, treatment_effect=1.5, random_state=42
    )

    X_df = pd.DataFrame(X_raw, columns=[f"feat_{i}" for i in range(3)])
    t_series = pd.Series(t_raw, name="treatment")
    y_series = pd.Series(y_raw, name="outcome")

    model = DeepDML(hidden_dim=16, epochs=5, n_splits=2, random_state=42)
    model.fit(X_df, t_series, y_series)

    assert isinstance(model.ate_, float)
    assert not math.isnan(model.ate_)


def test_binary_treatment():
    """Verify that binary treatments (0/1) are handled properly."""
    X, t, y, _ = make_synthetic_causal_data(
        n_samples=100, n_features=4, treatment_effect=1.8, binary_treatment=True, random_state=42
    )

    model = DeepDML(hidden_dim=16, epochs=5, n_splits=2, random_state=42)
    model.fit(X, t, y)

    assert isinstance(model.ate_, float)
    assert not math.isnan(model.ate_)


def test_mismatched_sample_counts():
    """Mismatched number of samples between X, treatment, and outcome must raise ValueError."""
    X = np.random.randn(50, 3)
    t = np.random.randn(50)
    y_short = np.random.randn(40)

    model = DeepDML()
    with pytest.raises(ValueError, match="Mismatched sample counts"):
        model.fit(X, t, y_short)

    y = np.random.randn(50)
    t_short = np.random.randn(45)
    with pytest.raises(ValueError, match="Mismatched sample counts"):
        model.fit(X, t_short, y)


def test_nan_inputs_raise():
    """Inputs containing NaN must raise ValueError."""
    X = np.random.randn(50, 3)
    t = np.random.randn(50)
    y = np.random.randn(50)

    X[2, 1] = np.nan
    model = DeepDML()
    with pytest.raises(ValueError, match="Input contains NaN values"):
        model.fit(X, t, y)

    X[2, 1] = 0.0
    t[10] = np.nan
    with pytest.raises(ValueError, match="Input contains NaN values"):
        model.fit(X, t, y)

    t[10] = 0.0
    y[5] = np.nan
    with pytest.raises(ValueError, match="Input contains NaN values"):
        model.fit(X, t, y)


def test_inf_inputs_raise():
    """Inputs containing Inf or -Inf must raise ValueError."""
    X = np.random.randn(50, 3)
    t = np.random.randn(50)
    y = np.random.randn(50)

    X[0, 0] = np.inf
    model = DeepDML()
    with pytest.raises(ValueError, match="Input contains infinite"):
        model.fit(X, t, y)


def test_invalid_n_splits():
    """Invalid n_splits parameter must raise ValueError."""
    X = np.random.randn(50, 3)
    t = np.random.randn(50)
    y = np.random.randn(50)

    model_1split = DeepDML(n_splits=1)
    with pytest.raises(ValueError, match="n_splits must be an integer >= 2"):
        model_1split.fit(X, t, y)

    model_insufficient = DeepDML(n_splits=60)
    with pytest.raises(ValueError, match="Insufficient observations for cross-fitting"):
        model_insufficient.fit(X, t, y)


def test_invalid_dimensions():
    """Non-2D X or non-1D treatment/outcome must raise ValueError."""
    X_1d = np.random.randn(50)
    t = np.random.randn(50)
    y = np.random.randn(50)

    model = DeepDML()
    with pytest.raises(ValueError, match="X must be a 2-dimensional array"):
        model.fit(X_1d, t, y)

    X_2d = np.random.randn(50, 3)
    t_2d_multival = np.random.randn(50, 2)
    with pytest.raises(ValueError, match="treatment must be a 1-dimensional vector"):
        model.fit(X_2d, t_2d_multival, y)


def test_summary_before_fit_raises():
    """Calling summary before fitting must raise RuntimeError."""
    model = DeepDML()
    with pytest.raises(RuntimeError, match="Estimator is not fitted yet"):
        model.summary()


def test_summary_after_fit(sample_causal_data):
    """Verify summary returns a valid dictionary and prints output."""
    X, t, y, _ = sample_causal_data
    model = DeepDML(hidden_dim=16, epochs=5, n_splits=2, random_state=42)
    model.fit(X, t, y)

    summary = model.summary(print_output=False)
    assert isinstance(summary, dict)
    assert summary["n_samples"] == len(X)
    assert summary["n_features"] == X.shape[1]
    assert summary["n_splits"] == 2
    assert summary["ate"] == model.ate_
    assert "treatment_residual_std" in summary
    assert "outcome_residual_std" in summary


def test_constant_treatment_raises():
    """Constant treatment vector with zero variance must raise ValueError."""
    n_samples = 50
    X = np.random.randn(n_samples, 2)
    t = np.ones(n_samples)
    y = np.random.randn(n_samples)

    model = DeepDML(hidden_dim=8, epochs=2, n_splits=2, random_state=42)
    with pytest.raises(ValueError, match="Treatment has zero variance"):
        model.fit(X, t, y)


def test_near_zero_denominator_protection(monkeypatch, sample_causal_data):
    """Verify estimator guards against division by near-zero denominator."""
    import deepdml.estimator
    X, t, y, _ = sample_causal_data
    model = DeepDML(hidden_dim=8, epochs=1, n_splits=2, random_state=42)

    class NumpyWithZeroSum:
        def __getattr__(self, name):
            return getattr(np, name)

        @staticmethod
        def sum(*args, **kwargs):
            return 0.0

    monkeypatch.setattr(deepdml.estimator, "np", NumpyWithZeroSum())

    with pytest.raises(ValueError, match="Treatment residual variance is virtually zero"):
        model.fit(X, t, y)
