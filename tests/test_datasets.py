"""
Tests for synthetic causal dataset generator.
"""

import numpy as np
import pytest
from deepdml.datasets import make_synthetic_causal_data


def test_synthetic_data_shapes():
    """Verify that dataset generator returns correct dimensions."""
    n_samples = 150
    n_features = 6
    treatment_effect = 2.5

    X, treatment, outcome, effect = make_synthetic_causal_data(
        n_samples=n_samples,
        n_features=n_features,
        treatment_effect=treatment_effect,
        binary_treatment=False,
        random_state=42,
    )

    assert isinstance(X, np.ndarray)
    assert X.shape == (n_samples, n_features)
    assert isinstance(treatment, np.ndarray)
    assert treatment.shape == (n_samples,)
    assert isinstance(outcome, np.ndarray)
    assert outcome.shape == (n_samples,)
    assert effect == pytest.approx(treatment_effect)


def test_synthetic_data_reproducibility():
    """Verify that identical random seeds produce identical datasets."""
    X1, t1, y1, e1 = make_synthetic_causal_data(
        n_samples=80, n_features=4, treatment_effect=1.5, random_state=99
    )
    X2, t2, y2, e2 = make_synthetic_causal_data(
        n_samples=80, n_features=4, treatment_effect=1.5, random_state=99
    )

    np.testing.assert_allclose(X1, X2)
    np.testing.assert_allclose(t1, t2)
    np.testing.assert_allclose(y1, y2)
    assert e1 == e2


def test_synthetic_data_different_seeds():
    """Verify that different seeds produce different outputs."""
    X1, t1, y1, _ = make_synthetic_causal_data(n_samples=50, random_state=1)
    X2, t2, y2, _ = make_synthetic_causal_data(n_samples=50, random_state=2)

    assert not np.allclose(X1, X2)
    assert not np.allclose(t1, t2)
    assert not np.allclose(y1, y2)


def test_synthetic_data_binary_treatment():
    """Verify that binary treatment option produces strictly 0 and 1 values."""
    _, treatment, _, _ = make_synthetic_causal_data(
        n_samples=200,
        n_features=5,
        binary_treatment=True,
        random_state=42,
    )

    unique_vals = set(np.unique(treatment))
    assert unique_vals.issubset({0.0, 1.0})
    assert len(unique_vals) == 2  # Both classes present with decent sample size


def test_synthetic_data_invalid_params():
    """Verify that invalid arguments raise ValueError."""
    with pytest.raises(ValueError, match="n_samples must be at least 1"):
        make_synthetic_causal_data(n_samples=0)

    with pytest.raises(ValueError, match="n_features must be at least 2"):
        make_synthetic_causal_data(n_samples=100, n_features=1)
