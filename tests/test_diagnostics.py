"""
Tests for Deep-DML residual diagnostics module.
"""

import math
import numpy as np
import pytest
from deepdml.datasets import make_synthetic_causal_data
from deepdml.diagnostics import residual_summary
from deepdml.estimator import DeepDML


def test_residual_summary_unfitted():
    """Calling residual_summary before fitting must raise ValueError."""
    model = DeepDML()
    with pytest.raises(ValueError, match="Estimator must be fitted"):
        residual_summary(model)


def test_residual_summary_fitted():
    """Verify residual_summary returns valid metrics on a fitted model."""
    X, t, y, _ = make_synthetic_causal_data(
        n_samples=100, n_features=4, treatment_effect=2.0, random_state=42
    )

    model = DeepDML(hidden_dim=16, epochs=5, n_splits=2, random_state=42)
    model.fit(X, t, y)

    summary = residual_summary(model)

    required_keys = [
        "mean_treatment_residual",
        "std_treatment_residual",
        "mean_outcome_residual",
        "std_outcome_residual",
        "variance_treatment_residual",
    ]

    for key in required_keys:
        assert key in summary
        assert isinstance(summary[key], float)
        assert not math.isnan(summary[key])
        assert not math.isinf(summary[key])

    # Check that variance matches square of standard deviation
    assert summary["variance_treatment_residual"] == pytest.approx(
        summary["std_treatment_residual"] ** 2, rel=1e-4
    )
