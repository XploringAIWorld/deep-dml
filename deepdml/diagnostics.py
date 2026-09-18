"""
Diagnostics utilities for Deep-DML fitted models.

This module provides post-estimation residual checks and summary statistics
to help assess nuisance model fit and residual balance.
"""

from typing import Any, Dict
import numpy as np


def residual_summary(model: Any) -> Dict[str, float]:
    """
    Compute summary statistics of nuisance residuals from a fitted DeepDML model.

    Parameters
    ----------
    model : DeepDML
        A fitted DeepDML estimator instance.

    Returns
    -------
    dict
        Dictionary containing:
        - "mean_treatment_residual": Mean of the treatment residuals.
        - "std_treatment_residual": Standard deviation of the treatment residuals.
        - "mean_outcome_residual": Mean of the outcome residuals.
        - "std_outcome_residual": Standard deviation of the outcome residuals.
        - "variance_treatment_residual": Variance of the treatment residuals.

    Raises
    ------
    ValueError
        If the estimator has not been fitted yet.
    """
    if not hasattr(model, "treatment_residuals_") or not hasattr(model, "outcome_residuals_"):
        raise ValueError(
            "Estimator must be fitted before computing residual diagnostics. "
            "Call 'fit' on the DeepDML instance first."
        )

    t_res = model.treatment_residuals_
    y_res = model.outcome_residuals_

    return {
        "mean_treatment_residual": float(np.mean(t_res)),
        "std_treatment_residual": float(np.std(t_res)),
        "mean_outcome_residual": float(np.mean(y_res)),
        "std_outcome_residual": float(np.std(y_res)),
        "variance_treatment_residual": float(np.var(t_res)),
    }
