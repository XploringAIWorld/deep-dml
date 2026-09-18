"""
Synthetic causal data generation for Double Machine Learning.

This module provides data generation functions for synthetic observational datasets
with known true Average Treatment Effects (ATE) for benchmarking, testing, and tutorial purposes.
"""

from typing import Optional, Tuple
import numpy as np


def make_synthetic_causal_data(
    n_samples: int = 1000,
    n_features: int = 10,
    treatment_effect: float = 2.0,
    binary_treatment: bool = False,
    random_state: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Generate synthetic observational data with known treatment effect and confounding.

    The partially linear structural equation model is:
        T = m(X) + ν
        Y = θ * T + h(X) + ε

    where:
        - X is a matrix of observed confounders drawn from N(0, I)
        - m(X) represents confounding on treatment
        - h(X) represents non-linear confounding on outcome
        - θ is the constant treatment effect (ATE)
        - ν, ε are independent Gaussian noise terms

    Parameters
    ----------
    n_samples : int, default=1000
        Number of observations to generate.
    n_features : int, default=10
        Number of confounder features. Must be >= 2.
    treatment_effect : float, default=2.0
        True Average Treatment Effect (ATE).
    binary_treatment : bool, default=False
        If True, generates a binary treatment T in {0, 1} using a logistic link.
        If False, generates a continuous treatment T.
    random_state : Optional[int], default=None
        Random seed for reproducible dataset generation.

    Returns
    -------
    X : np.ndarray of shape (n_samples, n_features)
        Confounders matrix.
    treatment : np.ndarray of shape (n_samples,)
        Treatment vector.
    outcome : np.ndarray of shape (n_samples,)
        Outcome vector.
    true_effect : float
        The true average treatment effect value.

    Raises
    ------
    ValueError
        If n_samples < 1 or n_features < 2.
    """
    if n_samples < 1:
        raise ValueError(f"n_samples must be at least 1, got {n_samples}.")
    if n_features < 2:
        raise ValueError(f"n_features must be at least 2, got {n_features}.")

    rng = np.random.RandomState(random_state)

    # 1. Generate observed confounders X
    X = rng.normal(loc=0.0, scale=1.0, size=(n_samples, n_features))

    # 2. Confounding function for treatment: m(X)
    # Uses linear combination and non-linear harmonic components
    m_X = 0.5 * X[:, 0] - 0.75 * X[:, 1]
    if n_features >= 3:
        m_X = m_X + 0.5 * np.sin(X[:, 2])

    treatment_noise = rng.normal(loc=0.0, scale=0.5, size=n_samples)

    if binary_treatment:
        # Logistic propensity score
        propensity = 1.0 / (1.0 + np.exp(-(m_X + treatment_noise)))
        # Clip propensity away from 0 and 1 for practical overlap
        propensity = np.clip(propensity, 0.05, 0.95)
        treatment = (rng.uniform(size=n_samples) < propensity).astype(np.float64)
    else:
        treatment = m_X + treatment_noise

    # 3. Structural confounding function for outcome: h(X)
    # Non-linear relationship with outcome
    h_X = 1.2 * X[:, 0] + 0.6 * (X[:, 1] ** 2)
    if n_features >= 4:
        h_X = h_X + 0.8 * np.cos(X[:, 3])

    outcome_noise = rng.normal(loc=0.0, scale=0.5, size=n_samples)

    # 4. Generate outcome Y = θ * T + h(X) + ε
    outcome = (treatment_effect * treatment) + h_X + outcome_noise

    return X, treatment, outcome, float(treatment_effect)
