"""
Core DeepDML Estimator for Partially Linear Causal Models.

This module implements the DeepDML estimator, which employs neural networks
as nuisance estimators combined with K-fold cross-fitting and Neyman-orthogonal
score residualization to estimate the Average Treatment Effect (ATE).
"""

from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from .networks import predict_nuisance_model, train_nuisance_model


def _safe_convert_and_validate(
    X: Union[np.ndarray, pd.DataFrame, list],
    treatment: Union[np.ndarray, pd.Series, pd.DataFrame, list],
    outcome: Union[np.ndarray, pd.Series, pd.DataFrame, list],
    n_splits: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Validate and convert inputs to standardized NumPy arrays.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Confounder matrix.
    treatment : array-like of shape (n_samples,)
        Treatment vector.
    outcome : array-like of shape (n_samples,)
        Outcome vector.
    n_splits : int
        Number of cross-fitting folds.

    Returns
    -------
    X_arr : np.ndarray of shape (n_samples, n_features)
    t_arr : np.ndarray of shape (n_samples,)
    y_arr : np.ndarray of shape (n_samples,)

    Raises
    ------
    ValueError
        If inputs are malformed, contain NaNs/Infs, have mismatched lengths,
        or fail cross-fitting criteria.
    """
    if not isinstance(n_splits, int) or n_splits < 2:
        raise ValueError(f"n_splits must be an integer >= 2, got {n_splits}.")

    # Convert X
    if isinstance(X, pd.DataFrame):
        X_arr = X.to_numpy(dtype=np.float64)
    else:
        try:
            X_arr = np.asarray(X, dtype=np.float64)
        except (ValueError, TypeError) as err:
            raise ValueError(f"Could not convert X to a numeric array: {err}") from err

    if X_arr.ndim != 2:
        raise ValueError(
            f"X must be a 2-dimensional array of shape (n_samples, n_features), "
            f"got {X_arr.ndim} dimensions."
        )

    n_samples, n_features = X_arr.shape

    if n_features == 0:
        raise ValueError("X must contain at least one feature column.")

    # Convert treatment
    if isinstance(treatment, (pd.Series, pd.DataFrame)):
        t_raw = treatment.to_numpy(dtype=np.float64)
    else:
        try:
            t_raw = np.asarray(treatment, dtype=np.float64)
        except (ValueError, TypeError) as err:
            raise ValueError(f"Could not convert treatment to a numeric array: {err}") from err

    if t_raw.ndim == 2 and t_raw.shape[1] == 1:
        t_arr = t_raw.reshape(-1)
    elif t_raw.ndim == 1:
        t_arr = t_raw
    else:
        raise ValueError(
            f"treatment must be a 1-dimensional vector of shape (n_samples,), "
            f"got shape {t_raw.shape}."
        )

    # Convert outcome
    if isinstance(outcome, (pd.Series, pd.DataFrame)):
        y_raw = outcome.to_numpy(dtype=np.float64)
    else:
        try:
            y_raw = np.asarray(outcome, dtype=np.float64)
        except (ValueError, TypeError) as err:
            raise ValueError(f"Could not convert outcome to a numeric array: {err}") from err

    if y_raw.ndim == 2 and y_raw.shape[1] == 1:
        y_arr = y_raw.reshape(-1)
    elif y_raw.ndim == 1:
        y_arr = y_raw
    else:
        raise ValueError(
            f"outcome must be a 1-dimensional vector of shape (n_samples,), "
            f"got shape {y_raw.shape}."
        )

    # Length consistency check
    if len(t_arr) != n_samples or len(y_arr) != n_samples:
        raise ValueError(
            f"Mismatched sample counts: X has {n_samples} samples, "
            f"treatment has {len(t_arr)}, and outcome has {len(y_arr)}."
        )

    # Missing / Infinite value checks
    if np.isnan(X_arr).any() or np.isnan(t_arr).any() or np.isnan(y_arr).any():
        raise ValueError(
            "Input contains NaN values. Deep-DML requires complete data without NaNs."
        )

    if np.isinf(X_arr).any() or np.isinf(t_arr).any() or np.isinf(y_arr).any():
        raise ValueError(
            "Input contains infinite (Inf or -Inf) values. Please clean inputs before fitting."
        )

    # Sample size vs splits check
    if n_samples < n_splits:
        raise ValueError(
            f"Insufficient observations for cross-fitting: dataset has {n_samples} samples, "
            f"which is less than n_splits={n_splits}."
        )

    # Treatment variance check
    if np.var(t_arr) < 1e-12:
        raise ValueError(
            "Treatment has zero variance (all values are identical). "
            "Cannot estimate treatment effect without treatment variation."
        )

    return X_arr, t_arr, y_arr


class DeepDML:
    """
    Neural Double Machine Learning estimator for the partially linear causal model.

    The partially linear model assumes:
        Y = θ * T + h(X) + ε
        T = m(X) + ν

    where:
        - X represents observed confounders
        - T represents treatment (continuous or binary)
        - Y represents outcome
        - h(X) is the structural effect of observed confounders on Y
        - m(X) = E[T|X] is the conditional treatment nuisance model
        - θ is the Average Treatment Effect (ATE)

    The outcome nuisance function is ell(X) = E[Y|X] = θ m(X) + h(X).
    DeepDML estimates ell(X) and m(X) with neural networks using K-fold cross-fitting,
    computes out-of-fold residuals:
        Y_tilde = Y - ell_hat(X)
        T_tilde = T - m_hat(X)
    and estimates the ATE via orthogonal residual regression:
        ATE = sum(T_tilde * Y_tilde) / sum(T_tilde^2)

    Parameters
    ----------
    hidden_dim : int, default=32
        Hidden layer width for neural nuisance models.
    epochs : int, default=50
        Number of training epochs per nuisance model.
    learning_rate : float, default=0.01
        Learning rate for Adam optimizer.
    batch_size : int, default=1024
        Batch size for training nuisance models. A larger default limits
        overfitting on typical cross-fitting folds.
    n_splits : int, default=2
        Number of cross-fitting folds (must be >= 2).
    random_state : Optional[int], default=None
        Random seed for reproducibility.
    device : str, default="cpu"
        Device on which PyTorch nuisance networks are trained.

    Attributes
    ----------
    ate_ : float
        Estimated Average Treatment Effect.
    n_samples_ : int
        Number of observations in the fitted dataset.
    n_features_ : int
        Number of confounder features.
    n_splits_ : int
        Number of cross-fitting folds used.
    outcome_residuals_ : np.ndarray of shape (n_samples,)
        Out-of-fold outcome residuals (Y - predicted_Y).
    treatment_residuals_ : np.ndarray of shape (n_samples,)
        Out-of-fold treatment residuals (T - predicted_T).
    nuisance_predictions_ : dict
        Dictionary of out-of-fold nuisance predictions with keys 'outcome' and 'treatment'.
    """

    def __init__(
        self,
        hidden_dim: int = 32,
        epochs: int = 50,
        learning_rate: float = 0.01,
        batch_size: int = 1024,
        n_splits: int = 2,
        random_state: Optional[int] = None,
        device: str = "cpu",
    ) -> None:
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_splits = n_splits
        self.random_state = random_state
        self.device = device

    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame, list],
        treatment: Union[np.ndarray, pd.Series, pd.DataFrame, list],
        outcome: Union[np.ndarray, pd.Series, pd.DataFrame, list],
    ) -> "DeepDML":
        """
        Fit the DeepDML estimator on observational data using K-fold cross-fitting.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Confounders matrix.
        treatment : array-like of shape (n_samples,)
            Treatment variable (continuous or binary).
        outcome : array-like of shape (n_samples,)
            Outcome variable.

        Returns
        -------
        self : DeepDML
            Fitted estimator instance.
        """
        X_arr, t_arr, y_arr = _safe_convert_and_validate(
            X=X, treatment=treatment, outcome=outcome, n_splits=self.n_splits
        )

        n_samples, n_features = X_arr.shape

        pred_outcome = np.zeros(n_samples, dtype=np.float64)
        pred_treatment = np.zeros(n_samples, dtype=np.float64)

        kf = KFold(
            n_splits=self.n_splits,
            shuffle=True,
            random_state=self.random_state,
        )

        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_arr)):
            # Deterministic fold-specific seed if random_state provided
            seed_y = None if self.random_state is None else (self.random_state + fold_idx * 101)
            seed_t = None if self.random_state is None else (self.random_state + fold_idx * 101 + 1)

            # Prevent data leakage: standardizer is fitted strictly on training fold
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_arr[train_idx])
            X_val_scaled = scaler.transform(X_arr[val_idx])

            # 1. Train conditional outcome model ell(X) = E[Y|X]
            outcome_model = train_nuisance_model(
                X_train=X_train_scaled,
                y_train=y_arr[train_idx],
                hidden_dim=self.hidden_dim,
                epochs=self.epochs,
                learning_rate=self.learning_rate,
                batch_size=self.batch_size,
                random_state=seed_y,
                device=self.device,
            )
            pred_outcome[val_idx] = predict_nuisance_model(
                model=outcome_model,
                X_val=X_val_scaled,
                device=self.device,
            )

            # 2. Train conditional treatment model m(X) = E[T|X]
            treatment_model = train_nuisance_model(
                X_train=X_train_scaled,
                y_train=t_arr[train_idx],
                hidden_dim=self.hidden_dim,
                epochs=self.epochs,
                learning_rate=self.learning_rate,
                batch_size=self.batch_size,
                random_state=seed_t,
                device=self.device,
            )
            pred_treatment[val_idx] = predict_nuisance_model(
                model=treatment_model,
                X_val=X_val_scaled,
                device=self.device,
            )

        # Compute Neyman-orthogonal score residuals
        outcome_residuals = y_arr - pred_outcome
        treatment_residuals = t_arr - pred_treatment

        # Estimate ATE = sum(T_tilde * Y_tilde) / sum(T_tilde^2)
        denominator = np.sum(treatment_residuals ** 2)
        if denominator < 1e-12:
            raise ValueError(
                "Treatment residual variance is virtually zero (denominator < 1e-12). "
                "Cannot reliably estimate ATE due to lack of treatment variation after conditioning."
            )

        numerator = np.sum(treatment_residuals * outcome_residuals)
        ate = numerator / denominator

        # Store fitted attributes
        self.ate_ = float(ate)
        self.n_samples_ = int(n_samples)
        self.n_features_ = int(n_features)
        self.n_splits_ = int(self.n_splits)
        self.outcome_residuals_ = outcome_residuals
        self.treatment_residuals_ = treatment_residuals
        self.nuisance_predictions_ = {
            "outcome": pred_outcome,
            "treatment": pred_treatment,
            "y": pred_outcome,
            "t": pred_treatment,
        }

        return self

    def summary(self, print_output: bool = True) -> Dict[str, Any]:
        """
        Generate and optionally display a summary of the estimation results.

        Parameters
        ----------
        print_output : bool, default=True
            If True, prints a clean human-readable summary table.

        Returns
        -------
        dict
            Dictionary containing summary metrics:
            - "n_samples": Number of samples
            - "n_features": Number of features
            - "n_splits": Number of cross-fitting folds
            - "ate": Estimated Average Treatment Effect
            - "treatment_residual_std": Standard deviation of treatment residuals
            - "outcome_residual_std": Standard deviation of outcome residuals

        Raises
        ------
        RuntimeError
            If called before the estimator has been fitted.
        """
        if not hasattr(self, "ate_"):
            raise RuntimeError(
                "Estimator is not fitted yet. Call 'fit' before calling 'summary'."
            )

        t_res_std = float(np.std(self.treatment_residuals_))
        y_res_std = float(np.std(self.outcome_residuals_))

        summary_dict = {
            "n_samples": self.n_samples_,
            "n_features": self.n_features_,
            "n_splits": self.n_splits_,
            "ate": self.ate_,
            "treatment_residual_std": t_res_std,
            "outcome_residual_std": y_res_std,
        }

        if print_output:
            print("\nDeep-DML Estimation Summary")
            print("----------------------------------------")
            print(f"Samples:                {self.n_samples_}")
            print(f"Features:               {self.n_features_}")
            print(f"Cross-fitting folds:    {self.n_splits_}")
            print(f"Estimated ATE:          {self.ate_:.4f}")
            print(f"Treatment residual std: {t_res_std:.4f}")
            print(f"Outcome residual std:   {y_res_std:.4f}")
            print("----------------------------------------\n")

        return summary_dict
