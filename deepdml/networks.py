"""
Neural network nuisance models for Deep-DML.

This module provides lightweight PyTorch multi-layer perceptron (MLP) architectures
and training routines for nuisance function estimation (conditional outcome E[Y|X]
and conditional treatment E[T|X]).

Note on Treatment Estimation:
    In Version 1.0.0, a regression-style nuisance network with Mean Squared Error (MSE)
    loss is used for both continuous and binary treatments. For binary treatments
    coded as 0 and 1, regression provides a direct plug-in estimate of the conditional
    expectation E[T|X] without requiring classification heuristics or probability clipping.
"""

from typing import Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class NuisanceMLP(nn.Module):
    """
    Lightweight Multi-Layer Perceptron (MLP) for nuisance function approximation.

    Architecture:
        Input -> Linear(in_features, hidden_dim) -> ReLU
              -> Linear(hidden_dim, hidden_dim) -> ReLU
              -> Linear(hidden_dim, 1)

    Parameters
    ----------
    in_features : int
        Number of input features (confounders).
    hidden_dim : int, default=32
        Number of hidden units in each intermediate layer.
    """

    def __init__(self, in_features: int, hidden_dim: int = 32) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass producing a 1D tensor of predictions.

        Parameters
        ----------
        x : torch.Tensor of shape (batch_size, in_features)
            Input features.

        Returns
        -------
        torch.Tensor of shape (batch_size,)
            Scalar predictions.
        """
        out = self.network(x)
        return out.squeeze(-1)


def train_nuisance_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    hidden_dim: int = 32,
    epochs: int = 50,
    learning_rate: float = 0.01,
    batch_size: int = 32,
    random_state: Optional[int] = None,
    device: str = "cpu",
) -> NuisanceMLP:
    """
    Train a nuisance MLP on training fold data.

    Parameters
    ----------
    X_train : np.ndarray of shape (n_samples, n_features)
        Standardized training confounders.
    y_train : np.ndarray of shape (n_samples,)
        Target vector (outcome Y or treatment T).
    hidden_dim : int, default=32
        Number of hidden units.
    epochs : int, default=50
        Number of training epochs.
    learning_rate : float, default=0.01
        Learning rate for Adam optimizer.
    batch_size : int, default=32
        Mini-batch size.
    random_state : Optional[int], default=None
        Deterministic random seed for PyTorch initializations.
    device : str, default="cpu"
        Computation device ("cpu" by default).

    Returns
    -------
    NuisanceMLP
        Trained model in evaluation mode.
    """
    if random_state is not None:
        torch.manual_seed(random_state)

    dev = torch.device(device)
    n_samples, in_features = X_train.shape

    model = NuisanceMLP(in_features=in_features, hidden_dim=hidden_dim).to(dev)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    effective_batch_size = max(1, min(batch_size, n_samples))
    x_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_tensor = torch.tensor(y_train, dtype=torch.float32)

    dataset = TensorDataset(x_tensor, y_tensor)
    
    # Use deterministic generator if random_state is provided
    if random_state is not None:
        generator = torch.Generator().manual_seed(random_state)
    else:
        generator = None

    loader = DataLoader(
        dataset,
        batch_size=effective_batch_size,
        shuffle=True,
        generator=generator,
    )

    model.train()
    for _ in range(epochs):
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(dev)
            batch_y = batch_y.to(dev)

            optimizer.zero_grad()
            preds = model(batch_x)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()

    model.eval()
    return model


def predict_nuisance_model(
    model: NuisanceMLP,
    X_val: np.ndarray,
    device: str = "cpu",
) -> np.ndarray:
    """
    Generate out-of-fold nuisance predictions using a trained model.

    Parameters
    ----------
    model : NuisanceMLP
        Trained neural network.
    X_val : np.ndarray of shape (n_samples, n_features)
        Standardized validation confounders.
    device : str, default="cpu"
        Computation device.

    Returns
    -------
    np.ndarray of shape (n_samples,)
        1D numpy array of predictions.
    """
    dev = torch.device(device)
    model.eval()
    with torch.no_grad():
        x_tensor = torch.tensor(X_val, dtype=torch.float32).to(dev)
        preds = model(x_tensor).cpu().numpy()

    # Ensure always 1D output
    preds_1d = np.asarray(preds, dtype=np.float64).reshape(-1)
    return preds_1d
