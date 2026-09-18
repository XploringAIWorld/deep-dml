"""
Demonstration of using custom NumPy arrays with Deep-DML.

This example illustrates how practitioners can supply their own observational arrays
directly to the DeepDML estimator without external wrapper classes.
"""

import numpy as np
from deepdml import DeepDML
from deepdml.diagnostics import residual_summary


def main() -> None:
    print("==================================================")
    print("      Deep-DML: Custom NumPy Data Example          ")
    print("==================================================")

    # 1. Create custom simulated observation arrays
    np.random.seed(123)
    n_samples = 600
    n_features = 4

    # User-provided confounders matrix (X)
    X = np.random.randn(n_samples, n_features)

    # User-provided treatment vector (e.g. continuous dosage or binary exposure)
    treatment = 0.8 * X[:, 0] - 0.4 * X[:, 1] + np.random.randn(n_samples) * 0.5

    # True treatment effect of interest
    true_ate = 1.5

    # User-provided outcome vector influenced by confounders and treatment
    outcome = true_ate * treatment + (X[:, 0] ** 2) + np.sin(X[:, 2]) + np.random.randn(n_samples) * 0.5

    print(f"Supplied confounders array X: shape {X.shape}, dtype {X.dtype}")
    print(f"Supplied treatment array:     shape {treatment.shape}, dtype {treatment.dtype}")
    print(f"Supplied outcome array:       shape {outcome.shape}, dtype {outcome.dtype}")
    print(f"True treatment effect:        {true_ate:.3f}")

    # 2. Instantiate DeepDML
    model = DeepDML(
        hidden_dim=24,
        epochs=30,
        learning_rate=0.01,
        n_splits=2,
        random_state=123,
    )

    # 3. Fit estimator
    print("\nFitting DeepDML on custom NumPy arrays...")
    model.fit(X, treatment, outcome)

    # 4. View results and diagnostics
    print(f"\nEstimated ATE: {model.ate_:.3f}")
    print(f"Absolute error from true ATE: {abs(model.ate_ - true_ate):.3f}")

    # Residual diagnostics
    diag = residual_summary(model)
    print("\nResidual Diagnostics:")
    for metric, value in diag.items():
        print(f"  {metric}: {value:+.4f}")

    # Summary table
    model.summary(print_output=True)


if __name__ == "__main__":
    main()
