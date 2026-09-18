"""
Basic demonstration of Deep-DML on synthetic observational causal data.

This example illustrates the end-to-end workflow:
1. Generating synthetic observational data with known treatment effect and non-linear confounding.
2. Initializing the DeepDML estimator.
3. Fitting with K-fold cross-fitting and neural nuisance models.
4. Comparing the estimated Average Treatment Effect (ATE) to ground truth.
5. Printing model estimation summary.
"""

from deepdml import DeepDML, make_synthetic_causal_data


def main() -> None:
    print("==================================================")
    print("      Deep-DML: Basic Synthetic Causal Example     ")
    print("==================================================")

    # 1. Generate synthetic causal observational data
    true_effect = 2.0
    print(f"\nGenerating observational dataset with true treatment effect: {true_effect:.3f}...")
    X, treatment, outcome, _ = make_synthetic_causal_data(
        n_samples=1000,
        n_features=8,
        treatment_effect=true_effect,
        binary_treatment=False,
        random_state=42,
    )

    print(f"Dataset shape: {X.shape[0]} samples, {X.shape[1]} confounders.")
    print(f"True treatment effect: {true_effect:.3f}")

    # 2. Initialize DeepDML estimator
    model = DeepDML(
        hidden_dim=32,
        epochs=40,
        learning_rate=0.01,
        batch_size=32,
        n_splits=3,
        random_state=42,
    )

    # 3. Fit estimator using cross-fitting
    print("\nFitting DeepDML with 3-fold cross-fitting and neural nuisance models...")
    model.fit(X, treatment, outcome)

    # 4. Display estimated ATE and error
    estimated_ate = model.ate_
    estimation_error = abs(estimated_ate - true_effect)

    print(f"\nEstimated ATE: {estimated_ate:.3f}")
    print(f"Absolute estimation error: {estimation_error:.3f}")

    # 5. Display model summary
    model.summary(print_output=True)


if __name__ == "__main__":
    main()
