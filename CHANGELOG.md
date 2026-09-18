# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026

Initial public release.

### Added
- **Neural nuisance models**: Lightweight PyTorch Multi-Layer Perceptrons (`NuisanceMLP`) for learning non-linear conditional expectations `E[Y|X]` and `E[T|X]`.
- **K-fold cross-fitting**: Implementation of sample-splitting and cross-fitting to reduce regularization and overfitting bias in nuisance estimation.
- **Residual-based ATE estimation**: Neyman-orthogonal score residualization for estimating the Average Treatment Effect (ATE) in partially linear models.
- **Synthetic causal data generator**: `make_synthetic_causal_data` function generating reproducible observational datasets with known ground-truth treatment effects and non-linear confounding.
- **Residual diagnostics**: Post-estimation diagnostic suite in `residual_summary` for inspecting treatment and outcome residual balance.
- **NumPy and pandas support**: Flexible handling of NumPy ndarrays, pandas DataFrames, pandas Series, and native Python lists.
- **Reproducible examples**: Standalone scripts in `examples/` demonstrating synthetic benchmark evaluations and custom NumPy workflows.
- **Automated tests**: Comprehensive test suite using pytest covering estimator behavior, edge cases, numerical safeguards, input validations, and reproducibility.
- **GitHub Actions CI**: Automated continuous integration workflow testing on Python 3.10, 3.11, and 3.12.
