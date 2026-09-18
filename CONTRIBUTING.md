# Contributing to Deep-DML

Thank you for your interest in contributing to **Deep-DML**! Deep-DML is an open-source Python toolkit designed to make neural Double Machine Learning and causal treatment-effect estimation accessible, clean, and extensible.

We welcome contributions of all types: bug fixes, algorithmic improvements, new tests, documentation enhancements, and research discussions.

---

## Development Setup

### 1. Clone the Repository

```bash
git clone https://github.com/<username>/deep-dml.git
cd deep-dml
```

### 2. Set Up a Virtual Environment

We recommend using Python 3.10 or newer:

**On Linux/macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**On Windows (PowerShell):**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install in Editable Mode with Development Dependencies

```bash
pip install --upgrade pip
pip install -e ".[dev]"
```

---

## Running Tests

Run the full test suite using `pytest`:

```bash
pytest -v
```

Ensure all tests pass before submitting a pull request.

---

## Contribution Workflow

1. **Fork and Branch**: Create a descriptive feature branch from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```
2. **Implement Changes**: Keep functions compact, well-typed, and focused on core causal mechanics.
3. **Add Tests**: Every new feature or bug fix must be accompanied by automated unit tests in `tests/`.
4. **Data Leakage Check**: In causal estimation workflows, ensure that any data preprocessing (such as feature scaling) is fitted strictly on the training fold during cross-fitting and never on the full dataset or validation fold.
5. **Run Verification**: Ensure tests pass cleanly:
   ```bash
   pytest -v
   python examples/basic_example.py
   python examples/custom_data_example.py
   ```
6. **Submit a Pull Request**: Open a pull request against the `main` branch using the provided PR template.

---

## Code Quality & Standards

- **Clarity & Simplicity**: Prioritize clean, readable code with clear variable names over overly abstract architectures.
- **Type Hints**: Include Python type hints for all public methods and functions.
- **Docstrings**: Document public classes and functions using NumPy-style docstrings, noting parameters, return types, and potential exceptions.
- **No Unnecessary Dependencies**: Keep runtime dependencies minimal (`numpy`, `pandas`, `scikit-learn`, `torch`).
- **Deterministic Seeding**: Ensure random generators accept and respect `random_state` arguments to facilitate reproducible benchmarks.

---

## Community Guidelines

By participating in this project, you agree to abide by our [Code of Conduct](CODE_OF_CONDUCT.md).
