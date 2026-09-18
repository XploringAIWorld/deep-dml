"""
Deep-DML: A lightweight neural Double Machine Learning toolkit for causal treatment-effect estimation.
"""

from .datasets import make_synthetic_causal_data
from .diagnostics import residual_summary
from .estimator import DeepDML

__version__ = "1.0.0"

__all__ = [
    "DeepDML",
    "make_synthetic_causal_data",
    "residual_summary",
    "__version__",
]
