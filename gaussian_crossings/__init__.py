"""Level-crossing statistics for smooth stationary Gaussian processes.

The analytical formulas are from Rawat, Morone, Heeger, and Martiniani,
*Exact Variance and Fano Factor for Arbitrary Level Crossings in Stationary
Gaussian Processes* (2026). Importing the package does not change PyTorch's
process-wide default dtype. Internal quadrature uses double precision.
"""

from .formula import (
    GaussianUpCrossings,
    GaussianUpCrossingsDimless,
    GaussianUpCrossingsDimless_minimal,
    NumericalIntegrationWarning,
)

__version__ = "0.1.0"

__all__ = [
    "GaussianUpCrossings",
    "GaussianUpCrossingsDimless",
    "GaussianUpCrossingsDimless_minimal",
    "NumericalIntegrationWarning",
]
