"""Analytical crossing statistics.

``GaussianUpCrossings`` (also available as ``GaussianCrossings``) is the
recommended interface. The two time-normalized classes are deprecated and kept
only so that existing code and archived results keep working.
"""

from .formula import GaussianCrossings, GaussianUpCrossings, NumericalIntegrationWarning
from .formula_dimless import GaussianUpCrossingsDimless
from .formula_dimless_minimal import GaussianUpCrossingsDimless_minimal
from .integration import IntegrationInfo

__all__ = [
    "GaussianCrossings",
    "GaussianUpCrossings",
    "GaussianUpCrossingsDimless",
    "GaussianUpCrossingsDimless_minimal",
    "NumericalIntegrationWarning",
    "IntegrationInfo",
]
