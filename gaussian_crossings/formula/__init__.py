"""Analytical formulae for level crossing statistics of Gaussian processes.

This subpackage contains classes that compute the exact mean, variance,
and Fano factor for upcrossings, downcrossings, and total crossings of
stationary Gaussian processes at arbitrary threshold levels. Implementations
are available in both dimensional and dimensionless formulations.
"""

from .formula import GaussianUpCrossings
from .formula_dimless import GaussianUpCrossingsDimless
from .formula_dimless_minimal import GaussianUpCrossingsDimless_minimal