"""Level-crossing statistics for smooth stationary Gaussian processes.

The analytical formulas are from Rawat, Morone, Heeger, and Martiniani,
*Exact Variance and Fano Factor for Arbitrary Level Crossings in Stationary
Gaussian Processes* (2026). Importing the package does not change PyTorch's
process-wide default dtype. Internal quadrature uses double precision.

Typical use::

    from gaussian_crossings import GaussianCrossings
    from gaussian_crossings.process import r_damped_harmonic_oscillator_noise

    model = GaussianCrossings(r_damped_harmonic_oscillator_noise, temp=1.0, omega0=1.0, zeta=0.5)
    model.fano_factor(u=0.5)           # long-time Fano factor of upcrossings
    model.fano_factor(u=0.5, T=100.0)  # Fano factor of counts in windows of length 100
"""

from .estimation import CoarseSamplingWarning, FanoEstimate, empirical_fano, windowed_counts
from .formula import (
    GaussianCrossings,
    GaussianUpCrossings,
    IntegrationInfo,
    NumericalIntegrationWarning,
)

__version__ = "0.2.0"

__all__ = [
    "GaussianCrossings",
    "GaussianUpCrossings",
    "empirical_fano",
    "windowed_counts",
    "FanoEstimate",
    "CoarseSamplingWarning",
    "NumericalIntegrationWarning",
    "IntegrationInfo",
]

_DEPRECATED = ("GaussianUpCrossingsDimless", "GaussianUpCrossingsDimless_minimal")


def __getattr__(name):
    # Deprecated classes stay importable from the top level without being listed
    # in __all__; instantiating them emits a DeprecationWarning.
    if name in _DEPRECATED:
        from . import formula

        return getattr(formula, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
