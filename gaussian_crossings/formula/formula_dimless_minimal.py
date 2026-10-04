"""Compatibility implementation of the original thresholded quadrature.

Deprecated: new work should use GaussianUpCrossings (or GaussianCrossings).
This class keeps the old sample-magnitude cutoff and endpoint defaults so
existing results remain reproducible; its numerical rule is not used for the
paper figures.
"""

import torch

from .formula_dimless import GaussianUpCrossingsDimless


class GaussianUpCrossingsDimless_minimal(GaussianUpCrossingsDimless):
    """Legacy dimensionless implementation with magnitude-filtered samples.

    .. deprecated:: 0.2.0
        Use :class:`GaussianUpCrossings` (or ``GaussianCrossings``).
    """

    _default_integration_method = "trapezoid"

    def _determinant(self, r0, r):
        return torch.abs(r0**2 - r**2)

    def _integration_mask(self, values):
        return ~(torch.abs(values) < 1e-5)

    def upcrossing_variance(
        self,
        T,
        u=None,
        epsilon_left=1e-5,
        epsilon_right=1e-2,
        num_points=1000,
        **integration_options,
    ):
        """Finite-window variance with the original minimum endpoint."""
        return self._variance_integral(
            T,
            u,
            epsilon_left,
            epsilon_right,
            num_points,
            total=False,
            minimum_endpoint=epsilon_right,
            **integration_options,
        )

    def upcrossing_variance_CLT_per_unit_time(
        self, u=None, epsilon_left=1e-5, epsilon_right=1e-2, num_points=1000, **integration_options
    ):
        """Asymptotic variance rate with the original tail cutoff."""
        return self._variance_integral(
            None, u, epsilon_left, epsilon_right, num_points, total=False, **integration_options
        )

    def upcrossing_variance_CLT(
        self,
        T,
        u=None,
        epsilon_left=1e-5,
        epsilon_right=1e-2,
        num_points=1000,
        **integration_options,
    ):
        """Asymptotic variance over T with the original tail cutoff."""
        return T * self.upcrossing_variance_CLT_per_unit_time(
            u, epsilon_left, epsilon_right, num_points, **integration_options
        )
