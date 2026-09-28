"""Time-normalized crossing statistics sharing the dimensional formula engine."""

from typing import Any, Callable, Optional

import torch

from .formula import GaussianUpCrossings


class GaussianUpCrossingsDimless(GaussianUpCrossings):
    """Evaluate a covariance at unit ``tau`` and restore physical time units.

    ``r_func`` is called with ``tau=1``. Its amplitude and other parameters
    are unchanged. Rates scale as ``1/tau``, finite windows use ``T/tau``,
    and asymptotic Fano factors are independent of the overall time scale.
    Integrand arguments are dimensionless lags. ``tau=None`` means unit time.

    The public methods and import path are retained from the original code.
    They share the adaptive default and ``method="trapezoid"`` compatibility
    option of :class:`GaussianUpCrossings`. Tolerances refer to Fano units.
    """

    def __init__(
        self,
        r_func: Callable[..., torch.Tensor],
        u: float = 0,
        tau: Optional[float] = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        self.tau = 1.0 if tau is None else tau
        value = torch.as_tensor(self.tau, dtype=torch.float64)
        if value.numel() != 1 or not bool(torch.isfinite(value) & (value > 0)):
            raise ValueError("tau must be a finite positive scalar")
        super().__init__(r_func, u, *args, **kwargs)

    @property
    def _time_scale(self):
        return self.tau

    def r(self, t: torch.Tensor) -> torch.Tensor:
        """Evaluate the covariance in units of the process correlation time."""
        t = torch.as_tensor(t, dtype=torch.float64)
        return self.r_func(t, *self.args, tau=1.0, **self.kwargs)

    def crossing_variance_CLT_per_unit_time(
        self, u=None, epsilon_left=1e-4, epsilon_right=1e-5, num_points=1000, **integration_options
    ):
        """Total-crossing variance rate, preserving the historical cutoff."""
        return super().crossing_variance_CLT_per_unit_time(
            u, epsilon_left, epsilon_right, num_points, **integration_options
        )

    def crossing_variance_CLT(
        self,
        T,
        u=None,
        epsilon_left=1e-4,
        epsilon_right=1e-5,
        num_points=1000,
        **integration_options,
    ):
        """Asymptotic total-crossing variance over a physical duration T."""
        return super().crossing_variance_CLT(
            T, u, epsilon_left, epsilon_right, num_points, **integration_options
        )
