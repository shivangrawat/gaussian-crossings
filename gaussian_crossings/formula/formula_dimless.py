"""Time-normalized crossing statistics sharing the dimensional formula engine."""

import inspect
import warnings
from typing import Any, Callable, Optional

import torch

from .formula import GaussianUpCrossings


def _accepts_keyword(fn: Callable[..., Any], name: str) -> bool:
    """Return True when ``fn`` accepts ``name`` as a keyword or takes ``**kwargs``."""
    try:
        parameters = inspect.signature(fn).parameters.values()
    except (TypeError, ValueError):  # No introspectable signature: keep the old call.
        return True
    keyword_kinds = (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    return any(
        p.kind is inspect.Parameter.VAR_KEYWORD or (p.name == name and p.kind in keyword_kinds)
        for p in parameters
    )


class GaussianUpCrossingsDimless(GaussianUpCrossings):
    """Evaluate a covariance at unit ``tau`` and restore physical time units.

    If ``r_func`` accepts a ``tau`` keyword, it is called with ``tau=1``;
    otherwise it is evaluated with its supplied parameters. Its amplitude
    and other parameters are unchanged. Rates scale as ``1/tau``, finite windows use ``T/tau``,
    and asymptotic Fano factors are independent of the overall time scale.
    Integrand arguments are dimensionless lags. ``tau=None`` means unit time.

    The public methods and import path are retained from the original code.
    They share the adaptive default and ``method="trapezoid"`` compatibility
    option of :class:`GaussianUpCrossings`. Tolerances refer to Fano units.

    .. deprecated:: 0.2.0
        Use :class:`GaussianUpCrossings` (also available as
        ``GaussianCrossings``). For a covariance with a ``tau`` parameter,
        pass the former external time scale directly as ``tau=...``. For
        a covariance without one, such as the oscillator, wrap it as
        ``lambda t: r_func(t / tau, **params)`` to preserve the time rescaling
        and amplitude. A pure time rescaling leaves long-time Fano factors
        unchanged, but changes rates and finite-window results.
    """

    def __init__(
        self,
        r_func: Callable[..., torch.Tensor],
        u: float = 0,
        tau: Optional[float] = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        warnings.warn(
            f"{type(self).__name__} is deprecated and will be removed in a future release. "
            "Use GaussianUpCrossings (or GaussianCrossings): pass tau to the covariance "
            "where supported, or rescale t/tau in a covariance callback.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.tau = 1.0 if tau is None else tau
        value = torch.as_tensor(self.tau, dtype=torch.float64)
        if value.numel() != 1 or not bool(torch.isfinite(value) & (value > 0)):
            raise ValueError("tau must be a finite positive scalar")
        # Kernels without a time-scale parameter (e.g. the oscillator) are evaluated as given.
        self._unit_time_kwargs = {"tau": 1.0} if _accepts_keyword(r_func, "tau") else {}
        super().__init__(r_func, u, *args, **kwargs)

    @property
    def _time_scale(self):
        return self.tau

    def r(self, t: torch.Tensor) -> torch.Tensor:
        """Evaluate the covariance in units of the process correlation time."""
        t = torch.as_tensor(t, dtype=torch.float64)
        return self.r_func(t, *self.args, **self._unit_time_kwargs, **self.kwargs)

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
