"""Exact crossing statistics for stationary Gaussian processes.

This module provides the ``GaussianUpCrossings`` class, which implements
the exact analytical formulae for the mean (Kac-Rice), variance, and Fano
factor of arbitrary-level upcrossings, downcrossings, and total crossings
of smooth, stationary Gaussian processes.  The variance and Fano factor
expressions are derived in Theorems III.1 and III.2 of Rawat, Morone, Heeger, and
Martiniani (2026) and involve the error function and Owen's T function.
"""

import math
import warnings
from typing import Any, Callable, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch.func import grad

from gaussian_crossings.utils.owensT import owensT

from .integration import adaptive_fano

#: One threshold or an array of thresholds.
Threshold = Union[float, Sequence[float], np.ndarray, torch.Tensor]
#: A float for a scalar threshold, otherwise a NumPy array.
Statistic = Union[float, np.ndarray]


class NumericalIntegrationWarning(RuntimeWarning):
    """The compatibility quadrature omitted nonfinite small-lag samples."""


_TORCH_CALLBACK_MESSAGE = (
    "The covariance callback returned {kind}. It must be written with PyTorch "
    "operations (for example torch.exp rather than np.exp) and return a torch.Tensor, "
    "because r'(t) and r''(t) are obtained by automatic differentiation. The kernels "
    "in gaussian_crossings.process follow this convention."
)


class GaussianUpCrossings:
    """Compute exact level crossing statistics for stationary Gaussian processes.

    Given a correlation function r(t), this class computes the mean number of
    crossings (via the Kac-Rice formula), the exact variance (via the single-
    integral formula of Theorem III.1), and the Fano factor for upcrossings,
    downcrossings, and total crossings at an arbitrary threshold level u.
    ``GaussianCrossings`` is the same class under a shorter name.

    The mean crossing rate depends only on the local properties r(0) and
    r''(0), whereas the variance and Fano factor encode the full correlation
    structure at all lag times. A Fano factor of 1 means equal count variance
    and mean, without establishing that the crossing process is Poisson.
    Values below or above 1 indicate underdispersion or overdispersion,
    respectively, relative to Poisson counts.

    Recommended interface:
        ``mean_rate(u, kind)``, ``mean(T, u, kind)``, ``variance(T, u, kind)``,
        ``variance_rate(u, kind)``, and ``fano_factor(u, T=None, kind)``, with
        ``kind`` one of ``"up"``, ``"down"``, or ``"total"``. ``T=None`` in
        ``fano_factor`` gives the long-time limit; a finite ``T`` gives the
        variance-to-mean ratio of counts in windows of that length, which is the
        quantity to compare with data. These methods return a Python float for a
        scalar threshold and a NumPy array otherwise.

        The older ``upcrossing_*``, ``downcrossing_*``, and ``crossing_*``
        methods remain available and return PyTorch tensors; use them with
        ``method="trapezoid"`` when gradients are needed. In their names,
        ``_CLT`` denotes the long-time limit.

    Numerical integration:
        Variance and Fano methods default to ``method="adaptive"``. They accept
        ``epsabs=2e-10``, ``epsrel=2e-10`` (in Fano units), ``limit=1200``, and an
        optional initial ``cutoff`` in normalized lag units for known tails.
        ``last_integration_info`` records error estimates and tail refinement.
        This CPU path is not differentiable. ``method="trapezoid"`` retains the
        historical differentiable grid, endpoint arguments, and return types.
        Custom covariances need a checked local Taylor expansion through
        order 14; supplied kernels use dedicated stable expressions.

    Attributes:
        r_func: Correlation function r(t) of the stationary Gaussian process.
        u: Threshold level for crossings.
        r0: Process variance, r(0).
        p0: First derivative of the correlation at t=0 (always 0 for a
            stationary process).
        q0: Negative second derivative of the correlation at t=0,
            i.e. q0 = -r''(0).
    """

    _default_integration_method = "adaptive"

    def __init__(
        self, r_func: Callable[..., torch.Tensor], u: float = 0, *args: Any, **kwargs: Any
    ) -> None:
        """Initialize the GaussianUpCrossings instance.

        Args:
            r_func: A function that accepts a time tensor t (and optionally
                additional parameters) and returns the correlation function
                r(t) as a torch.Tensor.
            u: The threshold level for crossings. Default is 0.
            *args: Additional positional arguments to be passed to r_func.
            **kwargs: Additional keyword arguments to be passed to r_func.

        Note:
            Upon initialization, the class evaluates r0 (correlation at t=0),
            p0 (first derivative at t=0), and q0 (negative second derivative
            at t=0).
        """
        self.r_func = r_func
        self.u = u
        self.args = args
        self.kwargs = kwargs
        self.last_integration_info = None

        # Retained compatibility attribute; not a numerical clipping tolerance.
        self.eps = 1e-15

        # Evaluate r, its first derivative (p), and the second derivative (q) at t = 0.
        self.r0 = self.r(torch.tensor(0.0, dtype=torch.float64))
        if not isinstance(self.r0, torch.Tensor):
            raise TypeError(_TORCH_CALLBACK_MESSAGE.format(kind=type(self.r0).__name__))
        self.p0 = torch.tensor(0.0, dtype=torch.float64)
        # Use a positive lag because abs(t) has an artificial autograd cusp at zero.
        try:
            self.q0 = self.q(torch.tensor(1e-40, dtype=torch.float64))
        except (RuntimeError, TypeError) as exc:
            if "numpy" not in str(exc).lower() and "grad" not in str(exc).lower():
                raise
            raise TypeError(
                _TORCH_CALLBACK_MESSAGE.format(kind="a non-differentiable value")
            ) from exc
        for name, value in (("r(0)", self.r0), ("-r''(0)", self.q0)):
            if value.numel() != 1 or not bool(torch.isfinite(value) & (value > 0)):
                raise ValueError(
                    f"{name} must be a finite positive scalar. Crossing formulae require "
                    "a nondegenerate smooth process; ordinary OU noise is not smooth."
                )

    def __repr__(self) -> str:
        kernel = getattr(self.r_func, "__name__", repr(self.r_func))
        arguments = [repr(value) for value in self.args]
        arguments += [f"{key}={value!r}" for key, value in self.kwargs.items()]
        return f"{type(self).__name__}({', '.join([kernel, f'u={self.u!r}', *arguments])})"

    # ------------------------------------------------------------------
    # Recommended interface: one method per statistic, NumPy outputs.
    # ------------------------------------------------------------------

    _KINDS = {
        "up": "upcrossing",
        "upcrossing": "upcrossing",
        "upcrossings": "upcrossing",
        "down": "downcrossing",
        "downcrossing": "downcrossing",
        "downcrossings": "downcrossing",
        "total": "crossing",
        "all": "crossing",
        "crossing": "crossing",
        "crossings": "crossing",
    }

    def _statistic(self, name: str, kind: str):
        try:
            prefix = self._KINDS[str(kind).lower()]
        except KeyError:
            raise ValueError("kind must be 'up', 'down', or 'total'") from None
        return getattr(self, f"{prefix}_{name}")

    def _threshold(self, u):
        return torch.as_tensor(self.u if u is None else u, dtype=torch.float64)

    @staticmethod
    def _to_numpy(value):
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        array = np.asarray(value, dtype=float)
        return float(array) if array.ndim == 0 else array

    def _integrated(self, name: str, kind: str, u, options: dict, *args) -> Statistic:
        """Evaluate an integrated statistic for one threshold or an array of thresholds."""
        statistic = self._statistic(name, kind)
        level = self._threshold(u)
        if level.ndim > 0 and self._integration_method(options) == "trapezoid":
            # The fixed-grid rule broadcasts the threshold against its integration grid,
            # so it evaluates one threshold per call.
            values = [statistic(*args, u=value, **options) for value in level.reshape(-1)]
            return self._to_numpy(torch.stack(values).reshape(level.shape))
        return self._to_numpy(statistic(*args, u=level, **options))

    def mean_rate(self, u: Optional[Threshold] = None, kind: str = "up") -> Statistic:
        """Mean number of crossings per unit time (Kac-Rice formula).

        Args:
            u: Threshold(s). Defaults to the threshold stored in the model.
            kind: ``"up"``, ``"down"``, or ``"total"`` crossings.

        Returns:
            A float for a scalar threshold, otherwise a NumPy array.
        """
        return self._to_numpy(self._statistic("mean_rate", kind)(u=self._threshold(u)))

    def mean(self, T: float, u: Optional[Threshold] = None, kind: str = "up") -> Statistic:
        """Mean number of crossings in a window of length ``T``.

        Args:
            T: Window length, in the time units of the covariance.
            u: Threshold(s). Defaults to the threshold stored in the model.
            kind: ``"up"``, ``"down"``, or ``"total"`` crossings.

        Returns:
            A float for a scalar threshold, otherwise a NumPy array.
        """
        return self._to_numpy(self._statistic("mean", kind)(T, u=self._threshold(u)))

    def variance(
        self,
        T: float,
        u: Optional[Threshold] = None,
        kind: str = "up",
        **integration_options: Any,
    ) -> Statistic:
        """Variance of the number of crossings in a window of length ``T``.

        Args:
            T: Window length, in the time units of the covariance.
            u: Threshold(s). Defaults to the threshold stored in the model.
            kind: ``"up"``, ``"down"``, or ``"total"`` crossings.
            **integration_options: ``method``, ``epsabs``, ``epsrel``, ``limit``,
                and ``cutoff``; see the class notes.

        Returns:
            A float for a scalar threshold, otherwise a NumPy array.
        """
        return self._integrated("variance", kind, u, integration_options, T)

    def variance_rate(
        self, u: Optional[Threshold] = None, kind: str = "up", **integration_options: Any
    ) -> Statistic:
        """Long-time variance per unit time, ``lim Var[N(T)] / T`` as ``T`` grows.

        Args:
            u: Threshold(s). Defaults to the threshold stored in the model.
            kind: ``"up"``, ``"down"``, or ``"total"`` crossings.
            **integration_options: ``method``, ``epsabs``, ``epsrel``, ``limit``,
                and ``cutoff``; see the class notes.

        Returns:
            A float for a scalar threshold, otherwise a NumPy array.
        """
        return self._integrated("variance_CLT_per_unit_time", kind, u, integration_options)

    def fano_factor(
        self,
        u: Optional[Threshold] = None,
        T: Optional[float] = None,
        kind: str = "up",
        **integration_options: Any,
    ) -> Statistic:
        """Fano factor ``Var[N] / E[N]`` of crossing counts.

        With ``T=None`` this is the long-time limit. With a finite ``T`` it is
        the variance-to-mean ratio of counts in windows of length ``T``, which
        is what an estimate from data windows measures; the two can differ
        noticeably when ``T`` is only a few correlation times long.

        Args:
            u: Threshold(s). Defaults to the threshold stored in the model.
            T: Window length, or None for the long-time limit.
            kind: ``"up"``, ``"down"``, or ``"total"`` crossings. In the
                long-time limit the total-crossing Fano factor is twice the
                upcrossing one, because up- and downcrossings alternate.
            **integration_options: ``method``, ``epsabs``, ``epsrel``, ``limit``,
                and ``cutoff``; see the class notes.

        Returns:
            A float for a scalar threshold, otherwise a NumPy array.
        """
        if T is None:
            return self._integrated("fano_factor_CLT", kind, u, integration_options)
        return self._integrated("fano_factor", kind, u, integration_options, T)

    def r(self, t: torch.Tensor) -> torch.Tensor:
        """Evaluate the correlation function r at the given time(s).

        Args:
            t: A tensor representing time(s) at which to evaluate r.

        Returns:
            The correlation function evaluated at t.
        """
        t = torch.as_tensor(t, dtype=torch.float64)
        return self.r_func(t, *self.args, **self.kwargs)

    def p(self, t: torch.Tensor) -> torch.Tensor:
        """Evaluate the first derivative of the correlation function r(t).

        Args:
            t: A tensor representing time(s) at which to compute the derivative.

        Returns:
            The first derivative of r(t) evaluated at t.
        """
        t = torch.as_tensor(t, dtype=torch.float64)

        def sum_func(t: torch.Tensor) -> torch.Tensor:
            return torch.sum(self.r(t))

        return grad(sum_func)(t)

    def q(self, t: torch.Tensor) -> torch.Tensor:
        """Evaluate the negative second derivative of r(t).

        This effectively computes q(t) = -r''(t).

        Args:
            t: A tensor representing time(s) at which to compute the derivative.

        Returns:
            The negative second derivative of r(t) evaluated at t.
        """
        t = torch.as_tensor(t, dtype=torch.float64)

        def sum_func(t: torch.Tensor) -> torch.Tensor:
            return torch.sum(self.p(t))

        return -grad(sum_func)(t)

    def _alpha(self, t: torch.Tensor) -> torch.Tensor:
        """Compute the auxiliary quantity alpha(t).

        The formula is:
            alpha(t) = -(r(t) + r0) / (2*(p(t)^2 + (q(t) - q0)*(r(t) + r0)))

        Args:
            t: A tensor representing time(s).

        Returns:
            The absolute value of alpha(t).
        """
        return torch.abs(
            -(self.r(t) + self.r0)
            / (2 * (self.p(t) ** 2 + (self.q(t) - self.q0) * (self.r(t) + self.r0)))
        )

    def _beta(self, t: torch.Tensor) -> torch.Tensor:
        """Compute the auxiliary quantity beta(t).

        The formula is:
            beta(t) = -(r0 - r(t)) / (2*(p(t)^2 + (q(t) + q0)*(r(t) - r0)))

        Args:
            t: A tensor representing time(s).

        Returns:
            The absolute value of beta(t).
        """
        return torch.abs(
            -(self.r0 - self.r(t))
            / (2 * (self.p(t) ** 2 + (self.q(t) + self.q0) * (self.r(t) - self.r0)))
        )

    def _gamma(
        self, t: torch.Tensor, u: Optional[Union[float, torch.Tensor]] = None
    ) -> torch.Tensor:
        """Compute the auxiliary quantity gamma(t).

        The formula is:
            gamma(t) = (sqrt(2) * p(t) / (r(t) + r0)) * u

        Args:
            t: A tensor representing time(s).
            u: The threshold level. If not provided, uses the instance's u.

        Returns:
            The value of gamma(t).
        """
        if u is None:
            u = self.u
        sqrt2 = torch.sqrt(torch.tensor(2.0, dtype=torch.float64))
        return (sqrt2 * self.p(t) / (self.r(t) + self.r0)) * u

    def _delta(self, t: torch.Tensor) -> torch.Tensor:
        """Compute the auxiliary quantity delta(t).

        The formula is:
            delta(t) = 1 / (r(t) + r0)

        Args:
            t: A tensor representing time(s).

        Returns:
            The value of delta(t).
        """
        return 1.0 / (self.r(t) + self.r0)

    def compute_all_quantities(
        self, t: torch.Tensor, u: Optional[Union[float, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute auxiliary quantities alpha, beta, gamma, and delta at time t.

        Args:
            t: A tensor representing time(s).
            u: The threshold level for gamma. If not provided, uses instance's u.

        Returns:
            A tuple (alpha, beta, gamma, delta) evaluated at t.
        """
        if u is None:
            u = self.u

        r_vals = self.r(t)
        p_vals = self.p(t)
        q_vals = self.q(t)
        r0 = self.r0
        alpha = torch.abs(-(r_vals + r0) / (2 * (p_vals**2 + (q_vals - self.q0) * (r_vals + r0))))
        beta = torch.abs(-(r0 - r_vals) / (2 * (p_vals**2 + (q_vals + self.q0) * (r_vals - r0))))
        sqrt2 = torch.sqrt(torch.tensor(2.0, dtype=torch.float64))
        gamma = (sqrt2 * p_vals / (r_vals + r0)) * u
        delta = 1.0 / (r_vals + r0)

        return alpha, beta, gamma, delta

    def upcrossing_mean_rate(self, u: Optional[Union[float, torch.Tensor]] = None) -> torch.Tensor:
        """Compute the mean rate of upcrossings per unit time (Kac-Rice formula).

        Implements the Kac-Rice formula (Rice, 1944):
            E[N_u^+] / T = (1 / (2*pi)) * sqrt(q0 / r0) * exp(-u^2 / (2*r0))

        This depends only on the local properties of the correlation function
        at the origin (r(0) and r''(0)) and is independent of the full
        correlation structure.

        Args:
            u: Threshold level. If not provided, uses the instance's u.

        Returns:
            The mean upcrossing rate per unit time.
        """
        if u is None:
            u = self.u
        return (
            (1 / (2 * torch.pi))
            * torch.sqrt(self.q0 / self.r0)
            * torch.exp(-(u**2) / (2 * self.r0))
            / self._time_scale
        )

    def downcrossing_mean_rate(
        self, u: Optional[Union[float, torch.Tensor]] = None
    ) -> torch.Tensor:
        """Compute the mean rate of downcrossings per unit time.

        For stationary Gaussian processes, this equals the upcrossing rate.

        Args:
            u: The threshold level. If not provided, uses instance's u.

        Returns:
            The mean rate of downcrossings per unit time.
        """
        return self.upcrossing_mean_rate(u=u)

    def crossing_mean_rate(self, u: Optional[Union[float, torch.Tensor]] = None) -> torch.Tensor:
        """Compute the mean rate of crossings per unit time.

        The crossings rate is twice the upcrossing rate.

        Args:
            u: The threshold level. If not provided, uses instance's u.

        Returns:
            The mean rate of crossings per unit time.
        """
        return 2 * self.upcrossing_mean_rate(u=u)

    def upcrossing_mean(
        self, T: float, u: Optional[Union[float, torch.Tensor]] = None
    ) -> torch.Tensor:
        """Compute the expected number of upcrossings over time interval T.

        Args:
            T: The length of the time interval.
            u: The threshold level. If not provided, uses instance's u.

        Returns:
            The expected number of upcrossings in time T.
        """
        return self.upcrossing_mean_rate(u=u) * T

    def downcrossing_mean(
        self, T: float, u: Optional[Union[float, torch.Tensor]] = None
    ) -> torch.Tensor:
        """Compute the expected number of downcrossings over time interval T.

        Args:
            T: The length of the time interval.
            u: The threshold level. If not provided, uses instance's u.

        Returns:
            The expected number of downcrossings in time T.
        """
        return self.downcrossing_mean_rate(u=u) * T

    def crossing_mean(
        self, T: float, u: Optional[Union[float, torch.Tensor]] = None
    ) -> torch.Tensor:
        """Compute the expected number of crossings over time interval T.

        Args:
            T: The length of the time interval.
            u: The threshold level. If not provided, uses instance's u.

        Returns:
            The expected number of crossings in time T.
        """
        return self.crossing_mean_rate(u=u) * T

    def upcrossing_integrand(
        self, t: torch.Tensor, u: Optional[Union[float, torch.Tensor]] = None
    ) -> torch.Tensor:
        """Compute the integrand I^+(t) for the upcrossing variance formula.

        Evaluates the closed-form integrand from Theorem III.1 (Eq. 17 of the
        paper), which is expressed in terms of the error function and Owen's
        T function via the auxiliary quantities alpha, beta, gamma, and delta.
        The variance is obtained by integrating this quantity over time.

        Args:
            t: Time lag(s) at which to evaluate the integrand.
            u: Threshold level. If not provided, uses the instance's u.

        Returns:
            The integrand value I^+(t) for the upcrossing variance.
        """
        if u is None:
            u = self.u

        r = self.r(t)
        r0 = self.r0
        q0 = self.q0

        alpha, beta, gamma, delta = self.compute_all_quantities(t, u)

        combined_exp = torch.exp(-alpha * beta * gamma**2 / (alpha + beta))
        erf_arg = alpha * gamma / torch.sqrt(alpha + beta)

        final_integral = (
            torch.exp(-delta * u**2)
            / (4 * torch.pi**2 * torch.sqrt(self._determinant(r0, r)))
            * (
                (
                    torch.exp(-alpha * gamma**2)
                    + math.sqrt(math.pi)
                    * gamma
                    * torch.sqrt(alpha + beta)
                    * combined_exp
                    * torch.special.erf(erf_arg)
                )
                / (2 * torch.sqrt(alpha * beta))
                + torch.pi
                * ((alpha - beta - 2 * alpha * beta * gamma**2) / (alpha * beta))
                * owensT(
                    gamma * torch.sqrt(2 * alpha * beta / (alpha + beta)), torch.sqrt(alpha / beta)
                )
            )
        ) - (1 / (4 * torch.pi**2)) * (q0 / r0) * torch.exp(-(u**2) / r0)

        return final_integral

    def downcrossing_integrand(
        self, t: torch.Tensor, u: Optional[Union[float, torch.Tensor]] = None
    ) -> torch.Tensor:
        """Compute the integrand for downcrossings variance.

        For stationary Gaussian processes, this equals the upcrossing integrand.

        Args:
            t: A tensor representing time(s) at which to evaluate the integrand.
            u: The threshold level. If not provided, uses instance's u.

        Returns:
            The value of the integrand for downcrossings variance.
        """
        return self.upcrossing_integrand(t, u=u)

    def crossing_integrand(
        self, t: torch.Tensor, u: Optional[Union[float, torch.Tensor]] = None
    ) -> torch.Tensor:
        """Compute the integrand I(t) for the total crossing variance formula.

        Evaluates the closed-form integrand from Theorem III.2 (Eq. 24 of the
        paper) for bidirectional crossings, expressed in terms of the error
        function and Owen's T function.  The variance of total crossings is
        obtained by integrating this quantity over time.

        Args:
            t: Time lag(s) at which to evaluate the integrand.
            u: Threshold level. If not provided, uses the instance's u.

        Returns:
            The integrand value I(t) for the total crossing variance.
        """
        if u is None:
            u = self.u

        r = self.r(t)
        r0 = self.r0
        q0 = self.q0

        alpha, beta, gamma, delta = self.compute_all_quantities(t, u)

        combined_exp = torch.exp(-alpha * beta * gamma**2 / (alpha + beta))
        erf_arg = alpha * gamma / torch.sqrt(alpha + beta)

        final_integral = (
            torch.exp(-delta * u**2)
            / (4 * torch.pi**2 * torch.sqrt(self._determinant(r0, r)))
            * (
                2
                * (
                    torch.exp(-alpha * gamma**2)
                    + math.sqrt(math.pi)
                    * gamma
                    * torch.sqrt(alpha + beta)
                    * combined_exp
                    * torch.special.erf(erf_arg)
                )
                / torch.sqrt(alpha * beta)
                + 4
                * torch.pi
                * ((alpha - beta - 2 * alpha * beta * gamma**2) / (alpha * beta))
                * (
                    owensT(
                        gamma * torch.sqrt(2 * alpha * beta / (alpha + beta)),
                        torch.sqrt(alpha / beta),
                    )
                    - 1 / 8
                )
            )
        ) - (1 / (torch.pi**2)) * (q0 / r0) * torch.exp(-(u**2) / r0)

        return final_integral

    def _determinant(self, r0, r):
        return r0**2 - r**2

    @property
    def _time_scale(self):
        """Physical time per internal time unit; subclasses normalize this."""
        return 1.0

    def _integration_method(self, options):
        method = options.get("method")
        if method is None:
            method = self._default_integration_method
        if method not in ("adaptive", "trapezoid"):
            raise ValueError("method must be 'adaptive' or 'trapezoid'")
        return method

    def _adaptive_fano(
        self, T, u, *, total=False, method=None, epsabs=2e-10, epsrel=2e-10, limit=1200, cutoff=None
    ):
        self.last_integration_info = None
        result, info = adaptive_fano(
            self, T, u, total=total, epsabs=epsabs, epsrel=epsrel, limit=limit, cutoff=cutoff
        )
        self.last_integration_info = info
        return result

    def _integration_mask(self, values):
        """Preserve the historical endpoint treatment and make omissions visible."""
        if torch.isinf(values).any():
            raise FloatingPointError(
                "Infinite crossing integrand; refine the grid or rescale the process."
            )
        valid = ~torch.isnan(values)
        if not valid.all():
            warnings.warn(
                "Nonfinite small-lag samples were omitted by the compatibility quadrature. "
                "Check convergence in epsilon_left and num_points; use method='adaptive' "
                "for controlled integration.",
                NumericalIntegrationWarning,
                stacklevel=3,
            )
        if valid.sum() < 2:
            raise FloatingPointError("Fewer than two finite integration samples.")
        return valid

    def _variance_integral(
        self,
        T,
        u,
        epsilon_left,
        epsilon_right,
        num_points,
        *,
        total,
        minimum_endpoint=None,
        method=None,
        epsabs=2e-10,
        epsrel=2e-10,
        limit=1200,
        cutoff=None,
    ):
        """Dispatch to adaptive integration or the historical trapezoidal rule.

        t = s/(1-s) maps the positive half-line to (0, 1). This is a
        fixed-grid approximation, not an adaptive error-controlled integral.
        The historical near-origin correction is retained for compatibility.
        """
        options = dict(method=method, epsabs=epsabs, epsrel=epsrel, limit=limit, cutoff=cutoff)
        if self._integration_method(options) == "adaptive":
            level = torch.as_tensor(self.u if u is None else u, dtype=torch.float64)
            if level.numel() == 0 or not bool(torch.isfinite(level).all()):
                raise ValueError("u must contain finite thresholds and cannot be empty")
            if T == 0:
                self.last_integration_info = None
                return self.upcrossing_mean_rate(level) * 0
            value = self._adaptive_fano(T, level, total=total, **options)
            mean = self.crossing_mean_rate(level) if total else self.upcrossing_mean_rate(level)
            return value * mean if T is None else value * mean * T
        self.last_integration_info = None
        if u is None:
            u = self.u
        if torch.as_tensor(u).numel() != 1:
            raise ValueError(
                "method='trapezoid' evaluates one threshold per call. Pass a scalar u, use the "
                "default adaptive method, or use variance/variance_rate/fano_factor, which "
                "evaluate an array of thresholds one at a time."
            )
        if not isinstance(num_points, int) or isinstance(num_points, bool) or num_points < 2:
            raise ValueError("num_points must be an integer of at least 2")
        if not 0 < epsilon_left < 1 or not 0 <= epsilon_right < 1:
            raise ValueError("epsilon_left must be in (0, 1) and epsilon_right in [0, 1)")
        scale = self._time_scale
        if T is not None and (not math.isfinite(float(T)) or T < 0):
            raise ValueError("T must be finite and nonnegative")
        mean_rate = self.crossing_mean_rate(u) if total else self.upcrossing_mean_rate(u)
        if T == 0:
            return mean_rate * 0
        if T is None:
            upper = 1 - epsilon_right
        else:
            duration = T / float(scale)
            upper = duration / (1 + duration)
            if minimum_endpoint is not None:
                upper = max(upper, minimum_endpoint)
        if upper <= epsilon_left:
            raise ValueError("epsilon_left must be below the transformed integration endpoint")
        grid = torch.linspace(
            epsilon_left, upper, num_points, dtype=self.r0.dtype, device=self.r0.device
        )
        lag = grid / (1 - grid)
        integrand = self.crossing_integrand if total else self.upcrossing_integrand
        values = integrand(lag, u=u) / (1 - grid) ** 2 / scale
        if T is not None:
            values = values * (1 - scale * lag / T)
        factor = 1 if total else 4
        left_limit = (
            -(1 / (factor * torch.pi**2))
            * (self.q0 / self.r0)
            * torch.exp(-(u**2) / self.r0)
            / scale
        )
        valid = self._integration_mask(values)
        integral = epsilon_left * left_limit + torch.trapezoid(values[valid], grid[valid])
        rate = mean_rate + 2 * integral
        return rate if T is None else T * rate

    def upcrossing_variance(
        self,
        T: float,
        u: Optional[Union[float, torch.Tensor]] = None,
        epsilon_left: float = 1e-5,
        num_points: int = 1000,
        **integration_options,
    ) -> torch.Tensor:
        """Compute the variance of upcrossings over time interval T.

        Adaptive integration is the default. Pass method="trapezoid" for
        the historical fixed-grid calculation.

        Args:
            T: The length of the time interval.
            u: The threshold level. If not provided, uses instance's u.
            epsilon_left: Small value to avoid singularity at 0.
            num_points: Grid size for method="trapezoid" only.
            **integration_options: method, epsabs, epsrel, limit, and cutoff; see class notes.

        Returns:
            The variance of upcrossings over time T.
        """
        return self._variance_integral(
            T, u, epsilon_left, 0.0, num_points, total=False, **integration_options
        )

    def upcrossing_variance_CLT_per_unit_time(
        self,
        u: Optional[Union[float, torch.Tensor]] = None,
        epsilon_left: float = 1e-5,
        epsilon_right: float = 1e-5,
        num_points: int = 1000,
        **integration_options,
    ) -> torch.Tensor:
        """Compute the variance per unit time of upcrossings (CLT formula).

        Adaptive integration is the default; method="trapezoid" retains the old grid.

        Args:
            u: The threshold level. If not provided, uses instance's u.
            epsilon_left: Small value to avoid singularity at left endpoint.
            epsilon_right: Small value to avoid singularity at right endpoint.
            num_points: Grid size for method="trapezoid" only.
            **integration_options: method, epsabs, epsrel, limit, and cutoff; see class notes.

        Returns:
            The variance per unit time of upcrossings.
        """
        return self._variance_integral(
            None, u, epsilon_left, epsilon_right, num_points, total=False, **integration_options
        )

    def upcrossing_variance_CLT(
        self,
        T: float,
        u: Optional[Union[float, torch.Tensor]] = None,
        epsilon_left: float = 1e-5,
        epsilon_right: float = 1e-5,
        num_points: int = 1000,
        **integration_options,
    ) -> torch.Tensor:
        """Compute variance of upcrossings over time T (CLT formula).

        Multiplies the per unit time variance by T.

        Args:
            T: The length of the time interval.
            u: The threshold level. If not provided, uses instance's u.
            epsilon_left: Small value to avoid singularity at left endpoint.
            epsilon_right: Small value to avoid singularity at right endpoint.
            num_points: Grid size for method="trapezoid" only.
            **integration_options: method, epsabs, epsrel, limit, and cutoff; see class notes.

        Returns:
            The variance of upcrossings over time T.
        """
        return T * (
            self.upcrossing_variance_CLT_per_unit_time(
                u=u,
                epsilon_left=epsilon_left,
                epsilon_right=epsilon_right,
                num_points=num_points,
                **integration_options,
            )
        )

    def downcrossing_variance(
        self,
        T: float,
        u: Optional[Union[float, torch.Tensor]] = None,
        epsilon_left: float = 1e-5,
        num_points: int = 1000,
        **integration_options,
    ) -> torch.Tensor:
        """Compute the variance of downcrossings over time interval T.

        For stationary Gaussian processes, this equals the upcrossing variance.

        Args:
            T: The length of the time interval.
            u: The threshold level. If not provided, uses instance's u.
            epsilon_left: Small value to avoid singularity at 0.
            num_points: Grid size for method="trapezoid" only.
            **integration_options: method, epsabs, epsrel, limit, and cutoff; see class notes.

        Returns:
            The variance of downcrossings over time T.
        """
        return self.upcrossing_variance(
            T, u=u, epsilon_left=epsilon_left, num_points=num_points, **integration_options
        )

    def downcrossing_variance_CLT_per_unit_time(
        self,
        u: Optional[Union[float, torch.Tensor]] = None,
        epsilon_left: float = 1e-5,
        epsilon_right: float = 1e-5,
        num_points: int = 1000,
        **integration_options,
    ) -> torch.Tensor:
        """Compute the variance per unit time of downcrossings (CLT formula).

        For stationary Gaussian processes, this equals the upcrossing variance.

        Args:
            u: The threshold level. If not provided, uses instance's u.
            epsilon_left: Small value to avoid singularity at left endpoint.
            epsilon_right: Small value to avoid singularity at right endpoint.
            num_points: Grid size for method="trapezoid" only.
            **integration_options: method, epsabs, epsrel, limit, and cutoff; see class notes.

        Returns:
            The variance per unit time of downcrossings.
        """
        return self.upcrossing_variance_CLT_per_unit_time(
            u=u,
            epsilon_left=epsilon_left,
            epsilon_right=epsilon_right,
            num_points=num_points,
            **integration_options,
        )

    def downcrossing_variance_CLT(
        self,
        T: float,
        u: Optional[Union[float, torch.Tensor]] = None,
        epsilon_left: float = 1e-5,
        epsilon_right: float = 1e-5,
        num_points: int = 1000,
        **integration_options,
    ) -> torch.Tensor:
        """Compute variance of downcrossings over time T (CLT formula).

        For stationary Gaussian processes, this equals the upcrossing variance.

        Args:
            T: The length of the time interval.
            u: The threshold level. If not provided, uses instance's u.
            epsilon_left: Small value to avoid singularity at left endpoint.
            epsilon_right: Small value to avoid singularity at right endpoint.
            num_points: Grid size for method="trapezoid" only.
            **integration_options: method, epsabs, epsrel, limit, and cutoff; see class notes.

        Returns:
            The variance of downcrossings over time T.
        """
        return self.upcrossing_variance_CLT(
            T,
            u=u,
            epsilon_left=epsilon_left,
            epsilon_right=epsilon_right,
            num_points=num_points,
            **integration_options,
        )

    def crossing_variance(
        self,
        T: float,
        u: Optional[Union[float, torch.Tensor]] = None,
        epsilon_left: float = 1e-5,
        num_points: int = 1000,
        **integration_options,
    ) -> torch.Tensor:
        """Compute the variance of crossings over time interval T.

        Adaptive integration is the default; method="trapezoid" retains the old grid.

        Args:
            T: The length of the time interval.
            u: The threshold level. If not provided, uses instance's u.
            epsilon_left: Small value to avoid singularity at 0.
            num_points: Grid size for method="trapezoid" only.
            **integration_options: method, epsabs, epsrel, limit, and cutoff; see class notes.

        Returns:
            The variance of crossings over time T.
        """
        return self._variance_integral(
            T, u, epsilon_left, 0.0, num_points, total=True, **integration_options
        )

    def crossing_variance_CLT_per_unit_time(
        self,
        u: Optional[Union[float, torch.Tensor]] = None,
        epsilon_left: float = 1e-5,
        epsilon_right: float = 1e-5,
        num_points: int = 1000,
        **integration_options,
    ) -> torch.Tensor:
        """Compute the variance per unit time of crossings (CLT formula).

        Adaptive integration is the default; method="trapezoid" retains the old grid.

        Args:
            u: The threshold level. If not provided, uses instance's u.
            epsilon_left: Small value to avoid singularity at left endpoint.
            epsilon_right: Small value to avoid singularity at right endpoint.
            num_points: Grid size for method="trapezoid" only.
            **integration_options: method, epsabs, epsrel, limit, and cutoff; see class notes.

        Returns:
            The variance per unit time of crossings.
        """
        return self._variance_integral(
            None, u, epsilon_left, epsilon_right, num_points, total=True, **integration_options
        )

    def crossing_variance_CLT(
        self,
        T: float,
        u: Optional[Union[float, torch.Tensor]] = None,
        epsilon_left: float = 1e-5,
        epsilon_right: float = 1e-5,
        num_points: int = 1000,
        **integration_options,
    ) -> torch.Tensor:
        """Compute variance of crossings over time T (CLT formula).

        Multiplies the per unit time variance by T.

        Args:
            T: The length of the time interval.
            u: The threshold level. If not provided, uses instance's u.
            epsilon_left: Small value to avoid singularity at left endpoint.
            epsilon_right: Small value to avoid singularity at right endpoint.
            num_points: Grid size for method="trapezoid" only.
            **integration_options: method, epsabs, epsrel, limit, and cutoff; see class notes.

        Returns:
            The variance of crossings over time T.
        """
        return T * (
            self.crossing_variance_CLT_per_unit_time(
                u=u,
                epsilon_left=epsilon_left,
                epsilon_right=epsilon_right,
                num_points=num_points,
                **integration_options,
            )
        )

    def upcrossing_fano_factor_CLT(
        self,
        u: Optional[Union[float, torch.Tensor]] = None,
        epsilon_left: float = 1e-5,
        epsilon_right: float = 1e-5,
        num_points: int = 1000,
        **integration_options,
    ) -> torch.Tensor:
        """Compute the asymptotic Fano factor F^+ for upcrossings.

        The Fano factor is the variance-to-mean ratio in the long-time
        (T -> infinity) limit.  It is a dimensionless measure of crossing
        regularity: F^+ = 1 for Poisson-distributed crossings, F^+ < 1
        indicates sub-Poissonian regularity (anti-bunching), and F^+ > 1
        indicates super-Poissonian clustering (bunching).

        Args:
            u: Threshold level. If not provided, uses the instance's u.
            epsilon_left: Small value to avoid singularity at left endpoint.
            epsilon_right: Small value to avoid singularity at right endpoint.
            num_points: Grid size for method="trapezoid" only.
            **integration_options: method, epsabs, epsrel, limit, and cutoff; see class notes.

        Returns:
            The asymptotic Fano factor for upcrossings.
        """
        if self._integration_method(integration_options) == "adaptive":
            return self._adaptive_fano(None, u, total=False, **integration_options)
        return self.upcrossing_variance_CLT_per_unit_time(
            u=u,
            epsilon_left=epsilon_left,
            epsilon_right=epsilon_right,
            num_points=num_points,
            **integration_options,
        ) / self.upcrossing_mean_rate(u=u)

    def downcrossing_fano_factor_CLT(
        self,
        u: Optional[Union[float, torch.Tensor]] = None,
        epsilon_left: float = 1e-5,
        epsilon_right: float = 1e-5,
        num_points: int = 1000,
        **integration_options,
    ) -> torch.Tensor:
        """Compute the Fano factor for downcrossings (CLT formula).

        The Fano factor is variance/mean.

        Args:
            u: The threshold level. If not provided, uses instance's u.
            epsilon_left: Small value to avoid singularity at left endpoint.
            epsilon_right: Small value to avoid singularity at right endpoint.
            num_points: Grid size for method="trapezoid" only.
            **integration_options: method, epsabs, epsrel, limit, and cutoff; see class notes.

        Returns:
            The Fano factor for downcrossings.
        """
        if self._integration_method(integration_options) == "adaptive":
            return self._adaptive_fano(None, u, total=False, **integration_options)
        return self.downcrossing_variance_CLT_per_unit_time(
            u=u,
            epsilon_left=epsilon_left,
            epsilon_right=epsilon_right,
            num_points=num_points,
            **integration_options,
        ) / self.downcrossing_mean_rate(u=u)

    def crossing_fano_factor_CLT(
        self,
        u: Optional[Union[float, torch.Tensor]] = None,
        epsilon_left: float = 1e-5,
        epsilon_right: float = 1e-5,
        num_points: int = 1000,
        **integration_options,
    ) -> torch.Tensor:
        """Compute the asymptotic Fano factor F for total crossings.

        The Fano factor is the variance-to-mean ratio in the long-time
        (T -> infinity) limit.  It is a dimensionless measure of crossing
        regularity: F = 1 for Poisson-distributed crossings, F < 1
        indicates sub-Poissonian regularity (anti-bunching), and F > 1
        indicates super-Poissonian clustering (bunching).

        Args:
            u: Threshold level. If not provided, uses the instance's u.
            epsilon_left: Small value to avoid singularity at left endpoint.
            epsilon_right: Small value to avoid singularity at right endpoint.
            num_points: Grid size for method="trapezoid" only.
            **integration_options: method, epsabs, epsrel, limit, and cutoff; see class notes.

        Returns:
            The asymptotic Fano factor for total crossings.
        """
        if self._integration_method(integration_options) == "adaptive":
            return self._adaptive_fano(None, u, total=True, **integration_options)
        return self.crossing_variance_CLT_per_unit_time(
            u=u,
            epsilon_left=epsilon_left,
            epsilon_right=epsilon_right,
            num_points=num_points,
            **integration_options,
        ) / self.crossing_mean_rate(u=u)

    def upcrossing_fano_factor(
        self,
        T: float,
        u: Optional[Union[float, torch.Tensor]] = None,
        epsilon_left: float = 1e-5,
        num_points: int = 1000,
        **integration_options,
    ) -> torch.Tensor:
        """Compute the Fano factor for upcrossings over time T.

        The Fano factor is variance/mean.

        Args:
            T: The length of the time interval.
            u: The threshold level. If not provided, uses instance's u.
            epsilon_left: Small value to avoid singularity at 0.
            num_points: Grid size for method="trapezoid" only.
            **integration_options: method, epsabs, epsrel, limit, and cutoff; see class notes.

        Returns:
            The Fano factor for upcrossings over time T.
        """
        if self._integration_method(integration_options) == "adaptive":
            return self._adaptive_fano(T, u, total=False, **integration_options)
        return self.upcrossing_variance(
            T, u=u, epsilon_left=epsilon_left, num_points=num_points, **integration_options
        ) / self.upcrossing_mean(T, u=u)

    def downcrossing_fano_factor(
        self,
        T: float,
        u: Optional[Union[float, torch.Tensor]] = None,
        epsilon_left: float = 1e-5,
        num_points: int = 1000,
        **integration_options,
    ) -> torch.Tensor:
        """Compute the Fano factor for downcrossings over time T.

        The Fano factor is variance/mean.

        Args:
            T: The length of the time interval.
            u: The threshold level. If not provided, uses instance's u.
            epsilon_left: Small value to avoid singularity at 0.
            num_points: Grid size for method="trapezoid" only.
            **integration_options: method, epsabs, epsrel, limit, and cutoff; see class notes.

        Returns:
            The Fano factor for downcrossings over time T.
        """
        if self._integration_method(integration_options) == "adaptive":
            return self._adaptive_fano(T, u, total=False, **integration_options)
        return self.downcrossing_variance(
            T, u=u, epsilon_left=epsilon_left, num_points=num_points, **integration_options
        ) / self.downcrossing_mean(T, u=u)

    def crossing_fano_factor(
        self,
        T: float,
        u: Optional[Union[float, torch.Tensor]] = None,
        epsilon_left: float = 1e-5,
        num_points: int = 1000,
        **integration_options,
    ) -> torch.Tensor:
        """Compute the Fano factor for crossings over time T.

        The Fano factor is variance/mean.

        Args:
            T: The length of the time interval.
            u: The threshold level. If not provided, uses instance's u.
            epsilon_left: Small value to avoid singularity at 0.
            num_points: Grid size for method="trapezoid" only.
            **integration_options: method, epsabs, epsrel, limit, and cutoff; see class notes.

        Returns:
            The Fano factor for crossings over time T.
        """
        if self._integration_method(integration_options) == "adaptive":
            return self._adaptive_fano(T, u, total=True, **integration_options)
        return self.crossing_variance(
            T, u=u, epsilon_left=epsilon_left, num_points=num_points, **integration_options
        ) / self.crossing_mean(T, u=u)

    def upcrossing_integrand_mean_level(self, t: torch.Tensor) -> torch.Tensor:
        """Compute the integrand for upcrossings variance at mean level (u=0).

        When the threshold equals the process mean (u=0), the general
        formula simplifies: Owen's T function reduces to an arctangent,
        recovering the classical results of Steinberg et al. (1955) and
        Leadbetter and Cryer (1965).

        Args:
            t: Time lag(s) at which to evaluate the integrand.

        Returns:
            The simplified integrand value for mean-level upcrossing variance.
        """
        r = self.r(t)
        r0 = self.r0
        q0 = self.q0

        alpha, beta, _, _ = self.compute_all_quantities(t, u=0)

        final_integral = (
            1
            / (8 * torch.pi**2 * torch.sqrt(self._determinant(r0, r)))
            * (
                1 / (torch.sqrt(alpha * beta))
                + (alpha - beta) / (alpha * beta) * torch.arctan(torch.sqrt(alpha / beta))
            )
        ) - (1 / (4 * torch.pi**2)) * (q0 / r0)

        return final_integral

    def downcrossing_integrand_mean_level(self, t: torch.Tensor) -> torch.Tensor:
        """Compute the integrand for downcrossings variance at mean level (u=0).

        For stationary Gaussian processes, equals the upcrossing integrand.

        Args:
            t: A tensor representing time(s) at which to evaluate the integrand.

        Returns:
            The value of the integrand for downcrossings variance at u=0.
        """
        return self.upcrossing_integrand_mean_level(t=t)

    def crossing_integrand_mean_level(self, t: torch.Tensor) -> torch.Tensor:
        """Compute the integrand for total crossing variance at mean level (u=0).

        When the threshold equals the process mean (u=0), the general
        formula for bidirectional crossings simplifies to an expression
        involving only arctangent terms.

        Args:
            t: Time lag(s) at which to evaluate the integrand.

        Returns:
            The simplified integrand value for mean-level crossing variance.
        """
        r = self.r(t)
        r0 = self.r0
        q0 = self.q0

        alpha, beta, _, _ = self.compute_all_quantities(t, u=0)

        expr = torch.sqrt(alpha / beta)

        final_integral = (
            1
            / (2 * torch.pi**2 * torch.sqrt(self._determinant(r0, r)))
            * (
                1 / (torch.sqrt(alpha * beta))
                + (alpha - beta) / (alpha * beta) * torch.arctan((expr - 1) / (expr + 1))
            )
        ) - (1 / (torch.pi**2)) * (q0 / r0)

        return final_integral


# Shorter name for the same class: it handles up-, down-, and total crossings.
GaussianCrossings = GaussianUpCrossings
