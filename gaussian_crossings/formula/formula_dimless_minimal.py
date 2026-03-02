"""Minimal dimensionless formulation of exact upcrossing statistics.

This module provides the ``GaussianUpCrossingsDimless_minimal`` class, a
lightweight version of ``GaussianUpCrossingsDimless`` that contains only the
essential methods for computing the mean and variance of upcrossings in
dimensionless time (tau=1).  It is intended for use cases where downcrossing,
total-crossing, and Fano factor methods are not needed.
"""

from typing import Any, Callable, Optional, Tuple, Union

import torch
from torch.func import grad, hessian
import numpy as np
import math
import scipy.special
from gaussian_crossings.utils.owensT import owensT

torch.set_default_dtype(torch.float64)


class GaussianUpCrossingsDimless_minimal:
    """Minimal class for dimensionless upcrossing statistics of Gaussian processes.

    A lightweight alternative to ``GaussianUpCrossingsDimless`` that provides
    only the mean rate, variance, and the upcrossing integrand I^+(t).
    Useful when only upcrossing statistics are needed and computational
    overhead should be minimized.

    Attributes:
        r_func: Correlation function r(t) of the stationary Gaussian process.
        u: Threshold level for crossings.
        tau: Physical timescale used to rescale rates and variances.
        r0: Process variance, r(0).
        p0: First derivative of the correlation at t=0 (always 0 for a
            stationary process).
        q0: Negative second derivative of the correlation at t=0,
            i.e. q0 = -r''(0).
    """

    def __init__(
        self,
        r_func: Callable[..., torch.Tensor],
        u: float = 0,
        tau: Optional[float] = None,
        *args: Any,
        **kwargs: Any
    ) -> None:
        """Initialize the GaussianUpCrossingsDimless_minimal instance.

        Args:
            r_func: A function that accepts a time tensor t (and optionally
                additional parameters) and returns the correlation function
                r(t) as a torch.Tensor.
            u: The threshold level for crossings. Default is 0.
            tau: The time constant of the process for scaling results.
            *args: Additional positional arguments to be passed to r_func.
            **kwargs: Additional keyword arguments to be passed to r_func.

        Note:
            Upon initialization, the class evaluates r0, p0, and q0 at t=0
            using dimensionless time (tau=1).
        """
        self.r_func = r_func
        self.u = u 
        self.tau = tau # the time constant of associated with the process
        self.args = args
        self.kwargs = kwargs

        # define a small constant to avoid gradient explosion
        self.eps = 1e-15

        # Evaluate r, its first derivative (p), and the second derivative (q) at t = 0.
        self.r0 = self.r(torch.tensor(0.0, dtype=torch.float64))
        self.p0 = torch.tensor(0.0, dtype=torch.float64)
        self.q0 = self.q(torch.tensor(1e-40, dtype=torch.float64)) # to avoid issues with autograd at t=0

    def r(self, t: torch.Tensor) -> torch.Tensor:
        """Evaluate the correlation function r at the given time(s).

        Uses tau=1.0 to make everything dimensionless.

        Args:
            t: A tensor representing time(s) at which to evaluate r.

        Returns:
            The correlation function evaluated at t.
        """
        return self.r_func(t, tau=1.0, *self.args, **self.kwargs)

    def p(self, t: torch.Tensor) -> torch.Tensor:
        """Evaluate the first derivative of r(t) at dimensionless time.

        Args:
            t: A tensor representing time(s) at which to compute the derivative.

        Returns:
            The first derivative of r(t) evaluated at t.
        """
        def sum_func(t: torch.Tensor) -> torch.Tensor:
            return torch.sum(self.r(t))
        return grad(sum_func)(t)

    def q(self, t: torch.Tensor) -> torch.Tensor:
        """Evaluate the negative second derivative of r(t) at dimensionless time.

        This computes q(t) = -r''(t) for tau=1.

        Args:
            t: A tensor representing time(s) at which to compute the derivative.

        Returns:
            The negative second derivative of r(t) evaluated at t.
        """
        def sum_func(t: torch.Tensor) -> torch.Tensor:
            return torch.sum(self.p(t))
        return - grad(sum_func)(t)

    def _alpha(self, t: torch.Tensor) -> torch.Tensor:
        """Compute the auxiliary quantity alpha(t) for dimensionless time.

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
        """Compute the auxiliary quantity beta(t) for dimensionless time.

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
        self,
        t: torch.Tensor,
        u: Optional[Union[float, torch.Tensor]] = None
    ) -> torch.Tensor:
        """Compute the auxiliary quantity gamma(t) for dimensionless time.

        The formula is:
            gamma(t) = (sqrt(2) * p(t) / (r(t) + r0)) * u

        Args:
            t: A tensor representing time(s).
            u: The threshold level. If not provided, uses instance's u.

        Returns:
            The value of gamma(t).
        """
        if u is None:
            u = self.u
        sqrt2 = torch.sqrt(torch.tensor(2.0, dtype=torch.float64))
        return (sqrt2 * self.p(t) / (self.r(t) + self.r0)) * u

    def _delta(self, t: torch.Tensor) -> torch.Tensor:
        """Compute the auxiliary quantity delta(t) for dimensionless time.

        The formula is:
            delta(t) = 1 / (r(t) + r0)

        Args:
            t: A tensor representing time(s).

        Returns:
            The value of delta(t).
        """
        return 1.0 / (self.r(t) + self.r0)

    def compute_all_quantities(
        self,
        t: torch.Tensor,
        u: Optional[Union[float, torch.Tensor]] = None
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
        # p0 and q0 were computed at t=0 in __init__, but are not used in the following lines.
        alpha = torch.abs(
            -(r_vals + r0) / (2 * (p_vals**2 + (q_vals - self.q0) * (r_vals + r0)))
        )
        beta = torch.abs(
            -(r0 - r_vals) / (2 * (p_vals**2 + (q_vals + self.q0) * (r_vals - r0)))
        )
        sqrt2 = torch.sqrt(torch.tensor(2.0, dtype=torch.float64))
        gamma = (sqrt2 * p_vals / (r_vals + r0)) * u
        delta = 1.0 / (r_vals + r0)

        return alpha, beta, gamma, delta

    def upcrossing_mean_rate(
        self,
        u: Optional[Union[float, torch.Tensor]] = None
    ) -> torch.Tensor:
        """Compute the mean rate of upcrossings per unit time (Kac-Rice formula).

        Implements the Kac-Rice formula rescaled by the physical timescale:
            E[N_u^+] / T = (1/tau) * (1/(2*pi)) * sqrt(q0/r0) * exp(-u^2/(2*r0))

        Args:
            u: Threshold level. If not provided, uses the instance's u.

        Returns:
            The mean upcrossing rate per unit time.
        """
        if u is None:
            u = self.u
        return (1 / self.tau) * (1 / (2 * torch.pi)) * torch.sqrt(self.q0 / self.r0) * torch.exp(-u**2 / (2 * self.r0))

    def upcrossing_mean(
        self,
        T: float,
        u: Optional[Union[float, torch.Tensor]] = None
    ) -> torch.Tensor:
        """Compute the expected number of upcrossings over time interval T.

        Args:
            T: The length of the time interval.
            u: The threshold level. If not provided, uses instance's u.

        Returns:
            The expected number of upcrossings in time T.
        """
        return self.upcrossing_mean_rate(u=u) * T

    def upcrossing_integrand(
        self,
        t: torch.Tensor,
        u: Optional[Union[float, torch.Tensor]] = None
    ) -> torch.Tensor:
        """Compute the integrand I^+(t) for the upcrossing variance formula.

        Evaluates the closed-form integrand from Theorem 1 (Eq. 13 of the
        paper) in dimensionless time, expressed in terms of the error function
        and Owen's T function via the auxiliary quantities alpha, beta, gamma,
        and delta.

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

        expr1 = alpha**2 * gamma**2 / (alpha + beta)
        erf_arg = alpha * gamma / torch.sqrt(alpha + beta)

        final_integral = (
            torch.exp(-delta * u**2)
            / (4 * torch.pi**2 * torch.sqrt(torch.abs(r0**2 - r**2)))
            * (
                (torch.exp(-alpha * gamma**2) / (2 * torch.sqrt(alpha * beta)))
                * (
                    1 + math.sqrt(math.pi) * gamma * torch.sqrt(alpha + beta) * torch.exp(expr1) * torch.special.erf(erf_arg)
                )
                + torch.pi * ((alpha - beta - 2 * alpha * beta * gamma**2) / (alpha * beta))
                * owensT(gamma * torch.sqrt(2 * alpha * beta / (alpha + beta)), torch.sqrt(alpha / beta))
            )
        ) - (1 / (4 * torch.pi**2)) * (q0 / r0) * torch.exp(-(u**2) / r0)

        return final_integral

    def upcrossing_variance(
        self,
        T: float,
        u: Optional[Union[float, torch.Tensor]] = None,
        epsilon_left: float = 1e-5,
        epsilon_right: float = 1e-2,
        num_points: int = 1000
    ) -> torch.Tensor:
        """Compute the variance of upcrossings over time interval T.

        The computation maps the integration domain into [0, 1] and evaluates
        the integral using the trapezoidal rule.

        Args:
            T: The length of the time interval.
            u: The threshold level. If not provided, uses instance's u.
            epsilon_left: Small value to avoid singularity at 0.
            epsilon_right: Small value to avoid singularity at right.
            num_points: Number of points for numerical integration.

        Returns:
            The variance of upcrossings over time T.
        """
        if u is None:
            u = self.u

        if isinstance(self.tau, torch.Tensor):
            new_tau = self.tau.item()
        else:
            new_tau = self.tau

        t_prime = torch.linspace(epsilon_left, max((T / new_tau) / (1 + (T / new_tau)), epsilon_right), num_points)
        x = t_prime / (1 - t_prime)
        new_integrand_vals = (1 / self.tau) * (1 - self.tau * x / T) * self.upcrossing_integrand(x, u=u) / (1 - t_prime)**2
        left_limit = - (1 / self.tau) * (1 / (4 * torch.pi**2)) * (self.q0 / self.r0) * torch.exp(-(u**2) / self.r0)
        idx_in_limits = torch.abs(new_integrand_vals) < 1e-5
        integral = epsilon_left * left_limit + torch.trapz(new_integrand_vals[~idx_in_limits], t_prime[~idx_in_limits])

        return T * (self.upcrossing_mean_rate(u=u) + 2 * integral)

    def upcrossing_variance_CLT_per_unit_time(
        self,
        u: Optional[Union[float, torch.Tensor]] = None,
        epsilon_left: float = 1e-5,
        epsilon_right: float = 1e-2,
        num_points: int = 1000
    ) -> torch.Tensor:
        """Compute the variance per unit time of upcrossings (CLT formula).

        The integration maps the time domain into [0, 1] using numerical integration.

        Args:
            u: The threshold level. If not provided, uses instance's u.
            epsilon_left: Small value to avoid singularity at left endpoint.
            epsilon_right: Small value to avoid singularity at right endpoint.
            num_points: Number of points for numerical integration.

        Returns:
            The variance per unit time of upcrossings.
        """
        if u is None:
            u = self.u

        t_prime = torch.linspace(epsilon_left, 1 - epsilon_right, num_points)
        x = t_prime / (1 - t_prime)
        new_integrand_vals = (1 / self.tau) * self.upcrossing_integrand(x, u=u) / (1 - t_prime)**2
        left_limit = - (1 / self.tau) * (1 / (4 * torch.pi**2)) * (self.q0 / self.r0) * torch.exp(-(u**2) / self.r0)
        idx_in_limits = torch.abs(new_integrand_vals) < 1e-5
        integral = epsilon_left * left_limit + torch.trapz(new_integrand_vals[~idx_in_limits], t_prime[~idx_in_limits])

        return self.upcrossing_mean_rate(u=u) + 2 * integral
    
    def upcrossing_variance_CLT(
        self,
        T: float,
        u: Optional[Union[float, torch.Tensor]] = None,
        epsilon_left: float = 1e-5,
        epsilon_right: float = 1e-2,
        num_points: int = 1000
    ) -> torch.Tensor:
        """Compute variance of upcrossings over time T (CLT formula).

        Multiplies the per unit time variance by T.

        Args:
            T: The length of the time interval.
            u: The threshold level. If not provided, uses instance's u.
            epsilon_left: Small value to avoid singularity at left endpoint.
            epsilon_right: Small value to avoid singularity at right endpoint.
            num_points: Number of points for numerical integration.

        Returns:
            The variance of upcrossings over time T.
        """
        return T * (self.upcrossing_variance_CLT_per_unit_time(u=u, epsilon_left=epsilon_left, epsilon_right=epsilon_right, num_points=num_points))
