"""Formula implementation for Gaussian process level crossing statistics.

This module provides the GaussianUpCrossings class which computes exact
mean, variance, and Fano factor for level crossings of stationary Gaussian
processes using the Kac-Rice formulas.
"""

from typing import Any, Callable, Optional, Tuple, Union

import torch
from torch.func import grad
import numpy as np
import math
import scipy.special
from gaussian_crossings.utils.owensT import owensT

torch.set_default_dtype(torch.float64)


class GaussianUpCrossings:
    """Compute level crossing statistics for stationary Gaussian processes.

    This class computes various quantities related to upcrossings, downcrossings,
    and crossings of a Gaussian process defined via its correlation function,
    including mean rates, variances, and Fano factors.

    Attributes:
        r_func: Correlation function of the process.
        u: Threshold level for crossings.
        r0: Correlation function value at t=0 (variance).
        p0: First derivative of correlation at t=0 (always 0 for stationary).
        q0: Negative second derivative of correlation at t=0.
    """

    def __init__(
        self,
        r_func: Callable[..., torch.Tensor],
        u: float = 0,
        *args: Any,
        **kwargs: Any
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

        # define a small constant to avoid gradient explosion
        self.eps = 1e-15

        # Evaluate r, its first derivative (p), and the second derivative (q) at t = 0.
        self.r0 = self.r(torch.tensor(0.0, dtype=torch.float64))
        self.p0 = torch.tensor(0.0, dtype=torch.float64)
        self.q0 = self.q(torch.tensor(1e-40, dtype=torch.float64)) # to avoid issues with autograd at t=0

    def r(self, t: torch.Tensor) -> torch.Tensor:
        """Evaluate the correlation function r at the given time(s).

        Args:
            t: A tensor representing time(s) at which to evaluate r.

        Returns:
            The correlation function evaluated at t.
        """
        return self.r_func(t, *self.args, **self.kwargs)

    def p(self, t: torch.Tensor) -> torch.Tensor:
        """Evaluate the first derivative of the correlation function r(t).

        Args:
            t: A tensor representing time(s) at which to compute the derivative.

        Returns:
            The first derivative of r(t) evaluated at t.
        """
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
        def sum_func(t: torch.Tensor) -> torch.Tensor:
            return torch.sum(self.p(t))
        return - grad(sum_func)(t)

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
        self,
        t: torch.Tensor,
        u: Optional[Union[float, torch.Tensor]] = None
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
        """Compute the mean rate of upcrossings per unit time.

        The rate is given by:
            (1 / (2*pi)) * sqrt(q0 / r0) * exp(-u^2 / (2*r0))

        Args:
            u: The threshold level. If not provided, uses instance's u.

        Returns:
            The mean rate of upcrossings per unit time.
        """
        if u is None:
            u = self.u
        return (1 / (2 * torch.pi)) * torch.sqrt(self.q0 / self.r0) * torch.exp(-u**2 / (2 * self.r0))

    def downcrossing_mean_rate(
        self,
        u: Optional[Union[float, torch.Tensor]] = None
    ) -> torch.Tensor:
        """Compute the mean rate of downcrossings per unit time.

        For stationary Gaussian processes, this equals the upcrossing rate.

        Args:
            u: The threshold level. If not provided, uses instance's u.

        Returns:
            The mean rate of downcrossings per unit time.
        """
        return self.upcrossing_mean_rate(u=u)

    def crossing_mean_rate(
        self,
        u: Optional[Union[float, torch.Tensor]] = None
    ) -> torch.Tensor:
        """Compute the mean rate of crossings per unit time.

        The crossings rate is twice the upcrossing rate.

        Args:
            u: The threshold level. If not provided, uses instance's u.

        Returns:
            The mean rate of crossings per unit time.
        """
        return 2 * self.upcrossing_mean_rate(u=u)

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

    def downcrossing_mean(
        self,
        T: float,
        u: Optional[Union[float, torch.Tensor]] = None
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
        self,
        T: float,
        u: Optional[Union[float, torch.Tensor]] = None
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
        self,
        t: torch.Tensor,
        u: Optional[Union[float, torch.Tensor]] = None
    ) -> torch.Tensor:
        """Compute the integrand for upcrossings variance.

        Evaluates the integral formula derived for the variance of upcrossings
        using the auxiliary quantities alpha, beta, gamma, and delta.

        Args:
            t: A tensor representing time(s) at which to evaluate the integrand.
            u: The threshold level. If not provided, uses instance's u.

        Returns:
            The value of the integrand for upcrossings variance.
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
            / (4 * torch.pi**2 * torch.sqrt(r0**2 - r**2))
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
    
    def downcrossing_integrand(
        self,
        t: torch.Tensor,
        u: Optional[Union[float, torch.Tensor]] = None
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
        self,
        t: torch.Tensor,
        u: Optional[Union[float, torch.Tensor]] = None
    ) -> torch.Tensor:
        """Compute the integrand for crossings variance.

        Evaluates the integral formula derived for the variance of crossings
        using the auxiliary quantities alpha, beta, gamma, and delta.

        Args:
            t: A tensor representing time(s) at which to evaluate the integrand.
            u: The threshold level. If not provided, uses instance's u.

        Returns:
            The value of the integrand for crossings variance.
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
            / (4 * torch.pi**2 * torch.sqrt(r0**2 - r**2))
            * (
                (2 * torch.exp(-alpha * gamma**2) / torch.sqrt(alpha * beta))
                * (
                    1 + math.sqrt(math.pi) * gamma * torch.sqrt(alpha + beta) * torch.exp(expr1) * torch.special.erf(erf_arg)
                )
                + 4 * torch.pi * ((alpha - beta - 2 * alpha * beta * gamma**2) / (alpha * beta))
                * (owensT(gamma * torch.sqrt(2 * alpha * beta / (alpha + beta)), torch.sqrt(alpha / beta)) - 1 / 8)
            )
        ) - (1 / (torch.pi**2)) * (q0 / r0) * torch.exp(-(u**2) / r0)

        return final_integral

    def upcrossing_variance(
        self,
        T: float,
        u: Optional[Union[float, torch.Tensor]] = None,
        epsilon_left: float = 1e-5,
        num_points: int = 1000
    ) -> torch.Tensor:
        """Compute the variance of upcrossings over time interval T.

        The computation maps the integration domain into [0, 1] and evaluates
        the integral using the trapezoidal rule.

        Args:
            T: The length of the time interval.
            u: The threshold level. If not provided, uses instance's u.
            epsilon_left: Small value to avoid singularity at 0.
            num_points: Number of points for numerical integration.

        Returns:
            The variance of upcrossings over time T.
        """
        if u is None:
            u = self.u

        t_prime = torch.linspace(epsilon_left, T / (1 + T), num_points)
        x = t_prime / (1 - t_prime)
        new_integrand_vals = (1 - x / T) * self.upcrossing_integrand(x, u=u) / (1 - t_prime)**2
        left_limit = - (1 / (4 * torch.pi**2)) * (self.q0 / self.r0) * torch.exp(-(u**2) / self.r0)
        integral = epsilon_left * left_limit + torch.trapz(new_integrand_vals[~torch.isnan(new_integrand_vals)], t_prime[~torch.isnan(new_integrand_vals)])

        return T * (self.upcrossing_mean_rate(u=u) + 2 * integral)

    def upcrossing_variance_CLT_per_unit_time(
        self,
        u: Optional[Union[float, torch.Tensor]] = None,
        epsilon_left: float = 1e-5,
        epsilon_right: float = 1e-5,
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
        new_integrand_vals = self.upcrossing_integrand(x, u=u) / (1 - t_prime)**2
        left_limit = - (1 / (4 * torch.pi**2)) * (self.q0 / self.r0) * torch.exp(-(u**2) / self.r0)
        integral = epsilon_left * left_limit + torch.trapz(new_integrand_vals[~torch.isnan(new_integrand_vals)], t_prime[~torch.isnan(new_integrand_vals)])

        return self.upcrossing_mean_rate(u=u) + 2 * integral
    
    def upcrossing_variance_CLT(
        self,
        T: float,
        u: Optional[Union[float, torch.Tensor]] = None,
        epsilon_left: float = 1e-5,
        epsilon_right: float = 1e-5,
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

    def downcrossing_variance(
        self,
        T: float,
        u: Optional[Union[float, torch.Tensor]] = None,
        epsilon_left: float = 1e-5,
        num_points: int = 1000
    ) -> torch.Tensor:
        """Compute the variance of downcrossings over time interval T.

        For stationary Gaussian processes, this equals the upcrossing variance.

        Args:
            T: The length of the time interval.
            u: The threshold level. If not provided, uses instance's u.
            epsilon_left: Small value to avoid singularity at 0.
            num_points: Number of points for numerical integration.

        Returns:
            The variance of downcrossings over time T.
        """
        return self.upcrossing_variance(T, u=u, epsilon_left=epsilon_left, num_points=num_points)

    def downcrossing_variance_CLT_per_unit_time(
        self,
        u: Optional[Union[float, torch.Tensor]] = None,
        epsilon_left: float = 1e-5,
        epsilon_right: float = 1e-5,
        num_points: int = 1000
    ) -> torch.Tensor:
        """Compute the variance per unit time of downcrossings (CLT formula).

        For stationary Gaussian processes, this equals the upcrossing variance.

        Args:
            u: The threshold level. If not provided, uses instance's u.
            epsilon_left: Small value to avoid singularity at left endpoint.
            epsilon_right: Small value to avoid singularity at right endpoint.
            num_points: Number of points for numerical integration.

        Returns:
            The variance per unit time of downcrossings.
        """
        return self.upcrossing_variance_CLT_per_unit_time(u=u, epsilon_left=epsilon_left, epsilon_right=epsilon_right, num_points=num_points)

    def downcrossing_variance_CLT(
        self,
        T: float,
        u: Optional[Union[float, torch.Tensor]] = None,
        epsilon_left: float = 1e-5,
        epsilon_right: float = 1e-5,
        num_points: int = 1000
    ) -> torch.Tensor:
        """Compute variance of downcrossings over time T (CLT formula).

        For stationary Gaussian processes, this equals the upcrossing variance.

        Args:
            T: The length of the time interval.
            u: The threshold level. If not provided, uses instance's u.
            epsilon_left: Small value to avoid singularity at left endpoint.
            epsilon_right: Small value to avoid singularity at right endpoint.
            num_points: Number of points for numerical integration.

        Returns:
            The variance of downcrossings over time T.
        """
        return self.upcrossing_variance_CLT(T, u=u, epsilon_left=epsilon_left, epsilon_right=epsilon_right, num_points=num_points)

    def crossing_variance(
        self,
        T: float,
        u: Optional[Union[float, torch.Tensor]] = None,
        epsilon_left: float = 1e-5,
        num_points: int = 1000
    ) -> torch.Tensor:
        """Compute the variance of crossings over time interval T.

        The integration domain is mapped into [0, 1] using numerical integration.

        Args:
            T: The length of the time interval.
            u: The threshold level. If not provided, uses instance's u.
            epsilon_left: Small value to avoid singularity at 0.
            num_points: Number of points for numerical integration.

        Returns:
            The variance of crossings over time T.
        """
        if u is None:
            u = self.u

        t_prime = torch.linspace(epsilon_left, T / (1 + T), num_points)
        x = t_prime / (1 - t_prime)
        new_integrand_vals = (1 - x / T) * self.crossing_integrand(x, u=u) / (1 - t_prime)**2
        left_limit = - (1 / torch.pi**2) * (self.q0 / self.r0) * torch.exp(-(u**2) / self.r0)
        integral = epsilon_left * left_limit + torch.trapz(new_integrand_vals[~torch.isnan(new_integrand_vals)], t_prime[~torch.isnan(new_integrand_vals)])

        return T * (self.crossing_mean_rate(u=u) + 2 * integral)
    
    def crossing_variance_CLT_per_unit_time(
        self,
        u: Optional[Union[float, torch.Tensor]] = None,
        epsilon_left: float = 1e-5,
        epsilon_right: float = 1e-5,
        num_points: int = 1000
    ) -> torch.Tensor:
        """Compute the variance per unit time of crossings (CLT formula).

        The integration maps the time domain into [0, 1] using numerical integration.

        Args:
            u: The threshold level. If not provided, uses instance's u.
            epsilon_left: Small value to avoid singularity at left endpoint.
            epsilon_right: Small value to avoid singularity at right endpoint.
            num_points: Number of points for numerical integration.

        Returns:
            The variance per unit time of crossings.
        """
        if u is None:
            u = self.u

        t_prime = torch.linspace(epsilon_left, 1 - epsilon_right, num_points)
        x = t_prime / (1 - t_prime)
        new_integrand_vals = self.crossing_integrand(x, u=u) / (1 - t_prime)**2
        left_limit = - (1 / torch.pi**2) * (self.q0 / self.r0) * torch.exp(-(u**2) / self.r0)
        integral = epsilon_left * left_limit + torch.trapz(new_integrand_vals[~torch.isnan(new_integrand_vals)], t_prime[~torch.isnan(new_integrand_vals)])

        return self.crossing_mean_rate(u=u) + 2 * integral
    
    def crossing_variance_CLT(
        self,
        T: float,
        u: Optional[Union[float, torch.Tensor]] = None,
        epsilon_left: float = 1e-5,
        epsilon_right: float = 1e-5,
        num_points: int = 1000
    ) -> torch.Tensor:
        """Compute variance of crossings over time T (CLT formula).

        Multiplies the per unit time variance by T.

        Args:
            T: The length of the time interval.
            u: The threshold level. If not provided, uses instance's u.
            epsilon_left: Small value to avoid singularity at left endpoint.
            epsilon_right: Small value to avoid singularity at right endpoint.
            num_points: Number of points for numerical integration.

        Returns:
            The variance of crossings over time T.
        """
        return T * (self.crossing_variance_CLT_per_unit_time(u=u, epsilon_left=epsilon_left, epsilon_right=epsilon_right, num_points=num_points))

    def upcrossing_fano_factor_CLT(
        self,
        u: Optional[Union[float, torch.Tensor]] = None,
        epsilon_left: float = 1e-5,
        epsilon_right: float = 1e-5,
        num_points: int = 1000
    ) -> torch.Tensor:
        """Compute the Fano factor for upcrossings (CLT formula).

        The Fano factor is variance/mean.

        Args:
            u: The threshold level. If not provided, uses instance's u.
            epsilon_left: Small value to avoid singularity at left endpoint.
            epsilon_right: Small value to avoid singularity at right endpoint.
            num_points: Number of points for numerical integration.

        Returns:
            The Fano factor for upcrossings.
        """
        return self.upcrossing_variance_CLT_per_unit_time(u=u, epsilon_left=epsilon_left, epsilon_right=epsilon_right, num_points=num_points) / self.upcrossing_mean_rate(u=u)

    def downcrossing_fano_factor_CLT(
        self,
        u: Optional[Union[float, torch.Tensor]] = None,
        epsilon_left: float = 1e-5,
        epsilon_right: float = 1e-5,
        num_points: int = 1000
    ) -> torch.Tensor:
        """Compute the Fano factor for downcrossings (CLT formula).

        The Fano factor is variance/mean.

        Args:
            u: The threshold level. If not provided, uses instance's u.
            epsilon_left: Small value to avoid singularity at left endpoint.
            epsilon_right: Small value to avoid singularity at right endpoint.
            num_points: Number of points for numerical integration.

        Returns:
            The Fano factor for downcrossings.
        """
        return self.downcrossing_variance_CLT_per_unit_time(u=u, epsilon_left=epsilon_left, epsilon_right=epsilon_right, num_points=num_points) / self.downcrossing_mean_rate(u=u)

    def crossing_fano_factor_CLT(
        self,
        u: Optional[Union[float, torch.Tensor]] = None,
        epsilon_left: float = 1e-5,
        epsilon_right: float = 1e-5,
        num_points: int = 1000
    ) -> torch.Tensor:
        """Compute the Fano factor for crossings (CLT formula).

        The Fano factor is variance/mean.

        Args:
            u: The threshold level. If not provided, uses instance's u.
            epsilon_left: Small value to avoid singularity at left endpoint.
            epsilon_right: Small value to avoid singularity at right endpoint.
            num_points: Number of points for numerical integration.

        Returns:
            The Fano factor for crossings.
        """
        return self.crossing_variance_CLT_per_unit_time(u=u, epsilon_left=epsilon_left, epsilon_right=epsilon_right, num_points=num_points) / self.crossing_mean_rate(u=u)

    def upcrossing_fano_factor(
        self,
        T: float,
        u: Optional[Union[float, torch.Tensor]] = None,
        epsilon_left: float = 1e-5,
        num_points: int = 1000
    ) -> torch.Tensor:
        """Compute the Fano factor for upcrossings over time T.

        The Fano factor is variance/mean.

        Args:
            T: The length of the time interval.
            u: The threshold level. If not provided, uses instance's u.
            epsilon_left: Small value to avoid singularity at 0.
            num_points: Number of points for numerical integration.

        Returns:
            The Fano factor for upcrossings over time T.
        """
        return self.upcrossing_variance(T, u=u, epsilon_left=epsilon_left, num_points=num_points) / self.upcrossing_mean(T, u=u)

    def downcrossing_fano_factor(
        self,
        T: float,
        u: Optional[Union[float, torch.Tensor]] = None,
        epsilon_left: float = 1e-5,
        num_points: int = 1000
    ) -> torch.Tensor:
        """Compute the Fano factor for downcrossings over time T.

        The Fano factor is variance/mean.

        Args:
            T: The length of the time interval.
            u: The threshold level. If not provided, uses instance's u.
            epsilon_left: Small value to avoid singularity at 0.
            num_points: Number of points for numerical integration.

        Returns:
            The Fano factor for downcrossings over time T.
        """
        return self.downcrossing_variance(T, u=u, epsilon_left=epsilon_left, num_points=num_points) / self.downcrossing_mean(T, u=u)

    def crossing_fano_factor(
        self,
        T: float,
        u: Optional[Union[float, torch.Tensor]] = None,
        epsilon_left: float = 1e-5,
        num_points: int = 1000
    ) -> torch.Tensor:
        """Compute the Fano factor for crossings over time T.

        The Fano factor is variance/mean.

        Args:
            T: The length of the time interval.
            u: The threshold level. If not provided, uses instance's u.
            epsilon_left: Small value to avoid singularity at 0.
            num_points: Number of points for numerical integration.

        Returns:
            The Fano factor for crossings over time T.
        """
        return self.crossing_variance(T, u=u, epsilon_left=epsilon_left, num_points=num_points) / self.crossing_mean(T, u=u)

    def upcrossing_integrand_mean_level(self, t: torch.Tensor) -> torch.Tensor:
        """Compute the integrand for upcrossings variance at mean level (u=0).

        This formula is specifically derived for mean level crossings.

        Args:
            t: A tensor representing time(s) at which to evaluate the integrand.

        Returns:
            The value of the integrand for upcrossings variance at u=0.
        """
        r = self.r(t)
        r0 = self.r0
        q0 = self.q0

        alpha, beta, _, _ = self.compute_all_quantities(t, u=0)

        final_integral = (
            1 / (8 * torch.pi**2 * torch.sqrt(r0**2 - r**2))
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
        """Compute the integrand for crossings variance at mean level (u=0).

        This formula is specifically derived for crossings at the mean level.

        Args:
            t: A tensor representing time(s) at which to evaluate the integrand.

        Returns:
            The value of the integrand for crossings variance at u=0.
        """
        r = self.r(t)
        r0 = self.r0
        q0 = self.q0

        alpha, beta, _, _ = self.compute_all_quantities(t, u=0)

        expr = torch.sqrt(alpha / beta)

        final_integral = (
            1 / (2 * torch.pi**2 * torch.sqrt(r0**2 - r**2))
            * (
                1 / (torch.sqrt(alpha * beta))
                + (alpha - beta) / (alpha * beta) * torch.arctan((expr - 1) / (expr + 1))
            )    
        ) - (1 / (torch.pi**2)) * (q0 / r0)

        return final_integral
