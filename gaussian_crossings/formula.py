import torch
from torch.func import grad
import numpy as np
import scipy.special

torch.set_default_dtype(torch.float64)


class GaussianUpCrossings:
    def __init__(self, r_func, u=0, *args, **kwargs):
        """
        Initialize the GaussianUpCrossings instance.

        This class computes various quantities related to upcrossings, downcrossings,
        and crossings of a Gaussian process defined via its correlation function.

        Parameters:
            r_func (callable): A function that accepts a time tensor `t` (and optionally additional
                parameters) and returns the correlation function r(t) as a torch.Tensor.
            u (float, optional): The threshold level for upcrossings. Default is 0.
            *args: Additional positional arguments to be passed to `r_func`.
            **kwargs: Additional keyword arguments to be passed to `r_func`.

        Notes:
            Upon initialization, the class evaluates:
                - r0: the value of r(t) at t = 0,
                - p0: the first derivative of r(t) at t = 0,
                - q0: the negative second derivative of r(t) at t = 0.
        """
        self.r_func = r_func
        self.u = u 
        self.args = args
        self.kwargs = kwargs

        # Evaluate r, its first derivative (p), and the second derivative (q) at t = 0.
        self.r0 = self.r(torch.tensor([1e-20], dtype=torch.float64))
        self.p0 = self.p(torch.tensor([0.0], dtype=torch.float64))
        self.q0 = self.q(torch.tensor([1e-20], dtype=torch.float64))

    def r(self, t):
        """
        Evaluate the correlation function r at the given time(s).

        Parameters:
            t (torch.Tensor): A tensor representing time(s) at which to evaluate r.

        Returns:
            torch.Tensor: The correlation function evaluated at t.
        """
        return self.r_func(t, *self.args, **self.kwargs)

    def p(self, t):
        """
        Evaluate the first derivative of the correlation function r(t) at the given time(s).

        Parameters:
            t (torch.Tensor): A tensor representing time(s) at which to compute the derivative.

        Returns:
            torch.Tensor: The first derivative of r(t) evaluated at t.
        """
        def sum_func(t):
            return torch.sum(self.r(t))
        return grad(sum_func)(t)

    def q(self, t):
        """
        Evaluate the negative derivative of p(t) with respect to t.

        This effectively computes the (negative) second derivative of r(t).

        Parameters:
            t (torch.Tensor): A tensor representing time(s) at which to compute the derivative.

        Returns:
            torch.Tensor: The negative derivative of p(t) evaluated at t.
        """
        def sum_func(t):
            return torch.sum(self.p(t))
        return - grad(sum_func)(t)

    def _alpha(self, t):
        """
        Compute the auxiliary quantity alpha(t).

        The formula is given by:
            alpha(t) = - (r(t) + r0) / (2*(p(t)**2 + (q(t) - q0)*(r(t) + r0)))
        where r0 and q0 are the correlation function and its second derivative evaluated at t = 0.

        Parameters:
            t (torch.Tensor): A tensor representing time(s).

        Returns:
            torch.Tensor: The absolute value of alpha(t).
        """
        return torch.abs(
            -(self.r(t) + self.r0)
            / (2 * (self.p(t) ** 2 + (self.q(t) - self.q0) * (self.r(t) + self.r0)))
        )

    def _beta(self, t):
        """
        Compute the auxiliary quantity beta(t).

        The formula is given by:
            beta(t) = - (r0 - r(t)) / (2*(p(t)**2 + (q(t) + q0)*(r(t) - r0)))
        where r0 and q0 are the correlation function and its second derivative evaluated at t = 0.

        Parameters:
            t (torch.Tensor): A tensor representing time(s).

        Returns:
            torch.Tensor: The absolute value of beta(t).
        """
        return torch.abs(
            -(self.r0 - self.r(t))
            / (2 * (self.p(t) ** 2 + (self.q(t) + self.q0) * (self.r(t) - self.r0)))
        )

    def _gamma(self, t, u=None):
        """
        Compute the auxiliary quantity gamma(t).

        The formula is given by:
            gamma(t) = (sqrt(2) * p(t) / (r(t) + r0)) * u,
        where u is a given constant (or tensor) and r0 is the value of the correlation function at t = 0.

        Parameters:
            t (torch.Tensor): A tensor representing time(s).
            u (float or torch.Tensor, optional): The threshold level. If not provided, the instance's u is used.

        Returns:
            torch.Tensor: The value of gamma(t).
        """
        if u is None:
            u = self.u
        # Create sqrt(2) as a double-precision tensor.
        sqrt2 = torch.sqrt(torch.tensor(2.0, dtype=torch.float64))
        return (sqrt2 * self.p(t) / (self.r(t) + self.r0)) * u

    def _delta(self, t):
        """
        Compute the auxiliary quantity delta(t).

        The formula is given by:
            delta(t) = 1 / (r(t) + r0)
        where r0 is the value of the correlation function at t = 0.

        Parameters:
            t (torch.Tensor): A tensor representing time(s).

        Returns:
            torch.Tensor: The value of delta(t).
        """
        return 1.0 / (self.r(t) + self.r0)

    def compute_all_quantities(self, t, u=None):
        """
        Compute the set of auxiliary quantities alpha, beta, gamma, and delta at time t.

        Parameters:
            t (torch.Tensor): A tensor representing time(s).
            u (float or torch.Tensor, optional): The threshold level for gamma. If not provided, the instance's u is used.

        Returns:
            tuple: A tuple containing (alpha, beta, gamma, delta) evaluated at t.
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

    def upcrossing_mean_rate(self, u=None):
        """
        Compute the mean rate (per unit time) of the upcrossings counting process.

        The rate is given by:
            (1 / (2*pi)) * sqrt(q0 / r0) * exp(-u**2 / (2*r0))

        Parameters:
            u (float or torch.Tensor, optional): The threshold level for upcrossings.
                If not provided, the instance's u is used.

        Returns:
            torch.Tensor: The mean rate of upcrossings per unit time.
        """
        if u is None:
            u = self.u
        return (1 / (2 * torch.pi)) * torch.sqrt(self.q0 / self.r0) * torch.exp(-u**2 / (2 * self.r0))
    
    def downcrossing_mean_rate(self, u=None):
        """
        Compute the mean rate (per unit time) of the downcrossings counting process.

        This is defined to be equal to the upcrossing mean rate.

        Parameters:
            u (float or torch.Tensor, optional): The threshold level for downcrossings.
                If not provided, the instance's u is used.

        Returns:
            torch.Tensor: The mean rate of downcrossings per unit time.
        """
        return self.upcrossing_mean_rate(u=u)
    
    def crossing_mean_rate(self, u=None):
        """
        Compute the mean rate (per unit time) of the crossings counting process.

        The crossings mean rate is defined as twice the upcrossing mean rate.

        Parameters:
            u (float or torch.Tensor, optional): The threshold level for crossings.
                If not provided, the instance's u is used.

        Returns:
            torch.Tensor: The mean rate of crossings per unit time.
        """
        return 2 * self.upcrossing_mean_rate(u=u)
    
    def upcrossing_mean(self, T, u=None):
        """
        Compute the expected number (mean) of upcrossings over a time interval T.

        Parameters:
            T (float): The length of the time interval.
            u (float or torch.Tensor, optional): The threshold level for upcrossings.
                If not provided, the instance's u is used.

        Returns:
            torch.Tensor: The expected number of upcrossings in time T.
        """
        return self.upcrossing_mean_rate(u=u) * T
    
    def downcrossing_mean(self, T, u=None):
        """
        Compute the expected number (mean) of downcrossings over a time interval T.

        Parameters:
            T (float): The length of the time interval.
            u (float or torch.Tensor, optional): The threshold level for downcrossings.
                If not provided, the instance's u is used.

        Returns:
            torch.Tensor: The expected number of downcrossings in time T.
        """
        return self.downcrossing_mean_rate(u=u) * T
    
    def crossing_mean(self, T, u=None):
        """
        Compute the expected number (mean) of crossings over a time interval T.

        Parameters:
            T (float): The length of the time interval.
            u (float or torch.Tensor, optional): The threshold level for crossings.
                If not provided, the instance's u is used.

        Returns:
            torch.Tensor: The expected number of crossings in time T.
        """
        return self.crossing_mean_rate(u=u) * T

    def upcrossing_integrand(self, t, u=None):
        """
        Compute the integrand for the variance of the upcrossings counting process.

        This function evaluates the integral formula derived for the variance of upcrossings.
        The formula involves the auxiliary quantities alpha, beta, gamma, and delta.

        Parameters:
            t (torch.Tensor): A tensor representing time(s) at which to evaluate the integrand.
            u (float or torch.Tensor, optional): The threshold level.
                If not provided, the instance's u is used.

        Returns:
            torch.Tensor: The value of the integrand for the upcrossings variance.
        """
        if u is None:
            u = self.u
        
        r = self.r(t)
        r0 = self.r0
        q0 = self.q0

        alpha, beta, gamma, delta = self.compute_all_quantities(t, u)

        expr1 = alpha**2 * gamma**2 / (alpha + beta)

        final_integral = (
            torch.exp(-delta * u**2)
            / (4 * torch.pi**2 * torch.sqrt(r0**2 - r**2))
            * (
                (torch.exp(-alpha * gamma**2) / (2 * torch.sqrt(alpha * beta)))
                * (
                    1 + np.sqrt(torch.pi) * gamma * torch.sqrt(alpha + beta) * torch.exp(expr1) * torch.special.erf(torch.sqrt(expr1))
                )
                + torch.pi * ((alpha - beta - 2 * alpha * beta * gamma**2) / (alpha * beta))
                * scipy.special.owens_t(gamma * torch.sqrt(2 * alpha * beta / (alpha + beta)), torch.sqrt(alpha / beta))
            )
        ) - (1 / (4 * torch.pi**2)) * (q0 / r0) * torch.exp(-(u**2) / r0)

        return final_integral
    
    def downcrossing_integrand(self, t, u=None):
        """
        Compute the integrand for the variance of the downcrossings counting process.

        This method is intended to be the same as the upcrossing integrand.

        Parameters:
            t (torch.Tensor): A tensor representing time(s) at which to evaluate the integrand.
            u (float or torch.Tensor, optional): The threshold level.
                If not provided, the instance's u is used.

        Returns:
            torch.Tensor: The value of the integrand for the downcrossings variance.
        """
        return self.I_upcrossing(t, u=u)
    
    def crossing_integrand(self, t, u=None):
        """
        Compute the integrand for the variance of the crossings counting process.

        This function evaluates the integral formula derived for the variance of crossings.
        The formula uses the auxiliary quantities alpha, beta, gamma, and delta.

        Parameters:
            t (torch.Tensor): A tensor representing time(s) at which to evaluate the integrand.
            u (float or torch.Tensor, optional): The threshold level.
                If not provided, the instance's u is used.

        Returns:
            torch.Tensor: The value of the integrand for the crossings variance.
        """
        if u is None:
            u = self.u
        
        r = self.r(t)
        r0 = self.r0
        q0 = self.q0

        alpha, beta, gamma, delta = self.compute_all_quantities(t, u)

        expr1 = alpha**2 * gamma**2 / (alpha + beta)

        final_integral = (
            torch.exp(-delta * u**2)
            / (4 * torch.pi**2 * torch.sqrt(r0**2 - r**2))
            * (
                (2 * torch.exp(-alpha * gamma**2) / torch.sqrt(alpha * beta))
                * (
                    1 + np.sqrt(torch.pi) * gamma * torch.sqrt(alpha + beta) * torch.exp(expr1) * torch.special.erf(torch.sqrt(expr1))
                )
                + 4 * torch.pi * ((alpha - beta - 2 * alpha * beta * gamma**2) / (alpha * beta))
                * (scipy.special.owens_t(gamma * torch.sqrt(2 * alpha * beta / (alpha + beta)), torch.sqrt(alpha / beta)) - 1 / 8)
            )
        ) - (1 / (torch.pi**2)) * (q0 / r0) * torch.exp(-(u**2) / r0)

        return final_integral

    def upcrossing_variance(self, T, u=None, epsilon_left=1e-5, num_points=1000):
        """
        Compute the variance of the upcrossings counting process over a time interval T.

        The computation involves mapping the integration domain into [0, 1] and evaluating
        the integral using a trapezoidal rule.

        Parameters:
            T (float): The length of the time interval.
            u (float or torch.Tensor, optional): The threshold level.
                If not provided, the instance's u is used.
            epsilon_left (float, optional): A small value to avoid the singularity at 0.
            num_points (int, optional): The number of points to use in the numerical integration.

        Returns:
            torch.Tensor: The variance of the upcrossings counting process over time T.
        """
        if u is None:
            u = self.u

        t_prime = torch.linspace(epsilon_left, T / (1 + T), num_points)
        x = t_prime / (1 - t_prime)
        new_integrand_vals = (1 - x / T) * self.upcrossing_integrand(x, u=u) / (1 - t_prime)**2
        left_limit = - (1 / (4 * torch.pi**2)) * (self.q0 / self.r0) * torch.exp(-(u**2) / self.r0)
        integral = epsilon_left * left_limit + torch.trapz(new_integrand_vals[~torch.isnan(new_integrand_vals)], t_prime[~torch.isnan(new_integrand_vals)])

        return T * (self.upcrossing_mean_rate(u=u) + 2 * integral)

    def upcrossing_variance_CLT_per_unit_time(self, u=None, epsilon_left=1e-4, epsilon_right=1e-5, num_points=1000):
        """
        Compute the variance per unit time of the upcrossings counting process based on the CLT formula.

        The integration is performed by mapping the time domain into [0, 1] and using numerical integration.

        Parameters:
            u (float or torch.Tensor, optional): The threshold level.
                If not provided, the instance's u is used.
            epsilon_left (float, optional): A small value to avoid singularity at the left endpoint.
            epsilon_right (float, optional): A small value to avoid singularity at the right endpoint.
            num_points (int, optional): The number of points for the numerical integration.

        Returns:
            torch.Tensor: The variance per unit time of the upcrossings counting process.
        """
        if u is None:
            u = self.u

        t_prime = torch.linspace(epsilon_left, 1 - epsilon_right, num_points)
        x = t_prime / (1 - t_prime)
        new_integrand_vals = self.upcrossing_integrand(x, u=u) / (1 - t_prime)**2
        left_limit = - (1 / (4 * torch.pi**2)) * (self.q0 / self.r0) * torch.exp(-(u**2) / self.r0)
        integral = epsilon_left * left_limit + torch.trapz(new_integrand_vals[~torch.isnan(new_integrand_vals)], t_prime[~torch.isnan(new_integrand_vals)])

        return self.upcrossing_mean_rate(u=u) + 2 * integral
    
    def upcrossing_variance_CLT(self, T, u=None, epsilon_left=1e-4, epsilon_right=1e-5, num_points=1000):
        """
        Compute the variance of the upcrossings counting process over time T based on the CLT formula.

        This method multiplies the per unit time variance by T.

        Parameters:
            T (float): The length of the time interval.
            u (float or torch.Tensor, optional): The threshold level.
                If not provided, the instance's u is used.
            epsilon_left (float, optional): A small value to avoid singularity at the left endpoint.
            epsilon_right (float, optional): A small value to avoid singularity at the right endpoint.
            num_points (int, optional): The number of points for the numerical integration.

        Returns:
            torch.Tensor: The variance of the upcrossings counting process over time T.
        """
        return T * (self.upcrossing_variance_CLT_per_unit_time(u=u, epsilon_left=epsilon_left, epsilon_right=epsilon_right, num_points=num_points))
    
    def downcrossing_variance(self, T, u=None, epsilon_left=1e-5, num_points=1000):
        """
        Compute the variance of the downcrossings counting process over a time interval T.

        This method is defined to be the same as the upcrossing variance.

        Parameters:
            T (float): The length of the time interval.
            u (float or torch.Tensor, optional): The threshold level.
                If not provided, the instance's u is used.
            epsilon_left (float, optional): A small value to avoid singularity at 0.
            num_points (int, optional): The number of points for the numerical integration.

        Returns:
            torch.Tensor: The variance of the downcrossings counting process over time T.
        """
        return self.upcrossing_variance(T, u=u, epsilon_left=epsilon_left, num_points=num_points)
    
    def downcrossing_variance_CLT_per_unit_time(self, u=None, epsilon_left=1e-4, epsilon_right=1e-5, num_points=1000):
        """
        Compute the variance per unit time of the downcrossings counting process based on the CLT formula.

        This is defined to be the same as the upcrossing CLT variance per unit time.

        Parameters:
            u (float or torch.Tensor, optional): The threshold level.
                If not provided, the instance's u is used.
            epsilon_left (float, optional): A small value to avoid singularity at the left endpoint.
            epsilon_right (float, optional): A small value to avoid singularity at the right endpoint.
            num_points (int, optional): The number of points for the numerical integration.

        Returns:
            torch.Tensor: The variance per unit time of the downcrossings counting process.
        """
        return self.upcrossing_variance_CLT_per_unit_time(u=u, epsilon_left=epsilon_left, epsilon_right=epsilon_right, num_points=num_points)
    
    def downcrossing_variance_CLT(self, T, u=None, epsilon_left=1e-4, epsilon_right=1e-5, num_points=1000):
        """
        Compute the variance of the downcrossings counting process over time T based on the CLT formula.

        This method multiplies the per unit time variance by T and is defined to be the same as the upcrossing variant.

        Parameters:
            T (float): The length of the time interval.
            u (float or torch.Tensor, optional): The threshold level.
                If not provided, the instance's u is used.
            epsilon_left (float, optional): A small value to avoid singularity at the left endpoint.
            epsilon_right (float, optional): A small value to avoid singularity at the right endpoint.
            num_points (int, optional): The number of points for the numerical integration.

        Returns:
            torch.Tensor: The variance of the downcrossings counting process over time T.
        """
        return self.upcrossing_variance_CLT(T, u=u, epsilon_left=epsilon_left, epsilon_right=epsilon_right, num_points=num_points)
    
    def crossing_variance(self, T, u=None, epsilon_left=1e-5, num_points=1000):
        """
        Compute the variance of the crossings counting process over a time interval T.

        The integration domain is mapped into [0, 1] and evaluated using numerical integration.

        Parameters:
            T (float): The length of the time interval.
            u (float or torch.Tensor, optional): The threshold level.
                If not provided, the instance's u is used.
            epsilon_left (float, optional): A small value to avoid singularity at 0.
            num_points (int, optional): The number of points for the numerical integration.

        Returns:
            torch.Tensor: The variance of the crossings counting process over time T.
        """
        if u is None:
            u = self.u

        t_prime = torch.linspace(epsilon_left, T / (1 + T), num_points)
        x = t_prime / (1 - t_prime)
        new_integrand_vals = (1 - x / T) * self.crossing_integrand(x, u=u) / (1 - t_prime)**2
        left_limit = - (1 / torch.pi**2) * (self.q0 / self.r0) * torch.exp(-(u**2) / self.r0)
        integral = epsilon_left * left_limit + torch.trapz(new_integrand_vals[~torch.isnan(new_integrand_vals)], t_prime[~torch.isnan(new_integrand_vals)])

        return T * (self.crossing_mean_rate(u=u) + 2 * integral)
    
    def crossing_variance_CLT_per_unit_time(self, u=None, epsilon_left=1e-4, epsilon_right=1e-5, num_points=1000):
        """
        Compute the variance per unit time of the crossings counting process based on the CLT formula.

        The integration is performed by mapping the time domain into [0, 1] and using numerical integration.

        Parameters:
            u (float or torch.Tensor, optional): The threshold level.
                If not provided, the instance's u is used.
            epsilon_left (float, optional): A small value to avoid singularity at the left endpoint.
            epsilon_right (float, optional): A small value to avoid singularity at the right endpoint.
            num_points (int, optional): The number of points for the numerical integration.

        Returns:
            torch.Tensor: The variance per unit time of the crossings counting process.
        """
        if u is None:
            u = self.u

        t_prime = torch.linspace(epsilon_left, 1 - epsilon_right, num_points)
        x = t_prime / (1 - t_prime)
        new_integrand_vals = self.crossing_integrand(x, u=u) / (1 - t_prime)**2
        left_limit = - (1 / torch.pi**2) * (self.q0 / self.r0) * torch.exp(-(u**2) / self.r0)
        integral = epsilon_left * left_limit + torch.trapz(new_integrand_vals[~torch.isnan(new_integrand_vals)], t_prime[~torch.isnan(new_integrand_vals)])

        return self.crossing_mean_rate(u=u) + 2 * integral
    
    def crossing_variance_CLT(self, T, u=None, epsilon_left=1e-4, epsilon_right=1e-5, num_points=1000):
        """
        Compute the variance of the crossings counting process over time T based on the CLT formula.

        This method multiplies the per unit time variance by T.

        Parameters:
            T (float): The length of the time interval.
            u (float or torch.Tensor, optional): The threshold level.
                If not provided, the instance's u is used.
            epsilon_left (float, optional): A small value to avoid singularity at the left endpoint.
            epsilon_right (float, optional): A small value to avoid singularity at the right endpoint.
            num_points (int, optional): The number of points for the numerical integration.

        Returns:
            torch.Tensor: The variance of the crossings counting process over time T.
        """
        return T * (self.crossing_variance_CLT_per_unit_time(u=u, epsilon_left=epsilon_left, epsilon_right=epsilon_right, num_points=num_points))
    
    def upcrossing_fano_factor_CLT(self, u=None, epsilon_left=1e-5, epsilon_right=1e-5, num_points=1000):
        """
        Compute the Fano factor (variance/mean) for the upcrossings counting process using the CLT formula.

        Parameters:
            u (float or torch.Tensor, optional): The threshold level.
                If not provided, the instance's u is used.
            epsilon_left (float, optional): A small value to avoid singularity at the left endpoint.
            epsilon_right (float, optional): A small value to avoid singularity at the right endpoint.
            num_points (int, optional): The number of points for the numerical integration.

        Returns:
            torch.Tensor: The Fano factor for the upcrossings counting process.
        """
        return self.upcrossing_variance_CLT_per_unit_time(u=u, epsilon_left=epsilon_left, epsilon_right=epsilon_right, num_points=num_points) / self.upcrossing_mean_rate(u=u)
    
    def downcrossing_fano_factor_CLT(self, u=None, epsilon_left=1e-5, epsilon_right=1e-5, num_points=1000):
        """
        Compute the Fano factor (variance/mean) for the downcrossings counting process using the CLT formula.

        Parameters:
            u (float or torch.Tensor, optional): The threshold level.
                If not provided, the instance's u is used.
            epsilon_left (float, optional): A small value to avoid singularity at the left endpoint.
            epsilon_right (float, optional): A small value to avoid singularity at the right endpoint.
            num_points (int, optional): The number of points for the numerical integration.

        Returns:
            torch.Tensor: The Fano factor for the downcrossings counting process.
        """
        return self.downcrossing_variance_CLT_per_unit_time(u=u, epsilon_left=epsilon_left, epsilon_right=epsilon_right, num_points=num_points) / self.downcrossing_mean_rate(u=u)
    
    def crossing_fano_factor_CLT(self, u=None, epsilon_left=1e-5, epsilon_right=1e-5, num_points=1000):
        """
        Compute the Fano factor (variance/mean) for the crossings counting process using the CLT formula.

        Parameters:
            u (float or torch.Tensor, optional): The threshold level.
                If not provided, the instance's u is used.
            epsilon_left (float, optional): A small value to avoid singularity at the left endpoint.
            epsilon_right (float, optional): A small value to avoid singularity at the right endpoint.
            num_points (int, optional): The number of points for the numerical integration.

        Returns:
            torch.Tensor: The Fano factor for the crossings counting process.
        """
        return self.crossing_variance_CLT_per_unit_time(u=u, epsilon_left=epsilon_left, epsilon_right=epsilon_right, num_points=num_points) / self.crossing_mean_rate(u=u)
    
    def upcrossing_fano_factor(self, T, u=None, epsilon_left=1e-5, num_points=1000):
        """
        Compute the Fano factor (variance/mean) for the upcrossings counting process over time T.

        Parameters:
            T (float): The length of the time interval.
            u (float or torch.Tensor, optional): The threshold level.
                If not provided, the instance's u is used.
            epsilon_left (float, optional): A small value to avoid singularity at 0.
            num_points (int, optional): The number of points for the numerical integration.

        Returns:
            torch.Tensor: The Fano factor for the upcrossings counting process over time T.
        """
        return self.upcrossing_variance(T, u=u, epsilon_left=epsilon_left, num_points=num_points) / self.upcrossing_mean(T, u=u)
    
    def downcrossing_fano_factor(self, T, u=None, epsilon_left=1e-5, num_points=1000):
        """
        Compute the Fano factor (variance/mean) for the downcrossings counting process over time T.

        Parameters:
            T (float): The length of the time interval.
            u (float or torch.Tensor, optional): The threshold level.
                If not provided, the instance's u is used.
            epsilon_left (float, optional): A small value to avoid singularity at 0.
            num_points (int, optional): The number of points for the numerical integration.

        Returns:
            torch.Tensor: The Fano factor for the downcrossings counting process over time T.
        """
        return self.downcrossing_variance(T, u=u, epsilon_left=epsilon_left, num_points=num_points) / self.downcrossing_mean(T, u=u)
    
    def crossing_fano_factor(self, T, u=None, epsilon_left=1e-5, num_points=1000):
        """
        Compute the Fano factor (variance/mean) for the crossings counting process over time T.

        Parameters:
            T (float): The length of the time interval.
            u (float or torch.Tensor, optional): The threshold level.
                If not provided, the instance's u is used.
            epsilon_left (float, optional): A small value to avoid singularity at 0.
            num_points (int, optional): The number of points for the numerical integration.

        Returns:
            torch.Tensor: The Fano factor for the crossings counting process over time T.
        """
        return self.crossing_variance(T, u=u, epsilon_left=epsilon_left, num_points=num_points) / self.crossing_mean(T, u=u)

    def upcrossing_integrand_mean_level(self, t):
        """
        Compute the integrand for the variance of the upcrossings counting process at the mean level (u = 0).

        This integral formula is specifically derived for mean level crossings (u = 0).

        Parameters:
            t (torch.Tensor): A tensor representing time(s) at which to evaluate the integrand.

        Returns:
            torch.Tensor: The value of the integrand for the upcrossings variance at u = 0.
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
    
    def downcrossing_integrand_mean_level(self, t):
        """
        Compute the integrand for the variance of the downcrossings counting process at the mean level (u = 0).

        This method is defined to be the same as the upcrossing integrand for mean level crossings.

        Parameters:
            t (torch.Tensor): A tensor representing time(s) at which to evaluate the integrand.

        Returns:
            torch.Tensor: The value of the integrand for the downcrossings variance at u = 0.
        """
        return self.upcrossing_integrand_mean_level(t=t)
    
    def crossing_integrand_mean_level(self, t):
        """
        Compute the integrand for the variance of the crossings counting process at the mean level (u = 0).

        This integral formula is specifically derived for the variance of crossings when u = 0.

        Parameters:
            t (torch.Tensor): A tensor representing time(s) at which to evaluate the integrand.

        Returns:
            torch.Tensor: The value of the integrand for the crossings variance at u = 0.
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
