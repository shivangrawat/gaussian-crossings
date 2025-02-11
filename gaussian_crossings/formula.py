import torch
from torch.func import grad
import numpy as np
import scipy.special

torch.set_default_dtype(torch.float64)


class GaussianUpCrossings:
    def __init__(self, r_func, u=0, *args, **kwargs):
        """
        Parameters:
          r_func: A function that takes a time tensor t (and possibly additional parameters)
                  and returns the correlation function r(t).
          *args, **kwargs: Optional extra parameters for r_func (if needed).
        """
        self.r_func = r_func
        self.u = u  # Level u for upcrossings.
        self.args = args
        self.kwargs = kwargs

        # Evaluate r, its first derivative (p), and the second derivative (q) at t = 0.
        self.r0 = self.r(torch.tensor([0.0], dtype=torch.float64))
        self.p0 = self.p(torch.tensor([0.0], dtype=torch.float64))
        self.q0 = self.q(torch.tensor([0.0], dtype=torch.float64))

    def r(self, t):
        """
        Evaluate the correlation function r at time t.
        """
        return self.r_func(t, *self.args, **self.kwargs)

    def p(self, t):
        """
        Evaluate the first derivative of r(t) at time t.
        """
        def sum_func(t):
            return torch.sum(self.r(t))
        return grad(sum_func)(t)

    def q(self, t):
        """
        Return the negative derivative of p(t) with respect to t.
        """
        def sum_func(t):
            return torch.sum(self.p(t))
        return - grad(sum_func)(t)

    def _alpha(self, t):
        """
        Compute alpha(t) = - (r(t) + r0) / (2*(p(t)**2 + (q(t) - q0)*(r(t) + r0)))
        where r0, q0 are evaluated at t = 0.
        """
        return torch.abs(
            -(self.r(t) + self.r0)
            / (2 * (self.p(t) ** 2 + (self.q(t) - self.q0) * (self.r(t) + self.r0)))
        )

    def _beta(self, t):
        """
        Compute beta(t) = - (r0 - r(t)) / (2*(p(t)**2 + (q(t) + q0)*(r(t) - r0)))
        where r0, q0 are evaluated at t = 0.
        """
        return torch.abs(
            -(self.r0 - self.r(t))
            / (2 * (self.p(t) ** 2 + (self.q(t) + self.q0) * (self.r(t) - self.r0)))
        )

    def _gamma(self, t, u=None):
        """
        Compute gamma(t) = (sqrt(2) * p(t) / (r(t) + r0)) * u,
        where u is a given constant (or tensor) and r0 is evaluated at t = 0.
        """
        if u is None:
            u = self.u
        # Create sqrt(2) as a double-precision tensor.
        sqrt2 = torch.sqrt(torch.tensor(2.0, dtype=torch.float64))
        return (sqrt2 * self.p(t) / (self.r(t) + self.r0)) * u

    def _delta(self, t):
        """
        Compute delta(t) = 1 / (r(t) + r0)
        where r0 is evaluated at t = 0.
        """
        return 1.0 / (self.r(t) + self.r0)

    def compute_all_quantities(self, t, u=None):
        """
        Compute alpha, beta, gamma, and delta at time t with a given u.
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
        """
        if u is None:
            u = self.u
        return (1 / (2 * torch.pi)) * torch.sqrt(self.q0 / self.r0) * torch.exp(-u**2 / (2 * self.r0))
    
    def downcrossing_mean_rate(self, u=None):
        """
        Compute the mean rate (per unit time) of the downcrossings counting process.
        """
        return self.upcrossing_mean_rate(u=u)
    
    def crossing_mean_rate(self, u=None):
        """
        Compute the mean rate (per unit time) of the crossings counting process.
        """
        return 2 * self.upcrossing_mean_rate(u=u)
    
    def upcrossing_mean(self, T, u=None):
        return self.upcrossing_mean_rate(u=u) * T
    
    def downcrossing_mean(self, T, u=None):
        return self.downcrossing_mean_rate(u=u) * T
    
    def crossing_mean(self, T, u=None):
        return self.crossing_mean_rate(u=u) * T

    def upcrossing_integrand(self, t, u=None):
        """
        Compute the integral formula we derived for the variance of upcrossings counting process.
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
        Ame as the integral for upcrossings.
        """
        return self.I_upcrossing(t, u=u)
    
    def crossing_integrand(self, t, u=None):
        """
        Compute the integral formula we derived for the variance of crossings counting process.
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
        Compute the variance of the upcrossings counting process per unit time by mapping into [0, 1].
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
        Compute the variance of the upcrossings counting process per unit time by mapping into [0, 1].
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
        Compute the variance of the upcrossings counting process per unit time by mapping into [0, 1].
        """
        return T * (self.upcrossing_variance_CLT_per_unit_time(u=u, epsilon_left=epsilon_left, epsilon_right=epsilon_right, num_points=num_points)) 

    def downcrossing_variance(self, T, u=None, epsilon_left=1e-5, num_points=1000):
        return self.upcrossing_variance(T, u=u, epsilon_left=epsilon_left, num_points=num_points)
    
    def downcrossing_variance_CLT_per_unit_time(self, u=None, epsilon_left=1e-4, epsilon_right=1e-5, num_points=1000):
        return self.upcrossing_variance_CLT_per_unit_time(u=u, epsilon_left=epsilon_left, epsilon_right=epsilon_right, num_points=num_points)
    
    def downcrossing_variance_CLT(self, T, u=None, epsilon_left=1e-4, epsilon_right=1e-5, num_points=1000):
        return self.upcrossing_variance_CLT(T, u=u, epsilon_left=epsilon_left, epsilon_right=epsilon_right, num_points=num_points)
    
    def crossing_variance(self, T, u=None, epsilon_left=1e-5, num_points=1000):
        if u is None:
            u = self.u

        t_prime = torch.linspace(epsilon_left, T / (1 + T), num_points)
        x = t_prime / (1 - t_prime)
        new_integrand_vals = (1 - x / T) * self.crossing_integrand(x, u=u) / (1 - t_prime)**2
        left_limit = - (1 / torch.pi**2) * (self.q0 / self.r0) * torch.exp(-(u**2) / self.r0)
        integral = epsilon_left * left_limit + torch.trapz(new_integrand_vals[~torch.isnan(new_integrand_vals)], t_prime[~torch.isnan(new_integrand_vals)])

        return T * (self.crossing_mean_rate(u=u) + 2 * integral)
    
    def crossing_variance_CLT_per_unit_time(self, u=None, epsilon_left=1e-4, epsilon_right=1e-5, num_points=1000):
        if u is None:
            u = self.u

        t_prime = torch.linspace(epsilon_left, 1 - epsilon_right, num_points)
        x = t_prime / (1 - t_prime)
        new_integrand_vals = self.crossing_integrand(x, u=u) / (1 - t_prime)**2
        left_limit = - (1 / torch.pi**2) * (self.q0 / self.r0) * torch.exp(-(u**2) / self.r0)
        integral = epsilon_left * left_limit + torch.trapz(new_integrand_vals[~torch.isnan(new_integrand_vals)], t_prime[~torch.isnan(new_integrand_vals)])

        return self.crossing_mean_rate(u=u) + 2 * integral
    
    def crossing_variance_CLT(self, T, u=None, epsilon_left=1e-4, epsilon_right=1e-5, num_points=1000):
        return T * (self.crossing_variance_CLT_per_unit_time(u=u, epsilon_left=epsilon_left, epsilon_right=epsilon_right, num_points=num_points))
    
    def upcrossing_fano_factor_CLT(self, u=None, epsilon_left=1e-5, epsilon_right=1e-5, num_points=1000):
        return self.upcrossing_variance_CLT_per_unit_time(u=u, epsilon_left=epsilon_left, epsilon_right=epsilon_right, num_points=num_points) / self.upcrossing_mean_rate(u=u)
    
    def downcrossing_fano_factor_CLT(self, u=None, epsilon_left=1e-5, epsilon_right=1e-5, num_points=1000):
        return self.downcrossing_variance_CLT_per_unit_time(u=u, epsilon_left=epsilon_left, epsilon_right=epsilon_right, num_points=num_points) / self.downcrossing_mean_rate(u=u)
    
    def crossing_fano_factor_CLT(self, u=None, epsilon_left=1e-5, epsilon_right=1e-5, num_points=1000):
        return self.crossing_variance_CLT_per_unit_time(u=u, epsilon_left=epsilon_left, epsilon_right=epsilon_right, num_points=num_points) / self.crossing_mean_rate(u=u)
    
    def upcrossing_fano_factor(self, T, u=None, epsilon_left=1e-5, num_points=1000):
        return self.upcrossing_variance(T, u=u, epsilon_left=epsilon_left, num_points=num_points) / self.upcrossing_mean(T, u=u)
    
    def downcrossing_fano_factor(self, T, u=None, epsilon_left=1e-5, num_points=1000):
        return self.downcrossing_variance(T, u=u, epsilon_left=epsilon_left, num_points=num_points) / self.downcrossing_mean(T, u=u)
    
    def crossing_fano_factor(self, T, u=None, epsilon_left=1e-5, num_points=1000):
        return self.crossing_variance(T, u=u, epsilon_left=epsilon_left, num_points=num_points) / self.crossing_mean(T, u=u)

    def upcrossing_integrand_mean_level(self, t):
        """
        Compute the integral formula we derived for the variance of upcrossings counting process. This one is specifically for the mean level crossings, i.e., u=0.
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
        Compute the integral formula we derived for the variance of downcrossings counting process. This one is specifically for the mean level crossings, i.e., u=0.
        """
        return self.upcrossing_integrand_mean_level(t=t)
    
    def crossing_integrand_mean_level(self, t):
        """
        Compute the integral formula we derived for the variance of crossings counting process. This one is specifically for the mean level crossings, i.e., u=0.
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
    
    
