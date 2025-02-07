import torch
from torch.func import grad

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

        return -grad(sum_func)(t)

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

    def I(self, t, u=None):
        """
        Compute the upcrossing intensity I(t) = alpha(t) + beta(t) * exp(-gamma(t)).
        """
        if u is None:
            u = self.u

        alpha, beta, gamma, delta = self.compute_all_quantities(t, u)
        return alpha + beta * torch.exp(-gamma)
