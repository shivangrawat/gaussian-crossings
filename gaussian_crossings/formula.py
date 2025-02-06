import torch

class GaussianUpcrossings:
    def __init__(self, r_func, *args, **kwargs):
        """
        Parameters:
          r_func: A function that takes a time tensor t (and possibly additional parameters)
                  and returns the correlation function r(t).
          *args, **kwargs: Optional extra parameters for r_func (if needed).
        """
        self.r_func = r_func
        # You may want to store extra parameters here if r_func requires them.
        self.args = args
        self.kwargs = kwargs

    def r(self, t):
        """
        Evaluate the correlation function r at time t.
        """
        return self.r_func(t, *self.args, **self.kwargs)

    def _compute_derivatives(self, t):
        """
        Given a scalar time t (a torch.tensor with requires_grad=True),
        compute:
          r_val = r(t)
          p_val = r'(t)
          q_val = -r''(t)
        using PyTorch's automatic differentiation.
        """
        # Ensure t is a differentiable tensor.
        t = t.clone().detach().requires_grad_(True)
        
        # Compute r(t)
        r_val = self.r(t)
        
        # Compute the first derivative: p(t) = r'(t)
        p_val = torch.autograd.grad(r_val, t, create_graph=True)[0]
        
        # Compute the second derivative: r''(t), then define q(t) = - r''(t)
        r_second = torch.autograd.grad(p_val, t, create_graph=True)[0]
        q_val = -r_second
        
        return r_val, p_val, q_val

    def derivatives_at(self, t):
        """
        Compute and return r(t), p(t) and q(t) at time t.
        """
        return self._compute_derivatives(t)

    def derivatives_at_zero(self):
        """
        Compute and return r(0), p(0) and q(0), denoted as r0, p0, and q0.
        """
        t0 = torch.tensor(0.0, dtype=torch.float, requires_grad=True)
        return self._compute_derivatives(t0)

    def alpha(self, t):
        """
        Compute alpha(t) = - (r(t) + r0) / (2*(p(t)**2 + (q(t) - q0)*(r(t) + r0)) )
        where r0, q0 are evaluated at t = 0.
        """
        r_val, p_val, q_val = self._compute_derivatives(t)
        r0, p0, q0 = self.derivatives_at_zero()
        return - (r_val + r0) / (2 * (p_val**2 + (q_val - q0) * (r_val + r0)))

    def beta(self, t):
        """
        Compute beta(t) = - (r0 - r(t)) / (2*(p(t)**2 + (q(t) + q0)*(r(t) - r0)) )
        where r0, q0 are evaluated at t = 0.
        """
        r_val, p_val, q_val = self._compute_derivatives(t)
        r0, p0, q0 = self.derivatives_at_zero()
        return - (r0 - r_val) / (2 * (p_val**2 + (q_val + q0) * (r_val - r0)))

    def gamma(self, t, u):
        """
        Compute gamma(t) = (sqrt(2) * p(t) / (r(t) + r0)) * u,
        where u is a given constant (or tensor) and r0 is evaluated at t = 0.
        """
        r_val, p_val, q_val = self._compute_derivatives(t)
        r0, p0, q0 = self.derivatives_at_zero()
        sqrt2 = torch.sqrt(torch.tensor(2.0))
        return (sqrt2 * p_val / (r_val + r0)) * u

    def delta(self, t):
        """
        Compute delta(t) = 1 / (r(t) + r0)
        where r0 is evaluated at t = 0.
        """
        r_val, p_val, q_val = self._compute_derivatives(t)
        r0, p0, q0 = self.derivatives_at_zero()
        return 1.0 / (r_val + r0)


# -------------------------
# Example usage:
# -------------------------
if __name__ == "__main__":
    # Define a sample correlation function:
    # For example, a squared exponential: r(t) = sigma^2 * exp( - (t/tau)^2 )
    def r_squared_exp(t, tau, sigma):
        return sigma**2 * torch.exp(- (t / tau)**2)

    # Define parameters (as tensors if you wish them to be differentiable)
    tau   = torch.tensor(1.0, dtype=torch.float)
    sigma = torch.tensor(2.0, dtype=torch.float)

    # Instantiate the GaussianUpcrossings class with the r_func and its parameters.
    gu = GaussianUpcrossings(r_squared_exp, tau, sigma)

    # Choose a time value to evaluate at
    t_val = torch.tensor(0.1, dtype=torch.float, requires_grad=True)

    # Compute the correlation function and its derivatives at t_val and at 0.
    r_val, p_val, q_val = gu.derivatives_at(t_val)
    r0, p0, q0 = gu.derivatives_at_zero()

    # Compute alpha, beta, gamma, and delta (with u given for gamma)
    u = torch.tensor(1.0)
    alpha_val = gu.alpha(t_val)
    beta_val  = gu.beta(t_val)
    gamma_val = gu.gamma(t_val, u)
    delta_val = gu.delta(t_val)

    # Print the computed quantities:
    print("At t =", t_val.item())
    print("r(t) =", r_val.item())
    print("p(t) =", p_val.item())
    print("q(t) =", q_val.item())
    print("r(0) =", r0.item())
    print("p(0) =", p0.item())
    print("q(0) =", q0.item())
    print("alpha =", alpha_val.item())
    print("beta  =", beta_val.item())
    print("gamma =", gamma_val.item())
    print("delta =", delta_val.item())
