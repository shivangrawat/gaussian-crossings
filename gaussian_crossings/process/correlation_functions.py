"""Autocorrelation functions for stationary Gaussian stochastic processes.

This module provides autocorrelation/autocovariance functions for several
commonly used stationary Gaussian processes, including the stochastic damped
harmonic oscillator, Ornstein-Uhlenbeck processes, and standard kernel
functions (rational quadratic, squared exponential, Matern).  These
correlation functions serve as inputs to the exact variance and Fano factor
formulae in the ``formula`` subpackage and are also used for simulation-based
validation of the analytical results.
"""

from typing import Any, Union

import numpy as np
import torch
from scipy.special import gamma, kv


def _lag_tensor(t):
    """Accept Python/NumPy inputs without detaching existing torch graphs."""
    return t if isinstance(t, torch.Tensor) else torch.as_tensor(t, dtype=torch.float64)


def _positive(name, value, *, allow_zero=False):
    tensor = torch.as_tensor(value, dtype=torch.float64)
    valid = tensor >= 0 if allow_zero else tensor > 0
    if tensor.numel() != 1 or not bool(torch.isfinite(tensor) & valid):
        sign = "nonnegative" if allow_zero else "positive"
        raise ValueError(f"{name} must be a finite {sign} scalar")


def r_damped_harmonic_oscillator_noise(
    t: Union[torch.Tensor, np.ndarray, float],
    temp: float,
    omega0: float,
    zeta: float,
    **kwargs: Any,
) -> torch.Tensor:
    """Compute the autocorrelation function of a stochastic damped harmonic oscillator.

    Computes the stationary autocorrelation function for a damped harmonic
    oscillator driven by thermal white noise, as described in Section IV.A
    of the paper.  The noise amplitude satisfies the fluctuation-dissipation
    theorem.  The correlation function has three qualitatively different
    forms depending on the damping regime: oscillatory decay (underdamped),
    critically damped, or bi-exponential decay (overdamped).

    Args:
        t: Time lag(s) at which to evaluate the correlation function.
        temp: Temperature of the system (controls noise strength, with k_B=1).
        omega0: Natural angular frequency of the oscillator.
        zeta: Damping ratio (zeta < 1: underdamped, zeta = 1: critically
            damped, zeta > 1: overdamped).
        **kwargs: Optional keyword arguments:
            - dtype: Torch dtype for computation (default: torch.float64).
            - device: Torch device for computation (default: None).

    Returns:
        Autocorrelation values r(t) at the specified time lag(s).

    Note:
        The variance of the process is r(0) = temp / omega0^2 in all
        damping regimes, as guaranteed by the equipartition theorem.
    """
    _positive("omega0", omega0, allow_zero=False)
    _positive("zeta", zeta, allow_zero=False)
    _positive("temp", temp, allow_zero=True)
    dtype = kwargs.get("dtype", torch.float64)
    device = kwargs.get("device", None)

    def to_tensor(x: Any) -> torch.Tensor:
        return x if isinstance(x, torch.Tensor) else torch.tensor(x, dtype=dtype, device=device)

    t = to_tensor(t)
    temp = to_tensor(temp)
    omega0 = to_tensor(omega0)
    zeta = to_tensor(zeta)

    t_abs = torch.abs(t)
    prefactor = temp * zeta / omega0**2
    decay = torch.exp(-zeta * omega0 * t_abs)

    discriminant = zeta**2 - 1.0

    # Under-damped case (zeta < 1)
    if discriminant < 0:
        decay = torch.exp(-zeta * omega0 * t_abs)
        omega_d = omega0 * torch.sqrt(1.0 - zeta**2)
        oscillatory_part = (
            torch.sin(omega_d * t_abs) / torch.sqrt(1.0 - zeta**2)
            + torch.cos(omega_d * t_abs) / zeta
        )
        result = prefactor * decay * oscillatory_part

    # Critically-damped case (zeta == 1)
    elif discriminant == 0:
        decay = torch.exp(-zeta * omega0 * t_abs)
        oscillatory_part = t_abs * omega0 + 1.0
        result = prefactor * decay * oscillatory_part

    # Over-damped case (zeta > 1)
    else:
        sqrt_disc = torch.sqrt(discriminant)
        term1 = (1 / sqrt_disc + 1 / zeta) * torch.exp(-omega0 * (zeta - sqrt_disc) * t_abs)
        term2 = (1 / zeta - 1 / sqrt_disc) * torch.exp(-omega0 * (zeta + sqrt_disc) * t_abs)
        exp_part = 0.5 * (term1 + term2)
        result = prefactor * exp_part

    return result


def r_filtered_OU(
    t: Union[torch.Tensor, np.ndarray, float], sigma: float, tau: float, kappa: float, **kwargs: Any
) -> torch.Tensor:
    """Compute the autocovariance function of a filtered Ornstein-Uhlenbeck process.

    The filtered OU process is obtained by passing an OU process through
    a first-order low-pass filter with time constant tau_f = kappa * tau.
    This bi-exponential correlation structure can be mapped to the
    overdamped SDHO correlation function.

    Args:
        t: Time lag(s) at which to evaluate the autocovariance.
        sigma: Noise amplitude parameter. The process standard deviation follows from r(0).
        tau: Exponential decay time constant (tau_e).
        kappa: Ratio of filter time constant to tau (tau_f = kappa * tau).
        **kwargs: Additional keyword arguments (unused, for API consistency).

    Returns:
        Autocovariance values at the specified time lag(s).

    Raises:
        ValueError: If kappa equals 1 (degenerate case).
    """
    t = _lag_tensor(t)
    _positive("sigma", sigma, allow_zero=True)
    _positive("tau", tau, allow_zero=False)
    _positive("kappa", kappa, allow_zero=False)
    if bool(torch.as_tensor(kappa) == 1):
        raise ValueError(
            "kappa=1 requires its limiting covariance; use an explicit smooth limit callback"
        )
    return (sigma**2 / (1 - kappa**2)) * (
        torch.exp(-torch.abs(t) / tau) - kappa * torch.exp(-torch.abs(t) / (kappa * tau))
    )


def r_OU_noise(
    t: Union[torch.Tensor, np.ndarray, float], sigma: float, tau: float, kappa: float, **kwargs: Any
) -> torch.Tensor:
    """Compute the autocovariance function of a mean-reverting process driven by OU noise.

    This represents the correlation function of the process y(t) from
    Section IV.B of the paper, where a mean-reverting process is driven
    by Ornstein-Uhlenbeck noise.  The resulting bi-exponential correlation
    structure supports both sub- and super-Poissonian crossing statistics.

    Args:
        t: Time lag(s) at which to evaluate the autocovariance.
        sigma: Noise amplitude parameter. The process standard deviation follows from r(0).
        tau: Exponential decay time constant (tau_e).
        kappa: Ratio of the filter time constant to tau (tau_f = kappa * tau).
        **kwargs: Additional keyword arguments (unused, for API consistency).

    Returns:
        Autocovariance values at the specified time lag(s).
    """
    t = _lag_tensor(t)
    _positive("sigma", sigma, allow_zero=True)
    _positive("tau", tau, allow_zero=False)
    _positive("kappa", kappa, allow_zero=False)
    if bool(torch.as_tensor(kappa) == 1):
        raise ValueError(
            "kappa=1 requires its limiting covariance; use an explicit smooth limit callback"
        )
    return (sigma**2 * kappa / (1 - kappa**2)) * (
        torch.exp(-torch.abs(t) / tau) - kappa * torch.exp(-torch.abs(t) / (kappa * tau))
    )


def r_OU(
    t: Union[torch.Tensor, np.ndarray, float], sigma: float, tau: float, **kwargs: Any
) -> torch.Tensor:
    """Compute the autocovariance function of an Ornstein-Uhlenbeck process.

    The OU process has an exponentially decaying autocorrelation function:
        r(t) = sigma^2 * exp(-|t| / tau)

    Args:
        t: Time lag(s) at which to evaluate the autocovariance.
        sigma: Noise amplitude parameter. The process standard deviation follows from r(0).
        tau: Exponential decay time constant (correlation time).
        **kwargs: Additional keyword arguments (unused, for API consistency).

    Returns:
        Autocovariance values at the specified time lag(s).
    """
    t = _lag_tensor(t)
    _positive("sigma", sigma, allow_zero=True)
    _positive("tau", tau, allow_zero=False)
    return sigma**2 * torch.exp(-torch.abs(t) / tau)


def r_rational_quadratic(
    t: Union[torch.Tensor, np.ndarray, float], sigma: float, tau: float, alpha: float, **kwargs: Any
) -> torch.Tensor:
    """Compute the rational quadratic autocorrelation function.

    The rational quadratic kernel can be seen as an infinite mixture of
    squared exponential kernels with different length scales.  As alpha -> inf,
    it converges to the squared exponential kernel.  This kernel is analyzed
    in Section IV.C of the paper, where the shape parameter alpha is shown
    to control crossing statistics together with the threshold. All five
    curves in the revised Figure 6 have a maximum above one, including
    the squared-exponential limit.

    Args:
        t: Time lag(s) at which to evaluate the autocorrelation.
        sigma: Noise amplitude parameter. The process standard deviation follows from r(0).
        tau: Length-scale parameter.
        alpha: Shape parameter controlling the mixture of scales.
            Larger alpha means the kernel is closer to squared exponential.
        **kwargs: Additional keyword arguments (unused, for API consistency).

    Returns:
        Autocorrelation values at the specified time lag(s).
    """
    t = _lag_tensor(t)
    _positive("sigma", sigma, allow_zero=True)
    _positive("tau", tau, allow_zero=False)
    _positive("alpha", alpha, allow_zero=False)
    return sigma**2 * (1 + (t / tau) ** 2 / (2 * alpha)) ** (-alpha)


def r_squared_exp(
    t: Union[torch.Tensor, np.ndarray, float], sigma: float, tau: float, **kwargs: Any
) -> torch.Tensor:
    """Compute the squared exponential (Gaussian/RBF) autocovariance function.

    The squared exponential kernel produces infinitely differentiable
    sample paths:
        r(t) = sigma^2 * exp(-t^2 / (2 * tau^2))

    Args:
        t: Time lag(s) at which to evaluate the autocovariance.
        sigma: Noise amplitude parameter. The process standard deviation follows from r(0).
        tau: Length-scale parameter (characteristic time scale).
        **kwargs: Additional keyword arguments (unused, for API consistency).

    Returns:
        Autocovariance values at the specified time lag(s).
    """
    t = _lag_tensor(t)
    _positive("sigma", sigma, allow_zero=True)
    _positive("tau", tau, allow_zero=False)
    return sigma**2 * torch.exp(-0.5 * (t / tau) ** 2)


def r_matern(
    t: Union[torch.Tensor, np.ndarray, float], sigma: float, tau: float, nu: float, **kwargs: Any
) -> torch.Tensor:
    """Compute the Matérn autocorrelation function.

    The Matérn kernel is a generalization that allows control over the
    smoothness of sample paths through the parameter nu. Special cases:
        - nu = 0.5: Ornstein-Uhlenbeck (exponential decay)
        - nu = 1.5: Once differentiable
        - nu = 2.5: Twice differentiable
        - nu -> inf: Squared exponential (infinitely differentiable)

    Args:
        t: Time lag(s) at which to evaluate the autocorrelation.
        sigma: Noise amplitude parameter. The process standard deviation follows from r(0).
        tau: Length-scale parameter.
        nu: Smoothness parameter (nu > 0). Controls differentiability
            of sample paths.
        **kwargs: Additional keyword arguments (unused, for API consistency).

    Returns:
        Autocorrelation values at the specified time lag(s).

    Note:
        Uses scipy.special.kv (modified Bessel function of the second kind)
        for computation.
    """
    t = _lag_tensor(t)
    _positive("sigma", sigma, allow_zero=True)
    _positive("tau", tau, allow_zero=False)
    _positive("nu", nu, allow_zero=False)
    t_abs = torch.abs(t)
    factor = np.sqrt(2 * nu) * t_abs / tau
    if nu == 0.5:
        return sigma**2 * torch.exp(-factor)
    if nu == 1.5:
        return sigma**2 * (1 + factor) * torch.exp(-factor)
    if nu == 2.5:
        return sigma**2 * (1 + factor + factor**2 / 3) * torch.exp(-factor)
    if factor.requires_grad:
        raise ValueError("Autograd for Matern covariance is supported for nu=0.5, 1.5, 2.5 only")
    corr = torch.zeros_like(t_abs)
    # Handle the t=0 case to avoid division by zero (the limit is sigma^2).
    nonzero = t_abs > 0
    corr[nonzero] = (
        sigma**2
        * (2 ** (1 - nu) / gamma(nu))
        * (factor[nonzero] ** nu)
        * torch.as_tensor(
            kv(nu, factor[nonzero].detach().cpu().numpy()), dtype=t_abs.dtype, device=t_abs.device
        )
    )
    corr[~nonzero] = sigma**2
    return corr
