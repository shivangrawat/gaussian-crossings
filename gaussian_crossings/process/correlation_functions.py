"""Correlation functions for various Gaussian stochastic processes.

This module provides autocorrelation/autocovariance functions for several
commonly used stationary Gaussian processes, including damped harmonic
oscillators, Ornstein-Uhlenbeck processes, and various kernel functions.
"""

from typing import Any, Union

import torch
import numpy as np
from scipy.special import kv, gamma


def r_damped_harmonic_oscillator_noise(
    t: Union[torch.Tensor, np.ndarray, float],
    temp: float,
    omega0: float,
    zeta: float,
    **kwargs: Any
) -> torch.Tensor:
    """Compute the autocorrelation function of a damped harmonic oscillator.

    Computes the autocorrelation function for a stochastic damped harmonic
    oscillator driven by thermal noise. The correlation function depends on
    the damping regime (underdamped, critically damped, or overdamped).

    Args:
        t: Time lag(s) at which to evaluate the correlation function.
        temp: Temperature of the system (controls noise strength).
        omega0: Natural angular frequency of the oscillator.
        zeta: Damping ratio (zeta < 1: underdamped, zeta = 1: critically
            damped, zeta > 1: overdamped).
        **kwargs: Optional keyword arguments:
            - dtype: Torch dtype for computation (default: torch.float64).
            - device: Torch device for computation (default: None).

    Returns:
        Autocorrelation values r(t) at the specified time lag(s).

    Note:
        The variance of the process is r(0) = temp / omega0^2.
    """
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
    t: Union[torch.Tensor, np.ndarray, float],
    sigma: float,
    tau: float,
    kappa: float,
    **kwargs: Any
) -> torch.Tensor:
    """Compute the autocovariance function of a filtered Ornstein-Uhlenbeck process.

    The filtered OU process is obtained by passing an OU process through
    a first-order low-pass filter with time constant tau_f = kappa * tau.

    Args:
        t: Time lag(s) at which to evaluate the autocovariance.
        sigma: Standard deviation (amplitude) of the process.
        tau: Exponential decay time constant (tau_e).
        kappa: Ratio of filter time constant to tau (tau_f = kappa * tau).
        **kwargs: Additional keyword arguments (unused, for API consistency).

    Returns:
        Autocovariance values at the specified time lag(s).

    Raises:
        ValueError: If kappa equals 1 (degenerate case).
    """
    return (sigma**2 / (1 - kappa**2)) * (
        torch.exp(-torch.abs(t) / tau) - kappa * torch.exp(-torch.abs(t) / (kappa * tau))
    )


def r_OU_noise(
    t: Union[torch.Tensor, np.ndarray, float],
    sigma: float,
    tau: float,
    kappa: float,
    **kwargs: Any
) -> torch.Tensor:
    """Compute the autocovariance function of a process driven by OU noise.

    This represents the correlation function of a mean-reverting process
    where the driving noise is itself an Ornstein-Uhlenbeck process.

    Args:
        t: Time lag(s) at which to evaluate the autocovariance.
        sigma: Standard deviation (amplitude) of the process.
        tau: Exponential decay time constant (tau_e).
        kappa: Ratio of the filter time constant to tau (tau_f = kappa * tau).
        **kwargs: Additional keyword arguments (unused, for API consistency).

    Returns:
        Autocovariance values at the specified time lag(s).
    """
    return (sigma**2 * kappa / (1 - kappa**2)) * (
        torch.exp(-torch.abs(t) / tau) - kappa * torch.exp(-torch.abs(t) / (kappa * tau))
    )


def r_OU(
    t: Union[torch.Tensor, np.ndarray, float],
    sigma: float,
    tau: float,
    **kwargs: Any
) -> torch.Tensor:
    """Compute the autocovariance function of an Ornstein-Uhlenbeck process.

    The OU process has an exponentially decaying autocorrelation function:
        r(t) = sigma^2 * exp(-|t| / tau)

    Args:
        t: Time lag(s) at which to evaluate the autocovariance.
        sigma: Standard deviation (amplitude) of the process.
        tau: Exponential decay time constant (correlation time).
        **kwargs: Additional keyword arguments (unused, for API consistency).

    Returns:
        Autocovariance values at the specified time lag(s).
    """
    return sigma**2 * torch.exp(-torch.abs(t) / tau)


def r_rational_quadratic(
    t: Union[torch.Tensor, np.ndarray, float],
    sigma: float,
    tau: float,
    alpha: float,
    **kwargs: Any
) -> torch.Tensor:
    """Compute the rational quadratic autocorrelation function.

    The rational quadratic kernel can be seen as an infinite mixture of
    squared exponential kernels with different length scales. As alpha -> inf,
    it converges to the squared exponential kernel.

    Args:
        t: Time lag(s) at which to evaluate the autocorrelation.
        sigma: Standard deviation (amplitude) of the process.
        tau: Length-scale parameter.
        alpha: Shape parameter controlling the mixture of scales.
            Larger alpha means the kernel is closer to squared exponential.
        **kwargs: Additional keyword arguments (unused, for API consistency).

    Returns:
        Autocorrelation values at the specified time lag(s).
    """
    return sigma**2 * (1 + (t / tau)**2 / (2 * alpha)) ** (-alpha)


def r_squared_exp(
    t: Union[torch.Tensor, np.ndarray, float],
    sigma: float,
    tau: float,
    **kwargs: Any
) -> torch.Tensor:
    """Compute the squared exponential (Gaussian/RBF) autocovariance function.

    The squared exponential kernel produces infinitely differentiable
    sample paths:
        r(t) = sigma^2 * exp(-t^2 / (2 * tau^2))

    Args:
        t: Time lag(s) at which to evaluate the autocovariance.
        sigma: Standard deviation (amplitude) of the process.
        tau: Length-scale parameter (characteristic time scale).
        **kwargs: Additional keyword arguments (unused, for API consistency).

    Returns:
        Autocovariance values at the specified time lag(s).
    """
    return sigma**2 * torch.exp(-0.5 * (t / tau)**2)


def r_matern(
    t: Union[torch.Tensor, np.ndarray, float],
    sigma: float,
    tau: float,
    nu: float,
    **kwargs: Any
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
        sigma: Standard deviation (amplitude) of the process.
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
    t_abs = torch.abs(t)
    factor = np.sqrt(2 * nu) * t_abs / tau
    corr = torch.zeros_like(t_abs)
    # Handle the t=0 case to avoid division by zero (the limit is sigma^2).
    nonzero = t_abs > 0
    corr[nonzero] = (
        sigma**2
        * (2 ** (1 - nu) / gamma(nu))
        * (factor[nonzero] ** nu)
        * kv(nu, factor[nonzero])
    )
    corr[~nonzero] = sigma**2
    return corr
