"""Autocorrelation functions for stationary Gaussian stochastic processes.

This module provides autocorrelation/autocovariance functions for several
commonly used stationary Gaussian processes, including the stochastic damped
harmonic oscillator, Ornstein-Uhlenbeck processes, and standard kernel
functions (rational quadratic, squared exponential, Matern).  These
correlation functions serve as inputs to the exact variance and Fano factor
formulae in the ``formula`` subpackage and are also used for simulation-based
validation of the analytical results.

Every function takes the time lag ``t`` first, followed by its named
parameters. Unknown keyword arguments raise ``TypeError`` instead of being
ignored, so a misspelled parameter cannot silently select a different model.
"""

from typing import Any, Optional, Union

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
    *,
    dtype: torch.dtype = torch.float64,
    device: Optional[Union[str, torch.device]] = None,
) -> torch.Tensor:
    """Compute the autocorrelation function of a stochastic damped harmonic oscillator.

    Computes the stationary autocorrelation function of the position of a
    damped harmonic oscillator driven by thermal white noise, the main example
    of the accompanying paper.  The noise amplitude satisfies the
    fluctuation-dissipation theorem.  The correlation function has three
    qualitatively different forms depending on the damping regime:
    oscillatory decay (underdamped), critically damped, or bi-exponential
    decay (overdamped).  The overdamped branch is evaluated in a form that
    stays accurate as ``zeta`` approaches one from above.

    Args:
        t: Time lag(s) at which to evaluate the correlation function.
        temp: Temperature of the system (controls noise strength, with k_B=1).
        omega0: Natural angular frequency of the oscillator.
        zeta: Damping ratio (zeta < 1: underdamped, zeta = 1: critically
            damped, zeta > 1: overdamped).
        dtype: Torch dtype used for non-tensor inputs (default: float64).
        device: Torch device used for non-tensor inputs (default: None).

    Returns:
        Autocorrelation values r(t) at the specified time lag(s).

    Note:
        The variance of the process is r(0) = temp / omega0^2 in all
        damping regimes, as guaranteed by the equipartition theorem.
    """
    _positive("omega0", omega0, allow_zero=False)
    _positive("zeta", zeta, allow_zero=False)
    _positive("temp", temp, allow_zero=True)

    def to_tensor(x: Any) -> torch.Tensor:
        return x if isinstance(x, torch.Tensor) else torch.tensor(x, dtype=dtype, device=device)

    t = to_tensor(t)
    temp = to_tensor(temp)
    omega0 = to_tensor(omega0)
    zeta = to_tensor(zeta)

    t_abs = torch.abs(t)
    prefactor = temp * zeta / omega0**2

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

    # Over-damped case (zeta > 1).
    else:
        w = torch.sqrt((zeta - 1.0) * (zeta + 1.0))
        if w < 0.1:
            # Near critical damping the two-exponential form cancels. With
            # slow = zeta - w and s = omega0 |t|, use the equivalent expression
            # r = (temp/omega0^2) exp(-slow s) [1 + slow (1 - exp(-2 w s)) / (2 w)].
            slow = 1.0 / (zeta + w)
            scaled = omega0 * t_abs
            envelope = torch.exp(-slow * scaled)
            spread = -torch.expm1(-2.0 * w * scaled) / (2.0 * w)
            result = (temp / omega0**2) * envelope * (1.0 + slow * spread)
        else:
            # Historical arithmetic, kept so archived compatibility values reproduce exactly.
            sqrt_disc = torch.sqrt(discriminant)
            term1 = (1 / sqrt_disc + 1 / zeta) * torch.exp(-omega0 * (zeta - sqrt_disc) * t_abs)
            term2 = (1 / zeta - 1 / sqrt_disc) * torch.exp(-omega0 * (zeta + sqrt_disc) * t_abs)
            exp_part = 0.5 * (term1 + term2)
            result = prefactor * exp_part

    return result


def r_filtered_OU(
    t: Union[torch.Tensor, np.ndarray, float], sigma: float, tau: float, kappa: float
) -> torch.Tensor:
    """Compute the autocovariance function of a filtered Ornstein-Uhlenbeck process.

    The filtered OU process is obtained by passing an OU process through
    a first-order low-pass filter with time constant tau_f = kappa * tau.
    This bi-exponential correlation structure can be mapped to the
    overdamped SDHO correlation function.

    Args:
        t: Time lag(s) at which to evaluate the autocovariance.
        sigma: Amplitude of the input OU process. The filtered process has
            variance r(0) = sigma**2 / (1 + kappa).
        tau: Exponential decay time constant (tau_e).
        kappa: Ratio of filter time constant to tau (tau_f = kappa * tau).

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
    t: Union[torch.Tensor, np.ndarray, float], sigma: float, tau: float, kappa: float
) -> torch.Tensor:
    """Compute the autocovariance function of a mean-reverting process driven by OU noise.

    The process y(t) relaxes toward zero with time constant tau and is driven
    by Ornstein-Uhlenbeck noise with correlation time kappa * tau, as in the
    accompanying paper.  The resulting bi-exponential correlation structure
    supports both sub- and super-Poissonian crossing statistics.

    Args:
        t: Time lag(s) at which to evaluate the autocovariance.
        sigma: Standard deviation of the driving OU noise. The process
            variance is r(0) = sigma**2 * kappa / (1 + kappa).
        tau: Exponential decay time constant (tau_e).
        kappa: Ratio of the filter time constant to tau (tau_f = kappa * tau).

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
    return (sigma**2 * kappa / (1 - kappa**2)) * (
        torch.exp(-torch.abs(t) / tau) - kappa * torch.exp(-torch.abs(t) / (kappa * tau))
    )


def r_OU(t: Union[torch.Tensor, np.ndarray, float], sigma: float, tau: float) -> torch.Tensor:
    """Compute the autocovariance function of an Ornstein-Uhlenbeck process.

    The OU process has an exponentially decaying autocorrelation function:
        r(t) = sigma^2 * exp(-|t| / tau)

    Its sample paths are not differentiable, so the crossing formulae do not
    apply to it directly; it is provided for simulation and as a building block.

    Args:
        t: Time lag(s) at which to evaluate the autocovariance.
        sigma: Standard deviation of the process (r(0) = sigma**2).
        tau: Exponential decay time constant (correlation time).

    Returns:
        Autocovariance values at the specified time lag(s).
    """
    t = _lag_tensor(t)
    _positive("sigma", sigma, allow_zero=True)
    _positive("tau", tau, allow_zero=False)
    return sigma**2 * torch.exp(-torch.abs(t) / tau)


def r_rational_quadratic(
    t: Union[torch.Tensor, np.ndarray, float], sigma: float, tau: float, alpha: float
) -> torch.Tensor:
    """Compute the rational quadratic autocorrelation function.

    The rational quadratic kernel can be seen as an infinite mixture of
    squared exponential kernels with different length scales.  As alpha -> inf,
    it converges to the squared exponential kernel.  Its covariance decays as
    |t|^(-2 alpha); smaller alpha means heavier-tailed, longer-range
    correlations.  The long-time crossing variance is finite at every
    threshold for alpha > 1/2.  At the mean level u = 0 the leading tail
    term, proportional to u^2 r(t), vanishes and alpha > 1/4 suffices.

    Args:
        t: Time lag(s) at which to evaluate the autocorrelation.
        sigma: Standard deviation of the process (r(0) = sigma**2).
        tau: Length-scale parameter.
        alpha: Shape parameter controlling the mixture of scales.
            Larger alpha means the kernel is closer to squared exponential.

    Returns:
        Autocorrelation values at the specified time lag(s).
    """
    t = _lag_tensor(t)
    _positive("sigma", sigma, allow_zero=True)
    _positive("tau", tau, allow_zero=False)
    _positive("alpha", alpha, allow_zero=False)
    return sigma**2 * (1 + (t / tau) ** 2 / (2 * alpha)) ** (-alpha)


def r_squared_exp(
    t: Union[torch.Tensor, np.ndarray, float], sigma: float, tau: float
) -> torch.Tensor:
    """Compute the squared exponential (Gaussian/RBF) autocovariance function.

    The squared exponential kernel produces infinitely differentiable
    sample paths:
        r(t) = sigma^2 * exp(-t^2 / (2 * tau^2))

    Args:
        t: Time lag(s) at which to evaluate the autocovariance.
        sigma: Standard deviation of the process (r(0) = sigma**2).
        tau: Length-scale parameter (characteristic time scale).

    Returns:
        Autocovariance values at the specified time lag(s).
    """
    t = _lag_tensor(t)
    _positive("sigma", sigma, allow_zero=True)
    _positive("tau", tau, allow_zero=False)
    return sigma**2 * torch.exp(-0.5 * (t / tau) ** 2)


def r_matern(
    t: Union[torch.Tensor, np.ndarray, float], sigma: float, tau: float, nu: float
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
        sigma: Standard deviation of the process (r(0) = sigma**2).
        tau: Length-scale parameter.
        nu: Smoothness parameter (nu > 0). Controls differentiability
            of sample paths.

    Returns:
        Autocorrelation values at the specified time lag(s).

    Note:
        Uses scipy.special.kv (modified Bessel function of the second kind)
        for general nu. Automatic differentiation, which the crossing formulae
        need, is supported only for nu = 0.5, 1.5, and 2.5.
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
