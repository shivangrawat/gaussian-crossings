import torch
from torch.func import grad
import numpy as np
import scipy.special
from scipy.special import kv, gamma


def r_damped_harmonic_oscillator_noise(t, temp, omega0, zeta, **kwargs):
    """
    Autocorrelation function of a damped harmonic oscillator with additive noise.

    Parameters:
        t : tensor or array
            Time differences.
        temp : float
            Tempreture of the system. (controls the strength of the noise)
        omega0 : float
            Natural frequency of the oscillator.
        zeta : float
            Damping ratio of the oscillator.

    Returns:
        Tensor: The autocorrelation computed at times t.
    """

    dtype = kwargs.get("dtype", torch.float64)
    device = kwargs.get("device", None)

    def to_tensor(x):
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
        oscillatory_part = (torch.sin(omega_d * t_abs) / torch.sqrt(1.0 - zeta**2) +
                            torch.cos(omega_d * t_abs) / zeta)
        result = prefactor * decay * oscillatory_part

    # Critically-damped case (zeta == 1)
    elif discriminant == 0:
        decay = torch.exp(-zeta * omega0 * t_abs)
        oscillatory_part = (t_abs * omega0 + 1.0)
        result = prefactor * decay * oscillatory_part

    # Over-damped case (zeta > 1)
    else:
        sqrt_disc = torch.sqrt(discriminant)
        term1 = (1/sqrt_disc + 1/zeta) * torch.exp(-omega0 * (zeta - sqrt_disc) * t_abs)
        term2 = (1/zeta - 1/sqrt_disc) * torch.exp(-omega0 * (zeta + sqrt_disc) * t_abs)
        exp_part = 0.5 * (term1 + term2)
        result = prefactor * exp_part

    return result

def r_filtered_OU(t, sigma, tau, kappa, **kwargs):
    """
    Filtered Ornstein-Uhlenbeck autocovariance function.

    Parameters:
        t : tensor or array
            Time differences.
        sigma : float
            Standard deviation (amplitude) of the process.
        tau : float
            Exponential decay time constant (tau_e).
        kappa : float
            Ratio of the filter time constant (tau_f) to tau (i.e., tau_f = kappa * tau).

    Returns:
        Tensor: The autocovariance computed at times t.
    """

    return (sigma ** 2 / (1 - kappa ** 2)) * (
        torch.exp(-torch.abs(t) / tau) - kappa * torch.exp(-torch.abs(t) / (kappa * tau))
    )

def r_OU_noise(t, sigma, tau, kappa, **kwargs):
    """
    Autocovariance function of a process with OU noise.

    Parameters:
        t : tensor or array
            Time differences.
        sigma : float
            Standard deviation (amplitude) of the process.
        tau : float
            Exponential decay time constant (tau_e).
        kappa : float
            Ratio of the filter time constant (tau_f) to tau (i.e., tau_f = kappa * tau).

    Returns:
        Tensor: The autocovariance computed at times t.
    """
    return (sigma ** 2 * kappa / (1 - kappa ** 2)) * (
        torch.exp(-torch.abs(t) / tau) - kappa * torch.exp(-torch.abs(t) / (kappa * tau))
    )

def r_OU(t, sigma, tau, **kwargs):
    """
    Ornstein-Uhlenbeck autocovariance function.
    
    Parameters:
        t : tensor or array
            Time differences.
        sigma : float
            Standard deviation (amplitude) of the process.
        tau : float
            Exponential decay time constant.
    
    Returns:
        Tensor: The autocovariance computed at times t.
    """
    return sigma ** 2 * torch.exp(- torch.abs(t) / tau)

def r_rational_quadratic(t, sigma, tau, alpha, **kwargs):
    """
    Rational quadratic autocorrelation function.

    Parameters:
        t (array-like): Time differences.
        sigma (float): Standard deviation (amplitude) of the process.
        tau (float): Length-scale parameter.
        alpha (float): Shape parameter controlling the relative weighting of different scales.

    Returns:
        array-like: Autocorrelation values at times t.
    """
    return sigma ** 2 * (1 + (t / tau) ** 2 / (2 * alpha)) ** (-alpha)

def r_squared_exp(t, sigma, tau, **kwargs):
    """
    Squared exponential autocovariance function.
    
    Parameters:
        t : tensor or array
            Time differences.
        sigma : float
            Standard deviation (amplitude) of the process.
        tau : float
            Exponential decay time constant.
    
    Returns:
        Tensor: The autocovariance computed at times t.
    """
    return sigma ** 2 * (torch.exp(- (t / (2 * tau)) ** 2))

def r_matern(t, sigma, tau, nu, **kwargs):
    """
    Matern autocorrelation function.

    Parameters:
        t (array-like): Time differences.
        sigma (float): Standard deviation (amplitude) of the process.
        tau (float): Length-scale parameter.
        nu (float): Smoothness parameter (nu > 0).

    Returns:
        array-like: Autocorrelation values at times t.
    """
    t_abs = torch.abs(t)
    factor = np.sqrt(2 * nu) * t_abs / tau
    corr = torch.zeros_like(t_abs)
    # Handle the t=0 case to avoid division by zero (the limit is sigma^2).
    nonzero = t_abs > 0
    corr[nonzero] = sigma**2 * (2**(1 - nu) / gamma(nu)) * (factor[nonzero]**nu) * kv(nu, factor[nonzero])
    corr[~nonzero] = sigma**2
    return corr
