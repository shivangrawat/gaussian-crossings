import torch
from torch.func import grad
import numpy as np
import scipy.special
from scipy.special import kv, gamma


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
