import torch
from torch.func import grad
import numpy as np
import scipy.special
from scipy.special import kv, gamma


def r_filtered_OU(t, sigma, tau_e, tau_f):
    """
    Filtered Ornstein-Uhlenbeck autocovariance function.
    
    Parameters:
        t : tensor or array
            Time differences.
        sigma : float
            Standard deviation (amplitude) of the process.
        tau_e : float
            Exponential decay time constant.
        tau_f : float
            Filter time constant.
    
    Returns:
        Tensor: The autocovariance computed at times t.
    """
    kappa = tau_f / tau_e
    return (sigma ** 2 / (1 - kappa ** 2)) * (
        torch.exp(- torch.abs(t) / tau_e) - kappa * torch.exp(- torch.abs(t) / tau_f)
    )

def r_mean_reverting_OU_noise(t, sigma, tau_e, tau_f):
    """
    Autocovariance function of the mean-reverting OU noise.
    
    Parameters:
        t : tensor or array
            Time differences.
        sigma : float
            Standard deviation (amplitude) of the process.
        tau_e : float
            Exponential decay time constant.
        tau_f : float
            Filter time constant.
    
    Returns:
        Tensor: The autocovariance computed at times t.
    """
    kappa = tau_f / tau_e
    return (sigma ** 2 * kappa / (1 - kappa ** 2)) * (
        torch.exp(- torch.abs(t) / tau_e) - kappa * torch.exp(- torch.abs(t) / tau_f)
    )

def r_OU(t, sigma, tau):
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

def r_rational_quadratic(t, tau, sigma, alpha):
    """
    Rational quadratic autocorrelation function.

    Parameters:
        t (array-like): Time differences.
        tau (float): Length-scale parameter.
        sigma (float): Standard deviation (amplitude) of the process.
        alpha (float): Shape parameter controlling the relative weighting of different scales.

    Returns:
        array-like: Autocorrelation values at times t.
    """
    return sigma ** 2 * (1 + (t / tau) ** 2 / (2 * alpha)) ** (-alpha)

def r_squared_exp(t, sigma, tau):
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

def r_matern(t, tau, sigma, nu):
    """
    Matern autocorrelation function.

    Parameters:
        t (array-like): Time differences.
        tau (float): Length-scale parameter.
        sigma (float): Standard deviation (amplitude) of the process.
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
