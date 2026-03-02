"""Stochastic process definitions and correlation functions.

This subpackage provides autocorrelation functions for several stationary
Gaussian processes (e.g., damped harmonic oscillator, Ornstein-Uhlenbeck,
rational quadratic) and SDE-based dynamical system models used for
numerical simulation and validation of the exact crossing formulae.
"""

from .correlation_functions import *
from .dynamical_equations import *