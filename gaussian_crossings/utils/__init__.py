"""Utility functions for simulation, crossing detection, and special functions.

This subpackage provides tools for simulating Gaussian process sample paths
(via FFT circulant embedding or Cholesky factorization), detecting threshold
crossings in time series, Euler-Maruyama SDE integration, and a differentiable
implementation of Owen's T function used in the exact variance expressions.
"""

from .simulation import simulate_gaussian_process_fft, simulate_gaussian_process_cholesky
from .utils import *
from .owensT import OwensT