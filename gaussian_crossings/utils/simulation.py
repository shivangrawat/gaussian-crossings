"""Simulation methods for stationary Gaussian stochastic processes.

This module provides functions for generating sample paths from stationary
Gaussian processes using either FFT-based circulant embedding (O(N log N))
or Cholesky factorization (O(N^3)).  These simulations are used for
numerical validation of the exact variance and Fano factor formulae by
comparing empirical crossing counts against the analytical predictions.
"""

from typing import Any, Callable, Tuple

import torch
from scipy.linalg import toeplitz


def simulate_gaussian_process_fft(
    r_func: Callable[..., torch.Tensor],
    T: float,
    dt: float,
    *args: Any,
    **kwargs: Any
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Simulate a stationary Gaussian process using FFT-based circulant embedding.

    Generates a sample path from a zero-mean stationary Gaussian process
    with the specified correlation function. This method is efficient for
    long time series as it uses FFT with O(N log N) complexity.

    Based on the algorithm described at:
    https://urbain.vaes.uk/static/teaching/lectures/build/lectures-w4.pdf

    Args:
        r_func: Correlation function r(tau) of the process. Must accept
            (tau, *args, **kwargs) and return a torch.Tensor.
        T: Total simulation time.
        dt: Time step for discretization.
        *args: Additional positional arguments passed to r_func.
        **kwargs: Additional keyword arguments passed to r_func.

    Returns:
        A tuple (t, x) where:
            - t: Tensor of shape (n,) containing time points.
            - x: Tensor of shape (n,) containing simulated process values.

    Example:
        >>> from gaussian_crossings.process import r_OU
        >>> t, x = simulate_gaussian_process_fft(r_OU, T=10.0, dt=0.01, sigma=1.0, tau=1.0)
    """
    # Basic discretization
    n = int(round(T / dt))
    N = 2 * n - 2
    # Time points
    t = torch.arange(n, dtype=torch.float64) * dt

    # Build discrete correlation array: r(0..N-1)
    try:
        r_vals = r_func(t, *args, **kwargs)
        if not isinstance(r_vals, torch.Tensor):
            r_vals = torch.tensor(r_vals, dtype=torch.float64)
        if r_vals.ndim != 1 or r_vals.shape[0] != n:
            raise ValueError("Correlation function r did not return the expected shape.")
    except Exception:
        r_vals = torch.tensor(
            [r_func(i * dt, *args, **kwargs) for i in range(N)],
            dtype=torch.float64
        )

    # r_full is [r(0), r(dt), ..., r((n-1)*dt), r((n-2)*dt), ..., r(dt)]
    r_full = torch.cat((r_vals, torch.flip(r_vals, dims=[0])[1:-1]))

    # FFT to get "eigenvalues" of the corresponding circulant covariance
    lam = torch.fft.fft(r_full)

    # Create random complex amplitudes with correct magnitude
    Z_real = torch.randn(N, dtype=torch.float64)
    Z_imag = torch.randn(N, dtype=torch.float64)
    Z = Z_real + 1j * Z_imag

    # Multiply by sqrt of lam / N
    # (We do not check positivity of lam here, assuming r_full is valid.)
    Y = torch.sqrt(lam / N) * Z

    # Inverse transform to get the time-domain signal
    V = torch.fft.fft(Y)

    # Extract the real part for the first n points
    x = torch.real(V[:n])

    return t, x


def simulate_gaussian_process_cholesky(
    r: Callable[..., torch.Tensor],
    T: float,
    dt: float,
    jitter: float = 1e-10,
    max_attempts: int = 5,
    *args: Any,
    **kwargs: Any
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Simulate a Gaussian process using Cholesky factorization.

    Generates a sample path by constructing the full covariance matrix
    and using Cholesky decomposition. This method is exact but has O(N³)
    complexity and O(N²) memory usage, making it suitable only for
    moderate-length time series.

    Args:
        r: Correlation function with signature r(t, *args, **kwargs).
            Must be defined for t >= 0.
        T: Total simulation time.
        dt: Time discretization step.
        jitter: Small diagonal increment added if the matrix is nearly
            singular. Default is 1e-10.
        max_attempts: Maximum number of times to attempt adding jitter
            before raising an error. Default is 5.
        *args: Additional positional arguments passed to r.
        **kwargs: Additional keyword arguments passed to r.

    Returns:
        A tuple (t, x) where:
            - t: Tensor of time points from 0 to T.
            - x: Tensor of simulated process values at times in t.

    Raises:
        RuntimeError: If Cholesky decomposition fails even after adding jitter.

    Example:
        >>> from gaussian_crossings.process import r_OU
        >>> t, x = simulate_gaussian_process_cholesky(r_OU, T=5.0, dt=0.1, sigma=1.0, tau=1.0)
    """
    # Determine number of time points
    N = int(round(T / dt))
    t = torch.arange(N, dtype=torch.float64) * dt

    # Attempt vectorized evaluation of r
    try:
        first_row_vals = r(t, *args, **kwargs)
        if not isinstance(first_row_vals, torch.Tensor):
            first_row_vals = torch.tensor(first_row_vals, dtype=torch.float64)
        if first_row_vals.ndim != 1 or first_row_vals.shape[0] != N:
            raise ValueError("Correlation function r did not return the expected shape.")
    except Exception:
        first_row_vals = torch.tensor(
            [r(i * dt, *args, **kwargs) for i in range(N)],
            dtype=torch.float64
        )

    # Construct the Toeplitz covariance matrix
    C = torch.from_numpy(toeplitz(first_row_vals.numpy()))

    # Attempt Cholesky with jitter if necessary
    attempt = 0
    while attempt < max_attempts:
        try:
            L = torch.linalg.cholesky(C)
            break
        except RuntimeError:
            # Add jitter to the diagonal
            jitter_value = jitter * (10 ** attempt)
            C = C + jitter_value * torch.eye(N, dtype=C.dtype)
            attempt += 1
    else:
        raise RuntimeError("Cholesky decomposition failed even after adding jitter.")

    # Generate the sample path: x = L @ z, where z ~ N(0, I)
    z = torch.randn(N, dtype=torch.float64)
    x = L @ z

    return t, x
