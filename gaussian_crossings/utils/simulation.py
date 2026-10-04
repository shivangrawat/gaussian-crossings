"""Stationary Gaussian paths on the grid ``arange(round(T / dt)) * dt``.

FFT sampling uses a positive semidefinite circulant embedding. Cholesky
sampling constructs the finite Toeplitz covariance directly. Both use
PyTorch's random generator; call ``torch.manual_seed`` for repeatable paths.
"""

import math
from typing import Any, Callable

import torch


def _time_grid(T: float, dt: float) -> torch.Tensor:
    if not math.isfinite(T) or not math.isfinite(dt) or T <= 0 or dt <= 0:
        raise ValueError("T and dt must be finite and positive")
    n = round(T / dt)
    if n < 2:
        raise ValueError("T / dt must give at least two sample points")
    return torch.arange(n, dtype=torch.float64) * dt


def _covariance_row(r_func, t, args, kwargs):
    """Evaluate a vectorized covariance, falling back to scalar callbacks."""
    try:
        values = torch.as_tensor(r_func(t, *args, **kwargs), dtype=t.dtype)
        if values.shape != t.shape:
            raise ValueError("Covariance callback returned the wrong shape")
    except (TypeError, ValueError, RuntimeError):
        values = torch.stack(
            [torch.as_tensor(r_func(float(lag), *args, **kwargs), dtype=t.dtype) for lag in t]
        )
    if values.shape != t.shape or not torch.isfinite(values).all():
        raise ValueError("Covariance must return one finite value per time lag")
    if values[0] <= 0:
        raise ValueError("Covariance at zero must be positive")
    return values


def simulate_gaussian_process_fft(
    r_func: Callable[..., torch.Tensor],
    T: float,
    dt: float,
    *args: Any,
    **kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a stationary Gaussian path with O(n log n) circulant embedding.

    The endpoint T is excluded when T/dt is an integer. Covariance parameters
    are passed through to ``r_func``. The covariance must be even and the
    reflected embedding must be positive semidefinite. Negative eigenvalues
    within floating-point tolerance are clipped to zero; larger violations
    raise ValueError instead of silently changing the target covariance.
    Increase the observation window or use Cholesky when embedding fails.
    """
    t = _time_grid(T, dt)
    row = _covariance_row(r_func, t, args, kwargs)
    embedded = torch.cat((row, row.flip(0)[1:-1]))
    n = embedded.numel()
    eigenvalues = torch.fft.fft(embedded).real
    tolerance = 1e-12 * float(eigenvalues.abs().max())
    if eigenvalues.min() < -tolerance:
        raise ValueError(
            "Circulant embedding is not positive semidefinite; increase T "
            "or use simulate_gaussian_process_cholesky."
        )
    eigenvalues = eigenvalues.clamp_min(0)
    real = torch.randn(n, dtype=row.dtype, device=row.device)
    imag = torch.randn(n, dtype=row.dtype, device=row.device)
    spectrum = torch.sqrt(eigenvalues / n) * (real + 1j * imag)
    path = torch.fft.fft(spectrum).real[: t.numel()]
    return t.to(row.device), path


def simulate_gaussian_process_cholesky(
    r: Callable[..., torch.Tensor],
    T: float,
    dt: float,
    *args: Any,
    jitter: float = 1e-10,
    max_attempts: int = 5,
    **kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample from the finite Toeplitz covariance in O(n^3) time/O(n^2) space.

    The first factorization uses the exact covariance. Failed attempts retry
    with diagonal jitter, beginning at ``jitter`` and increasing tenfold.
    Jitter changes the covariance by the stated diagonal increment. The
    time grid and callback conventions match the FFT sampler: extra positional
    and keyword arguments are passed to the covariance callback. ``jitter``
    and ``max_attempts`` are keyword-only so they cannot capture covariance
    parameters.
    """
    t = _time_grid(T, dt)
    if not math.isfinite(jitter) or jitter < 0:
        raise ValueError("jitter must be finite and nonnegative")
    if not isinstance(max_attempts, int) or max_attempts < 1:
        raise ValueError("max_attempts must be a positive integer")
    row = _covariance_row(r, t, args, kwargs)
    indices = torch.arange(t.numel(), device=row.device)
    covariance = row[(indices[:, None] - indices[None, :]).abs()]
    eye = torch.eye(t.numel(), dtype=row.dtype, device=row.device)
    for attempt in range(max_attempts):
        increment = 0 if attempt == 0 else jitter * 10 ** (attempt - 1)
        factor, info = torch.linalg.cholesky_ex(covariance + increment * eye)
        if info == 0:
            break
    else:
        raise RuntimeError("Cholesky decomposition failed even after adding jitter")
    noise = torch.randn(t.numel(), dtype=row.dtype, device=row.device)
    return t.to(row.device), factor @ noise
