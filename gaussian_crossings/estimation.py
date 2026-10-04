"""Estimate crossing-count statistics from sampled data.

Counts are formed in consecutive, non-overlapping windows of a uniformly
sampled series. When sampling resolves the crossings and the model matches
the data, their variance-to-mean ratio estimates the finite-window prediction
``GaussianCrossings.fano_factor(u, T=window, kind=kind)`` with a mean-centered
threshold. The long-time Fano factor is the limit of that prediction as the
window grows, and it can differ noticeably from the windowed value when
windows span only a few correlation times.
"""

import warnings
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import torch

__all__ = ["CoarseSamplingWarning", "FanoEstimate", "empirical_fano", "windowed_counts"]

ArrayLike = Union[np.ndarray, torch.Tensor, list, tuple]

_KINDS = {
    "up": "up",
    "upcrossing": "up",
    "upcrossings": "up",
    "down": "down",
    "downcrossing": "down",
    "downcrossings": "down",
    "total": "total",
    "all": "total",
    "crossing": "total",
    "crossings": "total",
}


class CoarseSamplingWarning(UserWarning):
    """Detected crossings change markedly when the sampling step is doubled."""


@dataclass(frozen=True)
class FanoEstimate:
    """Fano factor of windowed crossing counts with a bootstrap confidence interval.

    A scalar threshold produces float-valued statistics. An array of thresholds
    produces arrays with one entry per threshold, including a one-element array.

    Attributes:
        fano: Sample variance (``ddof=1``) divided by the mean count per window.
            NaN for thresholds with no crossings.
        ci_low: Lower end of the percentile bootstrap interval.
        ci_high: Upper end of the percentile bootstrap interval.
        stderr: Bootstrap standard error of ``fano``.
        mean: Mean number of crossings per window.
        variance: Sample variance (``ddof=1``) of the counts.
        n_windows: Number of windows used.
        window: Window length actually used, a whole number of sampling steps.
        confidence: Confidence level of the interval.
    """

    fano: Union[float, np.ndarray]
    ci_low: Union[float, np.ndarray]
    ci_high: Union[float, np.ndarray]
    stderr: Union[float, np.ndarray]
    mean: Union[float, np.ndarray]
    variance: Union[float, np.ndarray]
    n_windows: int
    window: float
    confidence: float


def _kind(kind: str) -> str:
    try:
        return _KINDS[str(kind).lower()]
    except KeyError:
        raise ValueError("kind must be 'up', 'down', or 'total'") from None


def _series(x: ArrayLike) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    values = np.asarray(x, dtype=float)
    if values.ndim == 1:
        values = values[None, :]
    if values.ndim != 2:
        raise ValueError("x must be one series (n_samples,) or a stack (n_series, n_samples)")
    if not np.all(np.isfinite(values)):
        raise ValueError("x must contain only finite values")
    return values


def _levels(u: ArrayLike):
    if isinstance(u, torch.Tensor):
        u = u.detach().cpu().numpy()
    levels = np.asarray(u, dtype=float)
    if levels.ndim > 1:
        raise ValueError("u must be a scalar or a one-dimensional array of thresholds")
    if not np.all(np.isfinite(levels)):
        raise ValueError("u must contain finite thresholds")
    return np.atleast_1d(levels), levels.ndim == 0


def _indicators(values: np.ndarray, level: float, kind: str) -> np.ndarray:
    """Crossing indicator for each interval between consecutive samples."""
    before, after = values[:, :-1], values[:, 1:]
    if kind == "up":
        return (before < level) & (after >= level)
    if kind == "down":
        return (before > level) & (after <= level)
    return ((before < level) & (after >= level)) | ((before > level) & (after <= level))


def _steps_per_window(window: float, dt: float) -> int:
    if not (np.isfinite(dt) and dt > 0):
        raise ValueError("dt must be finite and positive")
    if not (np.isfinite(window) and window > 0):
        raise ValueError("window must be finite and positive")
    steps = int(round(window / dt))
    if steps < 1:
        raise ValueError("window must be at least one sampling step long")
    return steps


def windowed_counts(
    x: ArrayLike, u: ArrayLike, window: float, dt: float = 1.0, kind: str = "up"
) -> np.ndarray:
    """Count crossings of one or more levels in consecutive, non-overlapping windows.

    A crossing is attributed to the interval between the two samples that
    bracket it, using the same rule as ``count_upcrossings`` (an upcrossing of
    ``u`` is ``x[i] < u <= x[i+1]``). With ``m = round(window / dt)`` steps per
    window, window ``k`` holds the intervals that start at samples ``k*m``
    through ``(k+1)*m - 1``, so it spans exactly ``m*dt`` and every crossing is
    counted in exactly one window. A final incomplete window is dropped.

    Args:
        x: Samples taken at a uniform step ``dt``: one series of shape
            ``(n_samples,)`` or independent series of shape ``(n_series, n_samples)``,
            whose windows are pooled.
        u: A threshold or a one-dimensional array of thresholds.
        window: Window length, in the same time units as ``dt``.
        dt: Sampling step.
        kind: ``"up"``, ``"down"``, or ``"total"`` crossings.

    Returns:
        Integer counts of shape ``(n_windows,)`` for a scalar threshold, or
        ``(n_windows, n_levels)`` for an array of thresholds.
    """
    values = _series(x)
    levels, scalar = _levels(u)
    direction = _kind(kind)
    steps = _steps_per_window(window, dt)
    per_series = (values.shape[1] - 1) // steps
    if per_series < 1:
        raise ValueError("each series must contain at least one complete window")
    span = per_series * steps
    counts = np.empty((values.shape[0] * per_series, levels.size), dtype=np.int64)
    for j, level in enumerate(levels):
        hits = _indicators(values, level, direction)[:, :span]
        counts[:, j] = hits.reshape(values.shape[0], per_series, steps).sum(axis=2).ravel()
    return counts[:, 0] if scalar else counts


def _fano(counts: np.ndarray) -> np.ndarray:
    mean = counts.mean(axis=-2)
    variance = counts.var(axis=-2, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(mean > 0, variance / mean, np.nan)


def _check_sampling(values, levels, direction, tolerance, stacklevel):
    # Compare the same observation span: every second sample, ending on the last even
    # index, against the full-rate samples up to that index. With an even number of
    # samples, the final full-rate interval has no coarse counterpart and is left out.
    span = 2 * ((values.shape[1] - 1) // 2)
    if span < 2:  # fewer than two coarse samples per series: nothing to compare
        return
    fine = values[:, : span + 1]
    coarse = fine[:, ::2]
    detected = np.array([_indicators(fine, level, direction).sum() for level in levels])
    halved = np.array([_indicators(coarse, level, direction).sum() for level in levels])
    with np.errstate(divide="ignore", invalid="ignore"):
        lost = np.where(detected >= 50, 1 - halved / detected, 0.0)
    worst = float(np.max(lost))
    if worst > tolerance:
        warnings.warn(
            f"Doubling the sampling step removes {100 * worst:.1f}% of the detected "
            "crossings, so crossings between samples are probably being missed and the "
            "counts are biased low. Sample more finely before counting crossings.",
            CoarseSamplingWarning,
            stacklevel=stacklevel,
        )


def empirical_fano(
    x: ArrayLike,
    u: ArrayLike,
    window: float,
    dt: float = 1.0,
    kind: str = "up",
    *,
    confidence: float = 0.95,
    n_bootstrap: int = 2000,
    seed: Optional[int] = None,
    check_sampling: bool = True,
    sampling_tolerance: float = 0.02,
) -> FanoEstimate:
    """Estimate the Fano factor of crossing counts in windows of length ``window``.

    Compare the estimate with the finite-window prediction
    ``GaussianCrossings.fano_factor(u, T=estimate.window, kind=kind)``, using
    the returned window length after rounding to whole sampling steps. The
    model assumes zero mean: for raw data with mean ``mu``, use ``u - mu`` as
    its threshold. Use the long-time limit only when its approximation is
    adequate for the chosen window.

    The confidence interval is a percentile bootstrap that resamples whole
    windows, which keeps the dependence between thresholds within each
    resample. The reported intervals are pointwise, not simultaneous. It
    treats windows as independent; dependence between windows can make the
    intervals too narrow or too wide. Long windows or one window per independent
    realization can support this assumption; check dependence between counts.

    Sampling a continuous process misses crossings that happen between
    samples. When ``check_sampling`` is true, the crossings are recounted using
    every second sample over the same observation span. At thresholds with at
    least 50 detected crossings, a loss larger than ``sampling_tolerance``
    issues a :class:`CoarseSamplingWarning`. Realizations shorter than three
    samples cannot be checked. This is a heuristic: both grids can miss events,
    so absence of a warning does not establish adequate sampling resolution.

    Args:
        x: Samples taken at a uniform step ``dt``: one series of shape
            ``(n_samples,)`` or independent series of shape ``(n_series, n_samples)``.
        u: A threshold or a one-dimensional array of thresholds.
        window: Window length, in the same time units as ``dt``.
        dt: Sampling step.
        kind: ``"up"``, ``"down"``, or ``"total"`` crossings.
        confidence: Confidence level of the bootstrap interval.
        n_bootstrap: Number of bootstrap resamples.
        seed: Seed for the bootstrap random number generator.
        check_sampling: Whether to test for missed crossings.
        sampling_tolerance: Fraction of crossings that may disappear when the
            step is doubled before a warning is issued.

    Returns:
        A :class:`FanoEstimate`.
    """
    if not 0 < confidence < 1:
        raise ValueError("confidence must lie strictly between 0 and 1")
    if not isinstance(n_bootstrap, (int, np.integer)) or n_bootstrap < 1:
        raise ValueError("n_bootstrap must be a positive integer")
    values = _series(x)
    levels, scalar = _levels(u)
    direction = _kind(kind)
    steps = _steps_per_window(window, dt)
    counts = windowed_counts(values, levels, window, dt, direction).astype(float)
    n_windows = counts.shape[0]
    if n_windows < 2:
        raise ValueError("at least two complete windows are needed to estimate a variance")
    if check_sampling:
        _check_sampling(values, levels, direction, sampling_tolerance, stacklevel=3)

    fano = _fano(counts)
    rng = np.random.default_rng(seed)
    # Resample in chunks of at most ~2e6 counts to bound memory.
    chunk = max(1, min(n_bootstrap, int(2e6 // max(1, n_windows * levels.size))))
    resampled = np.empty((n_bootstrap, levels.size))
    for start in range(0, n_bootstrap, chunk):
        stop = min(n_bootstrap, start + chunk)
        picks = rng.integers(0, n_windows, size=(stop - start, n_windows))
        resampled[start:stop] = _fano(counts[picks])
    alpha = (1 - confidence) / 2
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN columns for empty levels
        low, high = np.nanquantile(resampled, [alpha, 1 - alpha], axis=0)
        stderr = np.nanstd(resampled, axis=0, ddof=1)

    def shape(value):
        value = np.asarray(value, dtype=float)
        return float(value[0]) if scalar else value

    return FanoEstimate(
        fano=shape(fano),
        ci_low=shape(low),
        ci_high=shape(high),
        stderr=shape(stderr),
        mean=shape(counts.mean(axis=0)),
        variance=shape(counts.var(axis=0, ddof=1)),
        n_windows=int(n_windows),
        window=steps * dt,
        confidence=confidence,
    )
