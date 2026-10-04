"""Crossing detection, SDE stepping, and empirical covariance utilities."""

import math
from functools import wraps
from typing import Any, Callable

import numpy as np
import torch


def _series(x):
    values = torch.as_tensor(x)
    if not values.is_floating_point():
        values = values.to(torch.float64)
    if values.ndim != 1 or not torch.isfinite(values).all():
        raise ValueError("x must be a finite one-dimensional series")
    return values


def _mask(x, threshold, up):
    if up:
        return (x[:-1] < threshold) & (x[1:] >= threshold)
    return (x[:-1] > threshold) & (x[1:] <= threshold)


def _count(x, threshold, up):
    if isinstance(threshold, (list, tuple)):
        return [_count(x, level, up) for level in threshold]
    return int(_mask(_series(x), threshold, up).sum())


def count_upcrossings(x, threshold):
    """Count arrivals at/above a level from strictly below; lists give lists."""
    return _count(x, threshold, True)


def count_downcrossings(x, threshold):
    """Count arrivals at/below a level from strictly above; lists give lists."""
    return _count(x, threshold, False)


def count_crossings(x, threshold):
    """Count upcrossings plus downcrossings using the same endpoint rule."""
    if isinstance(threshold, (list, tuple)):
        return [count_crossings(x, level) for level in threshold]
    return count_upcrossings(x, threshold) + count_downcrossings(x, threshold)


def _crossing_times(x, t, threshold, up):
    if isinstance(threshold, (list, tuple)):
        return [_crossing_times(x, t, level, up) for level in threshold]
    x = _series(x)
    t = torch.as_tensor(t, dtype=x.dtype, device=x.device)
    if t.shape != x.shape or not torch.isfinite(t).all() or not torch.all(t[1:] > t[:-1]):
        raise ValueError("t must match x and contain finite, strictly increasing times")
    indices = torch.nonzero(_mask(x, threshold, up)).flatten()
    fraction = (threshold - x[indices]) / (x[indices + 1] - x[indices])
    return t[indices] + fraction * (t[indices + 1] - t[indices])


def upcrossing_times(x, t, threshold):
    """Linearly interpolate times of the events counted by count_upcrossings."""
    return _crossing_times(x, t, threshold, True)


def downcrossing_times(x, t, threshold):
    """Linearly interpolate times of events counted by count_downcrossings."""
    return _crossing_times(x, t, threshold, False)


def crossing_times(x, t, threshold):
    """Return sorted crossing times and directions (+1 up, -1 down)."""
    if isinstance(threshold, (list, tuple)):
        return [crossing_times(x, t, level) for level in threshold]
    up = upcrossing_times(x, t, threshold)
    down = downcrossing_times(x, t, threshold)
    times = torch.cat((up, down))
    directions = torch.cat((torch.ones_like(up), -torch.ones_like(down)))
    order = times.argsort()
    return times[order], directions[order]


def _sde_setup(model, time, dt, x0):
    if not math.isfinite(time) or not math.isfinite(dt) or time <= 0 or dt <= 0:
        raise ValueError("time and dt must be finite and positive")
    points = int(time / dt)
    if points < 1:
        raise ValueError("time must be at least dt")
    state = torch.as_tensor(model.steady_state() if x0 is None else x0).clone()
    if not state.is_floating_point():
        state = state.to(torch.float64)
    if state.shape != (model.dim,):
        raise ValueError("x0 must have shape (model.dim,)")
    noise = torch.as_tensor(model.noise_vector(), dtype=state.dtype, device=state.device)
    if noise.shape != state.shape:
        raise ValueError("noise_vector() must have shape (model.dim,)")
    return points, state, noise


def _sde_step(model, state, noise, dt):
    drift = model._dynamical_fun(None, state)
    increment = torch.randn(state.shape, dtype=state.dtype, device=state.device)
    return state + drift * dt + noise * increment * math.sqrt(dt)


def euler_maruyama_sde(model, time, dt, x0=None):
    """Integrate an SDE with fixed step dt, returning time and state arrays.

    Returns floor(time/dt) samples including the initial state. Times are
    actual integration times, starting at zero. By default the initial state
    is the model's deterministic equilibrium, not a stationary random draw;
    supply x0 or allow burn-in when stationary statistics are required.
    """
    points, state, noise = _sde_setup(model, time, dt, x0)
    t = torch.arange(points, dtype=state.dtype, device=state.device) * dt
    trajectory = state.new_empty((points, model.dim))
    trajectory[0] = state
    for i in range(1, points):
        state = _sde_step(model, state, noise, dt)
        trajectory[i] = state
    return t, trajectory


def euler_maruyama_upcrossings(model, time, dt, u_tensor, idx, x0=None):
    """Count upcrossings during the same SDE integration without storing paths."""
    points, state, noise = _sde_setup(model, time, dt, x0)
    if not 0 <= idx < model.dim:
        raise ValueError("idx must select a state component")
    levels = torch.as_tensor(u_tensor, dtype=state.dtype, device=state.device)
    counts = torch.zeros_like(levels, dtype=torch.int32)
    for _ in range(points - 1):
        next_state = _sde_step(model, state, noise, dt)
        counts += ((state[idx] < levels) & (next_state[idx] >= levels)).to(torch.int32)
        state = next_state
    return counts


def dynm_fun(f: Callable) -> Callable:
    """Compatibility decorator preserving a model drift method's metadata."""

    @wraps(f)
    def wrapper(self: Any, t: Any, x: torch.Tensor) -> torch.Tensor:
        return f(self, t, x)

    return wrapper


def autocorrelation(x):
    """Mean-centered covariance at nonnegative lags, divided by n-1.

    The zero-lag value is the unbiased sample variance. The denominator is
    fixed across lags for compatibility; this is not a per-lag unbiased or
    unit-normalized correlation estimator.
    """
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or x.size < 2 or not np.isfinite(x).all():
        raise ValueError("x must contain at least two finite scalar samples")
    centered = x - x.mean()
    return np.correlate(centered, centered, mode="full")[x.size - 1 :] / (x.size - 1)


def __getattr__(name):
    # MidpointNormalize moved to gaussian_crossings.plotting so that matplotlib
    # is only imported when plotting helpers are used.
    if name == "MidpointNormalize":
        from gaussian_crossings.plotting import MidpointNormalize

        return MidpointNormalize
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
