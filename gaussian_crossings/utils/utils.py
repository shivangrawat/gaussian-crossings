"""Utility functions for threshold crossing detection and SDE simulation.

This module provides functions for counting and locating threshold crossings
(upcrossings, downcrossings, and total crossings) in discrete time series,
Euler-Maruyama SDE integration for generating sample paths from dynamical
system models, empirical autocorrelation estimation, and visualization
helpers for Fano factor heatmaps.
"""

from typing import Any, Callable, List, Optional, Tuple, Union

import numpy as np
import torch
import math
import matplotlib.colors as mcolors


def count_upcrossings(
    x: torch.Tensor,
    threshold: Union[float, List[float], Tuple[float, ...]]
) -> Union[int, List[int]]:
    """Count the number of upcrossings of a threshold in a 1D signal.

    An upcrossing occurs when the signal transitions from below to at or
    above the threshold between consecutive samples.

    Args:
        x: 1D signal tensor.
        threshold: Threshold value(s) to check for crossings.

    Returns:
        Number of upcrossings for each threshold. Returns an int if a single
        threshold is provided, or a list of ints for multiple thresholds.
    """
    if isinstance(threshold, (list, tuple)):
        return [count_upcrossings(x, t) for t in threshold]

    upcrossings = torch.logical_and(x[:-1] < threshold, x[1:] >= threshold)
    return upcrossings.sum().item()


def count_downcrossings(
    x: torch.Tensor,
    threshold: Union[float, List[float], Tuple[float, ...]]
) -> Union[int, List[int]]:
    """Count the number of downcrossings of a threshold in a 1D signal.

    A downcrossing occurs when the signal transitions from above to at or
    below the threshold between consecutive samples.

    Args:
        x: 1D signal tensor.
        threshold: Threshold value(s) to check for crossings.

    Returns:
        Number of downcrossings for each threshold. Returns an int if a single
        threshold is provided, or a list of ints for multiple thresholds.
    """
    if isinstance(threshold, (list, tuple)):
        return [count_downcrossings(x, t) for t in threshold]

    downcrossings = torch.logical_and(x[:-1] > threshold, x[1:] <= threshold)
    return downcrossings.sum().item()


def count_crossings(
    x: torch.Tensor,
    threshold: Union[float, List[float], Tuple[float, ...]]
) -> Union[int, List[int]]:
    """Count the total number of crossings of a threshold in a 1D signal.

    A crossing is either an upcrossing or a downcrossing.

    Args:
        x: 1D signal tensor.
        threshold: Threshold value(s) to check for crossings.

    Returns:
        Total number of crossings (up + down) for each threshold. Returns
        an int if a single threshold is provided, or a list of ints for
        multiple thresholds.
    """
    if isinstance(threshold, (list, tuple)):
        return [count_crossings(x, t) for t in threshold]

    return count_upcrossings(x, threshold) + count_downcrossings(x, threshold)


def upcrossing_times(
    x: torch.Tensor,
    t: torch.Tensor,
    threshold: Union[float, List[float], Tuple[float, ...]]
) -> Union[torch.Tensor, List[torch.Tensor]]:
    """Find the times of upcrossings using linear interpolation.

    Locates the precise times at which the signal crosses upward through
    the threshold by linearly interpolating between sample points.

    Args:
        x: 1D signal tensor.
        t: 1D tensor of time values corresponding to the signal.
        threshold: Threshold value(s) to check for crossings.

    Returns:
        Interpolated times of upcrossings. Returns a tensor if a single
        threshold is provided, or a list of tensors for multiple thresholds.
    """
    if isinstance(threshold, (list, tuple)):
        return [upcrossing_times(x, t, thr) for thr in threshold]

    # Identify segments where x crosses upward through the threshold
    mask = (x[:-1] < threshold) & (x[1:] >= threshold)
    if not mask.any():
        return x.new_empty(0)
    idx = torch.nonzero(mask).squeeze(-1)

    # Compute the interpolation factor for each crossing
    alpha = (threshold - x[idx]) / (x[idx + 1] - x[idx])

    # Use the time vector to compute the exact crossing times
    t_cross = t[idx] + alpha * (t[idx + 1] - t[idx])
    return t_cross


def downcrossing_times(
    x: torch.Tensor,
    t: torch.Tensor,
    threshold: Union[float, List[float], Tuple[float, ...]]
) -> Union[torch.Tensor, List[torch.Tensor]]:
    """Find the times of downcrossings using linear interpolation.

    Locates the precise times at which the signal crosses downward through
    the threshold by linearly interpolating between sample points.

    Args:
        x: 1D signal tensor.
        t: 1D tensor of time values corresponding to the signal.
        threshold: Threshold value(s) to check for crossings.

    Returns:
        Interpolated times of downcrossings. Returns a tensor if a single
        threshold is provided, or a list of tensors for multiple thresholds.
    """
    if isinstance(threshold, (list, tuple)):
        return [downcrossing_times(x, t, thr) for thr in threshold]

    # Identify segments where x crosses downward through the threshold
    mask = (x[:-1] >= threshold) & (x[1:] < threshold)
    if not mask.any():
        return x.new_empty(0)
    idx = torch.nonzero(mask).squeeze(-1)

    # Compute the interpolation factor for each crossing
    alpha = (x[idx] - threshold) / (x[idx] - x[idx + 1])

    # Interpolate the crossing times using the time vector
    t_cross = t[idx] + alpha * (t[idx + 1] - t[idx])
    return t_cross


def crossing_times(
    x: torch.Tensor,
    t: torch.Tensor,
    threshold: Union[float, List[float], Tuple[float, ...]]
) -> Union[Tuple[torch.Tensor, torch.Tensor], List[Tuple[torch.Tensor, torch.Tensor]]]:
    """Find all threshold crossings with direction indicators.

    Locates both upcrossings and downcrossings using linear interpolation,
    returning the crossing times along with direction indicators.

    Args:
        x: 1D signal tensor.
        t: 1D tensor of time values corresponding to the signal.
        threshold: Threshold value(s) to check for crossings.

    Returns:
        For a single threshold, returns a tuple (times, directions) where:
            - times: 1D tensor of interpolated crossing times (sorted).
            - directions: 1D tensor of crossing directions (+1 for upcrossings,
              -1 for downcrossings).
        If multiple thresholds are provided, returns a list of such tuples.
    """
    if isinstance(threshold, (list, tuple)):
        return [crossing_times(x, t, thr) for thr in threshold]

    # Compute the upcrossing and downcrossing times
    up_times = upcrossing_times(x, t, threshold)
    down_times = downcrossing_times(x, t, threshold)

    # Create corresponding direction indicators
    up_dirs = torch.ones_like(up_times)
    down_dirs = -torch.ones_like(down_times)

    # Combine and sort the crossing times and directions
    times = torch.cat([up_times, down_times])
    directions = torch.cat([up_dirs, down_dirs])

    if times.numel() > 0:
        sort_idx = times.argsort()
        times = times[sort_idx]
        directions = directions[sort_idx]

    return times, directions


def euler_maruyama_sde(
    model: Any,
    time: float,
    dt: float,
    x0: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Integrate an SDE using the Euler-Maruyama method.

    Numerically integrates the stochastic differential equation:
        dX = f(X, t) dt + sigma dW

    where f is the drift and sigma is the diffusion coefficient.

    Args:
        model: Dynamical system model with methods:
            - _dynamical_fun(t, x): Returns the drift term.
            - noise_vector(): Returns the diffusion coefficients.
            - steady_state(): Returns the equilibrium state.
            - dim: Dimension of the state space.
        time: Total integration time.
        dt: Time step size.
        x0: Initial condition. If None, uses model.steady_state().

    Returns:
        A tuple (t, trajectory) where:
            - t: Tensor of time points.
            - trajectory: Tensor of shape (n_points, dim) containing
              the state at each time point.
    """
    n_points = int(time / dt)

    # Default initial condition if not supplied
    if x0 is None:
        x0 = model.steady_state()

    # Create time vector
    t = torch.linspace(0, time, n_points)

    # Prepare output
    trajectory = torch.zeros((n_points, 2), dtype=torch.float)
    trajectory[0] = x0

    sqrt_dt = math.sqrt(dt)

    for i in range(n_points - 1):
        # Drift term
        drift = model._dynamical_fun(None, trajectory[i])

        # Diffusion term
        noise = model.noise_vector() * torch.randn(model.dim)

        # Euler-Maruyama update
        trajectory[i + 1] = trajectory[i] + drift * dt + noise * sqrt_dt

    return t, trajectory


def euler_maruyama_upcrossings(
    model: Any,
    time: float,
    dt: float,
    u_tensor: torch.Tensor,
    idx: int,
    x0: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Count upcrossings during Euler-Maruyama integration.

    Integrates an SDE while simultaneously counting the number of
    upcrossings of multiple threshold levels for a specified state variable.

    Args:
        model: Dynamical system model with methods:
            - _dynamical_fun(t, x): Returns the drift term.
            - noise_vector(): Returns the diffusion coefficients.
            - steady_state(): Returns the equilibrium state.
            - dim: Dimension of the state space.
        time: Total integration time.
        dt: Time step size.
        u_tensor: 1D tensor of threshold levels for upcrossings.
        idx: Index of the state variable to monitor for crossings.
        x0: Initial condition. If None, uses model.steady_state().

    Returns:
        1D tensor containing the number of upcrossings for each threshold
        level in u_tensor.
    """
    n_points = int(time / dt)
    sqrt_dt = math.sqrt(dt)

    # Default initial condition if not supplied
    if x0 is None:
        x0 = model.steady_state()

    # Prepare the first point
    current_state = x0.clone()
    upcrossings = torch.zeros_like(u_tensor, dtype=torch.int32)

    # Simulation loop
    for _ in range(n_points - 1):
        # Compute drift and diffusion
        drift = model._dynamical_fun(None, current_state)
        noise = model.noise_vector() * torch.randn(model.dim)

        # Euler-Maruyama update
        next_state = current_state + drift * dt + noise * sqrt_dt

        # Check upcrossings for all u values simultaneously
        below = current_state[idx] < u_tensor
        above = next_state[idx] >= u_tensor

        # Increment upcrossings where there is an upcrossing
        upcrossings += (below & above).int()

        # Update state
        current_state = next_state

    return upcrossings


def dynm_fun(f: Callable) -> Callable:
    """Decorator for dynamical system drift functions.

    Wraps a bound method ``(self, t, x) -> dx/dt`` so that it can be
    called as ``model._dynamical_fun(t, x)`` by the Euler-Maruyama
    integrator without exposing the ``self`` binding.

    Args:
        f: A method with signature ``(self, t, x) -> torch.Tensor``
            representing the drift term of an SDE.

    Returns:
        A wrapper with signature ``(self, t, x) -> torch.Tensor``.
    """
    def wrapper(self: Any, t: Any, x: torch.Tensor) -> torch.Tensor:
        new_fun = lambda t, x: f(self, t, x)
        return new_fun(t, x)

    return wrapper


def autocorrelation(x: Union[np.ndarray, torch.Tensor, List[float]]) -> np.ndarray:
    """Compute the autocorrelation function of a 1D time series.

    Computes the full autocorrelation of the input signal after subtracting
    its mean, then extracts the non-negative lags and normalizes by (n - 1).
    The zero-lag value equals the unbiased sample variance.

    Args:
        x: A one-dimensional time series. Can be a list, numpy array,
            or torch tensor.

    Returns:
        A numpy array containing the autocorrelation values for non-negative
        lags, normalized by (n - 1).

    Example:
        >>> x = np.random.randn(100)
        >>> ac = autocorrelation(x)
        >>> # ac[0] equals the unbiased sample variance of x
        >>> ac_normalized = ac / ac[0]  # Now ac_normalized[0] == 1
    """
    # Convert to numpy array in case x is a torch tensor
    x = np.asarray(x)
    n = len(x)
    # Subtract the mean
    x = x - np.mean(x)
    # Compute the full autocorrelation using np.correlate
    ac_full = np.correlate(x, x, mode='full')
    # Keep only the second half (non-negative lags)
    ac = ac_full[n - 1:]
    # Normalize so that the zero lag equals the unbiased sample variance
    ac = ac / (n - 1)
    return ac


class MidpointNormalize(mcolors.Normalize):
    """Matplotlib normalizer with a specified midpoint for diverging colormaps.

    A color normalizer that maps a chosen midpoint to 0.5 in the colormap,
    producing a diverging color scale.  This is used for Fano factor heatmaps
    where the Poisson reference value (F=1) should appear at the center of
    the colormap, with sub-Poissonian (F<1) and super-Poissonian (F>1) regions
    mapped to opposite ends.

    Attributes:
        midpoint: The data value that maps to 0.5 in the normalized range.
    """

    def __init__(
        self,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        midpoint: Optional[float] = None,
        clip: bool = False
    ) -> None:
        """Initialize the normalizer.

        Args:
            vmin: Minimum data value.
            vmax: Maximum data value.
            midpoint: Value that maps to 0.5 in the normalized range.
            clip: Whether to clip values outside [vmin, vmax].

        Raises:
            ValueError: If vmin, vmax, or midpoint is not specified.
        """
        if vmin is None or vmax is None or midpoint is None:
            raise ValueError("vmin, vmax, and midpoint must all be specified")
        self.midpoint = midpoint
        super().__init__(vmin, vmax, clip)

    def __call__(
        self,
        value: Union[float, np.ndarray],
        clip: Optional[bool] = None
    ) -> np.ma.MaskedArray:
        """Normalize the value(s).

        Args:
            value: Value(s) to normalize.
            clip: Whether to clip (overrides instance setting if provided).

        Returns:
            Normalized value(s) in the range [0, 1].
        """
        # If the entire dataset is above the midpoint
        if self.vmin > self.midpoint:
            return np.ma.masked_array(
                np.interp(value, [self.vmin, self.vmax], [0.5, 1])
            )
        # If the entire dataset is below the midpoint
        elif self.vmax < self.midpoint:
            return np.ma.masked_array(
                np.interp(value, [self.vmin, self.vmax], [0, 0.5])
            )
        # Otherwise, use a diverging normalization with the midpoint in the center
        else:
            return np.ma.masked_array(
                np.interp(value, [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1])
            )
