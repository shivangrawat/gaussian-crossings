import numpy as np
import torch
import math


def count_upcrossings(x, threshold):
    """
    Count the number of upcrossings of a threshold in a 1D signal.

    Parameters
    ----------
    x : torch.Tensor
        1D signal.
    threshold : float
        Threshold value.

    Returns
    -------
    int
        Number of upcrossings.
    """
    upcrossings = torch.logical_and(x[:-1] < threshold, x[1:] >= threshold)
    return upcrossings.sum().item()

def count_downcrossings(x, threshold):
    """
    Count the number of downcrossings of a threshold in a 1D signal.

    Parameters
    ----------
    x : torch.Tensor
        1D signal.
    threshold : float
        Threshold value.

    Returns
    -------
    int
        Number of downcrossings.
    """
    downcrossings = torch.logical_and(x[:-1] > threshold, x[1:] <= threshold)
    return downcrossings.sum().item()

def count_crossings(x, threshold):
    """
    Count the number of crossings of a threshold in a 1D signal.

    Parameters
    ----------
    x : torch.Tensor
        1D signal.
    threshold : float
        Threshold value.

    Returns
    -------
    int
        Number of crossings.
    """
    return count_upcrossings(x, threshold) + count_downcrossings(x, threshold)

def upcrossing_times(x, t, threshold):
    """
    Find the times of upcrossings of a threshold in a 1D signal using linear interpolation.

    Parameters
    ----------
    x : torch.Tensor
        1D signal.
    t : torch.Tensor
        1D tensor of time values corresponding to the signal (assumed to be evenly spaced).
    threshold : float
        Threshold value.

    Returns
    -------
    torch.Tensor
        Interpolated times of upcrossings.
    """
    # Identify segments where x crosses upward through the threshold.
    mask = (x[:-1] < threshold) & (x[1:] >= threshold)
    if not mask.any():
        return x.new_empty(0)
    idx = torch.nonzero(mask).squeeze(-1)
    
    # Compute the interpolation factor alpha for each crossing.
    alpha = (threshold - x[idx]) / (x[idx+1] - x[idx])
    
    # Use the time vector to compute the exact crossing times.
    t_cross = t[idx] + alpha * (t[idx+1] - t[idx])
    return t_cross

def downcrossing_times(x, t, threshold):
    """
    Find the times of downcrossings of a threshold in a 1D signal using linear interpolation.

    Parameters
    ----------
    x : torch.Tensor
        1D signal.
    t : torch.Tensor
        1D tensor of time values corresponding to the signal (assumed to be evenly spaced).
    threshold : float
        Threshold value.

    Returns
    -------
    torch.Tensor
        Interpolated times of downcrossings.
    """
    # Identify segments where x crosses downward through the threshold.
    mask = (x[:-1] >= threshold) & (x[1:] < threshold)
    if not mask.any():
        return x.new_empty(0)
    idx = torch.nonzero(mask).squeeze(-1)
    
    # Compute the interpolation factor alpha for each crossing.
    alpha = (x[idx] - threshold) / (x[idx] - x[idx+1])
    
    # Interpolate the crossing times using the time vector.
    t_cross = t[idx] + alpha * (t[idx+1] - t[idx])
    return t_cross

def crossing_times(x, t, threshold):
    """
    Find all threshold crossings (both upcrossings and downcrossings) in a 1D signal using linear interpolation.
    
    This function reuses the `upcrossing_times` and `downcrossing_times` functions defined above.
    It returns the interpolated crossing times along with a direction indicator:
      +1 for upcrossings and -1 for downcrossings.

    Parameters
    ----------
    x : torch.Tensor
        1D signal.
    t : torch.Tensor
        1D tensor of time values corresponding to the signal (assumed to be evenly spaced).
    threshold : float
        Threshold value.

    Returns
    -------
    times : torch.Tensor
        1D tensor of interpolated crossing times (sorted in ascending order).
    directions : torch.Tensor
        1D tensor of crossing directions corresponding to each time 
        (+1 for upcrossings, -1 for downcrossings).
    """
    # Compute the upcrossing and downcrossing times.
    up_times = upcrossing_times(x, t, threshold)
    down_times = downcrossing_times(x, t, threshold)
    
    # Create corresponding direction indicators.
    up_dirs = torch.ones_like(up_times)
    down_dirs = -torch.ones_like(down_times)
    
    # Combine and sort the crossing times and directions.
    times = torch.cat([up_times, down_times])
    directions = torch.cat([up_dirs, down_dirs])
    
    if times.numel() > 0:
        sort_idx = times.argsort()
        times = times[sort_idx]
        directions = directions[sort_idx]
    
    return times, directions


def euler_maruyama_sde(model, time, dt, x0=None):
    """
    A simple Euler–Maruyama SDE integrator for:
        dX/dt = f(X, t) + sigma * dW
    where:
        - model._dynamical_fun(t, x) gives the drift f(x, t)
        - model.noise_vector() is the function which returns the strength of the noise
    Args:
        model:     filtered_process instance (with ._dynamical_fun and .eta)
        time:      time
        dt:        step size
        x0:        initial condition (tensor of shape (2,))
    Returns:
        trajectory (tensor of shape (n_points, 2)):
            trajectory[:, 0] = x(t)
            trajectory[:, 1] = y(t)
    """
    n_points = int(time / dt)

    # Default initial condition if not supplied
    if x0 is None:
        x0 = model.steady_state()

    # Prepare output
    trajectory = torch.zeros((n_points, 2), dtype=torch.float)
    trajectory[0] = x0

    sqrt_dt = math.sqrt(dt)

    for i in range(n_points - 1):
        # 1) Drift = f(t, X)
        drift = model._dynamical_fun(None, trajectory[i])

        # 2) Diffusion (noise only on x-component):
        noise = model.noise_vector() * torch.randn(model.dim)

        # 3) Euler–Maruyama update:
        trajectory[i + 1] = trajectory[i] + drift * dt + noise * sqrt_dt

    return trajectory

def euler_maruyama_upcrossings(model, time, dt, u_tensor, idx, x0=None):
    """
    Euler–Maruyama SDE integrator that counts the number of upcrossings for multiple threshold levels.

    Args:
        model:      filtered_process instance (with ._dynamical_fun and .noise_vector)
        time:       total simulation time
        dt:         time step size
        u_tensor:   1D tensor of threshold levels for upcrossings (relative to the mean)
        idx:        index of the simulation to check (e.g., 0 for the first variable)
        x0:         initial condition (tensor of shape (model.dim,))
    Returns:
        upcrossings (1D tensor): Number of upcrossings for each threshold level in u_tensor.
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

        # Euler–Maruyama update
        next_state = current_state + drift * dt + noise * sqrt_dt

        # Check upcrossings for all u values simultaneously
        below = current_state[idx] < u_tensor  # Current state below thresholds
        above = next_state[idx] >= u_tensor   # Next state above thresholds

        # Increment upcrossings where there is an upcrossing
        upcrossings += (below & above).int()

        # Update state
        current_state = next_state

    return upcrossings
