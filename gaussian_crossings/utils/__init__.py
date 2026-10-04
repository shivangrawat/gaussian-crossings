"""Simulation, crossing detection, covariance estimation, and special functions."""

from .owensT import OwensT, owensT
from .simulation import simulate_gaussian_process_cholesky, simulate_gaussian_process_fft
from .utils import (
    autocorrelation,
    count_crossings,
    count_downcrossings,
    count_upcrossings,
    crossing_times,
    downcrossing_times,
    dynm_fun,
    euler_maruyama_sde,
    euler_maruyama_upcrossings,
    upcrossing_times,
)

__all__ = [
    "OwensT",
    "owensT",
    "simulate_gaussian_process_cholesky",
    "simulate_gaussian_process_fft",
    "autocorrelation",
    "count_crossings",
    "count_downcrossings",
    "count_upcrossings",
    "crossing_times",
    "downcrossing_times",
    "dynm_fun",
    "euler_maruyama_sde",
    "euler_maruyama_upcrossings",
    "upcrossing_times",
]


def __getattr__(name):
    # Plotting helpers live in gaussian_crossings.plotting (optional matplotlib).
    if name == "MidpointNormalize":
        from gaussian_crossings.plotting import MidpointNormalize

        return MidpointNormalize
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
