"""Windowed crossing counts and empirical Fano factors."""

import warnings

import numpy as np
import pytest
import torch

from gaussian_crossings import (
    CoarseSamplingWarning,
    GaussianCrossings,
    empirical_fano,
    windowed_counts,
)
from gaussian_crossings.process import r_squared_exp
from gaussian_crossings.utils import (
    count_crossings,
    count_downcrossings,
    count_upcrossings,
    simulate_gaussian_process_fft,
)


def test_each_crossing_lands_in_exactly_one_window():
    # Crossings sit in every interval, including those that start at a window boundary.
    x = np.array([-1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0])
    np.testing.assert_array_equal(windowed_counts(x, 0.0, window=2, dt=1), [1, 1, 1])
    np.testing.assert_array_equal(windowed_counts(x, 0.0, window=2, kind="down"), [1, 1, 1])
    np.testing.assert_array_equal(windowed_counts(x, 0.0, window=2, kind="total"), [2, 2, 2])
    # A window of three steps covers six intervals as two windows; nothing is double counted.
    np.testing.assert_array_equal(windowed_counts(x, 0.0, window=3, kind="total"), [3, 3])


def test_window_totals_match_whole_series_counts():
    rng = np.random.default_rng(4)
    x = np.cumsum(rng.normal(size=1201)) * 0.1
    for kind, counter in [
        ("up", count_upcrossings),
        ("down", count_downcrossings),
        ("total", count_crossings),
    ]:
        counts = windowed_counts(x, 0.3, window=0.5, dt=0.01, kind=kind)
        assert counts.shape == (24,)
        assert counts.sum() == counter(torch.as_tensor(x), 0.3)


def test_multiple_levels_and_stacked_series():
    rng = np.random.default_rng(5)
    x = rng.normal(size=(3, 401))
    levels = np.array([-0.5, 0.0, 0.5])
    counts = windowed_counts(x, levels, window=40, dt=1)
    assert counts.shape == (30, 3)
    for j, level in enumerate(levels):
        np.testing.assert_array_equal(counts[:, j], windowed_counts(x, level, window=40, dt=1))
    np.testing.assert_array_equal(
        windowed_counts(x, 0.0, window=40, dt=1)[:10], windowed_counts(x[0], 0.0, window=40)
    )


@pytest.mark.parametrize(
    "kwargs,message",
    [
        (dict(window=0.5, dt=1.0), "at least one sampling step"),
        (dict(window=1e6, dt=1.0), "complete window"),
        (dict(window=10.0, dt=1.0, kind="sideways"), "kind must be"),
    ],
)
def test_invalid_windows_and_kinds(kwargs, message):
    with pytest.raises(ValueError, match=message):
        windowed_counts(np.zeros(100), 0.0, **kwargs)


def test_empirical_fano_statistics_and_reproducible_interval():
    rng = np.random.default_rng(6)
    x = rng.normal(size=20001)
    first = empirical_fano(x, 0.2, window=100, dt=1, seed=1, check_sampling=False)
    second = empirical_fano(x, 0.2, window=100, dt=1, seed=1, check_sampling=False)
    counts = windowed_counts(x, 0.2, window=100, dt=1)
    assert first.n_windows == 200 and first.window == 100
    assert first.fano == pytest.approx(counts.var(ddof=1) / counts.mean())
    assert first.ci_low < first.fano < first.ci_high
    assert (first.ci_low, first.ci_high) == (second.ci_low, second.ci_high)
    assert isinstance(first.fano, float)


def test_empirical_fano_vector_levels_and_empty_level():
    rng = np.random.default_rng(7)
    x = rng.normal(size=5001)
    estimate = empirical_fano(x, [0.0, 50.0], window=50, dt=1, seed=0, check_sampling=False)
    assert estimate.fano.shape == (2,)
    assert np.isfinite(estimate.fano[0]) and np.isnan(estimate.fano[1])


def test_coarse_sampling_warning():
    torch.manual_seed(11)
    _, fine = simulate_gaussian_process_fft(r_squared_exp, 400.0, 0.02, sigma=1.0, tau=1.0)
    with warnings.catch_warnings():
        warnings.simplefilter("error", CoarseSamplingWarning)
        empirical_fano(fine.numpy(), 0.5, window=20.0, dt=0.02, seed=0)
    with pytest.warns(CoarseSamplingWarning, match="Doubling the sampling step"):
        empirical_fano(fine.numpy()[::40], 0.5, window=20.0, dt=0.8, seed=0)


def test_sampling_check_ignores_the_unmatched_final_interval():
    # Even-length series whose only crossing lies in the final interval, which a series of
    # every second sample cannot cover; it must not count as a missed crossing.
    x = np.tile([-1.0, -1.0, -1.0, 1.0], (200, 1))
    with warnings.catch_warnings():
        warnings.simplefilter("error", CoarseSamplingWarning)
        estimate = empirical_fano(x, 0.0, window=1.0, dt=1.0, seed=0)
    assert estimate.n_windows == 600
    assert estimate.mean == pytest.approx(1 / 3)


def test_sampling_check_skips_series_too_short_to_downsample():
    rng = np.random.default_rng(8)
    x = rng.normal(size=(500, 2))  # independent two-sample realizations
    with warnings.catch_warnings():
        warnings.simplefilter("error", CoarseSamplingWarning)
        estimate = empirical_fano(x, 0.0, window=1.0, dt=1.0, seed=0)
    assert estimate.n_windows == 500


def test_estimate_agrees_with_finite_window_theory():
    torch.manual_seed(12)
    _, x = simulate_gaussian_process_fft(r_squared_exp, 20000.0, 0.05, sigma=1.0, tau=1.0)
    estimate = empirical_fano(x.numpy(), 1.0, window=50.0, dt=0.05, seed=0)
    theory = GaussianCrossings(r_squared_exp, sigma=1.0, tau=1.0).fano_factor(u=1.0, T=50.0)
    assert abs(estimate.fano - theory) < 4 * estimate.stderr
