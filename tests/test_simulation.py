"""Sampling and streaming integration checked against independent constructions."""

import math

import numpy as np
import pytest
import torch

from gaussian_crossings.process import damped_harmonic_oscillator_noise, r_OU
from gaussian_crossings.utils import (
    count_upcrossings,
    euler_maruyama_sde,
    euler_maruyama_upcrossings,
    simulate_gaussian_process_cholesky,
    simulate_gaussian_process_fft,
)


@pytest.mark.parametrize(
    "sampler", [simulate_gaussian_process_fft, simulate_gaussian_process_cholesky]
)
def test_scalar_callback_and_seed_reproducibility(sampler):
    def scalar_covariance(t):
        return math.exp(-abs(float(t)))

    torch.manual_seed(27)
    time, values = sampler(scalar_covariance, 2.0, 0.1)
    torch.manual_seed(27)
    time2, values2 = sampler(r_OU, 2.0, 0.1, sigma=1.0, tau=1.0)
    torch.testing.assert_close(time, torch.arange(20, dtype=torch.float64) * 0.1)
    torch.testing.assert_close(time, time2)
    torch.testing.assert_close(values, values2)


def test_fft_samples_target_stationary_covariance():
    torch.manual_seed(207)
    paths = torch.stack(
        [simulate_gaussian_process_fft(r_OU, 2.0, 0.1, sigma=1.2, tau=0.4)[1] for _ in range(3000)]
    )
    # Independent trajectories, six-standard-error bounds for Gaussian moments.
    variance = 1.2**2
    assert abs(paths[:, 0].mean()) < 6 * math.sqrt(variance / len(paths))
    for lag in [0, 1, 5, 12]:
        target = variance * math.exp(-lag * 0.1 / 0.4)
        estimate = (paths[:, 0] * paths[:, lag]).mean().item()
        se = math.sqrt((variance**2 + target**2) / len(paths))
        assert abs(estimate - target) < 6 * se


def test_cholesky_path_matches_known_covariance():
    n = 10
    t = torch.arange(n, dtype=torch.float64) * 0.1
    covariance = torch.exp(-(t[:, None] - t[None, :]).abs())
    torch.manual_seed(51)
    expected = torch.linalg.cholesky(covariance) @ torch.randn(n, dtype=torch.float64)
    torch.manual_seed(51)
    _, result = simulate_gaussian_process_cholesky(r_OU, 1.0, 0.1, sigma=1.0, tau=1.0)
    torch.testing.assert_close(result, expected)


@pytest.mark.parametrize("amplitude", [1.0, 1e-14])
def test_invalid_embedding_is_reported(amplitude):
    def invalid(t):
        return amplitude * torch.where(t == 0, 1.0, 2.0)

    with pytest.raises(ValueError, match="positive semidefinite"):
        simulate_gaussian_process_fft(invalid, 1.0, 0.1)
    with pytest.raises(RuntimeError, match="Cholesky"):
        simulate_gaussian_process_cholesky(invalid, 1.0, 0.1, jitter=0, max_attempts=2)


@pytest.mark.parametrize("T,dt", [(0.0, 0.1), (1.0, 0.0), (-1.0, 0.1), (0.01, 0.1)])
def test_invalid_sampling_grid(T, dt):
    with pytest.raises(ValueError):
        simulate_gaussian_process_fft(r_OU, T, dt, sigma=1.0, tau=1.0)


def test_streaming_sde_counts_match_stored_trajectory():
    model = damped_harmonic_oscillator_noise(zeta=0.5)
    levels = torch.tensor([-0.2, 0.0, 0.2], dtype=torch.float64)
    torch.manual_seed(31)
    times, paths = euler_maruyama_sde(model, 10.0, 0.01)
    torch.manual_seed(31)
    counts = euler_maruyama_upcrossings(model, 10.0, 0.01, levels, 0)
    np.testing.assert_array_equal(counts, count_upcrossings(paths[:, 0], levels.tolist()))
    torch.testing.assert_close(
        times[1:] - times[:-1], torch.full((999,), 0.01, dtype=torch.float64)
    )
    assert paths.dtype == torch.float64


def test_sde_uses_model_dimension_and_actual_step_times():
    class ThreeDimensional:
        dim = 3

        def steady_state(self):
            return torch.zeros(3, dtype=torch.float64)

        def noise_vector(self):
            return torch.zeros(3, dtype=torch.float64)

        def _dynamical_fun(self, t, x):
            return torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)

    t, x = euler_maruyama_sde(ThreeDimensional(), 1.0, 0.1)
    torch.testing.assert_close(x, t[:, None] * torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64))
