"""Inputs that previously failed silently or confusingly now behave predictably."""

import math

import numpy as np
import pytest
import torch

from gaussian_crossings import GaussianUpCrossings, process
from gaussian_crossings.formula.formula_dimless import GaussianUpCrossingsDimless
from gaussian_crossings.formula.integration import sdho_parts
from gaussian_crossings.utils import simulate_gaussian_process_cholesky

pytestmark = pytest.mark.filterwarnings(
    "ignore:GaussianUpCrossingsDimless.*is deprecated:DeprecationWarning"
)


@pytest.mark.parametrize(
    "kernel,params",
    [
        (process.r_squared_exp, dict(sigma=1.0, tau=1.0)),
        (process.r_rational_quadratic, dict(sigma=1.0, tau=1.0, alpha=2.0)),
        (process.r_damped_harmonic_oscillator_noise, dict(temp=1.0, omega0=1.0, zeta=0.5)),
    ],
)
def test_misspelled_kernel_parameter_is_rejected(kernel, params):
    with pytest.raises(TypeError, match="unexpected keyword"):
        GaussianUpCrossings(kernel, **params, tua=5.0)


def test_numpy_kernel_gets_an_explanatory_error():
    with pytest.raises(TypeError, match="PyTorch operations"):
        GaussianUpCrossings(lambda t: np.exp(-(np.asarray(t) ** 2) / 2))


def test_cholesky_forwards_positional_covariance_parameters():
    torch.manual_seed(3)
    _, positional = simulate_gaussian_process_cholesky(process.r_squared_exp, 1.0, 0.1, 1.3, 0.4)
    torch.manual_seed(3)
    _, keyword = simulate_gaussian_process_cholesky(
        process.r_squared_exp, 1.0, 0.1, sigma=1.3, tau=0.4
    )
    torch.testing.assert_close(positional, keyword)
    with pytest.raises(TypeError):
        simulate_gaussian_process_cholesky(process.r_squared_exp, 1.0, 0.1, 1.3, 0.4, 1e-8)


@pytest.mark.parametrize("excess", [1e-15, 1e-12, 1e-9, 1e-6])
def test_fano_factor_is_continuous_through_critical_damping(excess):
    def fano(zeta):
        model = GaussianUpCrossings(
            process.r_damped_harmonic_oscillator_noise, u=0.5, temp=1.0, omega0=1.0, zeta=zeta
        )
        return float(model.upcrossing_fano_factor_CLT())

    critical = fano(1.0)
    # dF/dzeta is O(1) at critical damping, so the change is of order the excess.
    assert abs(fano(1.0 + excess) - critical) < 10 * excess + 1e-10


@pytest.mark.parametrize("excess", [1e-12, 1e-6, 1e-3])
def test_near_critical_covariance_matches_critical_limit(excess):
    zeta = 1.0 + excess
    for t in [0.2, 1.0, 5.0]:
        r, p, *_ = sdho_parts(t, zeta)
        assert r == pytest.approx((1 + t) * math.exp(-t), abs=5 * excess)
        assert p == pytest.approx(-t * math.exp(-t), abs=5 * excess)
        kernel = process.r_damped_harmonic_oscillator_noise(t, temp=1.0, omega0=1.0, zeta=zeta)
        assert kernel.item() == pytest.approx((1 + t) * math.exp(-t), abs=5 * excess)


def test_unit_time_class_accepts_kernels_without_tau():
    oscillator = GaussianUpCrossingsDimless(
        process.r_damped_harmonic_oscillator_noise, u=0.5, temp=1.0, omega0=1.0, zeta=0.5
    )
    direct = GaussianUpCrossings(
        process.r_damped_harmonic_oscillator_noise, u=0.5, temp=1.0, omega0=1.0, zeta=0.5
    )
    assert float(oscillator.upcrossing_fano_factor_CLT()) == pytest.approx(
        float(direct.upcrossing_fano_factor_CLT()), abs=1e-12
    )


def test_model_class_aliases():
    assert process.filtered_OU is process.FilteredOU
    assert process.OU_noise is process.OUNoise
    assert process.damped_harmonic_oscillator_noise is process.DampedHarmonicOscillatorNoise
