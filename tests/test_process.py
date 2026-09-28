"""Covariances checked against exact linear-system propagation."""

import math

import numpy as np
import pytest
import torch
from scipy.linalg import expm, solve_continuous_lyapunov

from gaussian_crossings import process


@pytest.mark.parametrize("zeta", [0.5, 1.0, 2.0])
def test_sdho_covariance_from_matrix_exponential(zeta):
    omega, temp = 1.7, 2.3
    matrix = np.array([[0.0, 1.0], [-(omega**2), -2 * zeta * omega]])
    covariance = np.diag([temp / omega**2, temp])
    for t in [0.0, 0.01, 0.5, 3.0, 20.0]:
        expected = (expm(matrix * t) @ covariance)[0, 0]
        actual = process.r_damped_harmonic_oscillator_noise(t, temp=temp, omega0=omega, zeta=zeta)
        assert actual.item() == pytest.approx(expected, abs=1e-13)


@pytest.mark.parametrize(
    "model_type,acf",
    [(process.filtered_OU, process.r_filtered_OU), (process.OU_noise, process.r_OU_noise)],
)
def test_colored_noise_covariance_matches_sde(model_type, acf):
    model = model_type(tau_e=0.7, tau_f=0.21, sigma=1.2)
    A = model.J.numpy()
    diffusion = model.noise_vector().numpy()
    covariance = solve_continuous_lyapunov(A, -np.outer(diffusion, diffusion))
    for t in [0.0, 0.1, 0.7, 2.0]:
        expected = (expm(A * t) @ covariance)[1, 1]
        actual = acf(t, sigma=1.2, tau=0.7, kappa=0.3)
        assert actual.item() == pytest.approx(expected, abs=1e-13)


@pytest.mark.parametrize(
    "name,kwargs",
    [
        ("r_OU", {}),
        ("r_OU_noise", {"kappa": 0.3}),
        ("r_filtered_OU", {"kappa": 0.3}),
        ("r_squared_exp", {}),
        ("r_rational_quadratic", {"alpha": 1.5}),
        ("r_matern", {"nu": 1.7}),
    ],
)
def test_scalar_numpy_tensor_inputs_and_symmetry(name, kwargs):
    fn = getattr(process, name)
    params = dict(sigma=1.2, tau=0.7, **kwargs)
    a = fn(0.3, **params)
    b = fn(np.array([-0.3, 0.3]), **params)
    c = fn(torch.tensor([-0.3, 0.3], dtype=torch.float64), **params)
    torch.testing.assert_close(b, c)
    torch.testing.assert_close(b, a.expand(2))


@pytest.mark.parametrize("nu", [1.5, 2.5])
def test_matern_half_integer_derivatives(nu):
    from gaussian_crossings import GaussianUpCrossings

    model = GaussianUpCrossings(process.r_matern, sigma=1.0, tau=1.0, nu=nu)
    assert model.q0.item() == pytest.approx(nu / (nu - 1), rel=1e-13)
    assert model.upcrossing_mean_rate().item() == pytest.approx(
        math.sqrt(nu / (nu - 1)) / (2 * math.pi)
    )
