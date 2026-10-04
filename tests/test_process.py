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


@pytest.mark.parametrize(
    "model_type,acf",
    [(process.FilteredOU, process.r_filtered_OU), (process.OUNoise, process.r_OU_noise)],
)
@pytest.mark.parametrize(
    "kappa",
    [1 - 1e-4, 1 - 1e-8, np.nextafter(1.0, 0.0), np.nextafter(1.0, 2.0), 1 + 1e-8, 1 + 1e-4],
)
def test_near_equal_ou_time_scales_match_linear_system(model_type, acf, kappa):
    import mpmath as mp

    # Use high precision for the matrix exponential: the triangular double-
    # precision shortcut itself cancels when the drift eigenvalues nearly coincide.
    tau, sigma = 0.7, 1.2
    dynamics = model_type(tau_e=tau, tau_f=kappa * tau, sigma=sigma)
    A = dynamics.J.numpy()
    diffusion = dynamics.noise_vector().numpy()
    covariance = solve_continuous_lyapunov(A, -np.outer(diffusion, diffusion))
    for lag in (0.05, 0.5, 3.0):
        t = torch.tensor(lag, dtype=torch.float64, requires_grad=True)
        r = acf(t, sigma=sigma, tau=tau, kappa=kappa)
        p = torch.autograd.grad(r, t, create_graph=True)[0]
        q = -torch.autograd.grad(p, t)[0]
        with mp.workdps(60):
            propagator = np.array(mp.expm(mp.matrix((A * lag).tolist())).tolist(), dtype=float)
        propagated = propagator @ covariance
        expected = [propagated[1, 1], (A @ propagated)[1, 1], -(A @ A @ propagated)[1, 1]]
        np.testing.assert_allclose([r.item(), p.item(), q.item()], expected, rtol=2e-13, atol=2e-14)


@pytest.mark.parametrize("acf,driven", [(process.r_filtered_OU, False), (process.r_OU_noise, True)])
@pytest.mark.parametrize("offset", [-1e-8, 1e-8])
def test_near_equal_ou_origin_and_parameter_gradients(acf, driven, offset):
    from gaussian_crossings import GaussianUpCrossings

    sigma, tau = 1.2, 0.7
    kappa = torch.tensor(1 + offset, dtype=torch.float64, requires_grad=True)
    model = GaussianUpCrossings(acf, sigma=sigma, tau=tau, kappa=kappa)
    k = kappa.item()
    r0 = sigma**2 * (k if driven else 1) / (1 + k)
    q0 = sigma**2 / (tau**2 * (1 + k) * (1 if driven else k))
    assert model.r0.item() == pytest.approx(r0, rel=2e-14)
    assert model.q(0).item() == pytest.approx(q0, rel=2e-14)
    dr0 = torch.autograd.grad(model.r0, kappa)[0]
    dq0 = torch.autograd.grad(model.q(0), kappa)[0]
    expected_dr0 = sigma**2 / (1 + k) ** 2 * (1 if driven else -1)
    expected_dq0 = -(sigma**2) / (tau**2 * (1 + k) ** 2)
    if not driven:
        expected_dq0 *= (1 + 2 * k) / k**2
    assert dr0.item() == pytest.approx(expected_dr0, rel=2e-13)
    assert dq0.item() == pytest.approx(expected_dq0, rel=2e-13)


@pytest.mark.parametrize("acf", [process.r_filtered_OU, process.r_OU_noise])
def test_equal_ou_time_scales_still_require_limiting_callback(acf):
    with pytest.raises(ValueError, match="kappa=1"):
        acf(0.5, sigma=1.0, tau=1.0, kappa=1.0)
