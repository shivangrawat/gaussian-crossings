"""Independent identities, derivatives, scaling, and covariance constraints."""

import math
import subprocess
import sys

import pytest
import torch
from scipy.integrate import quad
from scipy.special import ndtr

from gaussian_crossings import GaussianUpCrossings, GaussianUpCrossingsDimless
from gaussian_crossings.process import r_damped_harmonic_oscillator_noise, r_OU, r_squared_exp
from gaussian_crossings.utils import owensT

pytestmark = pytest.mark.filterwarnings(
    "ignore:GaussianUpCrossingsDimless.*is deprecated:DeprecationWarning"
)


@pytest.mark.parametrize("zeta", [0.5, 1.0, 2.0])
@pytest.mark.parametrize("level", [-2.0, 0.0, 0.5, 2.0])
def test_rice_mean(zeta, level):
    model = GaussianUpCrossings(
        r_damped_harmonic_oscillator_noise, u=level, zeta=zeta, omega0=1.7, temp=2.0
    )
    expected = 1.7 / (2 * math.pi) * math.exp(-(level**2) * 1.7**2 / 4.0)
    assert model.upcrossing_mean_rate().item() == pytest.approx(expected, rel=2e-14)
    assert model.downcrossing_mean(3.0) == model.upcrossing_mean(3.0)
    assert model.crossing_mean(3.0) == 2 * model.upcrossing_mean(3.0)


@pytest.mark.parametrize("lag", [0.1, 0.5, 1.0, 4.0])
@pytest.mark.parametrize("level", [-2.0, 0.0, 2.0])
def test_pair_intensity_against_independent_conditional_gaussian_integral(lag, level):
    # RBF covariance, with analytical derivatives and q0=r0=1.
    r = math.exp(-(lag**2) / 2)
    p = -lag * r
    q = (1 - lag**2) * r
    a = (1 - q - p * p / (1 + r)) / 2
    b = (1 + q - p * p / (1 - r)) / 2
    mean_d = p * level / (1 + r)

    # S=(V0+Vt)/2 and D=(Vt-V0)/2 are conditionally independent.
    # Both derivatives positive iff S>|D|; V0*Vt=S^2-D^2.
    def conditional(z):
        d = mean_d + math.sqrt(a) * z
        h = abs(d) / math.sqrt(b)
        truncated = b * h * math.exp(-h * h / 2) / math.sqrt(2 * math.pi) + (b - d * d) * ndtr(-h)
        return truncated * math.exp(-z * z / 2) / math.sqrt(2 * math.pi)

    velocity_product = quad(
        conditional, -12.0, 12.0, epsabs=1e-12, points=[-mean_d / math.sqrt(a)]
    )[0]
    density = math.exp(-level * level / (1 + r)) / (2 * math.pi * math.sqrt(1 - r * r))
    nu = math.exp(-level * level / 2) / (2 * math.pi)
    reference = density * velocity_product - nu * nu
    model = GaussianUpCrossings(r_squared_exp, u=level, sigma=1.0, tau=1.0)
    assert model.upcrossing_integrand(
        torch.tensor(lag, dtype=torch.float64)
    ).item() == pytest.approx(reference, abs=2e-10)


@pytest.mark.parametrize("total", [False, True])
def test_mean_level_and_sign_symmetry(total):
    model = GaussianUpCrossings(r_squared_exp, sigma=1.0, tau=1.0)
    lag = torch.tensor([0.1, 0.5, 1.0, 3.0], dtype=torch.float64)
    prefix = "crossing" if total else "upcrossing"
    fn = getattr(model, prefix + "_integrand")
    torch.testing.assert_close(fn(lag, u=0), getattr(model, prefix + "_integrand_mean_level")(lag))
    torch.testing.assert_close(fn(lag, u=-2), fn(lag, u=2))


@pytest.mark.filterwarnings("ignore:Nonfinite small-lag samples:RuntimeWarning")
def test_time_and_amplitude_scaling():
    base = GaussianUpCrossingsDimless(r_squared_exp, u=0.5, tau=1.0, sigma=1.0)
    scaled = GaussianUpCrossingsDimless(r_squared_exp, u=1.0, tau=3.0, sigma=2.0)
    torch.testing.assert_close(scaled.upcrossing_mean_rate() * 3, base.upcrossing_mean_rate())
    torch.testing.assert_close(scaled.upcrossing_variance(30.0), base.upcrossing_variance(10.0))
    torch.testing.assert_close(
        scaled.upcrossing_fano_factor_CLT(), base.upcrossing_fano_factor_CLT()
    )


def test_parameter_gradient_and_broadcasting():
    h = torch.tensor([[-1.3], [0.4]], dtype=torch.float64, requires_grad=True)
    a = torch.tensor([[0.2, 0.7, 1.1]], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(owensT, (h, a))
    assert torch.autograd.gradgradcheck(owensT, (h, a))

    def value(sigma, level):
        model = GaussianUpCrossings(r_squared_exp, u=level, sigma=sigma, tau=1.0)
        return model.upcrossing_integrand(torch.tensor(0.7, dtype=torch.float64))

    assert torch.autograd.gradcheck(
        value,
        (
            torch.tensor(1.2, dtype=torch.float64, requires_grad=True),
            torch.tensor(0.5, dtype=torch.float64, requires_grad=True),
        ),
    )


def test_import_does_not_change_torch_default_dtype():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import torch; torch.set_default_dtype(torch.float32); import gaussian_crossings; assert torch.get_default_dtype() == torch.float32",
        ],
        check=True,
    )


def test_invalid_process_and_quadrature_inputs():
    with pytest.raises(ValueError, match="smooth"):
        GaussianUpCrossings(r_OU, sigma=1.0, tau=1.0)
    model = GaussianUpCrossings(r_squared_exp, sigma=1.0, tau=1.0)
    assert model.upcrossing_variance(0.0) == 0
    for options in ({"num_points": 1}, {"epsilon_left": 0}, {"epsilon_left": 0.9}):
        with pytest.raises(ValueError):
            model.upcrossing_variance(1.0, method="trapezoid", **options)
    with pytest.raises(ValueError):
        GaussianUpCrossingsDimless(r_squared_exp, sigma=1.0, tau=-1.0)


@pytest.mark.parametrize("level", [0.0, 0.5, 2.0])
def test_alternating_crossings_give_twice_the_long_time_fano_factor(level):
    model = GaussianUpCrossings(r_squared_exp, u=level, sigma=1.0, tau=1.0)
    # The identity is exact in the paper; allow for finite-grid truncation here.
    options = dict(epsilon_left=0.001, epsilon_right=1e-5, num_points=16000)
    up = model.upcrossing_fano_factor_CLT(**options).item()
    total = model.crossing_fano_factor_CLT(**options).item()
    assert total == pytest.approx(2 * up, abs=2e-6)
