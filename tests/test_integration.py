"""Accuracy, failure reporting, and compatibility of adaptive integration."""

import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.integrate import quad
from scipy.special import ndtr

from gaussian_crossings import GaussianUpCrossings, GaussianUpCrossingsDimless
from gaussian_crossings.formula.integration import rq_parts, sdho_parts
from gaussian_crossings.process import (
    r_damped_harmonic_oscillator_noise,
    r_filtered_OU,
    r_matern,
    r_OU_noise,
    r_rational_quadratic,
    r_squared_exp,
)

pytestmark = pytest.mark.filterwarnings(
    "ignore:GaussianUpCrossingsDimless.*is deprecated:DeprecationWarning"
)

ROOT = Path(__file__).resolve().parents[1]
REF = json.loads((ROOT / "tests/fixtures/paper_reference.json").read_text())
SPEC = importlib.util.spec_from_file_location("reference_pre", ROOT / "paper/pre_figures.py")
PRE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PRE)


@pytest.mark.parametrize("cls", [GaussianUpCrossings, GaussianUpCrossingsDimless])
def test_package_reproduces_paper_peaks(cls):
    for row in REF["sdho_peaks"]:
        model = cls(r_damped_harmonic_oscillator_noise, temp=1.0, omega0=1.0, zeta=row["zeta"])
        result = model.upcrossing_fano_factor_CLT(
            u=torch.tensor([0.0, row["a_peak"]], dtype=torch.float64), method="adaptive"
        )
        np.testing.assert_allclose(result, [row["f_at_zero"], row["f_peak"]], atol=2e-9, rtol=0)
        assert model.last_integration_info.estimated_error < 2e-9
    for row in REF["rq_peaks"]:
        shape = float(row["alpha"])
        kernel = r_squared_exp if math.isinf(shape) else r_rational_quadratic
        parameters = {} if math.isinf(shape) else {"alpha": shape}
        model = cls(kernel, sigma=1.0, tau=1.0, **parameters)
        result = model.upcrossing_fano_factor_CLT(
            u=torch.tensor([0.0, row["a_peak"]], dtype=torch.float64), method="adaptive"
        )
        np.testing.assert_allclose(result, [row["f_at_zero"], row["f_peak"]], atol=2e-9, rtol=0)
        assert result[1] > 1


def test_finite_window_paper_values():
    for row in REF["fig3_theory"]:
        model = GaussianUpCrossings(
            r_damped_harmonic_oscillator_noise, temp=1.0, omega0=1.0, zeta=row["zeta"]
        )
        u = torch.tensor([0.0, 0.25, 0.5], dtype=torch.float64)
        mean = model.upcrossing_mean(120.0, u=u)
        variance = model.upcrossing_variance(120.0, u=u, method="adaptive")
        fano = model.upcrossing_fano_factor(120.0, u=u, method="adaptive")
        np.testing.assert_allclose(
            torch.stack((mean, variance, fano), dim=-1), row["values"], atol=1e-8, rtol=0
        )


@pytest.mark.parametrize("kernel", [r_OU_noise, r_filtered_OU])
def test_dimensional_scaling_and_ou_mapping(kernel):
    u = torch.tensor([0.0, 0.5, 1.5], dtype=torch.float64)
    a = GaussianUpCrossings(kernel, sigma=1.7, tau=0.013, kappa=0.3)
    b = GaussianUpCrossingsDimless(kernel, sigma=1.7, tau=0.013, kappa=0.3)
    for name, args in [
        ("upcrossing_variance", (1.0,)),
        ("crossing_variance", (1.0,)),
        ("upcrossing_fano_factor_CLT", ()),
    ]:
        torch.testing.assert_close(
            getattr(a, name)(*args, u=u, method="adaptive"),
            getattr(b, name)(*args, u=u, method="adaptive"),
            atol=2e-10,
            rtol=2e-10,
        )


@pytest.mark.parametrize("zeta", [0.1, 0.5, 1.0, 2.0, 5.05, 50.0])
def test_small_lag_conditional_variances(zeta):
    # Independent high-precision covariance matrix, not a duplicated series.
    import mpmath as mp

    with mp.workdps(80):
        z = mp.mpf(str(zeta))

        def r(t):
            if z == 1:
                return (1 + t) * mp.exp(-t)
            w = mp.sqrt(z * z - 1)
            return mp.exp(-z * t) * (mp.cosh(w * t) + z / w * mp.sinh(w * t))

        for t in (1e-9, 1e-5, 0.01 / max(zeta, 1)):
            x = mp.mpf(str(t))
            rr, p, q = r(x), mp.diff(r, x), -mp.diff(r, x, 2)
            reference = [1 - rr, (1 - q - p * p / (1 + rr)) / 2, (1 + q - p * p / (1 - rr)) / 2]
            got = sdho_parts(t, zeta)[3:]
            np.testing.assert_allclose(got, [float(mp.re(v)) for v in reference], rtol=2e-8, atol=0)


@pytest.mark.parametrize("shape", [0.75, 1.0, 5.0, math.inf])
def test_rq_small_lag_covariance_and_high_levels(shape):
    for lag in (1e-9, 1e-5, 0.01, 1.0):
        np.testing.assert_allclose(
            rq_parts(lag, shape)[3:], PRE.rq_parts(lag, shape)[3:], rtol=2e-9, atol=0
        )
    kernel = r_squared_exp if math.isinf(shape) else r_rational_quadratic
    model = GaussianUpCrossings(
        kernel, sigma=1.0, tau=1.0, **({} if math.isinf(shape) else {"alpha": shape})
    )
    result = model.upcrossing_fano_factor_CLT(
        u=np.array([[-8.0, 0.0], [8.0, 40.0]]), method="adaptive"
    )
    assert result.shape == (2, 2)
    assert bool(torch.isfinite(result).all())
    assert result[0, 0] == result[1, 0]


def _independent_total_pair(t, u):
    # Condition on the two positions, then integrate E|V0*Vt| directly.
    r, p, _, d, a, b = PRE.rq_parts(t, math.inf)
    m = p * u / (1 + r)

    def integrand(z):
        v = m + math.sqrt(a) * z
        h = abs(v) / math.sqrt(b)
        mass = 2 * ndtr(h) - 1
        absolute_product = (b - v * v) * (1 - 2 * mass) + 4 * b * h * math.exp(
            -h * h / 2
        ) / math.sqrt(2 * math.pi)
        return absolute_product * math.exp(-z * z / 2) / math.sqrt(2 * math.pi)

    value = quad(integrand, -12, 12, points=[-m / math.sqrt(a)], epsabs=1e-11)[0]
    return value * math.exp(-u * u / (2 - d)) / (2 * math.pi * math.sqrt(d * (2 - d)))


@pytest.mark.parametrize("level", [0.0, 0.5, 2.0])
def test_total_finite_variance_against_independent_velocity_integral(level):
    T = 3.0
    model = GaussianUpCrossings(r_squared_exp, sigma=1.0, tau=1.0, u=level)
    nu = float(model.crossing_mean_rate())
    integral = quad(
        lambda t: (T - t) * (_independent_total_pair(t, level) - nu * nu),
        0.0,
        T,
        epsabs=2e-9,
        points=[0.01, 0.1, 1.0],
    )[0]
    expected = T * nu + 2 * integral
    got = model.crossing_variance(T, method="adaptive")
    assert float(got) == pytest.approx(expected, abs=3e-8)
    total = model.crossing_fano_factor_CLT(method="adaptive")
    up = model.upcrossing_fano_factor_CLT(method="adaptive")
    assert float(total) == pytest.approx(2 * float(up), abs=2e-10)


@pytest.mark.parametrize("T", [1e-8, 0.01, 2.0])
def test_custom_covariance_and_short_windows(T):
    def callback(t, sigma, tau):
        return sigma**2 * torch.exp(-((t / tau) ** 2) / 2)

    custom = GaussianUpCrossings(callback, sigma=1.3, tau=0.4)
    known = GaussianUpCrossings(r_squared_exp, sigma=1.3, tau=0.4)
    for prefix in ("upcrossing", "crossing"):
        a = getattr(custom, prefix + "_fano_factor")(T, u=0.5, method="adaptive")
        b = getattr(known, prefix + "_fano_factor")(T, u=0.5, method="adaptive")
        torch.testing.assert_close(a, b, rtol=2e-8, atol=2e-8)
    assert custom.upcrossing_fano_factor_CLT(u=0.5, method="adaptive") == pytest.approx(
        known.upcrossing_fano_factor_CLT(u=0.5, method="adaptive"), abs=2e-8
    )


def test_matern_custom_series():
    model = GaussianUpCrossings(r_matern, sigma=1.0, tau=1.0, nu=2.5)
    assert math.isfinite(float(model.upcrossing_fano_factor_CLT(method="adaptive")))


def test_invalid_options_and_nondifferentiable_path():
    model = GaussianUpCrossings(r_squared_exp, sigma=1.0, tau=1.0)
    for options in [
        {"method": "unknown"},
        {"epsabs": 0},
        {"epsrel": -1},
        {"limit": 1},
        {"cutoff": -1},
    ]:
        with pytest.raises(ValueError):
            model.upcrossing_fano_factor_CLT(**({"method": "adaptive"} | options))
    with pytest.raises(ArithmeticError, match="quadrature failed"):
        model.upcrossing_fano_factor_CLT(method="adaptive", limit=2)
    with pytest.raises(ValueError, match="T > 0"):
        model.upcrossing_fano_factor(0.0, method="adaptive")
    assert model.upcrossing_variance(0.0, method="adaptive") == 0
    with pytest.raises(ValueError, match="not differentiable"):
        model.upcrossing_fano_factor_CLT(u=torch.tensor(0.5, requires_grad=True), method="adaptive")
    level = torch.tensor(0.5, dtype=torch.float64, requires_grad=True)
    result = model.upcrossing_variance(2.0, u=level, method="trapezoid", epsilon_left=0.01)
    assert torch.isfinite(torch.autograd.grad(result, level)[0])


def test_nonintegrable_rq_tail_is_rejected():
    model = GaussianUpCrossings(r_rational_quadratic, sigma=1.0, tau=1.0, alpha=0.5, u=1.0)
    with pytest.raises(ValueError, match="diverges"):
        model.upcrossing_fano_factor_CLT(method="adaptive")
    assert torch.isfinite(model.upcrossing_variance(2.0, method="adaptive"))


def test_adaptive_default_and_legacy_selection():
    from gaussian_crossings import GaussianUpCrossingsDimless_minimal

    for cls in (GaussianUpCrossings, GaussianUpCrossingsDimless):
        model = cls(r_rational_quadratic, sigma=1.0, tau=1.0, alpha=1.0, u=2.0)
        default = model.upcrossing_fano_factor_CLT()
        assert model.last_integration_info.method == "adaptive"
        assert float(default) == pytest.approx(1.1199456330211173, abs=2e-9)
        legacy = model.upcrossing_fano_factor_CLT(method="trapezoid")
        assert model.last_integration_info is None
        assert float(legacy) == pytest.approx(1.1200499272101503, abs=2e-9)
        for prefix in ("upcrossing", "downcrossing", "crossing"):
            for suffix, args in [
                ("variance", (3.0,)),
                ("variance_CLT", (3.0,)),
                ("variance_CLT_per_unit_time", ()),
                ("fano_factor", (3.0,)),
                ("fano_factor_CLT", ()),
            ]:
                fn = getattr(model, prefix + "_" + suffix)
                torch.testing.assert_close(fn(*args), fn(*args, method="adaptive"))
    legacy_model = GaussianUpCrossingsDimless_minimal(r_squared_exp, sigma=1.0, tau=1.0)
    legacy_model.upcrossing_variance(2.0)
    assert legacy_model.last_integration_info is None
    legacy_model.upcrossing_variance(2.0, method="adaptive")
    assert legacy_model.last_integration_info.method == "adaptive"


@pytest.mark.parametrize("shape", [0.75, 1.0, 2.0, 5.0, math.inf])
def test_complete_rq_curve_with_tighter_tolerance(shape):
    kernel = r_squared_exp if math.isinf(shape) else r_rational_quadratic
    options = {} if math.isinf(shape) else {"alpha": shape}
    model = GaussianUpCrossings(kernel, sigma=1.0, tau=1.0, **options)
    levels = np.linspace(0.0, 8.0, 500)
    normal = model.upcrossing_fano_factor_CLT(u=levels)
    refined = model.upcrossing_fano_factor_CLT(u=levels, epsabs=2e-11, epsrel=2e-11, cutoff=4096.0)
    torch.testing.assert_close(refined, normal, atol=2e-9, rtol=0)
    assert model.last_integration_info.estimated_error <= 2e-11 * max(1.0, float(refined.max()))


def test_zero_duration_vector_thresholds():
    model = GaussianUpCrossings(r_squared_exp, sigma=1.0, tau=1.0)
    for method in (model.upcrossing_variance, model.crossing_variance):
        torch.testing.assert_close(method(0.0, u=[0.0, 0.5]), torch.zeros(2, dtype=torch.float64))
        for invalid in [[], [float("nan")]]:
            with pytest.raises(ValueError, match="thresholds"):
                method(0.0, u=invalid)
