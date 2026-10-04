"""The recommended interface agrees with the original methods and returns NumPy."""

import numpy as np
import pytest
import torch

import gaussian_crossings
from gaussian_crossings import GaussianCrossings, GaussianUpCrossings
from gaussian_crossings.process import r_damped_harmonic_oscillator_noise, r_squared_exp


@pytest.fixture(scope="module")
def model():
    return GaussianCrossings(
        r_damped_harmonic_oscillator_noise, u=0.5, temp=1.0, omega0=1.0, zeta=0.5
    )


def test_alias_is_the_same_class(model):
    assert GaussianCrossings is GaussianUpCrossings
    assert isinstance(model, GaussianUpCrossings)


def test_scalar_results_are_floats_and_match_original_methods(model):
    assert isinstance(model.fano_factor(), float)
    assert model.fano_factor() == pytest.approx(float(model.upcrossing_fano_factor_CLT()))
    assert model.fano_factor(T=120.0) == pytest.approx(float(model.upcrossing_fano_factor(120.0)))
    assert model.mean_rate() == pytest.approx(float(model.upcrossing_mean_rate()))
    assert model.mean(120.0) == pytest.approx(float(model.upcrossing_mean(120.0)))
    assert model.variance(120.0) == pytest.approx(float(model.upcrossing_variance(120.0)))
    assert model.variance_rate() == pytest.approx(
        float(model.upcrossing_variance_CLT_per_unit_time())
    )


def test_array_thresholds_return_arrays(model):
    levels = np.linspace(0.0, 2.0, 7)
    fano = model.fano_factor(u=levels)
    assert isinstance(fano, np.ndarray) and fano.shape == levels.shape
    np.testing.assert_allclose(fano[3], model.fano_factor(u=levels[3]), rtol=1e-12)
    rates = model.mean_rate(u=levels.reshape(7, 1))
    assert rates.shape == (7, 1)


@pytest.mark.parametrize("kind", ["down", "downcrossing", "DOWN"])
def test_downcrossings_equal_upcrossings(model, kind):
    assert model.fano_factor(kind=kind) == pytest.approx(model.fano_factor(kind="up"))
    assert model.mean_rate(kind=kind) == pytest.approx(model.mean_rate(kind="up"))


@pytest.mark.parametrize("kind", ["total", "all", "crossings"])
def test_total_crossings(model, kind):
    assert model.mean_rate(kind=kind) == pytest.approx(2 * model.mean_rate())
    assert model.fano_factor(kind=kind) == pytest.approx(2 * model.fano_factor(), rel=1e-9)
    assert model.fano_factor(T=50.0, kind=kind) == pytest.approx(
        float(model.crossing_fano_factor(50.0))
    )


def test_invalid_kind_is_rejected(model):
    with pytest.raises(ValueError, match="kind must be"):
        model.fano_factor(kind="sideways")


def test_original_methods_still_return_tensors(model):
    assert isinstance(model.upcrossing_fano_factor_CLT(), torch.Tensor)


def test_repr_names_kernel_and_parameters(model):
    text = repr(model)
    assert text.startswith("GaussianUpCrossings(r_damped_harmonic_oscillator_noise")
    assert "zeta=0.5" in text and "u=0.5" in text


@pytest.mark.parametrize(
    "name", ["GaussianUpCrossingsDimless", "GaussianUpCrossingsDimless_minimal"]
)
def test_deprecated_classes_warn_but_still_work(name):
    assert name not in gaussian_crossings.__all__
    cls = getattr(gaussian_crossings, name)
    with pytest.warns(DeprecationWarning, match=f"{name} is deprecated"):
        legacy = cls(r_squared_exp, u=0.5, sigma=1.0, tau=2.0)
    current = GaussianCrossings(r_squared_exp, u=0.5, sigma=1.0, tau=2.0)
    assert float(legacy.upcrossing_mean_rate()) == pytest.approx(current.mean_rate())
