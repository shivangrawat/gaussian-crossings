"""matplotlib is optional: importing the package must not require it."""

import subprocess
import sys

import numpy as np
import pytest


def test_core_import_does_not_load_matplotlib():
    code = (
        "import sys, gaussian_crossings, gaussian_crossings.utils, gaussian_crossings.process; "
        "print('matplotlib' in sys.modules)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "False"


def test_midpoint_normalize_is_available_from_old_and_new_paths():
    pytest.importorskip("matplotlib")
    from gaussian_crossings.plotting import MidpointNormalize
    from gaussian_crossings.utils import MidpointNormalize as legacy
    from gaussian_crossings.utils.utils import MidpointNormalize as deep

    assert legacy is MidpointNormalize
    assert deep is MidpointNormalize
    norm = MidpointNormalize(vmin=0.5, vmax=3.0, midpoint=1.0)
    assert float(norm(1.0)) == pytest.approx(0.5)


def test_midpoint_clipping_and_call_override():
    pytest.importorskip("matplotlib")
    from gaussian_crossings.plotting import MidpointNormalize

    norm = MidpointNormalize(vmin=0.5, vmax=3.0, midpoint=1.0)
    np.testing.assert_allclose(norm([0.0, 1.0, 4.0]), [-0.5, 0.5, 1.25])
    np.testing.assert_allclose(norm([0.0, 1.0, 4.0], clip=True), [0.0, 0.5, 1.0])
    norm.clip = True
    assert norm(4.0) == 1.0
    assert norm(4.0, clip=False) == 1.25


@pytest.mark.parametrize(
    "bounds,midpoint,colors",
    [
        ((0.5, 3.0), 1.0, [0.0, 1.0]),
        ((2.0, 3.0), 1.0, [0.5, 1.0]),
        ((-2.0, -1.0), 1.0, [0.0, 0.5]),
        ((1.0, 3.0), 1.0, [0.5, 1.0]),
        ((-2.0, 1.0), 1.0, [0.0, 0.5]),
    ],
)
def test_midpoint_inverse_preserves_asymmetric_and_one_sided_scales(bounds, midpoint, colors):
    pytest.importorskip("matplotlib")
    from gaussian_crossings.plotting import MidpointNormalize

    norm = MidpointNormalize(vmin=bounds[0], vmax=bounds[1], midpoint=midpoint)
    np.testing.assert_allclose(norm(bounds), colors)
    values = np.linspace(bounds[0] - 1, bounds[1] + 1, 17).reshape(1, 17)
    np.testing.assert_allclose(norm.inverse(norm(values)), values, atol=1e-14)
    if bounds[0] < midpoint < bounds[1]:
        assert norm.inverse(0.5) == midpoint


def test_midpoint_normalization_and_inverse_preserve_mask():
    pytest.importorskip("matplotlib")
    from gaussian_crossings.plotting import MidpointNormalize

    norm = MidpointNormalize(vmin=0.5, vmax=3.0, midpoint=1.0)
    values = np.ma.array([[0.0, 1.0], [2.0, 4.0]], mask=[[True, False], [False, True]])
    normalized = norm(values)
    np.testing.assert_array_equal(normalized.mask, values.mask)
    np.testing.assert_array_equal(norm(values, clip=True).mask, values.mask)
    restored = norm.inverse(normalized)
    np.testing.assert_array_equal(restored.mask, values.mask)
    np.testing.assert_allclose(restored.compressed(), values.compressed())
    assert np.ndim(norm(1.0)) == 0
    assert np.ndim(norm.inverse(0.5)) == 0


def test_midpoint_masked_scalars_preserve_mask_without_warnings(recwarn):
    pytest.importorskip("matplotlib")
    from gaussian_crossings.plotting import MidpointNormalize

    norm = MidpointNormalize(vmin=0.5, vmax=3.0, midpoint=1.0)
    for value in (np.ma.masked, np.ma.array(1.0, mask=True)):
        assert np.ma.is_masked(norm(value))
        assert np.ma.is_masked(norm(value, clip=True))
        assert np.ma.is_masked(norm.inverse(value))
    assert not recwarn


@pytest.mark.parametrize("value,color", [(0.0, 0.25), (1.0, 0.5), (2.0, 0.75)])
def test_midpoint_constant_data_normalization_and_colorbar(value, color):
    pytest.importorskip("matplotlib")
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    from gaussian_crossings.plotting import MidpointNormalize

    data = np.ma.array(np.full((2, 2), value), mask=[[False, True], [False, False]])
    norm = MidpointNormalize(vmin=data.min(), vmax=data.max(), midpoint=1.0)
    for clip in (False, True):
        normalized = norm(data, clip=clip)
        np.testing.assert_allclose(normalized.compressed(), color)
        np.testing.assert_array_equal(normalized.mask, data.mask)
        restored = norm.inverse(normalized)
        np.testing.assert_allclose(restored.compressed(), value)
        np.testing.assert_array_equal(restored.mask, data.mask)
    assert norm(value) == color
    assert norm.inverse(color) == value

    figure = Figure()
    canvas = FigureCanvasAgg(figure)
    axis = figure.subplots()
    artist = axis.imshow(data, norm=norm, cmap="PiYG")
    figure.colorbar(artist, ax=axis)
    canvas.draw()
    assert np.isfinite(norm(data).compressed()).all()
    assert norm(value) == pytest.approx(color)
