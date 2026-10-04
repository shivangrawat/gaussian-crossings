"""matplotlib is optional: importing the package must not require it."""

import subprocess
import sys

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
