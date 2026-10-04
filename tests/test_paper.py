"""Paper benchmarks run without private datasets or archived trajectories."""

import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("pre_figures", ROOT / "paper/pre_figures.py")
PRE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PRE)
REF = json.loads((Path(__file__).parent / "fixtures/paper_reference.json").read_text())


def test_sdho_paper_benchmarks():
    for row in REF["sdho_peaks"]:
        values, _ = PRE.sdho_fano(row["zeta"], [0.0, row["a_peak"]])
        np.testing.assert_allclose(values, [row["f_at_zero"], row["f_peak"]], atol=2e-9, rtol=0)
    values, _ = PRE.sdho_fano(1.0, [2.0])
    assert values[0] == pytest.approx(REF["sdho_zeta1_u2"]["correct"], abs=2e-9)


def test_rational_quadratic_and_squared_exponential_peaks():
    for row in REF["rq_peaks"]:
        shape = float(row["alpha"])
        values, _ = PRE.rq_fano(shape, [0.0, row["a_peak"]], end=4096.0)
        np.testing.assert_allclose(values, [row["f_at_zero"], row["f_peak"]], atol=2e-9, rtol=0)
        assert values[1] > 1


def test_ou_paper_curve():
    row = REF["ou_kappa_0.3"]
    kappa = 0.3
    zeta = (1 + kappa) / (2 * math.sqrt(kappa))
    levels = np.array(row["levels"]) * math.sqrt((1 + kappa) / kappa)
    actual, _ = PRE.sdho_fano(zeta, levels)
    np.testing.assert_allclose(actual, row["correct"], atol=2e-9, rtol=0)


def test_finite_window_figure3_theory():
    for row in REF["fig3_theory"]:
        actual = [PRE.fig3.theory(row["zeta"], level)[0] for level in [0.0, 0.25, 0.5]]
        np.testing.assert_allclose(actual, row["values"], atol=1e-8, rtol=0)


def test_all_retained_notebooks_are_figure_sources():
    expected = {value for value in PRE.NOTEBOOKS.values() if value is not None}
    expected.add("paper/PRE_regenerated_figures.ipynb")
    actual = {
        str(path.relative_to(ROOT))
        for path in (ROOT / "paper").rglob("*.ipynb")
        if ".ipynb_checkpoints" not in path.parts
    }
    assert actual == expected


@pytest.mark.reproduction
def test_full_local_archive_validation():
    if not (ROOT / "data/pre_figure3_10000/summary.json").exists():
        pytest.skip("Requires the untracked 50,000-trial Figure 3 archive")
    result = PRE.validate()
    assert result["fig3_counts_hashes_and_statistics_verified"]
    assert result["current_package_integrand_max_abs_error"] < 2e-9


def test_figure3_exact_transition_and_bootstrap_validation():
    assert PRE.fig3.validate()["status"] == "passed"


def test_cached_archive_verification_rejects_changed_source(tmp_path):
    import shutil

    source = ROOT / "paper/pre_figures.py"
    shutil.copyfile(source, tmp_path / "run_source.py")
    (tmp_path / "run.json").write_text(json.dumps({"source_sha256": PRE.digest(source)}))
    PRE.verify_archive(tmp_path)
    (tmp_path / "run_source.py").write_text(source.read_text() + "\n# changed\n")
    with pytest.raises(RuntimeError, match="source hash"):
        PRE.verify_archive(tmp_path)
