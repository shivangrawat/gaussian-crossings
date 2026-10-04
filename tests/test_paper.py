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
@pytest.mark.filterwarnings("ignore:GaussianUpCrossingsDimless.*is deprecated:DeprecationWarning")
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


@pytest.mark.parametrize("relative_paths", [False, True])
def test_comparison_reports_the_selected_archives_validation(tmp_path, monkeypatch, relative_paths):
    import subprocess

    archive = tmp_path / "archive"
    archive.mkdir()
    (archive / "comparisons.json").write_text(json.dumps(REF))
    (archive / "validation.json").write_text(
        json.dumps(
            {
                "current_package_integrand_max_abs_error": 1.2345e-8,
                "positive_full_integral_max_abs_error": 7.949196856316121e-14,
                "sdho_cutoff_and_tolerance_max_change": 3.5678e-7,
                "rq_2048_to_4096_max_change": 4.6789e-6,
            }
        )
    )
    figures = tmp_path / "figures" / "publication"
    report_build = figures.parent / "build" / "figure_comparison"
    manuscript = tmp_path / "manuscript"
    manuscript.mkdir()

    def compile_report(command, **kwargs):
        working_dir = Path(kwargs["cwd"]).resolve()
        assert working_dir == manuscript
        source = working_dir / command[-1]
        assert source == report_build / "figure_comparison.tex"
        assert source.is_file()
        output_argument = next(
            arg.removeprefix("-outdir=") for arg in command if arg.startswith("-outdir=")
        )
        effective_build = working_dir / output_argument
        assert effective_build == report_build
        (effective_build / "figure_comparison.pdf").write_bytes(b"test PDF")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(PRE.subprocess, "run", compile_report)
    arguments = (archive, figures, manuscript)
    if relative_paths:
        monkeypatch.chdir(tmp_path)
        arguments = tuple(path.relative_to(tmp_path) for path in arguments)
    report = PRE.comparison(*arguments)
    text = (report_build / "figure_comparison.tex").read_text()
    for expected in (
        r"1.235\times10^{-8}",
        r"7.949\times10^{-14}",
        r"3.568\times10^{-7}",
        r"4.679\times10^{-6}",
    ):
        assert expected in text
    assert "@FULL_INTEGRAL_ERROR@" not in text
    assert report.is_absolute()
    assert report == figures.parent / "figure_comparison.pdf"
    assert report.read_bytes() == b"test PDF"
