# Reproducing the paper

Everything specific to the paper is in the
[`paper/`](https://github.com/shivangrawat/gaussian-crossings/tree/main/paper) directory. Its
numerical Figures 2–6 are produced by `paper/pre_figures.py`, orchestrated by
`paper/PRE_regenerated_figures.ipynb`. Figure 1 and Supplemental Figure S1 use the original
illustrations; the runner can export alternatives for those two.

## Exact code used for the paper

The release tag
[`pre-revision-2026-09-27`](https://github.com/shivangrawat/gaussian-crossings/tree/pre-revision-2026-09-27)
contains the exact code that produced the figures of the revised manuscript (dated 27 September
2026), with the scripts under `examples/` and the archive under `reproduction/`. Check it out
for a byte-for-byte reproduction:

```bash
git checkout pre-revision-2026-09-27
```

Later versions of the package give the same numbers to within the stated integration
tolerances. Because the runner records hashes of the package source, a cached numerical archive
made with one version is not re-exported by a version with different source; recalculate into a
fresh directory instead.

## Environment and input data

Run `uv sync --all-extras --locked` from the repository root. Figure text uses LaTeX, so install a
TeX distribution with the Matplotlib font packages. `latexmk` is needed only for the optional
manuscript comparison.

Generated arrays and trajectory archives live in the ignored `data/` directory. The repository
ships the original inputs in a checksum-verified
[reproduction bundle](https://github.com/shivangrawat/gaussian-crossings/tree/main/paper/reproduction).
Restore them with

```bash
uv run python paper/restore_pre_archive.py
```

and re-export the figures from the archived arrays with

```bash
uv run python paper/pre_figures.py plot \
  --output data/pre_figures_20260927 --figures figures/publication --historical
```

The main analytical benchmark tests need no archive. Full figure generation requires the Figure 3
archive at `data/pre_figure3_10000`.

## Regenerating the Figure 3 simulations

To create a new ensemble, use a fresh output directory. Keep the stated seed, batch size (20 by
default), step, and trial count. A separate seeded stream is used for each damping value and
batch.

```bash
uv run python paper/damped_harmonic_oscillator/pre_figure3.py validate
uv run python paper/damped_harmonic_oscillator/pre_figure3.py simulate \
  --output data/pre_figure3_new --trials 10000 --dt 0.00125 --seed 20260927 --workers 3
uv run python paper/damped_harmonic_oscillator/pre_figure3.py analyze \
  --output data/pre_figure3_new --bootstrap 10000
```

This is the full simulation: stationary initial conditions, exact Gaussian transitions, nested
sampling grids, and whole-trial bootstrap resampling. The 95% intervals are pointwise.

A production run contains `counts_0.npz` through `counts_4.npz` (whole-trial counts for three
thresholds and three nested grids, plus stationary covariance probes and initial velocities),
`run.json` (parameters, versions, source hash, seeds, and validation results), `run_source.py`
(the source snapshot), and `summary.json` (statistics, bootstrap intervals, theoretical curves,
and convergence diagnostics).

The manuscript ensemble was generated as two seeded blocks of 5,000 trials per damping value; the
second block was added after the first had been analyzed, and all 10,000 trials are used.
Reanalysis of an archived run must use the source matching its recorded hash, which the archived
`run_source.py` provides.

## Calculating all figures

```bash
# New calculation: validates first and refuses to overwrite arrays.npz.
uv run python paper/pre_figures.py all \
  --output data/pre_figures_new --figures figures/publication

# Validation only.
uv run python paper/pre_figures.py validate
```

To compare a fresh calculation with the restored manuscript arrays, add
`--reference data/pre_figures_20260927`. Every stored grid point, all Figure 3 theory curves,
coordinates, and masked boundaries are compared, with absolute tolerances of `2e-8` for Fano
grids and `1e-8` for Figure 3 statistics; the result is saved in `comparison_to_reference.json`.

## Figure map

| Figure | Export | Numerical content | Source notebook |
|---|---|---|---|
| 1 (alternative) | `upcrossing_description.pdf` | Fixed-seed stationary oscillator trajectories | `paper/fig1_illustration.ipynb` |
| 2 | `sdho_phase.pdf` | 250 x 250 threshold/damping grid | `paper/damped_harmonic_oscillator/sdho_zeta.ipynb` |
| 3 | `sdho_comparison.pdf` | `T = 120` predictions; 10,000 trials at each of five damping values | `paper/damped_harmonic_oscillator/theory_simulation_upcrossings.ipynb` |
| 4a–c | `sdho_fanos.pdf` | Frequency and threshold scan at three damping ratios | `paper/damped_harmonic_oscillator/fano_scan_omega0.ipynb` |
| 4d–f | `sdho_fanos.pdf` | Temperature and threshold scan; zero temperature masked | `paper/damped_harmonic_oscillator/fano_scan_temp.ipynb` |
| 5 | `OU_noise_phase.pdf` | Ornstein–Uhlenbeck-driven mean reversion | `paper/OU_noise/OU_noise.ipynb` |
| 6 | `rational_quadratic_fano.pdf` | Four rational-quadratic shapes and the squared-exponential limit | `paper/rational_quadratic/rational_quadratic.ipynb` |
| S1 (alternative) | `positive_quadrant_new.pdf` | Integration-region change of variables | reconstructed by `paper/pre_figures.py` |

The figure notebooks document the earlier figure calculations and parameters; their meshes and
simulation settings differ from the revised manuscript. `paper/PRE_regenerated_figures.ipynb` is
the reproduction entry point.

## What is checked

All numerical figure grids and all 101 finite-window Figure 3 curve positions are recalculated
through the package. Archived trial counts and bootstrap intervals are preserved. The runner
checks small-lag covariance expressions with high precision, conditional-Gaussian pair intensities
independently of the Owen's-T expression, sign symmetry, the Ornstein–Uhlenbeck and oscillator
rescaling, and finite-window Figure 3 theory. Representative integrals are repeated with tighter
tolerances and longer cutoffs.

```bash
uv run pytest -m 'not reproduction'  # portable suite, no generated datasets
uv run pytest -m reproduction        # full local archive check, or an explicit skip
```

## Optional historical comparison

A separate review command compares the original manuscript exports with the revised figures. It
requires the manuscript checkout and builds under the numerical archive's `build/` directory:

```bash
uv run python paper/pre_figures.py comparison \
  --output data/pre_figures_20260927 --figures figures/publication --historical \
  --paper /path/to/manuscript
```
