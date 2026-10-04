# Paper reproduction

This directory reproduces the figures of S. Rawat, F. Morone, D. J. Heeger, and S. Martiniani,
*Exact Variance and Fano Factor for Arbitrary Level Crossings in Stationary Gaussian Processes*.
It is separate from the package and its user [tutorials](../examples/README.md).

| Path | Contents |
|---|---|
| `PRE_regenerated_figures.ipynb` | Entry point: recalculates and exports Figures 2–6 |
| `pre_figures.py` | Validated calculations and plotting for all figures |
| `damped_harmonic_oscillator/pre_figure3.py` | Figure 3 simulations, bootstrap intervals, and checks |
| `restore_pre_archive.py` | Restores the archived arrays and crossing counts into `data/` |
| `reproduction/` | Checksum-verified archive of the numerical inputs for Figures 2–6 |
| `fig1_illustration.ipynb`, `*/` notebooks | Original figure notebooks, kept as provenance records |

Three of the original figure notebooks use the deprecated `GaussianUpCrossingsDimless` class
and show a `DeprecationWarning` when re-run; this is expected. `PRE_regenerated_figures.ipynb`
uses the current interface. The historical notebooks retain their numerical code and saved
outputs; archival notes and corrected comments explain how they differ from the current workflow.

The exact code used for the revised manuscript is tagged
[`pre-revision-2026-09-27`](https://github.com/shivangrawat/gaussian-crossings/tree/pre-revision-2026-09-27).
Step-by-step instructions are in the
[reproduction guide](../docs/reproducibility.md).
