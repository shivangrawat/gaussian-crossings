# Changelog

All notable changes to this project are documented here. The project follows
[semantic versioning](https://semver.org/).

## 0.2.0 (unreleased)

### Added

- `GaussianCrossings`, a shorter name for `GaussianUpCrossings`.
- Methods `mean_rate`, `mean`, `variance`, `variance_rate`, and `fano_factor` with
  `kind="up" | "down" | "total"`. `fano_factor(u, T=None)` gives the long-time limit and a finite
  `T` gives the ratio for windows of that length. They return floats or NumPy arrays.
- `windowed_counts` and `empirical_fano` for estimating Fano factors from sampled data, with
  bootstrap confidence intervals and a heuristic `CoarseSamplingWarning` based on downsampling.
- `gaussian_crossings.plotting` for optional plotting helpers.
- Four executed tutorials with artificial data, a documentation site, and a release workflow for
  PyPI.

### Changed

- matplotlib is now an optional dependency (`plot`, `notebooks`, `dev`, and `reproduce` extras).
  `MidpointNormalize` moved to `gaussian_crossings.plotting`; the old import paths still work.
- Covariance functions no longer accept unknown keyword arguments, so a misspelled parameter
  raises `TypeError` instead of being ignored.
- `simulate_gaussian_process_cholesky` takes `jitter` and `max_attempts` as keyword-only
  arguments, so positional covariance parameters are passed to the covariance function.
- Covariance callbacks written with NumPy raise an explanatory `TypeError`.
- The state-space model classes are named `DampedHarmonicOscillatorNoise`, `OUNoise`, and
  `FilteredOU`; the lowercase names remain as aliases.
- The paper reproduction scripts, figure notebooks, and archive moved from `examples/` and
  `reproduction/` to `paper/`.

### Fixed

- `q(0)` now follows the right-hand derivative convention used by `q0`, including covariance
  callbacks written with `abs(t)`.
- OU-driven covariances accept representable `kappa` values close to one and evaluate nearby
  unequal time scales without subtracting nearly equal exponentials.
- `MidpointNormalize` honors clipping, preserves masks, and supplies an inverse consistent with
  its asymmetric and one-sided color scales. Equal bounds support constant-data plots.
- The figure-comparison report reads numerical validation errors from the selected archive and
  resolves relative paths before compiling from the manuscript checkout.
- Documentation and tutorials clarify model assumptions, mean-centered thresholds, bootstrap
  dependence, sampling diagnostics, and the historical reproduction workflow.
- The overdamped oscillator covariance lost precision near critical damping, and long-time Fano
  factors failed for damping ratios such as `1 + 1e-12`.
- With `method="trapezoid"`, the original variance and Fano methods broadcast an array of
  thresholds against the integration grid: they raised a shape error or, for arrays as long as the
  grid, returned incorrect values. They now raise `ValueError` for more than one threshold, and
  the new methods evaluate the thresholds one at a time.

### Deprecated

- `GaussianUpCrossingsDimless` and `GaussianUpCrossingsDimless_minimal`. Use `GaussianCrossings`
  and pass `tau` to covariances that support it; otherwise wrap the callback to evaluate at
  `t / tau`.

## 0.1.0 (2026-09-27)

Code accompanying the revision of the paper, tagged `pre-revision-2026-09-27`.

- Adaptive, error-controlled integration became the default; the original fixed-grid rule is
  available as `method="trapezoid"`. The paper figure runner uses the same implementation as the
  package.
- The error-function argument in the variance integrand keeps its sign; earlier code used its
  absolute value, which lowered the integrand where the argument was negative.
- Refactor preserving 60 representative earlier numerical results, with fixes for scalar
  covariance callbacks in FFT simulation, invalid covariance embeddings, broadcast gradients of
  Owen's T function, consistent crossing-count and crossing-time endpoints, SDE time labels and
  state dimension, and explicit precision.
- Reproduction archive of the arrays and crossing counts behind Figures 2–6 of the paper.
