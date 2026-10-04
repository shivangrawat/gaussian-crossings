# Gaussian Crossings

**Exact variance and Fano factor of level crossings for smooth stationary Gaussian processes.**

[![Tests](https://github.com/shivangrawat/gaussian-crossings/actions/workflows/tests.yml/badge.svg)](https://github.com/shivangrawat/gaussian-crossings/actions/workflows/tests.yml)
[![Documentation](https://img.shields.io/badge/docs-online-blue)](https://shivangrawat.github.io/gaussian-crossings/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-3776AB)](https://github.com/shivangrawat/gaussian-crossings/blob/main/pyproject.toml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](https://github.com/shivangrawat/gaussian-crossings/blob/main/LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-2605.25278-b31b1b)](https://arxiv.org/abs/2605.25278)

Given the autocovariance $r(t)$ of a stationary Gaussian process and a threshold $u$, this package
computes the mean, the exact variance, and the Fano factor of the number of upcrossings,
downcrossings, or all crossings of $u$, for finite observation windows and in the long-time limit.
It also estimates the Fano factor from sampled data, with confidence intervals, so predictions
can be compared directly with measurements.

![Mean crossing rate, variance rate, and Fano factor of a damped harmonic oscillator as the threshold and damping vary.](https://raw.githubusercontent.com/shivangrawat/gaussian-crossings/main/docs/assets/sdho_phase.png)

*The mean crossing rate (left) depends only on $r(0)$ and $r''(0)$ and is the same for every
damping ratio. The Fano factor (right) depends on the whole covariance and separates regular
crossings ($F<1$, purple) from clustered ones ($F>1$, green).*

## Installation

```bash
pip install gaussian-crossings
```

Until the first release is on PyPI, install from GitHub with
`pip install "git+https://github.com/shivangrawat/gaussian-crossings"`. Python 3.10 or newer is
required. See the [installation guide](https://shivangrawat.github.io/gaussian-crossings/installation/)
for optional extras and a smaller CPU-only install.

## Example

```python
from gaussian_crossings import GaussianCrossings
from gaussian_crossings.process import r_damped_harmonic_oscillator_noise

model = GaussianCrossings(r_damped_harmonic_oscillator_noise, temp=1.0, omega0=1.0, zeta=0.5)

model.mean_rate(u=0.5)              # 0.1405 upcrossings per unit time
model.fano_factor(u=0.5)            # 0.4071, long-time Fano factor (regular crossings)
model.fano_factor(u=0.5, T=120.0)   # 0.4145, Fano factor of counts in windows of length 120
model.fano_factor(u=0.5, kind="total")
```

Thresholds can be arrays, and `kind` is `"up"`, `"down"`, or `"total"`. You can also
[use your own covariance function](https://shivangrawat.github.io/gaussian-crossings/covariances/).

## Finite windows versus the long-time limit

When comparing with data, use the Fano factor for the **window length of the data**,
`fano_factor(u, T=window)`, rather than the long-time limit. The two can differ noticeably when
windows span only a few correlation times. For a process with a rational-quadratic covariance
($\alpha = 0.75$, $\tau = 1$) observed at $u = 1.75$ in windows of length 100, artificial data give
1.25 (95% interval 1.14–1.37), the finite-window prediction is 1.23, and the long-time limit is
1.34 ([tutorial 03](https://github.com/shivangrawat/gaussian-crossings/blob/main/examples/03_fano_from_data.ipynb)).

## Estimating from data

```python
from gaussian_crossings import empirical_fano

estimate = empirical_fano(x, u=[0.5, 1.0, 1.5], window=50.0, dt=0.01, seed=0)
estimate.fano, estimate.ci_low, estimate.ci_high       # one value per threshold
model.fano_factor([0.5, 1.0, 1.5], T=estimate.window)  # matching prediction
```

Counts are formed in non-overlapping windows of the sampled series `x`. For nonzero-mean data,
compare a physical threshold `u` with the model at `u - mu`, where `mu` is the process mean.
`CoarseSamplingWarning` is a heuristic based on downsampling, applied when at least 50 crossings
are detected at a tested threshold. No warning does not establish adequate sampling resolution;
see the [estimation guide](https://shivangrawat.github.io/gaussian-crossings/estimation/).

## Learn more

- [Documentation](https://shivangrawat.github.io/gaussian-crossings/): concepts, covariance
  functions, numerical methods, and the API reference.
- [Tutorials](https://github.com/shivangrawat/gaussian-crossings/blob/main/examples/README.md): quick start, custom covariances, estimating from data, and
  telling models apart, all with artificial data.
- [Paper reproduction](https://github.com/shivangrawat/gaussian-crossings/blob/main/paper/README.md): scripts and archived data for the figures of the paper.
- [Changelog](https://github.com/shivangrawat/gaussian-crossings/blob/main/CHANGELOG.md).

## Citation

If you use this package, please cite

> S. Rawat, F. Morone, D. J. Heeger, and S. Martiniani, *Exact Variance and Fano Factor for
> Arbitrary Level Crossings in Stationary Gaussian Processes*,
> [arXiv:2605.25278](https://arxiv.org/abs/2605.25278) (2026).

BibTeX and software citation metadata are in the
[documentation](https://shivangrawat.github.io/gaussian-crossings/citation/) and
[`CITATION.cff`](https://github.com/shivangrawat/gaussian-crossings/blob/main/CITATION.cff).

## Development

```bash
git clone https://github.com/shivangrawat/gaussian-crossings.git
cd gaussian-crossings
uv sync --all-extras
uv run pytest
uv run ruff check gaussian_crossings tests
uv run mkdocs serve        # documentation preview
```

Released under the [MIT license](https://github.com/shivangrawat/gaussian-crossings/blob/main/LICENSE).
