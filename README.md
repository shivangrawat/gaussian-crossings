# Gaussian Crossings

A Python library implementing **exact analytical expressions for the variance and Fano factor** of arbitrary level crossings in smooth, stationary Gaussian processes.

## Overview

While the celebrated Kac-Rice formula (1944) gives the **mean** number of level crossings, the mean depends only on the *local* properties of the autocorrelation function at the origin — `r(0)` and `r''(0)` — and is completely blind to the full correlation structure of the process.

This library implements the central results of:

> **Exact Variance and Fano Factor for Arbitrary Level Crossings in Stationary Gaussian Processes**
> Shivang Rawat, Flaviano Morone, David J. Heeger, and Stefano Martiniani

which provide, for the first time, **closed-form single-integral expressions for the variance** of upcrossings, downcrossings, and total crossings at an arbitrary threshold level `u`. Unlike the mean, the variance probes the *entire* correlation structure at all lag times, capturing the clustering or regularity of crossing events.

### Key insight

Two processes with identical `r(0)` and `r''(0)` (and hence identical mean crossing rates) can have dramatically different variance and Fano factor. For example, the stochastic damped harmonic oscillator has a mean upcrossing rate that is completely independent of the damping ratio `zeta`, yet:

- **Underdamped** (`zeta < 1`): oscillatory correlations impose regularity on crossing times, producing **sub-Poissonian** statistics (`F < 1`, anti-bunching)
- **Overdamped** (`zeta > 1`): long-lived monotonic correlations cause clustering, producing **super-Poissonian** statistics (`F > 1`, bunching)

The Fano factor `F = Var[N] / E[N]` quantifies this: `F = 1` for Poisson-distributed crossings, `F < 1` for regular crossings, and `F > 1` for clustered crossings.

## Mathematical Background

### Mean (Kac-Rice formula)

For a zero-mean stationary Gaussian process with autocorrelation `r(t)`:

$$\mathbb{E}[N_u^{\uparrow}] = T \cdot \frac{1}{2\pi}\sqrt{\frac{-r''(0)}{r(0)}} \exp\left(-\frac{u^2}{2r(0)}\right)$$

### Variance (this work)

$$\text{Var}[N_u^{\uparrow}] = \mathbb{E}[N_u^{\uparrow}] + 2T \int_0^T \left(1 - \frac{t}{T}\right) I^{\uparrow}(t) \, dt$$

where the integrand $I^{\uparrow}(t)$ is expressed in closed form via the **error function** and **Owen's T function** through auxiliary quantities $\alpha, \beta, \gamma, \delta$ that depend on $r(t)$, $r'(t)$, $r''(t)$, and the threshold $u$.

### Fano Factor

$$F_u^{\uparrow} = 1 + 4\pi \sqrt{\frac{r(0)}{-r''(0)}} \exp\left(\frac{u^2}{2r(0)}\right) \int_0^\infty I^{\uparrow}(t) \, dt$$

In the limit $u/\sqrt{r(0)} \to \infty$, crossings become rare and uncorrelated, so $F \to 1$ (Poisson limit).

## Installation

### Using uv (Recommended)

[uv](https://docs.astral.sh/uv/) is a fast Python package manager. Install it first if you haven't:

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Then clone and install the package:

```bash
git clone https://github.com/yourusername/gaussian-crossings.git
cd gaussian-crossings

# Create virtual environment and install dependencies
uv sync

# For development (includes Jupyter support)
uv sync --extra dev
```

To run scripts or notebooks:

```bash
# Run a Python script
uv run python your_script.py

# Start Jupyter
uv run jupyter notebook
```

### Using pip

```bash
git clone https://github.com/yourusername/gaussian-crossings.git
cd gaussian-crossings
pip install -e .

# For development
pip install -e ".[dev]"
```

## Repository Structure

```text
gaussian-crossings/
├── gaussian_crossings/           # Main package
│   ├── formula/                  # Analytical formulae (Theorems 1 & 2)
│   │   ├── formula.py            # GaussianUpCrossings class
│   │   ├── formula_dimless.py    # Dimensionless formulation
│   │   └── formula_dimless_minimal.py
│   ├── process/                  # Stochastic process definitions
│   │   ├── correlation_functions.py  # Correlation functions (7 types)
│   │   └── dynamical_equations.py    # SDE definitions for simulation
│   └── utils/                    # Utilities
│       ├── simulation.py         # FFT/Cholesky-based GP simulation
│       ├── utils.py              # Crossing detection, SDE integration
│       └── owensT.py             # Owen's T function (PyTorch, with autograd)
├── examples/                     # Jupyter notebooks with applications
└── figures/                      # Generated figures
```

## Quick Start

### Computing Variance and Fano Factor

```python
import torch
from gaussian_crossings.formula.formula import GaussianUpCrossings
from gaussian_crossings.process.correlation_functions import r_damped_harmonic_oscillator_noise

# Stochastic damped harmonic oscillator (underdamped)
crossings = GaussianUpCrossings(
    r_func=r_damped_harmonic_oscillator_noise,
    u=0.5,           # threshold level
    temp=1.0,         # temperature
    omega0=1.0,       # natural frequency
    zeta=0.5          # damping ratio (underdamped)
)

T = 100.0  # observation time

mean = crossings.upcrossing_mean(T)
variance = crossings.upcrossing_variance(T)
fano = crossings.upcrossing_fano_factor_CLT()

print(f"Mean crossings: {mean:.4f}")
print(f"Variance: {variance:.4f}")
print(f"Fano factor: {fano:.4f}")   # < 1 for underdamped (sub-Poisson)
```

### Simulating and Counting Crossings

```python
import numpy as np
from gaussian_crossings.utils.simulation import simulate_gaussian_process_fft
from gaussian_crossings.utils.utils import count_upcrossings
from gaussian_crossings.process.correlation_functions import r_OU

# Simulate an Ornstein-Uhlenbeck process
t, X = simulate_gaussian_process_fft(r_OU, T=100.0, dt=0.01, sigma=1.0, tau=1.0)

# Count crossings at a given threshold
threshold = 0.5
n_up = count_upcrossings(X, threshold)
print(f"Number of upcrossings: {n_up}")
```

## Supported Correlation Functions

| Function | Description |
| ---------- | ------------- |
| Damped Harmonic Oscillator | Underdamped, critically damped, overdamped regimes |
| Ornstein-Uhlenbeck | Exponentially decaying correlation |
| Mean-Reverting with OU Noise | Bi-exponential correlation (cascaded filtering) |
| Filtered OU | Cascaded exponential filtering |
| Rational Quadratic | Flexible kernel with scale mixture interpretation |
| Squared Exponential | Infinitely differentiable (smooth) processes |
| Matern | Parameterized smoothness |

## Simulation Methods

- **FFT-based circulant embedding**: O(N log N) for large-scale simulations
- **Cholesky factorization**: Exact O(N^3) sampling for smaller systems
- **Euler-Maruyama SDE integration**: For dynamical system models

## Examples

The `examples/` directory contains Jupyter notebooks demonstrating:

| Category | Description |
|----------|-------------|
| `sdho_*.ipynb` | Stochastic damped harmonic oscillator: variance, Fano factor, and comparison with simulations across damping regimes |
| `ou_noise_*.ipynb` | Mean-reverting process with OU noise: crossing statistics as a function of timescale ratio |
| `rational_quadratic_*.ipynb` | Rational quadratic kernel: effect of shape parameter on Fano factor |
| `kohn_data_*.ipynb` | Analysis of neurophysiological data |
| `spike_train_*.ipynb` | Point process / spike train analysis |

## PRE Figure 3: confidence intervals

`examples/damped_harmonic_oscillator/pre_figure3.py` reproduces the revised
mean, variance, and finite-window variance/mean comparison. From the repository
root, using the project's Python environment:

```sh
python examples/damped_harmonic_oscillator/pre_figure3.py validate
python examples/damped_harmonic_oscillator/pre_figure3.py simulate --output data/pre_figure3_10000 --trials 10000 --dt 0.00125 --seed 20260927 --workers 3
python examples/damped_harmonic_oscillator/pre_figure3.py analyze --output data/pre_figure3_10000 --bootstrap 10000
python examples/damped_harmonic_oscillator/pre_figure3.py plot --output data/pre_figure3_10000
```

The run contains 10,000 independent stationary trajectories at each of five
logarithmically spaced damping ratios from 0.5 to 2, with omega0 = temperature = 1
and endpoints exactly at 0 and 120 seconds. Each trajectory supplies crossing
counts at all three thresholds (0, 0.25, 0.5) and at steps 0.00125, 0.0025, and
0.005 seconds. The finest grid supplies the plotted points. The initial
5,000-trial ensemble was extended uniformly to 10,000 at every damping value;
the first 5,000 trials are retained exactly.

Simulation uses the exact Gaussian transition of the linear position/velocity
SDE, with stationary covariance equal to the identity. A block matrix
exponential computes the innovation covariance without cancellation. A
second-order filter accelerates the same state recurrence; validation compares
it against direct matrix updates with identical Gaussian increments. There is
no FFT covariance embedding, burn-in, or numerical SDE integration error at
the sampled times. Missed crossings between samples still require grid checks.

The theory uses the finite-window integral at T = 120. Conditional variances
are evaluated with a covariance Taylor series near zero lag, including the
finite positive SDHO pair-intensity limit. Elsewhere the exact covariance is
used. The paper's Owen's-T expression is checked against an independent,
positive, one-dimensional conditional-Gaussian integral; adaptive quadrature
is checked at a tighter tolerance. Invalid variances, NaNs, or failed
quadrature raise errors instead of being discarded.

The plotted intervals are pointwise 95% percentile bootstrap intervals from
10,000 resamples of **whole independent trial rows**. Every resample recomputes
the mean, sample variance (ddof=1), and their ratio. Thus dependence across
thresholds, grids, and mean/variance estimates is retained. Intervals quantify
sampling uncertainty, not discretization error, and are not simultaneous bands.
See SciPy's [bootstrap documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html)
for the percentile and paired-resampling definitions, and its
[filter documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.lfilter.html)
for the recurrence used by the accelerated simulator.

The ignored output directory contains five `counts_*.npz` files with per-trial
counts, stationary covariance probes, initial velocities, and seeds;
`run.json` records parameters, dependency versions, source hash, and deterministic
validation; `summary.json` records estimates, intervals, theoretical values,
paired grid differences, and covariance diagnostics; `sdho_comparison.pdf` is
the vector figure. Existing run directories are never overwritten. A changed
source hash requires a provenance review before reanalysis. The source that generated
the saved 10,000-trial run is preserved as `data/pre_figure3_10000/run_source.py`;
use that snapshot for reanalysis of these archived counts, and the current
`pre_figure3.py plot` command for the updated manuscript styling. The styling
revision changes only the plotting function. Archive these files with the source
version when sharing the numerical data.

Before plotting, the pipeline checks stationary moments against their sampling
errors, requires the 95% paired interval for the fine-versus-twice-step change
to lie within one quarter of the plotted interval's half-width, and bounds the
remaining mean-count bias using the exact discrete-grid crossing probability.
These checks concern numerical accuracy; theory lying inside every individual
confidence interval is not a pass condition.

## Validated regeneration of all PRE figures

`examples/PRE_regenerated_figures.ipynb` is the review notebook for the six
main figures and supplemental integration-region schematic. It calls
`examples/pre_figures.py`; the original notebooks remain unchanged and are
mapped, with source hashes, in the calculation's `run.json`.

The original notebooks import the signed-erf formula classes now present in
this repository. Their default transformed trapezoidal quadrature is not used
for these exports: the runner uses normalized conditional variances, stable
small-lag Taylor expansions and checked adaptive integration. The RQ tail
includes analytic integrals of the first two powers of the covariance, with
cutoff doubling as a numerical check. Independent positive Gaussian integrals,
high-precision small-lag covariance checks, both package formula classes,
sign symmetry and the OU/SDHO rescaling are checked before calculation.
Invalid covariances or quadrature failures raise errors.

With the paper repository beside this checkout, run from this directory:

```sh
PYTHONPATH=. python examples/pre_figures.py all --output data/pre_figures_new --figures ../upcrossing_theory_tex/pre_rebuttal/regenerated_figures
```

Use a fresh output directory for a new calculation. To re-export the checked
archive created on 27 September 2026, execute the notebook or use:

```sh
PYTHONPATH=. python examples/pre_figures.py plot --output data/pre_figures_20260927 --figures ../upcrossing_theory_tex/pre_rebuttal/regenerated_figures
PYTHONPATH=. python examples/pre_figures.py comparison --output data/pre_figures_20260927 --figures ../upcrossing_theory_tex/pre_rebuttal/regenerated_figures
```

Requirements beyond the main package: mpmath for high-precision validation;
nbformat, nbclient and ipykernel for notebook execution; LaTeX for figure fonts
and latexmk for the comparison PDF. Matplotlib's TeX cache and comparison
build products are kept under `build/` directories.

`data/pre_figures_20260927/` contains arrays, validation results, numerical
comparisons, source snapshots/hashes and export hashes. Its `build/` directory
contains the executed notebook. The current renderer's hash is recorded
separately in `render.json`, since plot-only refinements do not change the
archived numerical source. The notebook verifies the archived source and
numerical function definitions before reusing cached arrays.

The physical parameter ranges and RQ curve set match the original notebooks.
Fig. 4 and Fig. 5 heatmaps use denser meshes; every valid Fig. 4 pixel is
integrated directly. The zero-temperature boundary is explicitly masked because
its crossing Fano ratio is undefined. Fig. 1 uses a fixed-seed exact stationary
SDHO realization; Fig. S1 is reconstructed from its change of variables because
no original source notebook was found. Fig. 3 reuses all 50,000 archived trials
from `data/pre_figure3_10000/`, checking count hashes and estimates without
changing any trials or bootstrap intervals. That archive is a prerequisite;
the earlier Figure 3 section gives its reproduction commands.

The comparison PDF quantifies the changed interpretations: the underdamped
SDHO can exceed F=1 at higher thresholds; the plotted OU reentrant return
below one disappears; all five RQ curves have resolved maxima above one.
The manuscript and rebuttal require a separate prose revision before submission.

## Applications

The exact variance and Fano factor expressions provide tools for:

- **Neuroscience**: The Fano factor of threshold crossings by membrane potential fluctuations distinguishes regular from bursty neuronal firing patterns. Our theory gives exact predictions for Gaussian-modeled membrane dynamics.
- **Structural & Reliability Engineering**: The variance of load exceedances governs fatigue life predictions — the mean crossing rate alone can significantly underestimate failure risk when crossings are clustered.
- **Environmental Science**: Clustering of extreme events (e.g., temperature or precipitation exceeding critical thresholds) has direct implications for risk assessment and policy. The Fano factor quantifies this clustering.
- **Finance**: The variability and clustering of extreme-event crossings in mean-reverting processes carry implications for risk management.
- **Parameter Estimation**: The variance encodes information about the full correlation structure that the mean does not, enabling more robust parameter estimation and model discrimination.

## Dependencies

- Python >= 3.10
- NumPy
- SciPy
- PyTorch
- Matplotlib

## License

Apache 2.0 License

## Citation

If you use this code in your research, please cite the [arXiv preprint](https://arxiv.org/abs/2605.25278):

```bibtex
@misc{rawat2026exact,
  title={Exact Variance and Fano Factor for Arbitrary Level Crossings in Stationary Gaussian Processes},
  author={Rawat, Shivang and Morone, Flaviano and Heeger, David J. and Martiniani, Stefano},
  year={2026},
  eprint={2605.25278},
  archivePrefix={arXiv},
  primaryClass={math.PR},
  doi={10.48550/arXiv.2605.25278},
  url={https://arxiv.org/abs/2605.25278}
}
```
