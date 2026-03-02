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

If you use this code in your research, please cite:

```bibtex
@article{rawat2025exact,
  title={Exact Variance and Fano Factor for Arbitrary Level Crossings in Stationary Gaussian Processes},
  author={Rawat, Shivang and Morone, Flaviano and Heeger, David J. and Martiniani, Stefano},
  journal={Physical Review X},
  year={2025}
}
```
