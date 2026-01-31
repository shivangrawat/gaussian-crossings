# Gaussian Crossings

A Python library for computing exact analytical expressions for the variance and Fano factor of level crossings in smooth, stationary Gaussian processes.

## Overview

This repository provides computational tools accompanying the paper:

> **Exact Variance and Fano Factor for Arbitrary Level Crossings in Stationary Gaussian Processes**

Understanding the statistics of level crossings in stochastic processes is crucial across many scientific disciplines. While exact results for mean-level crossings exist (the Kac-Rice formula), this work provides **exact analytical expressions** for:

- **Variance** of arbitrary level up-crossings, down-crossings, and bidirectional crossings
- **Fano factor** (variance-to-mean ratio) characterizing the regularity of crossing events

These analytical solutions are validated across diverse stochastic processes:
- **Stochastic Damped Harmonic Oscillator** - fundamental model in statistical physics
- **Mean-reverting process driven by Ornstein-Uhlenbeck (OU) noise** - widely used in neuroscience and finance
- **Rational Quadratic correlation function** - flexible kernel for Gaussian process modeling

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

```
gaussian-crossings/
├── gaussian_crossings/           # Main package
│   ├── utils/                    # Simulation and utility functions
│   │   ├── simulation.py         # FFT/Cholesky-based GP simulation
│   │   ├── utils.py              # Crossing detection, SDE integration
│   │   └── owensT.py             # Owen's T function (PyTorch, with autograd)
│   ├── formula/                  # Analytical formulas
│   │   ├── formula.py            # Base class for crossing statistics
│   │   ├── formula_dimless.py    # Dimensionless formulation
│   │   └── formula_dimless_minimal.py
│   └── process/                  # Stochastic process definitions
│       ├── correlation_functions.py  # Correlation functions (7 types)
│       └── dynamical_equations.py    # SDE definitions
├── examples/                     # Jupyter notebooks with applications
└── figures/                      # Generated figures
```

## Features

### Analytical Computations

- **Mean crossing rate**: Kac-Rice formula implementation
- **Variance**: Exact finite-time and asymptotic expressions
- **Fano factor**: Measure of crossing regularity (sub/super-Poisson statistics)

### Supported Correlation Functions

| Function | Description |
|----------|-------------|
| Damped Harmonic Oscillator | Underdamped, critically damped, overdamped regimes |
| Ornstein-Uhlenbeck | Exponentially decaying correlation |
| Filtered OU | Cascaded exponential filtering |
| Rational Quadratic | Flexible kernel with scale mixture interpretation |
| Squared Exponential | Infinitely differentiable (smooth) processes |
| Matern | Parameterized smoothness |

### Simulation Methods

- **FFT-based**: Efficient circulant embedding for large-scale simulations
- **Cholesky**: Direct covariance sampling for smaller systems
- **SDE Integration**: Euler-Maruyama scheme for dynamical systems

## Quick Start

### Computing Crossing Statistics

```python
import torch
from gaussian_crossings.formula.formula import GaussianUpCrossings
from gaussian_crossings.process.correlation_functions import r_OU

# Define correlation function parameters
tau = 1.0  # correlation time

# Create crossing statistics calculator
crossings = GaussianUpCrossings(
    r=lambda t: r_OU(t, tau),
    r0=1.0  # variance of the process
)

# Compute mean and variance for threshold u over time T
u = 0.5  # threshold level
T = 100.0  # observation time

mean = crossings.upcrossing_mean(u, T)
variance = crossings.upcrossing_variance(u, T)
fano = crossings.upcrossing_fano_factor(u)

print(f"Mean crossings: {mean:.4f}")
print(f"Variance: {variance:.4f}")
print(f"Fano factor: {fano:.4f}")
```

### Simulating Gaussian Processes

```python
import numpy as np
from gaussian_crossings.utils.simulation import simulate_gaussian_process_fft
from gaussian_crossings.utils.utils import count_upcrossings

# Simulation parameters
dt = 0.01
T = 100.0
t = np.arange(0, T, dt)

# Define correlation function
def r(t):
    tau = 1.0
    return np.exp(-np.abs(t) / tau)

# Simulate
X = simulate_gaussian_process_fft(t, r)

# Count crossings
threshold = 0.5
n_up = count_upcrossings(X, threshold)
print(f"Number of upcrossings: {n_up}")
```

## Examples

The `examples/` directory contains Jupyter notebooks demonstrating:

| Category | Description |
|----------|-------------|
| `sdho_*.ipynb` | Stochastic damped harmonic oscillator analysis |
| `ou_noise_*.ipynb` | Ornstein-Uhlenbeck driven processes |
| `rational_quadratic_*.ipynb` | Rational quadratic kernel applications |
| `kohn_data_*.ipynb` | Analysis of neurophysiological data |
| `spike_train_*.ipynb` | Point process / spike train analysis |

## Mathematical Background

### Level Crossings

For a smooth, stationary, zero-mean Gaussian process $X_t$ with autocorrelation function $r(t)$:

- **Up-crossing** of level $u$: $X_t = u$ with $\dot{X}_t > 0$
- **Down-crossing** of level $u$: $X_t = u$ with $\dot{X}_t < 0$

### Key Results

**Mean (Kac-Rice formula):**
$$\mathbb{E}[N_u^+] = T \cdot \frac{1}{2\pi}\sqrt{\frac{-r''(0)}{r(0)}} \exp\left(-\frac{u^2}{2r(0)}\right)$$

**Fano Factor:**
$$F_u^+ = 1 + 4\pi \sqrt{\frac{r(0)}{-r''(0)}} \exp\left(\frac{u^2}{2r(0)}\right) \int_0^\infty I^+(t) \, dt$$

where $I^+(t)$ is expressed in terms of Owen's T function and the error function.

## Applications

- **Neuroscience**: Spike train analysis, neuronal threshold crossing statistics
- **Physics**: Thermal noise in oscillators, stochastic resonance
- **Signal Processing**: Detection thresholds, event rate estimation
- **Finance**: Level crossing in mean-reverting processes
- **Environmental Science**: Threshold exceedance in climate data

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
@article{gaussian-crossings,
  title={Exact Variance and Fano Factor for Arbitrary Level Crossings in Stationary Gaussian Processes},
  author={...},
  journal={...},
  year={...}
}
```
