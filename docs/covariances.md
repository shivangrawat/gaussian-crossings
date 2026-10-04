# Covariance functions

A model is built from a covariance function and its parameters:

```python
from gaussian_crossings import GaussianCrossings
from gaussian_crossings.process import r_rational_quadratic

model = GaussianCrossings(r_rational_quadratic, sigma=1.0, tau=0.5, alpha=2.0)
```

Parameters are passed by name and forwarded to the covariance function. Unknown names raise
`TypeError`.

## Built-in covariances

| Function | Parameters | Variance $r(0)$ | Crossing formulas |
|---|---|---|---|
| `r_damped_harmonic_oscillator_noise` | `temp`, `omega0`, `zeta` | `temp / omega0**2` | Yes, every `zeta > 0` |
| `r_OU_noise` | `sigma`, `tau`, `kappa` | `sigma**2 * kappa / (1 + kappa)` | Yes, `kappa != 1` |
| `r_filtered_OU` | `sigma`, `tau`, `kappa` | `sigma**2 / (1 + kappa)` | Yes, `kappa != 1` |
| `r_rational_quadratic` | `sigma`, `tau`, `alpha` | `sigma**2` | Yes; long-time results need `alpha > 1/2` (`alpha > 1/4` suffices at `u = 0`) |
| `r_squared_exp` | `sigma`, `tau` | `sigma**2` | Yes |
| `r_matern` | `sigma`, `tau`, `nu` | `sigma**2` | Only `nu = 1.5` and `2.5` |
| `r_OU` | `sigma`, `tau` | `sigma**2` | No: paths are not differentiable (simulation only) |

- `r_damped_harmonic_oscillator_noise`: position of a damped harmonic oscillator driven by
  thermal noise, $\ddot x + 2\zeta\omega_0\dot x + \omega_0^2 x = \sqrt{4\zeta\omega_0\vartheta}\,\eta(t)$.
- `r_OU_noise`: a process relaxing with time constant `tau` and driven by Ornstein–Uhlenbeck noise
  with correlation time `kappa * tau`. `r_filtered_OU` is the same two-time-scale form with a
  different amplitude convention.
- `r_rational_quadratic`: $\sigma^2 [1 + t^2/(2\alpha\tau^2)]^{-\alpha}$, a mixture of
  squared-exponential covariances over time scales. Small `alpha` gives heavy-tailed, long-range
  correlations.
- `r_squared_exp`: $\sigma^2 e^{-t^2/(2\tau^2)}$.

The oscillator, both Ornstein–Uhlenbeck-driven forms, rational-quadratic, squared-exponential,
and Matérn `nu = 1.5` covariances use dedicated, numerically stable expressions. Evaluate many
thresholds in one call to share their adaptive integration. Runtime depends on the parameters,
thresholds, and requested tolerances. Matérn `nu = 2.5` uses the generic expansion described below.

## Writing your own

For a custom function `r(t, **params)`, distinguish the mathematical assumptions from the
requirements of the numerical method:

1. Use **PyTorch** operations (`torch.exp`, `torch.cos`, ...) and return a tensor, because
   $r'(t)$ and $r''(t)$ are obtained by automatic differentiation;
2. Supply a valid stationary covariance, even in `t` and positive semidefinite.
3. The process must satisfy the smoothness, Geman, and joint nondegeneracy assumptions in
   [Concepts](concepts.md#when-the-formulas-apply). Finite $-r''(0)>0$ is necessary but is not
   by itself a guarantee of continuously differentiable sample paths or finite crossing variance.
4. For a finite long-time variance rate or Fano factor, the excess crossing-pair intensity must be
   integrable. Mere covariance decay is insufficient: a rational-quadratic covariance with
   `alpha = 0.4` decays, but its long-time Fano factor diverges at every nonzero threshold.
5. The default adaptive method additionally needs a checked right-hand covariance expansion
   through order 14. This numerical requirement is stronger than the smoothness needed by the
   crossing formula itself; see [Numerical methods](numerics.md).

```python
import torch

def r_quasi_periodic(t, sigma, tau, period):
    return sigma**2 * torch.exp(-0.5 * (t / tau) ** 2) * torch.cos(2 * torch.pi * t / period)

model = GaussianCrossings(r_quasi_periodic, sigma=1.0, tau=3.0, period=2.0)
levels = [0.0, 0.5, 1.0]
fano = model.fano_factor(levels)   # pass all thresholds in one call
```

For a covariance supplied by the user, each adaptive variance/Fano calculation, whether finite
window or long-time, first builds a checked small-lag series by automatic differentiation.
This can take several seconds. Evaluate many
thresholds in one call rather than in a loop. See
[tutorial 02](tutorials/02_custom_covariance.ipynb).

## Common errors

| Situation | Error |
|---|---|
| Process is not smooth (for example `r_OU`) | `ValueError: -r''(0) must be a finite positive scalar ...` |
| Covariance written with NumPy | `TypeError: The covariance callback returned ... It must be written with PyTorch operations ...` |
| Misspelled parameter | `TypeError: ... got an unexpected keyword argument ...` |
| Rational-quadratic long-time result whose tail diverges (`alpha <= 1/2` with `u != 0`, or `alpha <= 1/4` at `u = 0`) | `ValueError: The long-time crossing variance diverges ...` |

## Linear stochastic models

For simulation by Euler–Maruyama integration (`gaussian_crossings.utils.euler_maruyama_sde`), the
package also provides the corresponding state-space models: `DampedHarmonicOscillatorNoise`,
`OUNoise`, and `FilteredOU` in `gaussian_crossings.process`. Their earlier lowercase names remain
available.
