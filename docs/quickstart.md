# Quick start

## Build a model

A model is a covariance function plus its parameters. The built-in covariance functions are in
`gaussian_crossings.process`; you can also [write your own](covariances.md#writing-your-own).

```python
import numpy as np
from gaussian_crossings import GaussianCrossings
from gaussian_crossings.process import r_damped_harmonic_oscillator_noise

model = GaussianCrossings(r_damped_harmonic_oscillator_noise, temp=1.0, omega0=1.0, zeta=0.5)
```

The built-in covariance functions reject misspelled parameters with `TypeError`.

## Compute statistics

Every method takes the threshold `u` and `kind`, which is `"up"` (default), `"down"`, or
`"total"`.

| Method | Returns |
|---|---|
| `mean_rate(u)` | mean number of crossings per unit time |
| `mean(T, u)` | mean number of crossings in a window of length `T` |
| `variance(T, u)` | variance of the number of crossings in a window of length `T` |
| `variance_rate(u)` | long-time variance per unit time, $\lim_{T\to\infty} \mathrm{Var}[N(T)]/T$ |
| `fano_factor(u, T=None)` | Fano factor: long-time limit if `T` is `None`, otherwise for windows of length `T` |

```python
model.mean_rate(0.5)              # 0.1405
model.fano_factor(0.5)            # 0.4071  (long-time limit)
model.fano_factor(0.5, T=120.0)   # 0.4145  (windows of length 120)
model.fano_factor(0.5, kind="total")   # 0.8143 = 2 x 0.4071
```

A scalar threshold gives a float; an array gives a NumPy array of the same shape, computed in
one call:

```python
levels = np.linspace(0.0, 3.0, 61)
curve = model.fano_factor(levels)
```

## Which Fano factor to use

To compare with data that have been cut into windows of length $T$, use
`fano_factor(u, T=T)`. The long-time value is the limit for very long windows and can differ
noticeably when windows span only a few correlation times. See
[Concepts](concepts.md#finite-windows-versus-the-long-time-limit).

## Estimate from data

```python
from gaussian_crossings import empirical_fano

estimate = empirical_fano(x, u=0.5, window=100.0, dt=0.01)   # x sampled every dt
estimate.fano, estimate.ci_low, estimate.ci_high
model.fano_factor(0.5, T=estimate.window)                     # matching prediction
```

See [Estimating from data](estimation.md).

The model assumes zero-mean data. If the recording has mean `mu`, compare counts at a physical
threshold `u` with `model.fano_factor(u - mu, T=estimate.window)`, or center both the recording
and its threshold before counting.

## Original method names

Earlier versions exposed methods such as `upcrossing_fano_factor_CLT()` (long-time limit) and
`crossing_variance(T)`. These methods remain available and return PyTorch tensors. With
`method="trapezoid"`, they require a scalar threshold; the new methods evaluate threshold arrays
one element at a time. In the original names, `_CLT` denotes the long-time limit.
