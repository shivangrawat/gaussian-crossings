# Gaussian Crossings

**Exact variance and Fano factor of level crossings for smooth stationary Gaussian processes.**

Given the autocovariance $r(t)$ of a stationary Gaussian process and a threshold $u$,
`gaussian-crossings` computes, for upcrossings, downcrossings, or all crossings of $u$:

- the **mean** number of crossings (the Kac–Rice formula),
- the exact **variance** of the number of crossings in a window of length $T$,
- the **Fano factor** $F = \mathrm{Var}[N]/\mathbb{E}[N]$, for finite windows and in the long-time limit.

It also estimates the Fano factor from sampled data, with confidence intervals, so that
predictions can be compared directly with measurements.

![Mean rate, variance rate, and Fano factor of a damped harmonic oscillator as the threshold and damping vary.](assets/sdho_phase.png)

The mean crossing rate depends only on $r(0)$ and $r''(0)$. The variance and Fano factor depend
on the whole covariance, so processes with identical mean rates can have very different crossing
statistics. In the figure, every damping ratio has the same mean rate (left), while the Fano
factor (right) ranges from regular crossings ($F<1$, purple) to clustered ones ($F>1$, green).

## Install

```bash
pip install gaussian-crossings
```

See [Installation](installation.md) for optional extras and a smaller CPU-only install.

## Example

```python
from gaussian_crossings import GaussianCrossings
from gaussian_crossings.process import r_damped_harmonic_oscillator_noise

model = GaussianCrossings(r_damped_harmonic_oscillator_noise, temp=1.0, omega0=1.0, zeta=0.5)

model.mean_rate(u=0.5)               # upcrossings per unit time
model.variance(T=100.0, u=0.5)       # variance of the count in a window of length 100
model.fano_factor(u=0.5, T=100.0)    # Fano factor of counts in windows of length 100
model.fano_factor(u=0.5)             # long-time Fano factor
model.fano_factor(u=0.5, kind="total")
```

## Where next

- [Quick start](quickstart.md): the main functions in five minutes.
- [Concepts](concepts.md): what is computed, when the formulas apply, and why the window length matters.
- [Estimating from data](estimation.md): Fano factors from recordings, with confidence intervals.
- [Tutorials](tutorials/README.md): executed notebooks with artificial data.
- [API reference](api.md).

## Background

The formulas are derived in S. Rawat, F. Morone, D. J. Heeger, and S. Martiniani,
*Exact Variance and Fano Factor for Arbitrary Level Crossings in Stationary Gaussian Processes*
([arXiv:2605.25278](https://arxiv.org/abs/2605.25278)). Please [cite](citation.md) the paper if
you use the package.
