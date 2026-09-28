# API and numerical notes

## Stable interfaces

The original import paths and class names are retained:

```python
from gaussian_crossings import GaussianUpCrossings, GaussianUpCrossingsDimless
from gaussian_crossings.formula.formula import GaussianUpCrossings
from gaussian_crossings.formula.formula_dimless import GaussianUpCrossingsDimless
```

`GaussianUpCrossings(r_func, u=0, *args, **kwargs)` evaluates a dimensional covariance. The callback must accept a PyTorch lag tensor and return a differentiable covariance tensor. Extra arguments are forwarded to the callback.

`GaussianUpCrossingsDimless(r_func, u=0, tau=None, *args, **kwargs)` evaluates the callback with `tau=1`, then restores physical time units. `tau=None` means one. Means and variance rates scale as `1/tau`; the long-time Fano factor does not. The callback must accept `tau`, or accept and ignore additional keywords when appropriate. Lag arguments to its integrand methods are dimensionless.

Both classes share the same formula engine. They expose `upcrossing_`, `downcrossing_`, and `crossing_` methods for `mean_rate`, `mean`, `integrand`, `variance`, `variance_CLT_per_unit_time`, `variance_CLT`, `fano_factor`, and `fano_factor_CLT`. Finite-window methods take `T`; rate and asymptotic Fano methods do not. `u=None` selects the threshold stored in the model.

`GaussianUpCrossingsDimless_minimal` remains available at its original import path. It retains an old magnitude-filtered integration rule and different cutoffs. Use the full dimensionless class for new work.

## Exact formulas and numerical integration

The analytical expressions are exact under the paper's assumptions. Their numerical evaluation is approximate.

The package retains the original transformed trapezoidal integration rule and defaults to preserve existing results. It maps positive time to `s=t/(1+t)`. `num_points`, `epsilon_left`, and (for long-time quantities) `epsilon_right` control the grid and truncation. The historical near-zero correction is also retained. This rule does not provide a quadrature error estimate and can be sensitive to small-lag cancellation, rescaling, and slowly decaying tails.

The compatibility integrator issues `NumericalIntegrationWarning` if it omits NaN samples, and raises for infinite integrands or too few usable samples. Treat warnings as errors for strict exploratory checks:

```python
import warnings
from gaussian_crossings import NumericalIntegrationWarning
warnings.simplefilter("error", NumericalIntegrationWarning)
```

For the reported figures, use `examples/pre_figures.py`. It supplies separately validated, model-specific stable small-lag expressions and adaptive quadrature. Tests distinguish legacy numerical compatibility from independent accuracy checks and paper-reference values. Matching a compatibility snapshot is not an accuracy certificate.

`T=0` has zero count variance. Its variance-to-mean ratio is undefined. A very short window may require a smaller `epsilon_left`. Under the paper's linear-growth condition, the long-time total-crossing Fano factor is twice the upcrossing factor; their finite-window ratios need not obey that identity. The long-time formula requires an integrable excess pair intensity; it must not be assumed for an arbitrary covariance.

## Covariance conventions

| Function | Parameters | Variance $r(0)$ | Notes |
|---|---|---|---|
| `r_damped_harmonic_oscillator_noise` | `temp`, `omega0`, `zeta` | `temp / omega0**2` | Underdamped, critical, and overdamped branches |
| `r_OU_noise` | `sigma`, `tau`, `kappa` | `sigma**2 * kappa / (1+kappa)` | OU-driven mean reversion in the paper |
| `r_filtered_OU` | `sigma`, `tau`, `kappa` | `sigma**2 / (1+kappa)` | Different amplitude convention from `r_OU_noise` |
| `r_rational_quadratic` | `sigma`, `tau`, `alpha` | `sigma**2` | Shape controls the covariance tail |
| `r_squared_exp` | `sigma`, `tau` | `sigma**2` | `exp(-t**2 / (2*tau**2))` convention |
| `r_OU` | `sigma`, `tau` | `sigma**2` | Nonsmooth; for simulation, not smooth-path crossing formulas |
| `r_matern` | `sigma`, `tau`, `nu` | `sigma**2` | Autograd for half-integer `nu=0.5, 1.5, 2.5`; general `nu` for covariance evaluation/simulation |

The two bi-exponential covariance functions require `kappa != 1`; use an explicit limiting covariance at that degenerate parameterization. A covariance with zero variance, nonpositive derivative variance, or insufficient smoothness is outside the formula engine's domain. Validity of a custom covariance and the paper's joint nondegeneracy assumptions remain the caller's responsibility.

Python scalars and NumPy arrays are accepted by the supplied covariance functions. Formula lag evaluation and quadrature use double precision without changing `torch.get_default_dtype()`. CPU execution is the validated path. Owen's T transfers its forward calculation through SciPy on CPU and supplies analytical PyTorch gradients; this is not a GPU-native special-function implementation.

## Simulation and event conventions

- `simulate_gaussian_process_fft` uses a reflected circulant embedding. It rejects materially negative eigenvalues and only clips roundoff-sized negatives.
- `simulate_gaussian_process_cholesky` constructs the finite Toeplitz covariance. It attempts an exact factorization first, then bounded diagonal jitter if necessary.
- Both return `round(T/dt)` samples on `arange(n)*dt`. When `T/dt` is integral, the endpoint `T` is excluded.
- `euler_maruyama_sde` returns `floor(time/dt)` states on their actual integration times. Its default initial state is a deterministic equilibrium, not a draw from the stationary distribution. Figure 3 instead uses exact stationary Gaussian transitions in its dedicated runner.
- Crossing counters count an arrival at/above a level from strictly below, or at/below from strictly above. Linear-interpolation time functions use the same convention. Sampled counts may miss excursions between time steps.
- `euler_maruyama_upcrossings` counts during integration without storing trajectories. With the same initial state and random seed, its counts match the stored-trajectory calculation.
- `autocorrelation` subtracts the sample mean and divides every lag by `n-1`. Its zero-lag value equals the sample variance; it is not normalized to one or unbiased separately at each lag.

## Maintenance changes

The refactor preserves 60 representative pre-refactor numerical cases. Targeted edge-case fixes cover scalar covariance callbacks in FFT simulation, invalid covariance embeddings, broadcast gradients for Owen's T, crossing-count/time endpoint consistency, SDE time labels and state dimension, and explicit precision. The validated PRE numerical routines are unchanged. Operational runner changes keep exports inside this checkout by default and save the source for newly generated Figure 3 runs.
