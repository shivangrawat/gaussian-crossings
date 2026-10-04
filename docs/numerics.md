# Numerical methods

The analytical expressions are exact under the assumptions in [Concepts](concepts.md); their
evaluation is numerical. This page describes how the package integrates them and how to check
the accuracy of a result.

## Integration options

Variance and Fano-factor methods accept these keyword options:

| Option | Default | Meaning |
|---|---|---|
| `method` | `"adaptive"` | Adaptive quadrature, or `"trapezoid"` for the original fixed-grid rule |
| `epsabs`, `epsrel` | `2e-10`, `2e-10` | Absolute and relative tolerances, in units of the Fano factor |
| `limit` | `1200` | Maximum number of adaptive subintervals |
| `cutoff` | model-specific | Initial tail cutoff in normalized lag $t\sqrt{q_0/r_0}$; built-in covariances only |

After a call, `model.last_integration_info` reports the covariance strategy, the quadrature
error estimate, the change produced by extending the tail cutoff, the number of function
evaluations, and the final cutoff. These are estimates, not rigorous bounds. Errors are
expressed in Fano-factor units even for variance methods; multiply by the mean count or rate for
the corresponding variance error.

## How the integrand is evaluated

The evaluator combines exponential factors, keeps the sign of the error-function argument, and
factors the cancelling covariance differences at small lags.

- **Built-in covariances** (damped oscillator, both Ornstein–Uhlenbeck-driven forms,
  rational-quadratic, squared-exponential) use explicit normalized expressions and series at small
  lags. Their long-time tails are checked by extending the cutoff; rational-quadratic tails include
  an asymptotic correction, and non-integrable cases are rejected.
- **Custom covariances** use a right-hand Taylor expansion obtained by automatic differentiation
  through order 14. It is checked for local convergence and for agreement with the callback, and
  is followed by adaptive integration over the half-line. A covariance without a reliable local
  expansion raises an explanatory error. Building the expansion takes several seconds per call.

These checks cannot establish the smoothness, nondegeneracy, or integrability assumptions for an
arbitrary covariance; that remains the user's responsibility.

## Gradients and devices

The adaptive path runs on the CPU through SciPy and does not provide parameter gradients; inputs
that require gradients are rejected rather than silently detached. For differentiable results,
use the original methods (for example `upcrossing_variance`) with `method="trapezoid"`, which
keeps the historical fixed grid (`num_points`, `epsilon_left`, `epsilon_right`). Matching an
earlier result with this rule is not an accuracy certificate.

The recommended methods (`mean_rate`, `mean`, `variance`, `variance_rate`, `fano_factor`) return
floats or NumPy arrays. The original `upcrossing_*`, `downcrossing_*`, and `crossing_*` methods
return float64 PyTorch tensors. Importing the package does not change PyTorch's default dtype.

## Total crossings

Total-crossing statistics use the exact identity
$\mathrm{Var}[N_u(T)] = 4\,\mathrm{Var}[N^\uparrow_u(T)] - P_T$, where $P_T$ is the probability
that $X_0$ and $X_T$ lie on opposite sides of $u$. It is integrated in a scaled form that avoids
underflow, and gives a long-time total-crossing Fano factor of exactly twice the upcrossing one.
The closed-form total-crossing integrand is available as `crossing_integrand`.

## Edge cases

- A window of length `T = 0` has zero variance and an undefined Fano factor; adaptive Fano
  methods reject it.
- The two Ornstein–Uhlenbeck-driven covariances require `kappa != 1`; supply the limiting
  covariance as a custom function for that case.
- Critical and near-critical damping of the oscillator are evaluated without cancellation.

## Simulation and counting conventions

- `simulate_gaussian_process_fft` uses a reflected circulant embedding. It rejects materially
  negative eigenvalues and clips only roundoff-sized ones; increase `T` or use the Cholesky sampler
  if the embedding fails.
- `simulate_gaussian_process_cholesky` factorizes the finite Toeplitz covariance, adding bounded
  diagonal jitter only if the exact factorization fails. Its `jitter` and `max_attempts` options
  are keyword-only.
- Both samplers return `round(T/dt)` samples at `arange(n) * dt`.
- `euler_maruyama_sde` starts from the deterministic equilibrium unless `x0` is given, so allow a
  burn-in period before measuring stationary statistics.
- Crossing counters count an arrival at or above a level from strictly below (upcrossing), or at
  or below from strictly above (downcrossing). `windowed_counts` and `empirical_fano` use the same
  rule.
- `autocorrelation` subtracts the sample mean and divides every lag by `n - 1`; its zero-lag value
  is the sample variance.
