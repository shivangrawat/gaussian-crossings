# API reference

## Crossing statistics

`GaussianCrossings` and `GaussianUpCrossings` are the same class.

::: gaussian_crossings.formula.formula.GaussianUpCrossings
    options:
      members:
        - __init__
        - mean_rate
        - mean
        - variance
        - variance_rate
        - fano_factor

The original methods remain available and return PyTorch tensors:
`upcrossing_mean_rate`, `upcrossing_mean`, `upcrossing_variance`,
`upcrossing_variance_CLT_per_unit_time`, `upcrossing_variance_CLT`, `upcrossing_fano_factor`,
`upcrossing_fano_factor_CLT`, `upcrossing_integrand`, and the corresponding `downcrossing_*` and
`crossing_*` methods. Their trapezoid integration requires a scalar threshold; the new methods
handle arrays by evaluating each threshold separately. `_CLT` denotes the long-time limit.

::: gaussian_crossings.formula.integration.IntegrationInfo

## Estimation from data

::: gaussian_crossings.estimation.empirical_fano

::: gaussian_crossings.estimation.windowed_counts

::: gaussian_crossings.estimation.FanoEstimate

::: gaussian_crossings.estimation.CoarseSamplingWarning

## Covariance functions

::: gaussian_crossings.process.correlation_functions
    options:
      show_root_heading: false
      members:
        - r_damped_harmonic_oscillator_noise
        - r_OU_noise
        - r_filtered_OU
        - r_rational_quadratic
        - r_squared_exp
        - r_matern
        - r_OU

## Simulation

::: gaussian_crossings.utils.simulation
    options:
      show_root_heading: false
      members:
        - simulate_gaussian_process_fft
        - simulate_gaussian_process_cholesky

## Counting and covariance estimation

::: gaussian_crossings.utils.utils
    options:
      show_root_heading: false
      members:
        - count_upcrossings
        - count_downcrossings
        - count_crossings
        - upcrossing_times
        - downcrossing_times
        - crossing_times
        - autocorrelation
        - euler_maruyama_sde

## Plotting

Requires the `plot` extra.

::: gaussian_crossings.plotting.MidpointNormalize

## Deprecated

`GaussianUpCrossingsDimless` and `GaussianUpCrossingsDimless_minimal` evaluate a covariance in
units of its time scale. They emit `DeprecationWarning`; use `GaussianCrossings` instead.
For a covariance that accepts `tau`, pass the former external time scale as `tau=...`.
Otherwise, preserve the time rescaling with a callback such as
`lambda t: r_func(t / tau, **params)`. A pure time rescaling leaves long-time Fano factors
unchanged but changes rates and finite-window results.
