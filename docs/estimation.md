# Estimating from data

The theory predicts the Fano factor of crossing counts in windows of a given length. To compare
it with a recording, cut the recording into windows, count crossings in each, and compute the
variance-to-mean ratio of the counts. The comparison assumes a stationary Gaussian model and
sampling fine enough to resolve its continuous-time crossings. In this example, `x` has been
centered and `model` describes its covariance, so the thresholds are relative to the data mean.

```python
from gaussian_crossings import GaussianCrossings, empirical_fano, windowed_counts

counts = windowed_counts(x, u=1.0, window=50.0, dt=0.01)          # one count per window
estimate = empirical_fano(x, u=[0.5, 1.0, 1.5], window=50.0, dt=0.01, seed=0)

estimate.fano        # array, one value per threshold
estimate.ci_low      # bootstrap confidence interval
estimate.ci_high
estimate.stderr
estimate.window      # window length actually used (a whole number of sampling steps)

prediction = model.fano_factor([0.5, 1.0, 1.5], T=estimate.window)
```

`x` holds samples taken every `dt`: one series, or a two-dimensional array of independent
series of equal length whose windows are pooled. Each crossing is assigned to exactly one window;
an incomplete final window is dropped. `kind` selects `"up"`, `"down"`, or `"total"` crossings.

## Choosing windows

- Use windows long enough that their crossing counts are approximately independent, or count one
  window from each independent realization. Windows much longer than the correlation time are a useful
  starting point; check dependence between the resulting counts.
- Use **many windows**, at least a few hundred, for a stable variance estimate.
- Compare with `fano_factor(u, T=estimate.window, kind=kind)` for the **same window length and
  crossing direction**, not with the long-time value. Requested window lengths are rounded to
  whole sampling steps; see [Concepts](concepts.md#finite-windows-versus-the-long-time-limit).

## Sampling resolution

Two crossings that both fall between consecutive samples are not seen, which biases the counts
low. `empirical_fano` recounts using every second sample over the same observation span. For each
threshold with at least 50 detected crossings, losing more than 2% triggers a
`CoarseSamplingWarning`; realizations shorter than three samples cannot be checked. This is a
heuristic: no warning does not establish adequate resolution, because both grids can miss events.
When finer data are available, check that the counts converge as the sampling step decreases.
Processes whose velocity varies rapidly, such as the damped harmonic oscillator, need finer
sampling than very smooth ones.

## Confidence intervals

The interval is a percentile bootstrap that resamples whole windows, which preserves the
dependence between thresholds within each resample. The reported intervals are pointwise, not
simultaneous intervals across thresholds. The bootstrap treats windows as independent. Dependence
between windows can make these intervals too narrow or too wide; positive dependence often makes
them too narrow. Use longer windows, independent realizations, or a resampling method that
preserves the relevant dependence. To assess coverage, repeat the analysis on many simulated
datasets from the fitted model; this checks calibration under that model.

## Matching the model to the data

The prediction depends on the full covariance, including its amplitude and time scale. Fit the
covariance to the data first, for example by comparing `gaussian_crossings.utils.autocorrelation(x)`
with candidate models, and express thresholds in the same units as the data. This utility centers
the data when estimating covariance, but the crossing counters do not. If the original recording
has mean `mu`, compare its crossings at a physical threshold `u` with the zero-mean model at
`u - mu`. Equivalently, subtract `mu` from both the recording and its thresholds before counting.
Estimating a covariance alone does not account for this mean shift.

Because the Fano factor responds to correlations that the mean crossing rate cannot see, it can
also help discriminate between models; see [tutorial 04](tutorials/04_model_discrimination.ipynb).

## Simulating data

`simulate_gaussian_process_fft(r, T, dt, **params)` draws a zero-mean stationary Gaussian path
when the covariance's reflected circulant embedding is positive semidefinite, up to the stated
floating-point tolerance. A valid covariance can fail this embedding check for a particular
window; increase the window or use `simulate_gaussian_process_cholesky` for short series. The
Cholesky sampler first uses the exact finite covariance matrix, then retries failed factorizations
with diagonal jitter, which changes that covariance. See the sampler docstrings for tolerances
and retry settings.

Both use PyTorch's random number generator; call `torch.manual_seed` for reproducible paths.
`count_upcrossings`, `count_downcrossings`, and `count_crossings` count events in a single series.
