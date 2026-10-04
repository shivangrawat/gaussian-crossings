# Estimating from data

The theory predicts the Fano factor of crossing counts in windows of a given length. To compare
it with a recording, cut the recording into windows, count crossings in each, and compute the
variance-to-mean ratio of the counts.

```python
from gaussian_crossings import GaussianCrossings, empirical_fano, windowed_counts

counts = windowed_counts(x, u=1.0, window=50.0, dt=0.01)          # one count per window
estimate = empirical_fano(x, u=[0.5, 1.0, 1.5], window=50.0, dt=0.01, seed=0)

estimate.fano        # array, one value per threshold
estimate.ci_low      # bootstrap confidence interval
estimate.ci_high
estimate.stderr
estimate.window      # window length actually used (a whole number of samples)

prediction = model.fano_factor([0.5, 1.0, 1.5], T=estimate.window)
```

`x` holds samples taken every `dt`: one series, or a two-dimensional array of independent
series of equal length whose windows are pooled. Each crossing is assigned to exactly one window;
an incomplete final window is dropped. `kind` selects `"up"`, `"down"`, or `"total"` crossings.

## Choosing windows

- Make windows **much longer than the correlation time** of the process. Windows are then nearly
  independent, which the confidence interval assumes.
- Use **many windows**, at least a few hundred, for a stable variance estimate.
- Compare with `fano_factor(u, T=window)` for the **same window length**, not with the long-time
  value; see [Concepts](concepts.md#finite-windows-versus-the-long-time-limit).

## Sampling resolution

Two crossings that both fall between consecutive samples are not seen, which biases the counts
low. `empirical_fano` recounts using every second sample; if that removes more than 2% of the
crossings, it issues a `CoarseSamplingWarning`. Processes whose velocity varies rapidly, such as
the damped harmonic oscillator, need finer sampling than very smooth ones.

## Confidence intervals

The interval is a percentile bootstrap that resamples whole windows, which preserves the
dependence between thresholds. It treats windows as independent. If windows are not much longer
than the correlation time, the interval is too narrow. For calibrated uncertainties, repeat the
analysis on many simulated datasets from the fitted model.

## Matching the model to the data

The prediction depends on the full covariance, including its amplitude and time scale. Fit the
covariance to the data first, for example by comparing `gaussian_crossings.utils.autocorrelation(x)`
with candidate models, and express thresholds in the same units as the data. Because the Fano
factor responds to correlations that the mean crossing rate cannot see, it can also be used to
discriminate between models; see [tutorial 04](tutorials/04_model_discrimination.ipynb).

## Simulating data

`simulate_gaussian_process_fft(r, T, dt, **params)` draws a stationary path from any covariance
by circulant embedding, and `simulate_gaussian_process_cholesky` does the same exactly for short
series. Both use PyTorch's random number generator; call `torch.manual_seed` for reproducible
paths. `count_upcrossings`, `count_downcrossings`, and `count_crossings` count events in a single
series.
