# Concepts

## Crossing counts

Let $X_t$ be a stationary, zero-mean Gaussian process with autocovariance $r(t)$, and let
$N^\uparrow_u(T)$ be the number of **upcrossings** of the level $u$ during $[0, T]$: times where
$X_t = u$ with positive slope. Downcrossings $N^\downarrow_u(T)$ have negative slope, and
$N_u(T) = N^\uparrow_u(T) + N^\downarrow_u(T)$ counts **all crossings**.

If your data have a nonzero mean, subtract it and measure thresholds from the mean.

## Mean: the Kac–Rice formula

With $r_0 = r(0)$ and $q_0 = -r''(0)$, the mean upcrossing rate is

$$
\nu_u = \frac{1}{2\pi}\sqrt{\frac{q_0}{r_0}}\, e^{-u^2/(2 r_0)},
\qquad \mathbb{E}[N^\uparrow_u(T)] = \nu_u T .
$$

It depends only on the variance and curvature of the covariance at zero lag.

## Variance and Fano factor

The variance of the count in a window of length $T$ is

$$
\mathrm{Var}[N^\uparrow_u(T)] = \nu_u T + 2\int_0^T (T - t)\, I^\uparrow(t)\, \mathrm{d}t ,
$$

where $I^\uparrow(t)$ is the joint intensity of upcrossings separated by a lag $t$, minus
$\nu_u^2$. The package evaluates the closed form of $I^\uparrow$, written with the error function
and Owen's $T$ function, and integrates it numerically. Because $I^\uparrow$ depends on $r$, $r'$,
and $r''$ at every lag, the variance encodes the whole correlation structure.

The **Fano factor** of counts in windows of length $T$ is
$F_T = \mathrm{Var}[N(T)] / \mathbb{E}[N(T)]$. Its long-time limit is

$$
F = \lim_{T\to\infty} F_T = 1 + \frac{2}{\nu_u}\int_0^\infty I^\uparrow(t)\, \mathrm{d}t ,
$$

when the integral converges.

- $F < 1$: crossings are more regular than a Poisson process, as with oscillatory correlations.
- $F > 1$: crossings cluster, as when slow relaxation produces repeated recrossings near the threshold.
- $F = 1$ does not by itself make the crossings a Poisson process.

## Finite windows versus the long-time limit

$F_T$ approaches $F$ only when $T$ is long compared with the time over which crossings remain
correlated. For slowly decaying covariances the difference can be large. In
[tutorial 03](tutorials/03_fano_from_data.ipynb), a process with a rational-quadratic covariance
($\alpha = 0.75$, $\tau = 1$) is observed at $u = 1.75$ in windows of length $T = 100$:

| | Fano factor |
|---|---|
| Estimate from artificial data (95% interval) | 1.25 (1.14 – 1.37) |
| Prediction for windows of length 100: `fano_factor(1.75, T=100)` | 1.23 |
| Long-time limit: `fano_factor(1.75)` | 1.34 |

The data agree with the finite-window prediction. Comparing them with the long-time limit would
suggest a discrepancy that is not there. **When comparing with data, use the window length of the
data.**

## Upcrossings, downcrossings, and all crossings

For a stationary Gaussian process, downcrossings have the same statistics as upcrossings
(`kind="down"`). Up- and downcrossings alternate, so their counts differ by at most one in any
window. As a result, for all crossings (`kind="total"`)

- the mean rate is $2\nu_u$;
- in the long-time limit, the Fano factor is exactly twice the upcrossing Fano factor;
- in a finite window, $\mathrm{Var}[N_u(T)] = 4\,\mathrm{Var}[N^\uparrow_u(T)] - P_T$, where
  $P_T$ is the probability that $X_0$ and $X_T$ lie on opposite sides of $u$.

At high thresholds crossings come from rare, isolated excursions, so the upcrossing Fano factor
approaches one and the total-crossing Fano factor approaches two.

## When the formulas apply

The results hold for a stationary Gaussian process that is

- **smooth**: sample paths are continuously differentiable, which requires a finite $q_0 = -r''(0)$.
  A process with exponential covariance (Ornstein–Uhlenbeck) is not smooth, and the package
  rejects it;
- **regular at short lags** (Geman's condition): $\int_0^\epsilon [r''(t) - r''(0)]/t \,\mathrm{d}t < \infty$,
  so that the variance is finite. All built-in smooth covariances satisfy it;
- **nondegenerate**: $(X_0, X_t, \dot X_0, \dot X_t)$ has a nonsingular covariance for $t \neq 0$.

Long-time results additionally need the correlations to decay quickly enough. It suffices that
$r$, $r'$, and $r''$ are integrable at large lags. The rational-quadratic covariance decays as
$|t|^{-2\alpha}$, so $\alpha > 1/2$ gives finite long-time results at every threshold. At the mean
level $u = 0$ the leading tail term, proportional to $u^2 r(t)$, vanishes, and $\alpha > 1/4$
suffices. The package raises an error when the tail integral diverges. Finite-window results do
not need these conditions.
