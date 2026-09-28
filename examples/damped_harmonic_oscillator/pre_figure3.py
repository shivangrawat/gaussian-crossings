"""Reproduce PRE Fig. 3 with trial bootstrap intervals and grid checks.

The experiment fixes omega0 = temperature = 1, T = 120. Exact stationary
Gaussian state transitions replace the original unvalidated FFT embedding.
All three thresholds and nested sampling grids use the same trial paths.
Run `python examples/damped_harmonic_oscillator/pre_figure3.py --help`.
Outputs live in the ignored data/pre_figure3 directory; no paths are discarded
or rerun based on agreement with theory. NumPy/SciPy are the only numerical
dependencies. See README.md for the run and validation protocol.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import platform
import subprocess
import time
import warnings

import numpy as np
import scipy
from numpy.polynomial.polynomial import polymul, polyval
from scipy.integrate import IntegrationWarning, quad
from scipy.linalg import expm
from scipy.signal import lfilter
from scipy.special import erf, ndtr, owens_t


T = 120.0
LEVELS = np.array([0.0, 0.25, 0.5])
ZETAS = np.geomspace(0.5, 2.0, 5)
STRIDES = np.array([1, 2, 4])
PROBE_TIMES = np.array([0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 60.0, 120.0])


def transition(zeta: float, dt: float):
    """Return M, Q for d(x,v) = A(x,v)dt + (0,sqrt(4*zeta))dW.

    The block exponential evaluates Q = integral exp(As) D exp(A.T*s) ds
    without subtracting two nearly equal position variances at small dt.
    C = I is the stationary covariance for this experiment's unit parameters.
    """
    if not (zeta > 0 and dt > 0):
        raise ValueError("zeta and dt must be positive")
    A = np.array([[0.0, 1.0], [-1.0, -2.0 * zeta]])
    D = np.diag([0.0, 4.0 * zeta])
    block = np.block([[A, D], [np.zeros((2, 2)), -A.T]])
    E = expm(block * dt)
    M = E[:2, :2]
    Q = E[:2, 2:] @ M.T
    Q = (Q + Q.T) / 2
    if np.linalg.eigvalsh(Q)[0] <= 0:
        raise FloatingPointError("nonpositive innovation covariance")
    np.testing.assert_allclose(M @ M.T + Q, np.eye(2), atol=2e-13, rtol=0)
    return M, Q


def filtered_positions(M, initial, eta):
    """Fast exact state recurrence, algebraically reduced to an ARMA(2,1).

    eta has shape (trials, 2, steps). The initial state is independent of eta.
    x_n - tr(M)x_(n-1) + det(M)x_(n-2)
      = eta_x,n - M22*eta_x,n-1 + M12*eta_v,n-1.
    The filter delays below encode x0,v0, including the first transition.
    """
    drive = eta[:, 0, :].copy()
    drive[:, 1:] += M[0, 1] * eta[:, 1, :-1] - M[1, 1] * eta[:, 0, :-1]
    det = np.linalg.det(M)
    zi = np.column_stack((initial @ M[0, :], -det * initial[:, 0]))
    x, _ = lfilter([1.0], [1.0, -np.trace(M), det], drive, axis=1, zi=zi)
    return np.column_stack((initial[:, 0], x))


def crossing_counts(x, levels=LEVELS, strides=STRIDES):
    counts = np.empty((len(x), len(strides), len(levels)), dtype=np.int32)
    for j, stride in enumerate(strides):
        y = x[:, ::stride]
        for k, level in enumerate(levels):
            counts[:, j, k] = ((y[:, :-1] < level) & (y[:, 1:] >= level)).sum(axis=1)
    return counts


@lru_cache(maxsize=256)
def covariance_series(zeta):
    """Taylor series of r'' + 2*zeta*r' + r = 0 at the positive-lag origin."""
    c = np.zeros(34)
    c[0] = 1
    for n in range(len(c) - 2):
        c[n + 2] = -(2 * zeta * (n + 1) * c[n + 1] + c[n]) / ((n + 2) * (n + 1))
    # 1-r = t^2 D(t), r' = t P(t); form the cancelling numerator of b as
    # a polynomial, so its exactly zero constant term never enters arithmetic.
    D = -c[2:]
    P = np.arange(2, len(c)) * c[2:]
    q = -np.arange(2, len(c)) * np.arange(1, len(c) - 1) * c[2:]
    one_plus_q = q.copy()
    one_plus_q[0] += 1
    numerator = polymul(one_plus_q, D) - polymul(P, P)
    assert numerator[0] == 0
    return c, D, P, q, numerator[1:]


def covariance_parts(t, zeta):
    """r,p,q and positive conditional variances a=Var((Vt-V0)/2), b=Var((Vt+V0)/2)."""
    if t <= 0:
        raise ValueError("positive lag required")
    if t < 0.05:
        _, D, P, qcoef, bnum = covariance_series(float(zeta))
        d = t * t * polyval(t, D)
        r = 1 - d
        p = t * polyval(t, P)
        q = polyval(t, qcoef)
        b = t * polyval(t, bnum) / (2 * polyval(t, D))
    else:
        if zeta < 1:
            w = math.sqrt(1 - zeta * zeta)
            decay = math.exp(-zeta * t)
            r = decay * (math.cos(w * t) + zeta / w * math.sin(w * t))
            p = -decay * math.sin(w * t) / w
        elif zeta == 1:
            r = (1 + t) * math.exp(-t)
            p = -t * math.exp(-t)
        else:
            w = math.sqrt(zeta * zeta - 1)
            slow, fast = zeta - w, zeta + w
            r = (fast * math.exp(-slow * t) - slow * math.exp(-fast * t)) / (2 * w)
            p = (-math.exp(-slow * t) + math.exp(-fast * t)) / (2 * w)
        q = r + 2 * zeta * p
        d = 1 - r
        b = (1 + q - p * p / d) / 2
    a = (1 - q - p * p / (2 - d)) / 2
    if min(a, b, d, 2 - d) <= 0:
        raise FloatingPointError((t, zeta, a, b, d))
    return r, p, q, d, a, b


def pair_intensity(t, zeta, u):
    """The paper's closed form, in conditional-variance coordinates.

    alpha=1/(4a), beta=1/(4b), gamma=sqrt(2)*m. This rearrangement
    preserves the expression, avoids large intermediate alpha/beta, and
    uses the nonzero SDHO limit at t=0. No NaNs or negative variances are
    masked, and no quadrature terms are silently removed.
    """
    if t == 0:
        return zeta * math.exp(-u * u / 2) * (math.sqrt(3) - math.pi / 3) / (6 * math.pi**2)
    _, p, _, d, a, b = covariance_parts(t, zeta)
    m = p * u / (2 - d)
    bracket = (2 * math.sqrt(a * b) * math.exp(-m * m / (2 * a))
               + math.sqrt(2 * math.pi) * m * math.sqrt(a + b)
               * math.exp(-m * m / (2 * (a + b)))
               * erf(m * math.sqrt(b / (2 * a * (a + b))))
               + 4 * math.pi * (b - a - m * m)
               * owens_t(m / math.sqrt(a + b), math.sqrt(b / a)))
    value = math.exp(-u * u / (2 - d)) * bracket / (4 * math.pi**2 * math.sqrt(d * (2 - d)))
    if not math.isfinite(value) or value < -1e-13:
        raise FloatingPointError((t, zeta, u, value))
    return value


def direct_pair_intensity(t, zeta, u):
    """Independent positive 1D Gaussian integral, with no Owen's T formula.

    S=(V0+Vt)/2 ~ N(0,b), D=(Vt-V0)/2 ~ N(m,a), independently.
    E[(S^2-D^2)1_{S>|D|}] integrates out S first, then integrates |D|.
    """
    _, p, _, d, a, b = covariance_parts(t, zeta)
    m = p * u / (2 - d)
    def integrand(y):
        phi = math.exp(-y * y / 2) / math.sqrt(2 * math.pi)
        h = (1 - y * y) * ndtr(-y) + y * phi
        z1, z2 = (math.sqrt(b) * y - m) / math.sqrt(a), (math.sqrt(b) * y + m) / math.sqrt(a)
        return h * (math.exp(-z1 * z1 / 2) + math.exp(-z2 * z2 / 2)) / math.sqrt(2 * math.pi)
    integral = quad(integrand, 0, 12, epsabs=2e-13, epsrel=2e-11)[0]
    moment = b * math.sqrt(b / a) * integral
    return math.exp(-u * u / (2 - d)) * moment / (2 * math.pi * math.sqrt(d * (2 - d)))


def theory(zeta, u, duration=T, tol=2e-10):
    nu = math.exp(-u * u / 2) / (2 * math.pi)
    def integrand(t):
        return (1 - t / duration) * (pair_intensity(t, zeta, u) - nu * nu)
    breaks = [0.0] + [t for t in (0.01, 0.05, 0.2, 1, 4, 12, 40) if t < duration] + [duration]
    value, error = 0.0, 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("error", IntegrationWarning)
        for left, right in zip(breaks[:-1], breaks[1:]):
            v, e = quad(integrand, left, right, epsabs=tol, epsrel=tol, limit=150)
            value += v
            error += e
    mean = duration * nu
    variance = duration * (nu + 2 * value)
    if variance <= 0:
        raise FloatingPointError("nonpositive theoretical variance")
    return np.array([mean, variance, variance / mean]), 2 * duration * error


def sampled_mean(zeta, u, dt, duration=T):
    """Exact expectation for sign changes on a discrete stationary grid.

    With S=(X0+Xh)/2 and D=(Xh-X0)/2 independent, integrate the width
    [u-D,u+D] for D>0. The small width is evaluated by an 8-point
    Gauss-Legendre rule to avoid subtracting nearly equal normal CDFs.
    """
    _, _, _, d, _, _ = covariance_parts(dt, zeta)
    sdD, sdS = math.sqrt(d / 2), math.sqrt(1 - d / 2)
    nodes, weights = np.polynomial.legendre.leggauss(8)
    def integrand(y):
        width = sdD * y
        z = (u + width * nodes) / sdS
        mass = width / sdS * np.dot(weights, np.exp(-z * z / 2)) / math.sqrt(2 * math.pi)
        return mass * math.exp(-y * y / 2) / math.sqrt(2 * math.pi)
    prob = quad(integrand, 0, 12, epsabs=1e-14)[0]
    return duration / dt * prob


def statistics(counts, axis=0):
    means = np.mean(counts, axis=axis)
    variances = np.var(counts, axis=axis, ddof=1)
    if np.any(means <= 0):
        raise ValueError("nonpositive sample mean")
    return np.stack((means, variances, variances / means), axis=-1)


def bootstrap(counts, seed, n_resamples=10000):
    rng = np.random.default_rng(seed)
    samples = np.empty((n_resamples,) + counts.shape[1:] + (3,))
    for first in range(0, n_resamples, 64):
        size = min(64, n_resamples - first)
        indices = rng.integers(0, len(counts), size=(size, len(counts)))
        samples[first:first + size] = statistics(counts[indices], axis=1)
    return samples


def validate():
    """Independent deterministic checks required before the production run."""
    errors = {"filter_max_abs_error": 0.0, "pair_integral_max_abs_error": 0.0,
              "covariance_max_abs_error": 0.0, "quadrature_max_abs_change": 0.0}
    rng = np.random.default_rng(70117)
    for zeta in ZETAS:
        for dt in (0.000625, 0.00125, 0.005, 0.05):
            M, Q = transition(float(zeta), dt)
            initial = rng.normal(size=(3, 2))
            eta = np.einsum('ij,bjn->bin', np.linalg.cholesky(Q), rng.normal(size=(3, 2, 16000)))
            fast = filtered_positions(M, initial, eta)
            direct = np.empty_like(fast)
            direct[:, 0] = initial[:, 0]
            state = initial.copy()
            for n in range(eta.shape[-1]):
                state = state @ M.T + eta[:, :, n]
                direct[:, n + 1] = state[:, 0]
            error = np.max(np.abs(fast - direct))
            errors["filter_max_abs_error"] = max(errors["filter_max_abs_error"], float(error))
            np.testing.assert_allclose(fast, direct, atol=3e-8, rtol=0)
            np.testing.assert_array_equal(crossing_counts(fast), crossing_counts(direct))
            M2, Q2 = transition(float(zeta), 2 * dt)
            np.testing.assert_allclose(M2, M @ M, atol=2e-14, rtol=0)
            np.testing.assert_allclose(Q2, Q + M @ Q @ M.T, atol=2e-14, rtol=0)
        A = np.array([[0., 1.], [-1., -2 * zeta]])
        for t in (1e-6, 0.001, 0.04999, 0.05001, 0.2, 1., 4., 12., 40.):
            r, p, q, _, _, _ = covariance_parts(t, float(zeta))
            reference = expm(A * t)
            error = np.max(np.abs(np.array([r, p, q]) - reference[[0, 1, 1], [0, 0, 1]]))
            errors["covariance_max_abs_error"] = max(errors["covariance_max_abs_error"], float(error))
            assert error < 2e-13
            for u in LEVELS:
                closed, direct = pair_intensity(t, float(zeta), u), direct_pair_intensity(t, float(zeta), u)
                error = abs(closed - direct)
                errors["pair_integral_max_abs_error"] = max(errors["pair_integral_max_abs_error"], error)
                np.testing.assert_allclose(closed, direct, atol=2e-12, rtol=2e-9)
        for u in LEVELS:
            coarse, _ = theory(float(zeta), u)
            fine, _ = theory(float(zeta), u, tol=2e-12)
            error = np.max(np.abs(coarse - fine))
            errors["quadrature_max_abs_change"] = max(errors["quadrature_max_abs_change"], float(error))
            assert error < 1e-7
            np.testing.assert_allclose(pair_intensity(1e-7, float(zeta), u), pair_intensity(0, float(zeta), u), rtol=2e-6)
        for dt in (0.00125, 0.005):
            d = covariance_parts(dt, float(zeta))[3]
            exact_zero = T / (2 * math.pi * dt) * 2 * math.asin(math.sqrt(d / 2))
            np.testing.assert_allclose(sampled_mean(float(zeta), 0., dt), exact_zero, rtol=1e-12)
    # Known hand-counted paths and explicit resampling verify grid/threshold axes.
    paths = np.array([[-1., 1., -1., 1., -1.], [1., 1., -1., -1., 1.]])
    np.testing.assert_array_equal(crossing_counts(paths, [0., 0.5], [1, 2])[:, :, 0], [[2, 0], [1, 1]])
    counts = np.arange(1, 25).reshape(8, 1, 3)
    boot = bootstrap(counts, 17, 11)
    indices = np.random.default_rng(17).integers(0, 8, (11, 8))
    ref = np.stack([statistics(counts[i]) for i in indices])
    np.testing.assert_array_equal(boot, ref)
    errors["status"] = "passed"
    return errors


def simulate_one(zidx, output, n_trials, dt, seed, batch_size):
    zeta = float(ZETAS[zidx])
    n_steps = round(T / dt)
    if n_steps % 4 or not math.isclose(n_steps * dt, T, rel_tol=0, abs_tol=1e-12):
        raise ValueError("dt must divide T exactly, with steps divisible by four")
    M, Q = transition(zeta, dt)
    L = np.linalg.cholesky(Q)
    counts = np.empty((n_trials, len(STRIDES), len(LEVELS)), dtype=np.int32)
    probe_indices = np.rint(PROBE_TIMES / dt).astype(int)
    probes = np.empty((n_trials, len(probe_indices)))
    initial_velocities = np.empty(n_trials)
    start = time.monotonic()
    for first in range(0, n_trials, batch_size):
        size = min(batch_size, n_trials - first)
        rng = np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(zidx, first // batch_size)))
        initial = rng.normal(size=(size, 2))
        eta = np.einsum('ij,bjn->bin', L, rng.normal(size=(size, 2, n_steps)))
        x = filtered_positions(M, initial, eta)
        counts[first:first + size] = crossing_counts(x)
        probes[first:first + size] = x[:, probe_indices]
        initial_velocities[first:first + size] = initial[:, 1]
        if first // batch_size % max(1, 500 // batch_size) == 0:
            print(f'zeta={zeta:.6g}: {first+size}/{n_trials} trials, {time.monotonic()-start:.1f}s', flush=True)
    if np.any(np.diff(counts, axis=1) > 0):
        raise AssertionError("subsampling increased a crossing count")
    path = output / f'counts_{zidx}.npz'
    np.savez_compressed(path, counts=counts, probes=probes, initial_velocities=initial_velocities,
                        probe_times=probe_indices * dt, zeta=zeta, dt=dt, duration=T,
                        strides=STRIDES, levels=LEVELS, seed=seed, batch_size=batch_size)
    print(f'zeta={zeta:.6g}: saved {path.name}, {time.monotonic()-start:.1f}s', flush=True)
    return str(path)


def json_write(path, data):
    def convert(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(type(value))
    path.write_text(json.dumps(data, indent=2, default=convert, allow_nan=False) + '\n')


def source_hash():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def analyze(output, n_resamples):
    metadata = json.loads((output / 'run.json').read_text())
    if metadata['source_sha256'] != source_hash():
        raise RuntimeError("source changed after simulation; review provenance before analysis")
    results = []
    for zidx, zeta in enumerate(ZETAS):
        path = output / f'counts_{zidx}.npz'
        data = np.load(path, allow_pickle=False)
        c, probes, dt = data['counts'], data['probes'], float(data['dt'])
        np.testing.assert_array_equal(data['levels'], LEVELS)
        np.testing.assert_array_equal(data['strides'], STRIDES)
        assert float(data['zeta']) == zeta and float(data['duration']) == T
        assert len(c) == metadata['n_trials'] and dt == metadata['dt']
        assert int(data['seed']) == metadata['seed'] and int(data['batch_size']) == metadata['batch_size']
        estimate = statistics(c)
        samples = bootstrap(c, np.random.SeedSequence(metadata['seed'], spawn_key=(100, zidx)), n_resamples)
        ci = np.quantile(samples, [0.025, 0.975], axis=0)
        se = np.std(samples, axis=0, ddof=1)
        delta = estimate[0] - estimate[1]
        delta_ci = np.quantile(samples[:, 0] - samples[:, 1], [0.025, 0.975], axis=0)
        halfwidth = (ci[1, 0] - ci[0, 0]) / 2
        ref = np.array([theory(float(zeta), u)[0] for u in LEVELS])
        discrete_means = np.array([[sampled_mean(float(zeta), u, dt * s) for u in LEVELS] for s in STRIDES])
        probe_times = data['probe_times']
        probe_mean_z = probes.mean(axis=0) * math.sqrt(len(c))
        probe_var_z = (np.mean(probes**2, axis=0) - 1) / math.sqrt(2 / len(c))
        expected_cov = np.array([1. if t == 0 else covariance_parts(t, float(zeta))[0] for t in probe_times])
        cov_z = ((probes[:, :1] * probes).mean(axis=0) - expected_cov) / np.sqrt((1 + expected_cov**2) / len(c))
        velocity = data['initial_velocities']
        velocity_z = np.array([velocity.mean() * math.sqrt(len(c)),
                               (np.mean(velocity**2) - 1) / math.sqrt(2 / len(c)),
                               np.mean(velocity * probes[:, 0]) * math.sqrt(len(c))])
        diagnostics = {
            'probe_mean_z': probe_mean_z, 'probe_second_moment_z': probe_var_z,
            'probe_covariance_z': cov_z, 'initial_velocity_z': velocity_z,
            'max_moment_abs_z': max(np.max(np.abs(a)) for a in (probe_mean_z, probe_var_z, cov_z, velocity_z)),
            'fine_minus_twice_dt': delta, 'paired_difference_ci95': delta_ci,
            'max_grid_shift_over_ci_halfwidth': np.max(np.abs(delta) / halfwidth),
            'max_grid_difference_ci_bound_over_halfwidth': np.max(np.max(np.abs(delta_ci), axis=0) / halfwidth),
            'exact_discrete_means': discrete_means,
            'max_remaining_mean_bias_over_ci_halfwidth': np.max(np.abs(discrete_means[0] - ref[:, 0]) / halfwidth[:, 0]),
            'theory_residual_in_bootstrap_se': (estimate[0] - ref) / se[0],
            'theory_inside_pointwise_ci95': (ref >= ci[0, 0]) & (ref <= ci[1, 0]),
        }
        results.append({'zeta': zeta, 'counts_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                        'estimate': estimate, 'ci95': ci, 'bootstrap_se': se, 'theory': ref,
                        'diagnostics': diagnostics})
        print(f'zeta={zeta:.6g}: bootstrap complete; grid shift/CI halfwidth={diagnostics["max_grid_shift_over_ci_halfwidth"]:.3f}', flush=True)
    curve_zeta = np.geomspace(0.5, 2., 101)
    curve = np.array([[theory(float(z), u)[0] for u in LEVELS] for z in curve_zeta])
    checks = {'moment_checks_pass': all(r['diagnostics']['max_moment_abs_z'] < 5 for r in results),
              'grid_changes_small': all(r['diagnostics']['max_grid_difference_ci_bound_over_halfwidth'] < 0.25 for r in results),
              'remaining_mean_bias_small': all(r['diagnostics']['max_remaining_mean_bias_over_ci_halfwidth'] < 0.2 for r in results)}
    summary = {'run': metadata, 'n_bootstrap': n_resamples, 'confidence': 0.95, 'method': 'pointwise percentile bootstrap of whole trial rows',
               'ddof': 1, 'metrics': ['mean_count', 'variance_count', 'variance_over_mean'],
               'checks': checks, 'points': results, 'curve_zeta': curve_zeta, 'curve': curve}
    json_write(output / 'summary.json', summary)
    if not all(checks.values()):
        raise RuntimeError(f'Investigate convergence/validation before plotting: {checks}')
    return summary


def plot(output):
    """Use the submitted Figure 3 design with the revised data and intervals."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FuncFormatter, LogLocator, MaxNLocator, NullLocator
    summary = json.loads((output / 'summary.json').read_text())
    if not all(summary['checks'].values()):
        raise RuntimeError('unresolved numerical checks')
    # Match cells 14 and 18 of theory_simulation_upcrossings.ipynb. These
    # sizes are specified for the original 24-by-7-inch plotting canvas.
    plt.rcParams.update({'text.usetex': True, 'font.family': 'serif',
                         'font.serif': ['Computer Modern Roman'], 'pdf.fonttype': 42})
    fig, axes = plt.subplots(1, 3, figsize=(24, 7))
    colors = plt.get_cmap('tab10')
    labels = ['Mean', 'Variance', r'$\mathrm{F}^{\uparrow}_T$']
    label_pads = [40, 50, 30]
    curve, curve_zeta = np.array(summary['curve']), np.array(summary['curve_zeta'])
    for j, ax in enumerate(axes):
        for k, u in enumerate(LEVELS):
            ax.plot(curve_zeta, curve[:, k, j], color=colors(k), lw=3.5)
            y = np.array([p['estimate'][0][k][j] for p in summary['points']])
            lo = np.array([p['ci95'][0][0][k][j] for p in summary['points']])
            hi = np.array([p['ci95'][1][0][k][j] for p in summary['points']])
            ax.errorbar(ZETAS, y, yerr=[y-lo, hi-y], fmt='o', color=colors(k),
                        ms=10, mfc='none', mew=2, elinewidth=2, capsize=5, capthick=2, zorder=4)
        ax.set_xscale('log')
        ax.set_xlim(0.465, 2.15)
        ax.xaxis.set_major_locator(LogLocator(base=2.0, numticks=5))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f'{value:g}'))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.yaxis.set_major_locator(MaxNLocator(4))
        ax.set_xlabel(r'$\zeta$', fontsize=30)
        ax.set_ylabel(labels[j], fontsize=30, rotation=0, va='center', labelpad=label_pads[j])
        ax.tick_params(axis='both', which='major', labelsize=24, width=2,
                       length=3.5, direction='out', top=False, right=False)
        for spine in ax.spines.values():
            spine.set_linewidth(2)
        # The other manuscript figures use bare bold letters outside the box.
        ax.text(-0.20, 1.03, rf'\textbf{{{chr(97+j)}}}', transform=ax.transAxes,
                fontsize=48, ha='center', va='top', clip_on=False)
    handles = [Line2D([0], [0], color=colors(k), marker='o', mfc='none', mew=2,
                      ms=10, linestyle='None', label=f'u = {u}') for k, u in enumerate(LEVELS)]
    axes[1].legend(handles=handles, loc='upper left', fontsize=24)
    fig.tight_layout()
    fig.savefig(output / 'sdho_comparison.pdf', bbox_inches='tight', pad_inches=0.035)
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['validate', 'simulate', 'analyze', 'plot'])
    parser.add_argument('--output', type=Path, default=Path('data/pre_figure3'))
    parser.add_argument('--trials', type=int, default=5000)
    parser.add_argument('--dt', type=float, default=0.00125)
    parser.add_argument('--seed', type=int, default=20260927)
    parser.add_argument('--batch-size', type=int, default=20)
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--bootstrap', type=int, default=10000)
    args = parser.parse_args()
    if args.action == 'validate':
        print(json.dumps(validate(), indent=2))
    elif args.action == 'simulate':
        if args.trials < 2 or args.batch_size < 1 or args.workers < 1:
            raise ValueError('invalid ensemble size or concurrency')
        if args.output.exists() and any(args.output.iterdir()):
            raise FileExistsError('Use a new output directory; archived runs are never overwritten')
        validation = validate()
        args.output.mkdir(parents=True, exist_ok=True)
        head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
        metadata = {'created_utc': datetime.now(timezone.utc).isoformat(), 'git_head': head,
                    'source_sha256': source_hash(), 'python': platform.python_version(),
                    'numpy': np.__version__, 'scipy': scipy.__version__, 'n_trials': args.trials,
                    'dt': args.dt, 'duration': T, 'strides': STRIDES, 'levels': LEVELS, 'zetas': ZETAS,
                    'seed': args.seed, 'batch_size': args.batch_size,
                    'simulation': 'exact stationary Gaussian state transition; second-order filter acceleration',
                    'validation': validation,
                    'criteria': {'max_moment_abs_z': 5, 'max_paired_grid_ci_over_sampling_halfwidth': 0.25,
                                 'max_mean_grid_bias_over_sampling_halfwidth': 0.2}}
        json_write(args.output / 'run.json', metadata)
        (args.output / 'run_source.py').write_bytes(Path(__file__).read_bytes())
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(simulate_one, j, args.output, args.trials, args.dt, args.seed, args.batch_size) for j in range(5)]
            for future in futures:
                future.result()
    elif args.action == 'analyze':
        analyze(args.output, args.bootstrap)
    elif args.action == 'plot':
        plot(args.output)


if __name__ == '__main__':
    main()
