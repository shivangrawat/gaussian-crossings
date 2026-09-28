"""Stable, CPU adaptive quadrature for the crossing formula.

Time and level are normalized to unit process/derivative variance. Known
covariances use analytic small-lag series; custom callbacks use checked
right-hand Taylor coefficients obtained by PyTorch differentiation.
"""

import inspect
import math
from dataclasses import dataclass
from functools import lru_cache, partial

import numpy as np
import torch
from numpy.polynomial.polynomial import polymul, polyval
from scipy.integrate import quad_vec
from scipy.special import beta, betainc, erf, erfc, owens_t

from gaussian_crossings.process import correlation_functions as kernels


@dataclass(frozen=True)
class IntegrationInfo:
    """Diagnostics in Fano-factor units; estimates are not rigorous bounds.

    ``tail_change`` is the change after extending a model-specific cutoff,
    not a bound on the uncomputed tail. Generic covariances are integrated
    on the infinite interval by QUADPACK-style adaptive subdivision.
    """

    method: str
    covariance: str
    estimated_error: float
    quadrature_error: float
    tail_change: float
    evaluations: int
    cutoff: float


def _check_parts(parts):
    if not np.all(np.isfinite(parts)) or min(parts[3], 2 - parts[3], *parts[4:]) <= 0:
        raise FloatingPointError(
            "Nonpositive conditional covariance or nonfinite integrand. "
            "Check covariance nondegeneracy and small-lag precision."
        )
    return parts


@lru_cache(maxsize=128)
def _series(coefficients):
    c = np.asarray(coefficients)
    d = -c[2:]
    p = np.arange(2, len(c)) * c[2:]
    q = -np.arange(2, len(c)) * np.arange(1, len(c) - 1) * c[2:]
    qp = q.copy()
    qp[0] += 1
    bn = polymul(qp, d) - polymul(p, p)
    bn[0] = 0  # Exact cancellation since c[2] = -1/2.
    if c[3] == 0:
        bn[1:3] = 0  # The t and t^2 terms vanish when r'''(0+) = 0.
    # a = ((1-q)(2-delta)-p(t)^2)/(2*(2-delta)).
    qm = -q.copy()
    qm[0] += 1
    two_minus_d = np.concatenate(([2.0, 0.0], -d))
    an = polymul(qm, two_minus_d)
    pp = np.concatenate(([0.0, 0.0], polymul(p, p)))
    an = np.pad(an, (0, max(0, len(pp) - len(an))))
    an[: len(pp)] -= pp
    return d, p, q, an, bn


def _factored_polynomial(t, c):
    nonzero = np.flatnonzero(c)
    if not len(nonzero):
        return 0.0
    n = nonzero[0]
    return t ** int(n) * polyval(t, c[n:])


def _series_parts(t, coefficients):
    dc, pc, qc, an, bn = _series(tuple(coefficients))
    d = t * t * polyval(t, dc)
    p = t * polyval(t, pc)
    q = polyval(t, qc)
    a = _factored_polynomial(t, an) / (2 * (2 - d))
    b = _factored_polynomial(t, bn) / (2 * polyval(t, dc))
    return _check_parts((1 - d, p, q, d, a, b))


@lru_cache(maxsize=128)
def _sdho_coefficients(zeta):
    c = np.zeros(34)
    c[0] = 1
    for n in range(len(c) - 2):
        c[n + 2] = -(2 * zeta * (n + 1) * c[n + 1] + c[n]) / ((n + 2) * (n + 1))
    return tuple(c)


def sdho_parts(t, zeta):
    """Unit-variance SDHO conditional covariance at normalized positive lag."""
    if t <= 0:
        raise ValueError("positive lag required")
    if t < 0.05 / max(1.0, zeta):
        return _series_parts(t, _sdho_coefficients(zeta))
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
        fast = zeta + w
        slow = 1 / fast
        r = (fast * math.exp(-slow * t) - slow * math.exp(-fast * t)) / (2 * w)
        p = (-math.exp(-slow * t) + math.exp(-fast * t)) / (2 * w)
    q = r + 2 * zeta * p
    d = 1 - r
    a = (1 - q - p * p / (2 - d)) / 2
    b = (1 + q - p * p / d) / 2
    return _check_parts((r, p, q, d, a, b))


@lru_cache(maxsize=32)
def _rq_coefficients(shape):
    c = np.zeros(40)
    c[0] = 1
    for n in range(1, len(c) // 2):
        c[2 * n] = -c[2 * n - 2] * (
            1 / (2 * n) if math.isinf(shape) else (shape + n - 1) / (2 * shape * n)
        )
    return tuple(c)


def rq_parts(t, shape):
    """RQ (or squared-exponential) covariance in unit time/amplitude units."""
    if t <= 0:
        raise ValueError("positive lag required")
    if t < 0.05 * min(1.0, math.sqrt(shape)):
        return _series_parts(t, _rq_coefficients(shape))
    if math.isinf(shape):
        r = math.exp(-t * t / 2)
        d = -math.expm1(-t * t / 2)
        p = -t * r
        q = (1 - t * t) * r
    else:
        logc = math.log1p(t * t / (2 * shape))
        r = math.exp(-shape * logc)
        d = -math.expm1(-shape * logc)
        p = -t * math.exp(-(shape + 1) * logc)
        q = math.exp(-(shape + 1) * logc) - (shape + 1) / shape * t * t * math.exp(
            -(shape + 2) * logc
        )
    a = (1 - q - p * p / (2 - d)) / 2
    b = (1 + q - p * p / d) / 2
    return _check_parts((r, p, q, d, a, b))


def normalized_pair(parts, levels):
    """K(t;u)/nu(u), with signed erf and exclusively nonpositive exponents."""
    _, p, _, d, a, b = parts
    u = np.asarray(levels)
    m = p * u / (2 - d)
    arg = m * math.sqrt(b / (2 * a * (a + b)))
    bracket = (
        2 * math.sqrt(a * b) * np.exp(-m * m / (2 * a))
        + math.sqrt(2 * math.pi) * m * math.sqrt(a + b) * np.exp(-m * m / (2 * (a + b))) * erf(arg)
        + 4 * math.pi * (b - a - m * m) * owens_t(m / math.sqrt(a + b), math.sqrt(b / a))
    )
    value = np.exp(-u * u * d / (2 * (2 - d))) * bracket / (2 * math.pi * math.sqrt(d * (2 - d)))
    if not np.all(np.isfinite(value)) or np.min(value) < -2e-13:
        raise FloatingPointError("Nonfinite or negative pair crossing intensity")
    return value


class _GenericParts:
    """Local right-hand Taylor expansion, checked against the callback."""

    def __init__(self, model, unit):
        self.model, self.unit = model, unit
        c = [1.0, 0.0, -0.5]

        def fn(t):
            return model.r(t * unit) / model.r0

        derivative = fn
        origin = torch.tensor(1e-40, dtype=torch.float64)
        try:
            for n in range(1, 15):
                derivative = torch.func.grad(derivative)
                value = float(derivative(origin).detach()) / math.factorial(n)
                if n >= 3:
                    # Right derivatives of an even analytic covariance have
                    # zero odd coefficients; remove only origin-shift noise.
                    c.append(0.0 if abs(value) < 1e-30 else value)
        except (RuntimeError, TypeError) as exc:
            raise ValueError(
                "Adaptive integration of a custom covariance requires a stable "
                "local Taylor expansion (autograd through order 14). "
                "Use a supplied covariance or method='trapezoid'."
            ) from exc
        if not np.all(np.isfinite(c)):
            raise ValueError("Custom covariance has no finite local Taylor expansion")
        self.coefficients = tuple(c)
        growth = max([1.0] + [abs(c[n]) ** (1 / n) for n in range(3, len(c))])
        self.boundary = 0.05 / growth
        for _ in range(12):
            t = self.boundary
            full = np.asarray(_series_parts(t, self.coefficients))
            short = np.asarray(_series_parts(t, self.coefficients[:-2]))
            if np.max(np.abs((full[3:] - short[3:]) / full[3:])) < 2e-10:
                break
            self.boundary /= 2
        else:
            raise ValueError("Custom covariance small-lag series did not converge")
        # Validate r, r', r'' where both representations are well conditioned.
        raw = self._raw(self.boundary)
        if not np.allclose(full[:3], raw[:3], rtol=2e-10, atol=2e-12):
            raise ValueError("Custom covariance disagrees with its local Taylor expansion")

    def _raw(self, t):
        model, unit = self.model, self.unit
        r = float((model.r(t * unit) / model.r0).detach())
        p = float((model.p(t * unit) * unit / model.r0).detach())
        q = float((model.q(t * unit) * unit * unit / model.r0).detach())
        d = 1 - r
        return _check_parts((r, p, q, d, (1 - q - p * p / (2 - d)) / 2, (1 + q - p * p / d) / 2))

    def __call__(self, t):
        return _series_parts(t, self.coefficients) if t < self.boundary else self._raw(t)


def _provider(model, unit):
    fn = model.r_func
    args, kwargs = model.args, model.kwargs.copy()
    if isinstance(fn, partial):
        args = fn.args + args
        kwargs = {**(fn.keywords or {}), **kwargs}
        fn = fn.func
    known = (
        kernels.r_damped_harmonic_oscillator_noise,
        kernels.r_OU_noise,
        kernels.r_filtered_OU,
        kernels.r_rational_quadratic,
        kernels.r_squared_exp,
        kernels.r_matern,
    )
    if fn in known:
        # Time normalization changes tau but never the shape parameters.
        bound = inspect.signature(fn).bind_partial(None, *args, **kwargs).arguments
        if fn is kernels.r_damped_harmonic_oscillator_noise:
            shape = float(bound["zeta"])
            return partial(sdho_parts, zeta=shape), "sdho", shape
        if fn in (kernels.r_OU_noise, kernels.r_filtered_OU):
            kappa = float(bound["kappa"])
            shape = (1 + kappa) / (2 * math.sqrt(kappa))
            return partial(sdho_parts, zeta=shape), "sdho", shape
        if fn in (kernels.r_rational_quadratic, kernels.r_squared_exp):
            shape = float(bound["alpha"]) if fn is kernels.r_rational_quadratic else math.inf
            return partial(rq_parts, shape=shape), "rq", shape
        if fn is kernels.r_matern and float(bound["nu"]) == 1.5:
            return partial(sdho_parts, zeta=1.0), "sdho", 1.0
    return _GenericParts(model, unit), "custom", None


def _rq_tail(shape, end, u):
    nu = np.exp(-u * u / 2) / (2 * math.pi)
    coefficients = (u * u, (u * u - 1) ** 2 / 2)
    value = np.zeros_like(u)
    for j, coefficient in enumerate(coefficients, start=1):
        if not np.any(coefficient):
            continue
        if math.isinf(shape):
            integral = math.sqrt(math.pi / (2 * j)) * erfc(math.sqrt(j / 2) * end)
        else:
            if j * shape <= 0.5:
                raise ValueError("The long-time crossing variance diverges for this RQ shape/level")
            x = 1 / (1 + end * end / (2 * shape))
            integral = (
                math.sqrt(shape / 2) * beta(j * shape - 0.5, 0.5) * betainc(j * shape - 0.5, 0.5, x)
            )
        value += coefficient * integral
    return 2 * nu * (value + math.pi / 2 * rq_parts(end, shape)[1])


def _requires_grad(value):
    if isinstance(value, torch.Tensor):
        return value.requires_grad
    if isinstance(value, dict):
        return any(_requires_grad(v) for v in value.values())
    if isinstance(value, (tuple, list)):
        return any(_requires_grad(v) for v in value)
    return False


def adaptive_fano(model, T, u, *, total=False, epsabs=2e-10, epsrel=2e-10, limit=1200, cutoff=None):
    """Evaluate Fano ratios; return a float64 tensor and integration diagnostics."""
    if not math.isfinite(epsabs) or not math.isfinite(epsrel) or epsabs <= 0 or epsrel <= 0:
        raise ValueError("epsabs and epsrel must be finite and positive")
    if not isinstance(limit, int) or isinstance(limit, bool) or limit < 2:
        raise ValueError("limit must be an integer of at least 2")
    if T is not None and (not math.isfinite(float(T)) or T <= 0):
        raise ValueError("A finite-window Fano factor requires finite T > 0")
    u = model.u if u is None else u
    if _requires_grad((T, u, model.args, model.kwargs, model.r0, model.q0, model._time_scale)):
        raise ValueError(
            "Adaptive quadrature is CPU-only and not differentiable; use method='trapezoid'"
        )
    if cutoff is not None and (not math.isfinite(float(cutoff)) or cutoff <= 0):
        raise ValueError("cutoff must be finite and positive in normalized time units")
    levels = torch.as_tensor(u, dtype=torch.float64)
    if levels.numel() == 0 or not bool(torch.isfinite(levels).all()):
        raise ValueError("u must contain finite thresholds and cannot be empty")
    unit = math.sqrt(float(model.r0) / float(model.q0))
    normalized_u = levels.detach().cpu().numpy() / math.sqrt(float(model.r0))
    original_shape = normalized_u.shape
    normalized_u = normalized_u.reshape(-1)
    duration = None if T is None else float(T) / (unit * float(model._time_scale))
    parts, family, shape = _provider(model, unit)
    nu = np.exp(-normalized_u * normalized_u / 2) / (2 * math.pi)
    evaluations = 0

    def integrand(t):
        value = 2 * (normalized_pair(parts(t), normalized_u) - nu)
        return value if duration is None else value * (1 - t / duration)

    def integrate(end):
        nonlocal evaluations
        breaks = [
            x
            for x in (0.0001, 0.001, 0.01, 0.05, 0.2, 1.0, 4.0, 12.0, 40.0, 128.0, 512.0, 2048.0)
            if x < end
        ]
        value, error, info = quad_vec(
            integrand,
            0.0,
            end,
            epsabs=epsabs / (2 if total else 1),
            epsrel=epsrel / (2 if total else 1),
            norm="max",
            points=breaks,
            limit=limit,
            full_output=True,
        )
        evaluations += info.neval
        if not info.success or not np.all(np.isfinite(value)):
            raise ArithmeticError(
                f"Adaptive quadrature failed: {info.message}; estimated error={error}"
            )
        return value, float(error)

    tail_change = 0.0
    if duration is not None:
        end = duration
        value, error = integrate(end)
    elif family == "custom":
        if cutoff is not None:
            raise ValueError("cutoff is only supported for supplied SDHO/RQ covariance tails")
        end = math.inf
        value, error = integrate(end)
    else:
        if family == "sdho":
            slow = shape if shape <= 1 else 1 / (shape + math.sqrt(shape * shape - 1))
            end = 40 / slow
        else:
            end = 2048.0
        if cutoff is not None:
            end = float(cutoff)

        def corrected(end):
            value, error = integrate(end)
            if family == "rq":
                value = value + _rq_tail(shape, end, normalized_u)
            return value, error

        value, error = corrected(end)
        for _ in range(12):
            end = end * 2 if family == "rq" else end + 20 / slow
            refined, refined_error = corrected(end)
            tail_change = float(np.max(np.abs(refined - value)))
            value, error = refined, refined_error
            if tail_change <= max(epsabs, epsrel * float(np.max(np.abs(1 + value)))) / 4:
                break
        else:
            raise ArithmeticError("Long-time tail failed cutoff-refinement convergence")
    value = 1 + value
    if total:
        # N_total = N_up + N_down and N_up-N_down = H(X_T-u)-H(X_0-u).
        # Var(N_total)=4 Var(N_up)-P(endpoint indicators differ).
        value = 2 * value
        error *= 2
        tail_change *= 2
        if duration is not None:
            d = parts(duration)[3]
            angle = math.atan(math.sqrt(d / (2 - d)))
            endpoint, endpoint_error, info = quad_vec(
                lambda theta: np.exp(-(normalized_u**2) * math.tan(theta) ** 2 / 2),
                0.0,
                angle,
                epsabs=epsabs * duration / 8,
                epsrel=epsrel / 4,
                norm="max",
                limit=limit,
                full_output=True,
            )
            if not info.success:
                raise ArithmeticError("Endpoint-probability integration failed")
            value -= 2 * endpoint / duration
            error += 2 * float(endpoint_error) / duration
            evaluations += info.neval
    if not np.all(np.isfinite(value)) or np.min(value) < -max(epsabs, 1e-12):
        raise FloatingPointError("Nonfinite or negative crossing Fano factor")
    target = max(epsabs, epsrel * float(np.max(np.abs(value))))
    if error + tail_change > target:
        raise ArithmeticError(
            f"Estimated Fano error {error + tail_change} exceeds requested tolerance {target}"
        )
    diagnostics = IntegrationInfo(
        "adaptive", family, error + tail_change, error, tail_change, evaluations, end
    )
    result = torch.as_tensor(
        value.reshape(original_shape), dtype=torch.float64, device=model.r0.device
    )
    return result, diagnostics
