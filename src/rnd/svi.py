"""Gatheral's raw SVI parameterisation and its calibration.

SVI buys two things a spline cannot. Its derivatives are analytic, so the
density is exact rather than the output of a differentiation scheme; and its
five parameters have enough structure that no-arbitrage can be imposed
*during* the fit instead of checked afterwards and patched.

Raw form::

    w(k) = a + b * (rho * (k - m) + sqrt((k - m)**2 + sigma**2))
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from .smile import LEE_SLOPE_BOUND, Smile, durrleman_g

#: Penalty weights tried in order. A soft penalty is enough on a well-behaved
#: chain and cheaper to optimise; noisy wings need a heavier hand. Escalating
#: only when the check fails keeps the common case fast without giving up the
#: guarantee in the uncommon one.
PENALTY_LADDER = (50.0, 500.0, 5000.0, 50000.0)
VARIANCE_PENALTY = 100.0
#: Grid the penalty is evaluated on during calibration.
_PENALTY_GRID = np.linspace(-3.0, 3.0, 241)
#: Finer, wider grid used to verify the fitted smile afterwards.
_VERIFICATION_GRID = np.linspace(-6.0, 6.0, 2401)
#: Steps in the shrink search used to repair a fit the penalty left just short
#: of feasible.
SHRINK_TRIALS = 81


def svi_arrays(k, a: float, b: float, rho: float, m: float, sigma: float):
    """Total variance and its first two derivatives, for raw parameters.

    Free of the admissibility checks that :class:`SVISmile` enforces, because a
    least-squares step has to be allowed to walk through inadmissible territory
    on its way somewhere better. The penalty terms in the objective, not an
    exception, are what push it back.
    """
    x = np.asarray(k, dtype=float) - m
    root = np.sqrt(x * x + sigma * sigma)
    total_variance = a + b * (rho * x + root)
    return total_variance, b * (rho + x / root), b * sigma**2 / root**3


@dataclass(frozen=True)
class SVIParams:
    """The five raw-SVI parameters, with their admissibility conditions."""

    a: float
    b: float
    rho: float
    m: float
    sigma: float

    def validate(self) -> None:
        if self.b < 0.0:
            raise ValueError("b must be non-negative")
        if not -1.0 < self.rho < 1.0:
            raise ValueError("rho must lie strictly inside (-1, 1)")
        if self.sigma <= 0.0:
            raise ValueError("sigma must be positive")
        if self.a + self.b * self.sigma * np.sqrt(1.0 - self.rho**2) < 0.0:
            raise ValueError("parameters imply negative total variance at the minimum")

    @property
    def wing_slope(self) -> float:
        """Steeper of the two asymptotic wing slopes; Lee caps this at 2."""
        return self.b * (1.0 + abs(self.rho))

    def as_tuple(self) -> tuple[float, ...]:
        return (self.a, self.b, self.rho, self.m, self.sigma)


class SVISmile(Smile):
    """A :class:`~rnd.smile.Smile` backed by raw SVI, differentiated in closed form."""

    def __init__(self, params: SVIParams, expiry: float, forward: float):
        super().__init__(expiry, forward)
        params.validate()
        self.params = params

    def total_variance(self, k):
        return svi_arrays(k, *self.params.as_tuple())[0]

    def d_total_variance(self, k):
        return svi_arrays(k, *self.params.as_tuple())[1]

    def d2_total_variance(self, k):
        return svi_arrays(k, *self.params.as_tuple())[2]


def _residuals(theta, k, w, weights, penalty):
    fitted, _, _ = svi_arrays(k, *theta)
    residual = weights * (fitted - w)
    if penalty <= 0.0:
        return residual
    grid_variance, first, second = svi_arrays(_PENALTY_GRID, *theta)
    butterfly = penalty * np.minimum(
        durrleman_g(_PENALTY_GRID, grid_variance, first, second), 0.0
    )
    positivity = VARIANCE_PENALTY * np.minimum(grid_variance, 0.0)
    return np.concatenate([residual, butterfly, positivity])


def _initial_guesses(k, w) -> list[tuple[float, ...]]:
    w_min = float(np.min(w))
    k_at_min = float(k[int(np.argmin(w))])
    spread = max(float(np.ptp(k)), 0.05)
    return [
        (0.5 * w_min, 0.1, rho, k_at_min, 0.1 * spread)
        for rho in (-0.7, -0.3, 0.0, 0.3)
    ]


def fit_svi(
    k,
    total_variance,
    expiry: float,
    forward: float,
    weights=None,
    enforce_no_arbitrage: bool = True,
) -> SVISmile:
    """Calibrate raw SVI to observed total variances by penalised least squares.

    Bounds keep the fit inside the admissible parameter set; the penalty term
    pushes Durrleman's ``g`` back above zero wherever the unconstrained optimum
    would have implied negative probability. Several starting points are tried
    because the SVI objective is not convex and a single start lands in a local
    minimum often enough to matter.

    The penalty weight is escalated until a fine-grid audit comes back clean,
    and anything the penalty cannot close is removed exactly by
    :func:`_repair_butterfly`. The density this smile produces is therefore
    non-negative by construction rather than by inspection.
    """
    k = np.asarray(k, dtype=float)
    w = np.asarray(total_variance, dtype=float)
    if k.size < 5:
        raise ValueError("SVI has five parameters and needs at least five quotes")
    weights = np.ones_like(w) if weights is None else np.asarray(weights, dtype=float)

    max_b = LEE_SLOPE_BOUND if enforce_no_arbitrage else 10.0
    bounds = (
        [-np.inf, 1e-8, -0.999, float(np.min(k)) - 1.0, 1e-4],
        [float(np.max(w)) * 2.0 + 1e-6, max_b, 0.999, float(np.max(k)) + 1.0, 5.0],
    )

    starts = [np.clip(guess, bounds[0], bounds[1]) for guess in _initial_guesses(k, w)]
    ladder = PENALTY_LADDER if enforce_no_arbitrage else (0.0,)

    best, best_violation = None, -np.inf
    for penalty in ladder:
        candidate = _best_of_starts(starts, bounds, k, w, weights, penalty)
        violation = _worst_butterfly(candidate)
        if violation > best_violation:
            best, best_violation = candidate, violation
        if violation >= 0.0:
            break

    if best_violation < 0.0:
        best = _repair_butterfly(best)
    params = SVIParams(*best)
    try:
        params.validate()
    except ValueError as exc:
        raise ValueError(
            "SVI calibration did not reach an admissible smile; the quotes are "
            f"probably not arbitrage-free ({exc})"
        ) from exc
    return SVISmile(params, expiry, forward)


def _best_of_starts(starts, bounds, k, w, weights, penalty):
    """Lowest-cost solution across the multi-start set at one penalty weight."""
    best, best_cost = None, np.inf
    for start in starts:
        result = least_squares(
            _residuals, start, bounds=bounds, method="trf",
            args=(k, w, weights, penalty),
        )
        if result.cost < best_cost:
            best, best_cost = result.x, result.cost
    return best


def _repair_butterfly(theta):
    """Shrink the smile until the butterfly condition holds exactly.

    A penalty approaches the no-arbitrage boundary from the wrong side: the
    violation shrinks with every increase in weight but never reaches zero, so
    a fit that looks converged can still carry a negative probability a
    fraction of a basis point deep. Rather than declare that close enough,
    ``b`` is scaled back -- which flattens the smile and lifts ``g`` toward the
    flat-smile value of one -- with ``a`` raised to hold the at-the-money
    variance fixed, so the level the whole density is anchored on does not move.

    The search is a descending scan, so it returns the largest scale that
    verifies clean. A scale of zero gives a flat smile, which always satisfies
    the condition, so it cannot fail to terminate.
    """
    a, b, rho, m, sigma = theta
    atm_anchor = float(-rho * m + np.hypot(m, sigma))
    for scale in np.linspace(1.0, 0.0, SHRINK_TRIALS):
        candidate = np.array([a + b * (1.0 - scale) * atm_anchor, b * scale, rho, m, sigma])
        if _worst_butterfly(candidate) >= 0.0:
            return candidate
    return np.array([a + b * atm_anchor, 0.0, rho, m, sigma])  # pragma: no cover


def _worst_butterfly(theta) -> float:
    """Most negative value of ``g`` on the verification grid; ``0`` when clean."""
    total_variance, first, second = svi_arrays(_VERIFICATION_GRID, *theta)
    g = durrleman_g(_VERIFICATION_GRID, total_variance, first, second)
    return float(min(0.0, np.min(g)))
