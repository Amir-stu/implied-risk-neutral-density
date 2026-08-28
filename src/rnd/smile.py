"""Smile representations: total implied variance as a smooth function of log-strike.

A smile is the only object the density estimator needs. Whatever produced it --
a spline through cleaned quotes, a parametric SVI fit, or a textbook flat
volatility -- it must answer three questions at any log-moneyness ``k``:
the total implied variance ``w(k)``, and its first two derivatives. Those three
numbers are enough to write the risk-neutral density in closed form, which is
why nothing in this package differentiates a noisy price curve twice.

Log-moneyness is always ``k = log(K / F)`` against the *forward*, not spot.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from scipy.interpolate import UnivariateSpline

from .black import call_price

#: Roger Lee's moment formula caps the asymptotic slope of total implied
#: variance in log-strike at 2. A steeper wing implies a distribution with no
#: finite moments of any order and, in practice, negative probabilities.
LEE_SLOPE_BOUND = 2.0

MIN_TOTAL_VARIANCE = 1e-10


def durrleman_g(k, total_variance, first_derivative, second_derivative):
    """Durrleman's function, from a smile and its first two derivatives.

    It is non-negative if and only if the smile admits no butterfly arbitrage,
    and it is the numerator of the risk-neutral density. Kept as a free
    function so that a calibration can evaluate it on raw parameters, before
    anything has been validated or wrapped in an object.
    """
    k = np.asarray(k, dtype=float)
    w = np.maximum(np.asarray(total_variance, dtype=float), MIN_TOTAL_VARIANCE)
    dw = np.asarray(first_derivative, dtype=float)
    drift = (1.0 - k * dw / (2.0 * w)) ** 2
    convexity = 0.25 * dw**2 * (1.0 / w + 0.25)
    return drift - convexity + 0.5 * np.asarray(second_derivative, dtype=float)


class Smile(ABC):
    """Total implied variance ``w(k)`` and the derivatives the density needs."""

    def __init__(self, expiry: float, forward: float):
        if expiry <= 0.0:
            raise ValueError("expiry must be positive")
        if forward <= 0.0:
            raise ValueError("forward must be positive")
        self.expiry = float(expiry)
        self.forward = float(forward)

    @abstractmethod
    def total_variance(self, k):
        """Total implied variance ``sigma(k)**2 * T``."""

    @abstractmethod
    def d_total_variance(self, k):
        """First derivative of the total variance in log-strike."""

    @abstractmethod
    def d2_total_variance(self, k):
        """Second derivative of the total variance in log-strike."""

    def implied_vol(self, k):
        """Annualised Black implied volatility at log-moneyness ``k``."""
        return np.sqrt(np.maximum(self.total_variance(k), MIN_TOTAL_VARIANCE) / self.expiry)

    def log_moneyness(self, strike):
        """Convert strikes to log-moneyness against this smile's forward."""
        return np.log(np.asarray(strike, dtype=float) / self.forward)

    def call_price(self, strike, discount: float = 1.0):
        """Black-76 call price implied by the smile at the given strikes."""
        k = self.log_moneyness(strike)
        return call_price(self.forward, np.asarray(strike, dtype=float),
                          self.total_variance(k), discount)

    def durrleman_g(self, k):
        """Durrleman's ``g(k)`` for this smile."""
        k = np.asarray(k, dtype=float)
        return durrleman_g(k, self.total_variance(k), self.d_total_variance(k),
                           self.d2_total_variance(k))

    def butterfly_violation(self, k_grid=None) -> float:
        """Worst (most negative) value of ``g`` on a grid; ``0.0`` when clean."""
        grid = np.linspace(-1.5, 1.5, 601) if k_grid is None else np.asarray(k_grid, dtype=float)
        return float(min(0.0, np.min(self.durrleman_g(grid))))


class FlatSmile(Smile):
    """Constant volatility -- the Black-Scholes special case, useful as a control."""

    def __init__(self, vol: float, expiry: float, forward: float):
        super().__init__(expiry, forward)
        if vol <= 0.0:
            raise ValueError("vol must be positive")
        self.vol = float(vol)

    def total_variance(self, k):
        return np.full_like(np.asarray(k, dtype=float), self.vol**2 * self.expiry)

    def d_total_variance(self, k):
        return np.zeros_like(np.asarray(k, dtype=float))

    def d2_total_variance(self, k):
        return np.zeros_like(np.asarray(k, dtype=float))


WING_LENGTH = 8.0
WING_GRID_POINTS = 401
SLOPE_TRIAL_COUNT = 41


def _linear_wing_g(offsets, k_join: float, w_join: float, slope: float):
    """Durrleman's ``g`` along a wing that continues linearly in total variance."""
    k = k_join + offsets
    w = np.maximum(w_join + slope * offsets, MIN_TOTAL_VARIANCE)
    return (1.0 - k * slope / (2.0 * w)) ** 2 - 0.25 * slope**2 * (1.0 / w + 0.25)


def _admissible_wing_slope(k_join: float, w_join: float, raw_slope: float, sign: float) -> float:
    """Steepest wing slope that keeps the extrapolation free of butterfly arbitrage.

    Two constraints bind. Lee caps the magnitude at 2, and the sign is fixed so
    that total variance never falls away from the data, which would eventually
    drive it through zero. Within what is left, the raw slope is scaled back
    until ``g`` is non-negative across the whole wing. A flat wing always
    satisfies the condition, so the search cannot fail; when it has to go that
    far it means the spline was bending hard at the edge of the quoted range,
    and extrapolating that bend is exactly what should not be done.
    """
    capped = float(np.clip(raw_slope, 0.0, LEE_SLOPE_BOUND) if sign > 0
                   else np.clip(raw_slope, -LEE_SLOPE_BOUND, 0.0))
    offsets = sign * np.linspace(0.0, WING_LENGTH, WING_GRID_POINTS)
    for scale in np.linspace(1.0, 0.0, SLOPE_TRIAL_COUNT):
        slope = capped * scale
        if np.min(_linear_wing_g(offsets, k_join, w_join, slope)) >= 0.0:
            return float(slope)
    return 0.0


class SplineSmile(Smile):
    """Smoothing cubic spline in ``(k, w)`` with arbitrage-aware linear wings.

    Inside the quoted strike range the spline does the work. Outside it the
    smile continues linearly in total variance. Two rules govern that
    continuation: Lee's bound caps the slope at 2 in absolute value, and the
    slope is scaled back further, if needed, until Durrleman's ``g`` stays
    non-negative along the whole wing.

    The second rule is not redundant. A linear wing has no curvature, so it
    loses the ``w'' / 2`` term that was holding ``g`` up on the spline side of
    the join; a smile bending sharply at the last quoted strike will go
    butterfly-negative immediately outside it if the slope is simply carried
    over. The price of the fix is a small discontinuity in the density's
    curvature at the edge of the quoted range, which is the honest place for it:
    beyond the last quote there is no information, only extrapolation.
    """

    def __init__(self, spline: UnivariateSpline, k_min: float, k_max: float,
                 expiry: float, forward: float):
        super().__init__(expiry, forward)
        self._spline = spline
        self._d1 = spline.derivative(1)
        self._d2 = spline.derivative(2)
        self.k_min = float(k_min)
        self.k_max = float(k_max)
        self._left_value = float(spline(k_min))
        self._right_value = float(spline(k_max))
        self._left_slope = _admissible_wing_slope(
            self.k_min, self._left_value, float(self._d1(k_min)), sign=-1.0
        )
        self._right_slope = _admissible_wing_slope(
            self.k_max, self._right_value, float(self._d1(k_max)), sign=1.0
        )

    def _regions(self, k):
        k = np.atleast_1d(np.asarray(k, dtype=float))
        return k, k < self.k_min, k > self.k_max

    def total_variance(self, k):
        k, left, right = self._regions(k)
        core = np.clip(k, self.k_min, self.k_max)
        out = np.asarray(self._spline(core), dtype=float)
        out = np.where(left, self._left_value + self._left_slope * (k - self.k_min), out)
        out = np.where(right, self._right_value + self._right_slope * (k - self.k_max), out)
        return np.maximum(out, MIN_TOTAL_VARIANCE)

    def d_total_variance(self, k):
        k, left, right = self._regions(k)
        out = np.asarray(self._d1(np.clip(k, self.k_min, self.k_max)), dtype=float)
        out = np.where(left, self._left_slope, out)
        return np.where(right, self._right_slope, out)

    def d2_total_variance(self, k):
        k, left, right = self._regions(k)
        out = np.asarray(self._d2(np.clip(k, self.k_min, self.k_max)), dtype=float)
        return np.where(left | right, 0.0, out)


def fit_spline_smile(
    k,
    total_variance,
    expiry: float,
    forward: float,
    weights=None,
    smoothing: float | None = None,
) -> SplineSmile:
    """Fit a weighted smoothing spline to observed total variances.

    ``smoothing`` is the usual spline penalty: ``0`` interpolates every quote
    and inherits every tick of its noise, which is the classic way to produce a
    density that oscillates through zero. When weights are supplied as inverse
    standard errors, the residual sum entering the penalty is a chi-square with
    roughly one degree of freedom per quote, so the default is the sample size:
    the fit is allowed to miss each quote by about its own uncertainty and no
    more.
    """
    k = np.asarray(k, dtype=float)
    w = np.asarray(total_variance, dtype=float)
    if k.size < 4:
        raise ValueError("a cubic spline needs at least 4 quotes")
    order = np.argsort(k)
    k, w = k[order], w[order]
    weights = None if weights is None else np.asarray(weights, dtype=float)[order]
    if smoothing is None:
        smoothing = float(k.size) if weights is not None else 1e-6 * k.size
    spline = UnivariateSpline(k, w, w=weights, k=3, s=smoothing, ext="const")
    return SplineSmile(spline, k[0], k[-1], expiry, forward)
