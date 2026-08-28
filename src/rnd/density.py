"""Breeden-Litzenberger: from a call-price surface to a probability density.

The result of Breeden and Litzenberger (1978) is that the risk-neutral density
is the second derivative of the discounted call price in the strike::

    q(K) = exp(r * T) * d2C / dK2

Applied literally to quoted prices that statement is close to useless: two
numerical derivatives amplify the bid-ask noise until the "density" oscillates
through zero. The fix used here is to differentiate the *smile* instead. Given
total implied variance ``w(k)`` and its first two derivatives, the density of
``k = log(K / F)`` is available in closed form,

    p(k) = g(k) / sqrt(2 * pi * w(k)) * exp(-d2(k) ** 2 / 2)

where ``g`` is the Durrleman function (see :meth:`rnd.smile.Smile.durrleman_g`).
This is the same object as the finite-difference estimate --
:func:`density_from_prices` is kept so the two can be compared -- but it is
exact, and its non-negativity is a checkable property of the smile rather than
an accident of the grid.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.integrate import cumulative_trapezoid, trapezoid

from .smile import Smile

MIN_TOTAL_VARIANCE = 1e-10


@dataclass(frozen=True)
class DensityMoments:
    """Summary statistics of a risk-neutral density."""

    mean: float
    std: float
    skewness: float
    excess_kurtosis: float

    def as_dict(self) -> dict[str, float]:
        return {
            "mean": self.mean,
            "std": self.std,
            "skewness": self.skewness,
            "excess_kurtosis": self.excess_kurtosis,
        }


class RiskNeutralDensity:
    """A density over terminal price, tabulated on a strike grid.

    The grid is truncated by construction, so :attr:`total_mass` is reported
    rather than assumed to be one. A mass far from one means the grid missed
    part of the distribution, and every statistic below inherits that error.
    """

    def __init__(self, strikes, density, forward: float, expiry: float):
        strikes = np.asarray(strikes, dtype=float)
        density = np.asarray(density, dtype=float)
        if strikes.ndim != 1 or strikes.size != density.size:
            raise ValueError("strikes and density must be 1-D arrays of equal length")
        if np.any(np.diff(strikes) <= 0.0):
            raise ValueError("strikes must be strictly increasing")
        self.strikes = strikes
        self.density = density
        self.forward = float(forward)
        self.expiry = float(expiry)

    @property
    def total_mass(self) -> float:
        """Integral of the density over the grid; ``1`` for an untruncated density."""
        return float(trapezoid(self.density, self.strikes))

    @property
    def min_density(self) -> float:
        """Most negative value on the grid. Anything below zero is a red flag."""
        return float(np.min(self.density))

    def normalised(self) -> RiskNeutralDensity:
        """Copy rescaled to unit mass, with any negative values clipped away."""
        clipped = np.maximum(self.density, 0.0)
        mass = float(trapezoid(clipped, self.strikes))
        if mass <= 0.0:
            raise ValueError("density integrates to zero; nothing to normalise")
        return RiskNeutralDensity(self.strikes, clipped / mass, self.forward, self.expiry)

    def cdf(self):
        """Cumulative distribution on the strike grid."""
        return np.concatenate([[0.0], cumulative_trapezoid(self.density, self.strikes)])

    def prob_below(self, level: float) -> float:
        """Risk-neutral probability that the terminal price ends below ``level``."""
        return float(np.interp(level, self.strikes, self.cdf()))

    def prob_between(self, low: float, high: float) -> float:
        """Probability of finishing inside the interval."""
        if high < low:
            raise ValueError("high must not be below low")
        return self.prob_below(high) - self.prob_below(low)

    def quantile(self, probability: float) -> float:
        """Inverse CDF. The 5% point is the risk-neutral downside level."""
        if not 0.0 < probability < 1.0:
            raise ValueError("probability must lie strictly inside (0, 1)")
        cdf = self.cdf()
        return float(np.interp(probability * cdf[-1], cdf, self.strikes))

    def raw_moment(self, order: int) -> float:
        """Expectation of the terminal price raised to ``order``, over the grid."""
        return float(trapezoid(self.strikes**order * self.density, self.strikes))

    def moments(self) -> DensityMoments:
        """Mean, standard deviation, skewness and excess kurtosis."""
        mass = self.total_mass
        mean = self.raw_moment(1) / mass
        centred = self.strikes - mean
        variance = float(trapezoid(centred**2 * self.density, self.strikes)) / mass
        std = float(np.sqrt(max(variance, 0.0)))
        if std <= 0.0:
            return DensityMoments(mean, 0.0, float("nan"), float("nan"))
        third = float(trapezoid(centred**3 * self.density, self.strikes)) / mass
        fourth = float(trapezoid(centred**4 * self.density, self.strikes)) / mass
        return DensityMoments(mean, std, third / std**3, fourth / std**4 - 3.0)

    def martingale_error(self) -> float:
        """Relative gap between the density mean and the forward.

        Under the risk-neutral measure the forward *is* the expected terminal
        price. A large error means the smile, the forward or the grid is wrong;
        it is the single most informative check in the package.
        """
        return self.moments().mean / self.forward - 1.0

    def to_frame(self) -> pd.DataFrame:
        """Tabulate strike, density and CDF."""
        return pd.DataFrame(
            {"strike": self.strikes, "density": self.density, "cdf": self.cdf()}
        )


MAX_LOG_HALF_WIDTH = 6.0


def strike_grid(smile: Smile, n_points: int = 601, width: float = 6.0) -> np.ndarray:
    """A log-uniform strike grid spanning ``width`` standard deviations.

    Sizing the grid off the at-the-money volatility alone truncates a skewed
    density: the left wing of an index smile can trade ten volatility points
    above the money, so a grid built for the centre cuts off exactly the tail
    the exercise is about. The width is therefore expanded until it is
    consistent with the volatility at its own edges.
    """
    if n_points < 3:
        raise ValueError("need at least three grid points")
    variance = float(np.atleast_1d(smile.total_variance(0.0))[0])
    half_width = width * float(np.sqrt(max(variance, MIN_TOTAL_VARIANCE)))
    for _ in range(8):
        edge = float(np.max(smile.total_variance(np.array([-half_width, half_width]))))
        candidate = min(width * float(np.sqrt(max(edge, MIN_TOTAL_VARIANCE))),
                        MAX_LOG_HALF_WIDTH)
        if candidate <= half_width * 1.001:
            break
        half_width = candidate
    return smile.forward * np.exp(np.linspace(-half_width, half_width, n_points))


def density_from_smile(
    smile: Smile, strikes=None, n_points: int = 601, width: float = 6.0
) -> RiskNeutralDensity:
    """Risk-neutral density implied by a smile, in closed form.

    Returned in strike space as ``q(K) = p(log(K / F)) / K``.
    """
    if strikes is None:
        strikes = strike_grid(smile, n_points, width)
    else:
        strikes = np.asarray(strikes, dtype=float)
    k = np.log(strikes / smile.forward)
    w = np.maximum(smile.total_variance(k), MIN_TOTAL_VARIANCE)
    sqrt_w = np.sqrt(w)
    d2 = -k / sqrt_w - 0.5 * sqrt_w
    log_density = smile.durrleman_g(k) * np.exp(-0.5 * d2**2) / np.sqrt(2.0 * np.pi * w)
    return RiskNeutralDensity(strikes, log_density / strikes, smile.forward, smile.expiry)


def density_from_prices(
    strikes, call_prices, forward: float, expiry: float, discount: float = 1.0
) -> RiskNeutralDensity:
    """Breeden-Litzenberger by finite differences on a call-price curve.

    Included as a reference implementation and a cross-check on
    :func:`density_from_smile`. On raw quotes it is exactly as fragile as the
    literature says; on a smoothed curve the two agree to grid accuracy.
    """
    strikes = np.asarray(strikes, dtype=float)
    prices = np.asarray(call_prices, dtype=float)
    if strikes.size < 3:
        raise ValueError("need at least three strikes to take a second difference")
    if np.any(np.diff(strikes) <= 0.0):
        raise ValueError("strikes must be strictly increasing")
    first = np.gradient(prices, strikes, edge_order=2)
    second = np.gradient(first, strikes, edge_order=2)
    return RiskNeutralDensity(strikes, second / discount, forward, expiry)
