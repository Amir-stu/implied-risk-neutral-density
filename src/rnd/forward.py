"""Implied forward and discount factor, backed out of the chain itself.

Nothing else in the package asks the user for a risk-free rate or a dividend
yield, and that is deliberate. Put-call parity holds on every listed pair,

    C(K) - P(K) = D * (F - K),

so a regression of the call-put spread on the strike gives the discount factor
as minus the slope and the forward as the intercept over that slope. What comes
out is the forward the options market is actually trading, including borrow
costs, hard-to-borrow spreads and the dividends the market expects rather than
the ones a data vendor has on file.

Using a textbook ``S * exp((r - q) * T)`` instead is the most common way to get
a visibly wrong density: a forward that is off by half a percent tilts the whole
distribution and shows up as a martingale error the size of the skew you were
trying to measure.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MIN_PARITY_PAIRS = 3
#: A discount factor above one means a negative implied rate. That is a real
#: state of the world rather than an error, so the ceiling is loose enough to
#: admit it and tight enough to still catch mismatched or mixed-expiry quotes.
MAX_DISCOUNT = 1.05


@dataclass(frozen=True)
class ForwardEstimate:
    """Forward, discount factor and the quality of the parity regression."""

    forward: float
    discount: float
    expiry: float
    n_pairs: int
    r_squared: float

    @property
    def implied_rate(self) -> float:
        """Continuously compounded rate implied by the discount factor."""
        return -np.log(self.discount) / self.expiry

    def as_dict(self) -> dict[str, float]:
        return {
            "forward": self.forward,
            "discount": self.discount,
            "implied_rate": self.implied_rate,
            "n_pairs": float(self.n_pairs),
            "r_squared": self.r_squared,
        }


def _select_atm_band(strikes: np.ndarray, spread: np.ndarray, band: int) -> np.ndarray:
    """Indices of the strikes closest to the money.

    Parity is exact everywhere in theory and reliable only near the money in
    practice: deep in-the-money quotes are wide, stale and mostly intrinsic, so
    including them lets one bad print set the forward.
    """
    if band <= 0 or band >= strikes.size:
        return np.arange(strikes.size)
    atm_index = int(np.argmin(np.abs(spread)))
    lo = max(0, atm_index - band)
    return np.arange(lo, min(strikes.size, lo + 2 * band + 1))


def implied_forward(
    strikes,
    call_prices,
    put_prices,
    expiry: float,
    weights=None,
    band: int = 8,
) -> ForwardEstimate:
    """Fit put-call parity across matched call and put quotes.

    ``band`` limits the regression to roughly that many strikes either side of
    the money. Pass ``0`` to use every pair supplied.
    """
    strikes = np.asarray(strikes, dtype=float)
    calls = np.asarray(call_prices, dtype=float)
    puts = np.asarray(put_prices, dtype=float)
    if not strikes.size == calls.size == puts.size:
        raise ValueError("strikes, calls and puts must have the same length")
    if expiry <= 0.0:
        raise ValueError("expiry must be positive")
    spread = calls - puts

    order = np.argsort(strikes)
    strikes, spread = strikes[order], spread[order]
    weights = np.ones_like(strikes) if weights is None else np.asarray(weights, float)[order]

    keep = _select_atm_band(strikes, spread, band)
    strikes, spread, weights = strikes[keep], spread[keep], weights[keep]
    if strikes.size < MIN_PARITY_PAIRS:
        raise ValueError(f"need at least {MIN_PARITY_PAIRS} call-put pairs to fit a forward")

    design = np.column_stack([np.ones_like(strikes), strikes])
    sqrt_w = np.sqrt(np.maximum(weights, 0.0))[:, None]
    coefficients, *_ = np.linalg.lstsq(design * sqrt_w, spread * sqrt_w[:, 0], rcond=None)
    intercept, slope = float(coefficients[0]), float(coefficients[1])

    discount = -slope
    if not 0.0 < discount <= MAX_DISCOUNT:
        raise ValueError(
            f"parity regression implied a discount factor of {discount:.4f}, outside "
            f"(0, {MAX_DISCOUNT}]; the quotes are inconsistent with a single expiry"
        )

    fitted = design @ coefficients
    residual_ss = float(np.sum(weights * (spread - fitted) ** 2))
    total_ss = float(np.sum(weights * (spread - np.average(spread, weights=weights)) ** 2))
    r_squared = 1.0 - residual_ss / total_ss if total_ss > 0.0 else 1.0

    return ForwardEstimate(
        forward=intercept / discount,
        discount=discount,
        expiry=float(expiry),
        n_pairs=int(strikes.size),
        r_squared=float(r_squared),
    )


def forward_from_carry(spot: float, expiry: float, rate: float, dividend_yield: float = 0.0):
    """Textbook cost-of-carry forward, for when no put quotes exist.

    A fallback, not a default. It assumes the rate and dividend yield handed to
    it are the ones the option market is pricing, which is exactly the
    assumption :func:`implied_forward` exists to avoid.
    """
    if spot <= 0.0 or expiry <= 0.0:
        raise ValueError("spot and expiry must be positive")
    discount = float(np.exp(-rate * expiry))
    forward = float(spot * np.exp((rate - dividend_yield) * expiry))
    return ForwardEstimate(forward, discount, float(expiry), n_pairs=0, r_squared=float("nan"))
