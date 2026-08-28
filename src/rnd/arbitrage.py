"""Static no-arbitrage checks on a call-price curve.

Three conditions have to hold across strikes at a single expiry, and each one
maps onto a property of the density that Breeden-Litzenberger would produce:

* price bounds -- the call sits between its intrinsic value and the discounted
  forward, or the implied distribution puts mass outside ``[0, inf)``;
* monotonicity -- the call price falls as the strike rises, with slope no
  steeper than the discount factor, or the implied CDF leaves ``[0, 1]``;
* convexity -- the second difference is non-negative, or the density is
  negative somewhere. This is the butterfly condition: the price of the spread
  *is* the probability, so a negative butterfly is a negative probability
  quoted on screen.

Run this before fitting. A chain that fails convexity in the middle of the
board has a data problem, and no amount of smoothing will fix it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

TOLERANCE = 1e-8


@dataclass(frozen=True)
class ArbitrageReport:
    """What the checks found, with the worst offender in each category.

    ``offending_strikes`` lists where each violated condition is *centred*, not
    which quote is to blame. A single bad print breaks the two butterflies on
    either side of it, so the culprit is usually the strike the flagged ones
    surround.
    """

    n_strikes: int
    bound_violations: int
    monotonicity_violations: int
    convexity_violations: int
    worst_butterfly: float
    offending_strikes: list[float] = field(default_factory=list)

    @property
    def is_clean(self) -> bool:
        """True when the curve admits no static arbitrage."""
        return (
            self.bound_violations == 0
            and self.monotonicity_violations == 0
            and self.convexity_violations == 0
        )

    def summary(self) -> str:
        """One-line verdict for logs and the CLI."""
        if self.is_clean:
            return f"no static arbitrage across {self.n_strikes} strikes"
        return (
            f"{self.bound_violations} bound, {self.monotonicity_violations} monotonicity "
            f"and {self.convexity_violations} convexity violations across "
            f"{self.n_strikes} strikes (worst butterfly {self.worst_butterfly:.6f})"
        )

    def as_dict(self) -> dict:
        return {
            "n_strikes": self.n_strikes,
            "bound_violations": self.bound_violations,
            "monotonicity_violations": self.monotonicity_violations,
            "convexity_violations": self.convexity_violations,
            "worst_butterfly": self.worst_butterfly,
            "is_clean": self.is_clean,
            "offending_strikes": self.offending_strikes,
        }


def butterfly_values(strikes, call_prices):
    """Price of the unit butterfly centred on each interior strike.

    Uneven strike spacing is handled properly: the wings are weighted so the
    payoff is a genuine long-short-short-long combination rather than a plain
    second difference, which understates violations on a board that thins out
    in the wings.
    """
    strikes = np.asarray(strikes, dtype=float)
    prices = np.asarray(call_prices, dtype=float)
    if strikes.size < 3:
        raise ValueError("a butterfly needs three strikes")
    left_gap = strikes[1:-1] - strikes[:-2]
    right_gap = strikes[2:] - strikes[1:-1]
    left_weight = right_gap / (left_gap + right_gap)
    right_weight = left_gap / (left_gap + right_gap)
    return left_weight * prices[:-2] + right_weight * prices[2:] - prices[1:-1]


def check_call_curve(
    strikes, call_prices, forward: float, discount: float = 1.0, tolerance: float = TOLERANCE
) -> ArbitrageReport:
    """Run all three static checks over a call curve sorted by strike."""
    strikes = np.asarray(strikes, dtype=float)
    prices = np.asarray(call_prices, dtype=float)
    if strikes.size != prices.size:
        raise ValueError("strikes and prices must have the same length")
    if np.any(np.diff(strikes) <= 0.0):
        raise ValueError("strikes must be strictly increasing")

    intrinsic = discount * np.maximum(forward - strikes, 0.0)
    upper = discount * forward
    bound_bad = (prices < intrinsic - tolerance) | (prices > upper + tolerance)

    slope = np.diff(prices) / np.diff(strikes)
    slope_bad = (slope > tolerance) | (slope < -discount - tolerance)

    butterflies = butterfly_values(strikes, prices)
    convexity_bad = butterflies < -tolerance

    offenders = sorted(
        set(strikes[bound_bad].tolist())
        | set(strikes[1:][slope_bad].tolist())
        | set(strikes[1:-1][convexity_bad].tolist())
    )
    return ArbitrageReport(
        n_strikes=int(strikes.size),
        bound_violations=int(bound_bad.sum()),
        monotonicity_violations=int(slope_bad.sum()),
        convexity_violations=int(convexity_bad.sum()),
        worst_butterfly=float(np.min(butterflies)),
        offending_strikes=[float(k) for k in offenders],
    )
