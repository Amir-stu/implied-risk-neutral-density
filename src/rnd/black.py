"""Black-76 pricing on the forward, plus a robust implied-volatility solver.

Everything downstream is expressed in terms of the forward ``F`` and the total
implied variance ``w = sigma**2 * T`` rather than spot and an annualised vol.
That choice is not cosmetic: the no-arbitrage conditions on a smile are
statements about ``w`` as a function of log-moneyness, and the density formula
in :mod:`rnd.density` falls out of the same parameterisation.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

MIN_TOTAL_VARIANCE = 1e-12
MAX_IMPLIED_VOL = 5.0
MIN_IMPLIED_VOL = 1e-6
IV_SOLVER_TOLERANCE = 1e-10


def d1_d2(forward: np.ndarray, strike: np.ndarray, total_variance: np.ndarray):
    """Return the Black-76 ``d1``/``d2`` terms for a given total variance."""
    w = np.maximum(np.asarray(total_variance, dtype=float), MIN_TOTAL_VARIANCE)
    sqrt_w = np.sqrt(w)
    log_moneyness = np.log(np.asarray(forward, dtype=float) / np.asarray(strike, dtype=float))
    d1 = log_moneyness / sqrt_w + 0.5 * sqrt_w
    return d1, d1 - sqrt_w


def call_price(forward, strike, total_variance, discount=1.0):
    """Undiscounted-forward Black-76 call, scaled by the discount factor."""
    d1, d2 = d1_d2(forward, strike, total_variance)
    undiscounted = forward * norm.cdf(d1) - strike * norm.cdf(d2)
    return discount * undiscounted


def put_price(forward, strike, total_variance, discount=1.0):
    """Black-76 put, by put-call parity on the forward."""
    call = call_price(forward, strike, total_variance, discount)
    return call - discount * (np.asarray(forward, dtype=float) - np.asarray(strike, dtype=float))


def vega(forward, strike, total_variance, expiry, discount=1.0):
    """Sensitivity of the option price to a one-unit move in annualised vol."""
    d1, _ = d1_d2(forward, strike, total_variance)
    return discount * forward * norm.pdf(d1) * np.sqrt(max(float(expiry), MIN_TOTAL_VARIANCE))


def _intrinsic(forward: float, strike: float, discount: float, is_call: bool) -> float:
    sign = 1.0 if is_call else -1.0
    return discount * max(sign * (forward - strike), 0.0)


def implied_vol(
    price: float,
    forward: float,
    strike: float,
    expiry: float,
    discount: float = 1.0,
    is_call: bool = True,
) -> float:
    """Invert Black-76 for annualised volatility.

    Returns ``nan`` when the quote sits outside the no-arbitrage price bounds,
    which is the honest answer for a stale or crossed quote. Callers are
    expected to drop those points rather than fit through them.
    """
    if not np.isfinite(price) or price <= 0.0 or expiry <= 0.0:
        return float("nan")

    lower = _intrinsic(forward, strike, discount, is_call)
    upper = discount * (forward if is_call else strike)
    if price <= lower + IV_SOLVER_TOLERANCE or price >= upper:
        return float("nan")

    pricer = call_price if is_call else put_price

    def objective(sigma: float) -> float:
        return float(pricer(forward, strike, sigma * sigma * expiry, discount)) - price

    if objective(MIN_IMPLIED_VOL) > 0.0 or objective(MAX_IMPLIED_VOL) < 0.0:
        return float("nan")
    return float(brentq(objective, MIN_IMPLIED_VOL, MAX_IMPLIED_VOL, xtol=IV_SOLVER_TOLERANCE))


def implied_vol_vector(prices, forward, strikes, expiry, discount=1.0, is_call=True):
    """Element-wise :func:`implied_vol` over arrays of prices, strikes and flags."""
    prices = np.atleast_1d(np.asarray(prices, dtype=float))
    strikes = np.atleast_1d(np.asarray(strikes, dtype=float))
    flags = np.broadcast_to(np.atleast_1d(np.asarray(is_call)), prices.shape)
    return np.array(
        [
            implied_vol(float(p), forward, float(k), expiry, discount, bool(flag))
            for p, k, flag in zip(prices, strikes, flags, strict=True)
        ]
    )
