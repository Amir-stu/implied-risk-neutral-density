"""A market with a known answer.

Estimating a density from real quotes has no ground truth: whatever comes out,
there is nothing to compare it against. So the test suite builds chains from a
mixture of lognormals, which has closed-form option prices *and* a closed-form
density. The estimator is then asked to recover a distribution that is known
exactly, including its skew and its left tail.

Two lognormal components with different forwards is also the cheapest honest
model of what an index smile is pricing: a benign regime, and a smaller
probability of a sharp drop.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import lognorm

from ..black import call_price, put_price
from ..chain import CALL, PUT, OptionChain


@dataclass(frozen=True)
class LognormalMixture:
    """Terminal price distribution as a weighted mixture of lognormals.

    Each component is described by its own forward and total variance, so the
    mixture forward is just the weighted sum of the component forwards and the
    martingale property holds by construction.
    """

    weights: np.ndarray
    forwards: np.ndarray
    total_variances: np.ndarray

    def __post_init__(self):
        weights = np.asarray(self.weights, dtype=float)
        forwards = np.asarray(self.forwards, dtype=float)
        variances = np.asarray(self.total_variances, dtype=float)
        if not (weights.size == forwards.size == variances.size):
            raise ValueError("weights, forwards and variances must have equal length")
        if np.any(weights < 0.0) or not np.isclose(weights.sum(), 1.0):
            raise ValueError("weights must be non-negative and sum to one")
        if np.any(forwards <= 0.0) or np.any(variances <= 0.0):
            raise ValueError("component forwards and variances must be positive")
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "forwards", forwards)
        object.__setattr__(self, "total_variances", variances)

    @classmethod
    def crash_scenario(
        cls,
        forward: float,
        expiry: float,
        base_vol: float = 0.16,
        crash_probability: float = 0.05,
        crash_size: float = 0.15,
        crash_vol: float = 0.32,
    ) -> LognormalMixture:
        """A calm regime plus a jump down, calibrated to a given forward.

        The benign component is scaled up so the mixture still prices to
        ``forward``; without that the synthetic market would fail its own
        martingale test before the estimator ever ran.
        """
        if not 0.0 <= crash_probability < 1.0:
            raise ValueError("crash probability must lie in [0, 1)")
        if not 0.0 < crash_size < 1.0:
            raise ValueError("crash size must be a fraction strictly inside (0, 1)")
        calm_forward = forward / (1.0 - crash_probability * crash_size)
        return cls(
            weights=np.array([1.0 - crash_probability, crash_probability]),
            forwards=np.array([calm_forward, calm_forward * (1.0 - crash_size)]),
            total_variances=np.array([base_vol**2 * expiry, crash_vol**2 * expiry]),
        )

    @property
    def forward(self) -> float:
        """Expected terminal price under the mixture."""
        return float(np.dot(self.weights, self.forwards))

    def _component_scales(self):
        sigma = np.sqrt(self.total_variances)
        return sigma, self.forwards * np.exp(-0.5 * self.total_variances)

    def pdf(self, price):
        """Exact risk-neutral density at the given terminal prices."""
        price = np.asarray(price, dtype=float)
        sigma, scale = self._component_scales()
        components = [
            w * lognorm.pdf(price, s, scale=sc)
            for w, s, sc in zip(self.weights, sigma, scale, strict=True)
        ]
        return np.sum(components, axis=0)

    def cdf(self, price):
        """Exact risk-neutral cumulative distribution."""
        price = np.asarray(price, dtype=float)
        sigma, scale = self._component_scales()
        components = [
            w * lognorm.cdf(price, s, scale=sc)
            for w, s, sc in zip(self.weights, sigma, scale, strict=True)
        ]
        return np.sum(components, axis=0)

    def call_price(self, strike, discount: float = 1.0):
        """Closed-form call price: the mixture prices as a mixture of Blacks."""
        strike = np.asarray(strike, dtype=float)
        legs = [
            w * call_price(f, strike, v, discount)
            for w, f, v in zip(self.weights, self.forwards, self.total_variances, strict=True)
        ]
        return np.sum(legs, axis=0)

    def put_price(self, strike, discount: float = 1.0):
        """Closed-form put price."""
        strike = np.asarray(strike, dtype=float)
        legs = [
            w * put_price(f, strike, v, discount)
            for w, f, v in zip(self.weights, self.forwards, self.total_variances, strict=True)
        ]
        return np.sum(legs, axis=0)


def synthetic_chain(
    model: LognormalMixture,
    expiry: float,
    strikes=None,
    discount: float = 1.0,
    relative_spread: float = 0.02,
    tick: float = 0.01,
    noise_bps: float = 0.0,
    seed: int | None = None,
) -> OptionChain:
    """Quote a full chain off the model, with a spread and optional noise.

    ``relative_spread`` and ``tick`` reproduce the two features of a real board
    that matter most for this problem: prices are only known inside a spread,
    and they are rounded. ``noise_bps`` adds independent mid-price error on top,
    which is what makes the naive second difference fall apart.
    """
    if strikes is None:
        atm_sigma = float(np.sqrt(np.max(model.total_variances)))
        strikes = np.round(model.forward * np.exp(np.linspace(-2.5, 2.0, 41) * atm_sigma), 2)
    strikes = np.asarray(strikes, dtype=float)
    rng = np.random.default_rng(seed)

    records = []
    for strike in strikes:
        for option_type, pricer in ((CALL, model.call_price), (PUT, model.put_price)):
            mid = float(pricer(strike, discount))
            if noise_bps > 0.0:
                mid *= 1.0 + rng.normal(scale=noise_bps * 1e-4)
            half = max(0.5 * relative_spread * mid, tick)
            bid = max(np.floor((mid - half) / tick) * tick, 0.0)
            ask = np.ceil((mid + half) / tick) * tick
            records.append(
                {"strike": strike, "option_type": option_type, "bid": bid, "ask": ask}
            )
    return OptionChain.from_records(records, expiry, spot=model.forward * discount,
                                    underlying="SYNTH")
