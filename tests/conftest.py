"""Shared fixtures: one synthetic market with a known answer, reused everywhere."""

from __future__ import annotations

import pytest

from rnd.data.synthetic import LognormalMixture, synthetic_chain
from rnd.pipeline import estimate_density

EXPIRY = 0.25
FORWARD = 100.0
DISCOUNT = 0.99


@pytest.fixture(scope="session")
def mixture() -> LognormalMixture:
    """A calm regime plus a 5% chance of a 15% drop, priced to a forward of 100."""
    return LognormalMixture.crash_scenario(forward=FORWARD, expiry=EXPIRY)


@pytest.fixture(scope="session")
def clean_chain(mixture):
    """A chain quoted straight off the model, with a spread but no noise."""
    return synthetic_chain(mixture, EXPIRY, discount=DISCOUNT, relative_spread=0.02,
                           tick=0.01, seed=1)


@pytest.fixture(scope="session")
def noisy_chain(mixture):
    """The same chain with independent mid-price error on every quote."""
    return synthetic_chain(mixture, EXPIRY, discount=DISCOUNT, relative_spread=0.03,
                           tick=0.05, noise_bps=40.0, seed=2)


@pytest.fixture(scope="session")
def svi_result(clean_chain):
    return estimate_density(clean_chain, method="svi")


@pytest.fixture(scope="session")
def spline_result(clean_chain):
    return estimate_density(clean_chain, method="spline")
