"""Recovering the forward and the discount factor from put-call parity."""

from __future__ import annotations

import numpy as np
import pytest

from rnd.black import call_price, put_price
from rnd.forward import forward_from_carry, implied_forward

EXPIRY = 0.75
FORWARD = 250.0
DISCOUNT = 0.972
TOTAL_VARIANCE = 0.22**2 * EXPIRY
STRIKES = np.linspace(200.0, 300.0, 21)


@pytest.fixture(scope="module")
def parity_quotes():
    return (
        STRIKES,
        call_price(FORWARD, STRIKES, TOTAL_VARIANCE, DISCOUNT),
        put_price(FORWARD, STRIKES, TOTAL_VARIANCE, DISCOUNT),
    )


def test_implied_forward_when_quotes_satisfy_parity_recovers_both_inputs(parity_quotes):
    estimate = implied_forward(*parity_quotes, EXPIRY)
    assert estimate.forward == pytest.approx(FORWARD, rel=1e-10)
    assert estimate.discount == pytest.approx(DISCOUNT, rel=1e-10)


def test_implied_forward_when_quotes_are_clean_reports_a_perfect_fit(parity_quotes):
    assert implied_forward(*parity_quotes, EXPIRY).r_squared == pytest.approx(1.0, abs=1e-9)


def test_implied_rate_when_derived_from_the_discount_factor_inverts_it(parity_quotes):
    estimate = implied_forward(*parity_quotes, EXPIRY)
    assert np.exp(-estimate.implied_rate * EXPIRY) == pytest.approx(DISCOUNT, rel=1e-9)


def test_as_dict_when_serialised_carries_the_whole_estimate(parity_quotes):
    payload = implied_forward(*parity_quotes, EXPIRY).as_dict()
    assert set(payload) == {"forward", "discount", "implied_rate", "n_pairs", "r_squared"}


def test_implied_forward_when_a_wing_quote_is_bad_the_atm_band_contains_the_damage(parity_quotes):
    strikes, calls, puts = parity_quotes
    corrupted = calls.copy()
    corrupted[0] += 12.0
    narrow = implied_forward(strikes, corrupted, puts, EXPIRY, band=4)
    wide = implied_forward(strikes, corrupted, puts, EXPIRY, band=0)
    assert abs(narrow.forward - FORWARD) < abs(wide.forward - FORWARD)


def test_implied_forward_when_band_is_zero_uses_every_pair(parity_quotes):
    assert implied_forward(*parity_quotes, EXPIRY, band=0).n_pairs == STRIKES.size


def test_implied_forward_when_the_band_exceeds_the_sample_uses_every_pair(parity_quotes):
    assert implied_forward(*parity_quotes, EXPIRY, band=500).n_pairs == STRIKES.size


def test_implied_forward_when_weights_are_supplied_still_recovers_the_forward(parity_quotes):
    strikes, calls, puts = parity_quotes
    weights = np.linspace(1.0, 5.0, strikes.size)
    assert implied_forward(strikes, calls, puts, EXPIRY, weights=weights).forward == pytest.approx(
        FORWARD, rel=1e-9
    )


def test_implied_forward_when_inputs_have_different_lengths_raises():
    with pytest.raises(ValueError, match="same length"):
        implied_forward([100.0, 110.0], [5.0], [3.0, 9.0], EXPIRY)


def test_implied_forward_when_expiry_is_not_positive_raises(parity_quotes):
    with pytest.raises(ValueError, match="expiry must be positive"):
        implied_forward(*parity_quotes, 0.0)


def test_implied_forward_when_there_are_too_few_pairs_raises():
    with pytest.raises(ValueError, match="at least 3"):
        implied_forward([100.0, 110.0], [8.0, 3.0], [2.0, 7.0], EXPIRY)


def test_implied_forward_when_the_spread_rises_with_strike_rejects_the_fit():
    strikes = np.array([90.0, 100.0, 110.0, 120.0])
    with pytest.raises(ValueError, match="discount factor"):
        implied_forward(strikes, strikes * 0.1, np.zeros_like(strikes), EXPIRY)


def test_forward_from_carry_when_there_is_no_dividend_matches_the_formula():
    estimate = forward_from_carry(spot=100.0, expiry=2.0, rate=0.05)
    assert estimate.forward == pytest.approx(100.0 * np.exp(0.10))
    assert estimate.discount == pytest.approx(np.exp(-0.10))
    assert estimate.n_pairs == 0


def test_forward_from_carry_when_the_dividend_equals_the_rate_returns_the_spot():
    assert forward_from_carry(100.0, 1.0, 0.04, 0.04).forward == pytest.approx(100.0)


@pytest.mark.parametrize(("spot", "expiry"), [(0.0, 1.0), (100.0, 0.0)])
def test_forward_from_carry_when_an_input_is_not_positive_raises(spot, expiry):
    with pytest.raises(ValueError, match="must be positive"):
        forward_from_carry(spot, expiry, 0.03)
