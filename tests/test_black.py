"""Black-76 pricing and the implied-volatility inversion."""

from __future__ import annotations

import numpy as np
import pytest

from rnd.black import (
    call_price,
    d1_d2,
    implied_vol,
    implied_vol_vector,
    put_price,
    vega,
)

FORWARD = 100.0
EXPIRY = 0.5
VOL = 0.25
DISCOUNT = 0.985
TOTAL_VARIANCE = VOL**2 * EXPIRY


def test_parity_when_priced_off_the_same_smile_holds_exactly():
    strikes = np.array([70.0, 100.0, 130.0])
    call = call_price(FORWARD, strikes, TOTAL_VARIANCE, DISCOUNT)
    put = put_price(FORWARD, strikes, TOTAL_VARIANCE, DISCOUNT)
    expected = DISCOUNT * (FORWARD - strikes)
    np.testing.assert_allclose(call - put, expected, atol=1e-12)


def test_call_price_when_volatility_vanishes_collapses_to_intrinsic():
    price = call_price(FORWARD, 90.0, 1e-14, DISCOUNT)
    assert price == pytest.approx(DISCOUNT * 10.0, abs=1e-6)


def test_call_price_when_strike_rises_decreases_monotonically():
    strikes = np.linspace(50.0, 200.0, 60)
    prices = call_price(FORWARD, strikes, TOTAL_VARIANCE, DISCOUNT)
    assert np.all(np.diff(prices) < 0.0)


def test_d1_d2_when_at_the_money_forward_are_symmetric():
    d1, d2 = d1_d2(FORWARD, FORWARD, TOTAL_VARIANCE)
    assert d1 == pytest.approx(-d2)


def test_vega_when_at_the_money_is_the_largest_across_strikes():
    strikes = np.array([70.0, 100.0, 140.0])
    vegas = vega(FORWARD, strikes, TOTAL_VARIANCE, EXPIRY, DISCOUNT)
    assert np.argmax(vegas) == 1


@pytest.mark.parametrize("strike", [60.0, 85.0, 100.0, 125.0, 180.0])
@pytest.mark.parametrize("is_call", [True, False])
def test_implied_vol_when_inverting_a_black_price_recovers_the_input(strike, is_call):
    pricer = call_price if is_call else put_price
    price = float(pricer(FORWARD, strike, TOTAL_VARIANCE, DISCOUNT))
    recovered = implied_vol(price, FORWARD, strike, EXPIRY, DISCOUNT, is_call)
    assert recovered == pytest.approx(VOL, abs=1e-8)


@pytest.mark.parametrize(
    "price",
    [-1.0, 0.0, float("nan"), 200.0],
    ids=["negative", "zero", "not-a-number", "above-upper-bound"],
)
def test_implied_vol_when_the_quote_is_unarbitrageable_returns_nan(price):
    assert np.isnan(implied_vol(price, FORWARD, 100.0, EXPIRY, DISCOUNT))


def test_implied_vol_when_the_price_is_at_intrinsic_returns_nan():
    intrinsic = DISCOUNT * (FORWARD - 80.0)
    assert np.isnan(implied_vol(intrinsic, FORWARD, 80.0, EXPIRY, DISCOUNT))


def test_implied_vol_when_expiry_has_passed_returns_nan():
    assert np.isnan(implied_vol(5.0, FORWARD, 100.0, 0.0, DISCOUNT))


def test_implied_vol_when_volatility_exceeds_the_search_range_returns_nan():
    extreme = call_price(FORWARD, 100.0, 30.0**2 * EXPIRY, DISCOUNT)
    assert np.isnan(implied_vol(float(extreme), FORWARD, 100.0, EXPIRY, DISCOUNT))


def test_implied_vol_vector_when_given_mixed_rights_inverts_each_one():
    strikes = np.array([80.0, 120.0])
    prices = np.array(
        [
            float(put_price(FORWARD, 80.0, TOTAL_VARIANCE, DISCOUNT)),
            float(call_price(FORWARD, 120.0, TOTAL_VARIANCE, DISCOUNT)),
        ]
    )
    vols = implied_vol_vector(prices, FORWARD, strikes, EXPIRY, DISCOUNT, [False, True])
    np.testing.assert_allclose(vols, VOL, atol=1e-8)
