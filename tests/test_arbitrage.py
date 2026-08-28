"""Static no-arbitrage checks on a call curve."""

from __future__ import annotations

import numpy as np
import pytest

from rnd.arbitrage import butterfly_values, check_call_curve
from rnd.black import call_price

FORWARD = 100.0
DISCOUNT = 0.99
TOTAL_VARIANCE = 0.2**2 * 0.5
STRIKES = np.linspace(60.0, 160.0, 21)


@pytest.fixture(scope="module")
def clean_prices():
    return call_price(FORWARD, STRIKES, TOTAL_VARIANCE, DISCOUNT)


def test_check_call_curve_when_prices_come_from_black_reports_no_arbitrage(clean_prices):
    report = check_call_curve(STRIKES, clean_prices, FORWARD, DISCOUNT)
    assert report.is_clean
    assert report.offending_strikes == []
    assert report.n_strikes == STRIKES.size


def test_summary_when_the_curve_is_clean_says_so(clean_prices):
    assert "no static arbitrage" in check_call_curve(
        STRIKES, clean_prices, FORWARD, DISCOUNT
    ).summary()


def test_as_dict_when_serialised_carries_every_counter(clean_prices):
    payload = check_call_curve(STRIKES, clean_prices, FORWARD, DISCOUNT).as_dict()
    assert payload["is_clean"] is True
    assert set(payload) >= {"bound_violations", "monotonicity_violations",
                            "convexity_violations", "worst_butterfly"}


def test_check_call_curve_when_a_price_dips_flags_convexity(clean_prices):
    """A dip at one strike breaks the butterflies centred on its neighbours."""
    prices = np.asarray(clean_prices).copy()
    prices[10] -= 1.5
    report = check_call_curve(STRIKES, prices, FORWARD, DISCOUNT)
    assert report.convexity_violations > 0
    assert report.worst_butterfly < 0.0
    assert set(report.offending_strikes) >= {STRIKES[9], STRIKES[11]}
    assert not report.is_clean


def test_check_call_curve_when_a_price_spikes_flags_monotonicity(clean_prices):
    prices = np.asarray(clean_prices).copy()
    prices[5] += 25.0
    assert check_call_curve(STRIKES, prices, FORWARD, DISCOUNT).monotonicity_violations > 0


def test_check_call_curve_when_a_price_is_below_intrinsic_flags_the_bound(clean_prices):
    prices = np.asarray(clean_prices).copy()
    prices[0] = 0.5 * DISCOUNT * (FORWARD - STRIKES[0])
    assert check_call_curve(STRIKES, prices, FORWARD, DISCOUNT).bound_violations > 0


def test_check_call_curve_when_a_price_exceeds_the_forward_flags_the_bound(clean_prices):
    prices = np.asarray(clean_prices).copy()
    prices[0] = DISCOUNT * FORWARD * 1.5
    assert check_call_curve(STRIKES, prices, FORWARD, DISCOUNT).bound_violations > 0


def test_summary_when_the_curve_is_dirty_reports_the_counts(clean_prices):
    prices = np.asarray(clean_prices).copy()
    prices[10] -= 1.5
    assert "convexity violations" in check_call_curve(
        STRIKES, prices, FORWARD, DISCOUNT
    ).summary()


def test_butterfly_values_when_strikes_are_evenly_spaced_match_the_second_difference():
    strikes = np.array([90.0, 100.0, 110.0])
    prices = np.array([12.0, 5.0, 2.0])
    expected = 0.5 * prices[0] + 0.5 * prices[2] - prices[1]
    assert butterfly_values(strikes, prices)[0] == pytest.approx(expected)


def test_butterfly_values_when_spacing_is_uneven_weights_the_wings():
    strikes = np.array([90.0, 100.0, 120.0])
    prices = np.array([12.0, 5.0, 1.0])
    expected = (20 / 30) * 12.0 + (10 / 30) * 1.0 - 5.0
    assert butterfly_values(strikes, prices)[0] == pytest.approx(expected)


def test_butterfly_values_when_given_fewer_than_three_strikes_raises():
    with pytest.raises(ValueError, match="three strikes"):
        butterfly_values([100.0, 110.0], [5.0, 3.0])


@pytest.mark.parametrize(
    ("strikes", "prices", "message"),
    [
        ([90.0, 100.0, 110.0], [12.0, 5.0], "same length"),
        ([110.0, 100.0, 90.0], [2.0, 5.0, 12.0], "strictly increasing"),
    ],
    ids=["mismatched-length", "unsorted"],
)
def test_check_call_curve_when_inputs_are_invalid_raises(strikes, prices, message):
    with pytest.raises(ValueError, match=message):
        check_call_curve(strikes, prices, FORWARD, DISCOUNT)
