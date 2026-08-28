"""The density itself: closed form against the lognormal, and its statistics."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import lognorm

from rnd.density import (
    RiskNeutralDensity,
    density_from_prices,
    density_from_smile,
    strike_grid,
)
from rnd.smile import FlatSmile, fit_spline_smile

EXPIRY = 0.5
FORWARD = 100.0
VOL = 0.25
DISCOUNT = 0.98


@pytest.fixture(scope="module")
def flat_smile() -> FlatSmile:
    return FlatSmile(VOL, EXPIRY, FORWARD)


@pytest.fixture(scope="module")
def flat_density(flat_smile) -> RiskNeutralDensity:
    return density_from_smile(flat_smile, n_points=4001, width=9.0)


def _lognormal_pdf(strikes):
    scale = VOL * np.sqrt(EXPIRY)
    return lognorm.pdf(strikes, scale, scale=FORWARD * np.exp(-0.5 * scale**2))


def test_density_from_smile_when_the_smile_is_flat_matches_the_lognormal(flat_density):
    np.testing.assert_allclose(flat_density.density, _lognormal_pdf(flat_density.strikes),
                               atol=1e-12)


def test_density_from_smile_when_integrated_has_unit_mass(flat_density):
    assert flat_density.total_mass == pytest.approx(1.0, abs=1e-5)


def test_density_from_smile_when_asked_for_the_mean_reprices_the_forward(flat_density):
    assert flat_density.martingale_error() == pytest.approx(0.0, abs=1e-6)


def test_density_from_smile_when_the_smile_is_arbitrage_free_stays_non_negative(flat_density):
    assert flat_density.min_density >= 0.0


def test_density_from_smile_when_given_explicit_strikes_uses_them(flat_smile):
    strikes = np.linspace(60.0, 160.0, 51)
    density = density_from_smile(flat_smile, strikes=strikes)
    np.testing.assert_allclose(density.strikes, strikes)


def test_density_from_prices_when_prices_come_from_a_smile_matches_the_closed_form(flat_smile):
    strikes = np.linspace(40.0, 220.0, 3001)
    prices = flat_smile.call_price(strikes, DISCOUNT)
    numeric = density_from_prices(strikes, prices, FORWARD, EXPIRY, DISCOUNT)
    analytic = density_from_smile(flat_smile, strikes=strikes)
    np.testing.assert_allclose(numeric.density, analytic.density, atol=1e-6)


def test_density_from_prices_when_the_curve_is_noisy_goes_negative(flat_smile):
    """The failure the whole package exists to avoid, demonstrated on purpose."""
    strikes = np.linspace(70.0, 140.0, 71)
    rng = np.random.default_rng(3)
    prices = flat_smile.call_price(strikes, DISCOUNT) + rng.normal(scale=0.02, size=strikes.size)
    assert density_from_prices(strikes, prices, FORWARD, EXPIRY, DISCOUNT).min_density < 0.0


@pytest.mark.parametrize(
    ("strikes", "prices", "message"),
    [
        ([100.0, 110.0], [5.0, 3.0], "at least three"),
        ([100.0, 90.0, 110.0], [5.0, 8.0, 3.0], "strictly increasing"),
    ],
    ids=["too-few-strikes", "unsorted-strikes"],
)
def test_density_from_prices_when_the_grid_is_invalid_raises(strikes, prices, message):
    with pytest.raises(ValueError, match=message):
        density_from_prices(strikes, prices, FORWARD, EXPIRY)


def test_quantiles_when_read_off_the_flat_density_match_the_lognormal(flat_density):
    for probability in (0.05, 0.5, 0.95):
        expected = lognorm.ppf(probability, VOL * np.sqrt(EXPIRY),
                               scale=FORWARD * np.exp(-0.5 * VOL**2 * EXPIRY))
        assert flat_density.quantile(probability) == pytest.approx(expected, rel=1e-4)


def test_prob_below_when_evaluated_at_the_median_returns_one_half(flat_density):
    assert flat_density.prob_below(flat_density.quantile(0.5)) == pytest.approx(0.5, abs=1e-4)


def test_prob_between_when_the_interval_is_the_whole_grid_returns_the_total_mass(flat_density):
    span = flat_density.prob_between(flat_density.strikes[0], flat_density.strikes[-1])
    assert span == pytest.approx(flat_density.total_mass, abs=1e-9)


def test_prob_between_when_the_bounds_are_reversed_raises(flat_density):
    with pytest.raises(ValueError, match="high must not be below low"):
        flat_density.prob_between(120.0, 80.0)


@pytest.mark.parametrize("probability", [0.0, 1.0, -0.1, 1.5])
def test_quantile_when_the_probability_is_outside_the_unit_interval_raises(
    flat_density, probability
):
    with pytest.raises(ValueError, match="strictly inside"):
        flat_density.quantile(probability)


def test_moments_when_computed_on_the_flat_density_match_the_lognormal(flat_density):
    variance = VOL**2 * EXPIRY
    moments = flat_density.moments()
    assert moments.mean == pytest.approx(FORWARD, rel=1e-6)
    assert moments.std == pytest.approx(FORWARD * np.sqrt(np.exp(variance) - 1.0), rel=1e-4)
    expected_skew = (np.exp(variance) + 2.0) * np.sqrt(np.exp(variance) - 1.0)
    assert moments.skewness == pytest.approx(expected_skew, rel=1e-3)


def test_moments_as_dict_when_serialised_carries_every_field(flat_density):
    assert set(flat_density.moments().as_dict()) == {
        "mean", "std", "skewness", "excess_kurtosis"
    }


def test_raw_moment_when_order_is_one_returns_the_forward(flat_density):
    assert flat_density.raw_moment(1) == pytest.approx(FORWARD, rel=1e-5)


def test_normalised_when_the_density_has_negative_values_clips_and_rescales():
    strikes = np.linspace(80.0, 120.0, 41)
    values = np.where(strikes < 90.0, -0.001, 0.02)
    normalised = RiskNeutralDensity(strikes, values, FORWARD, EXPIRY).normalised()
    assert normalised.min_density >= 0.0
    assert normalised.total_mass == pytest.approx(1.0)


def test_normalised_when_nothing_is_positive_raises():
    strikes = np.linspace(80.0, 120.0, 5)
    density = RiskNeutralDensity(strikes, np.full(5, -1.0), FORWARD, EXPIRY)
    with pytest.raises(ValueError, match="integrates to zero"):
        density.normalised()


def test_to_frame_when_exported_carries_strike_density_and_cdf(flat_density):
    frame = flat_density.to_frame()
    assert list(frame.columns) == ["strike", "density", "cdf"]
    assert len(frame) == flat_density.strikes.size


@pytest.mark.parametrize(
    ("strikes", "values", "message"),
    [
        ([100.0, 110.0], [0.1], "equal length"),
        ([110.0, 100.0], [0.1, 0.2], "strictly increasing"),
    ],
    ids=["mismatched-length", "unsorted"],
)
def test_risk_neutral_density_when_inputs_are_invalid_raises(strikes, values, message):
    with pytest.raises(ValueError, match=message):
        RiskNeutralDensity(strikes, values, FORWARD, EXPIRY)


def test_strike_grid_when_the_smile_is_flat_is_centred_on_the_forward(flat_smile):
    grid = strike_grid(flat_smile, n_points=101, width=4.0)
    assert np.sqrt(grid[0] * grid[-1]) == pytest.approx(FORWARD)
    assert grid.size == 101


def test_strike_grid_when_the_wings_are_steep_widens_beyond_the_atm_width():
    k = np.linspace(-0.3, 0.3, 21)
    steep = 0.01 + 0.6 * np.abs(k)
    smile = fit_spline_smile(k, steep, EXPIRY, FORWARD)
    flat = FlatSmile(float(np.sqrt(0.01 / EXPIRY)), EXPIRY, FORWARD)
    assert strike_grid(smile, width=5.0)[0] < strike_grid(flat, width=5.0)[0]


def test_strike_grid_when_too_few_points_are_requested_raises(flat_smile):
    with pytest.raises(ValueError, match="three grid points"):
        strike_grid(flat_smile, n_points=2)
