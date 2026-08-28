"""Smile base behaviour, the flat control case and the spline fit."""

from __future__ import annotations

import numpy as np
import pytest

from rnd.smile import LEE_SLOPE_BOUND, FlatSmile, fit_spline_smile

EXPIRY = 0.5
FORWARD = 100.0
VOL = 0.2
STEP = 1e-5


def _numeric_first(smile, k):
    return (smile.total_variance(k + STEP) - smile.total_variance(k - STEP)) / (2 * STEP)


def _numeric_second(smile, k):
    return (
        smile.total_variance(k + STEP)
        - 2 * smile.total_variance(k)
        + smile.total_variance(k - STEP)
    ) / STEP**2


@pytest.fixture
def skewed_smile():
    k = np.linspace(-0.35, 0.30, 30)
    total_variance = VOL**2 * EXPIRY - 0.05 * k + 0.25 * k**2
    return fit_spline_smile(k, total_variance, EXPIRY, FORWARD)


def test_flat_smile_when_evaluated_returns_the_constant_total_variance():
    smile = FlatSmile(VOL, EXPIRY, FORWARD)
    np.testing.assert_allclose(smile.total_variance([-1.0, 0.0, 1.0]), VOL**2 * EXPIRY)


def test_flat_smile_when_asked_for_durrleman_g_returns_one_everywhere():
    smile = FlatSmile(VOL, EXPIRY, FORWARD)
    np.testing.assert_allclose(smile.durrleman_g(np.linspace(-2, 2, 9)), 1.0)


def test_flat_smile_when_checked_for_butterflies_reports_no_violation():
    assert FlatSmile(VOL, EXPIRY, FORWARD).butterfly_violation() == 0.0


def test_flat_smile_when_priced_matches_black_with_the_same_vol():
    smile = FlatSmile(VOL, EXPIRY, FORWARD)
    from rnd.black import call_price

    expected = call_price(FORWARD, 110.0, VOL**2 * EXPIRY, 0.99)
    assert smile.call_price(110.0, 0.99) == pytest.approx(float(expected))


def test_implied_vol_when_converted_back_from_total_variance_round_trips():
    smile = FlatSmile(VOL, EXPIRY, FORWARD)
    np.testing.assert_allclose(smile.implied_vol([-0.2, 0.4]), VOL)


def test_log_moneyness_when_strike_equals_the_forward_is_zero():
    assert FlatSmile(VOL, EXPIRY, FORWARD).log_moneyness(FORWARD) == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("expiry", "forward", "vol"),
    [(0.0, FORWARD, VOL), (EXPIRY, 0.0, VOL), (EXPIRY, FORWARD, 0.0)],
    ids=["expiry", "forward", "vol"],
)
def test_flat_smile_when_a_parameter_is_not_positive_raises(expiry, forward, vol):
    with pytest.raises(ValueError):
        FlatSmile(vol, expiry, forward)


def test_spline_smile_when_fitted_to_a_smooth_curve_reproduces_it(skewed_smile):
    k = np.linspace(-0.3, 0.25, 40)
    expected = VOL**2 * EXPIRY - 0.05 * k + 0.25 * k**2
    np.testing.assert_allclose(skewed_smile.total_variance(k), expected, atol=1e-5)


def test_spline_smile_when_differentiated_matches_finite_differences(skewed_smile):
    k = np.array([-0.2, 0.0, 0.15])
    np.testing.assert_allclose(skewed_smile.d_total_variance(k), _numeric_first(skewed_smile, k),
                               atol=1e-5)
    np.testing.assert_allclose(skewed_smile.d2_total_variance(k), _numeric_second(skewed_smile, k),
                               atol=1e-3)


def test_spline_smile_when_extrapolated_keeps_total_variance_positive(skewed_smile):
    far = np.linspace(-8.0, 8.0, 200)
    assert np.all(skewed_smile.total_variance(far) > 0.0)


def test_spline_smile_when_extrapolated_respects_the_lee_slope_bound(skewed_smile):
    slopes = skewed_smile.d_total_variance(np.array([-6.0, 6.0]))
    assert np.all(np.abs(slopes) <= LEE_SLOPE_BOUND + 1e-12)


def test_spline_smile_when_extrapolated_has_no_curvature_in_the_wings(skewed_smile):
    np.testing.assert_allclose(skewed_smile.d2_total_variance([-5.0, 5.0]), 0.0)


def test_spline_smile_when_extrapolated_stays_free_of_butterfly_arbitrage(skewed_smile):
    assert skewed_smile.butterfly_violation(np.linspace(-6.0, 6.0, 2001)) == 0.0


def test_fit_spline_smile_when_given_too_few_quotes_raises():
    with pytest.raises(ValueError, match="at least 4"):
        fit_spline_smile([0.0, 0.1, 0.2], [0.04, 0.04, 0.04], EXPIRY, FORWARD)


def test_fit_spline_smile_when_quotes_are_unsorted_sorts_them_first():
    k = np.array([0.2, -0.3, 0.0, -0.1, 0.1])
    total_variance = 0.04 + 0.1 * k**2
    smile = fit_spline_smile(k, total_variance, EXPIRY, FORWARD)
    assert smile.k_min == pytest.approx(-0.3)
    assert smile.k_max == pytest.approx(0.2)


def test_fit_spline_smile_when_weights_are_given_uses_the_chi_square_default():
    k = np.linspace(-0.3, 0.3, 20)
    total_variance = 0.04 + 0.1 * k**2
    weights = np.full_like(k, 500.0)
    noisy = total_variance + np.tile([1e-4, -1e-4], 10)
    smooth = fit_spline_smile(k, noisy, EXPIRY, FORWARD, weights=weights)
    rough = fit_spline_smile(k, noisy, EXPIRY, FORWARD, weights=weights, smoothing=0.0)
    curvature = np.abs(smooth.d2_total_variance(k)).max()
    assert curvature < np.abs(rough.d2_total_variance(k)).max()
