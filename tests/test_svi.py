"""Raw SVI: parameter admissibility, analytic derivatives and calibration."""

from __future__ import annotations

import numpy as np
import pytest

from rnd.smile import LEE_SLOPE_BOUND
from rnd.svi import SVIParams, SVISmile, fit_svi

EXPIRY = 0.5
FORWARD = 100.0
STEP = 1e-5
TRUE = SVIParams(a=0.015, b=0.09, rho=-0.55, m=0.02, sigma=0.11)


@pytest.fixture(scope="module")
def true_smile() -> SVISmile:
    return SVISmile(TRUE, EXPIRY, FORWARD)


@pytest.fixture(scope="module")
def quoted(true_smile):
    k = np.linspace(-0.45, 0.35, 30)
    return k, true_smile.total_variance(k)


def test_svi_params_when_admissible_validate_passes():
    TRUE.validate()


@pytest.mark.parametrize(
    "params",
    [
        SVIParams(0.02, -0.1, -0.5, 0.0, 0.1),
        SVIParams(0.02, 0.1, -1.5, 0.0, 0.1),
        SVIParams(0.02, 0.1, -0.5, 0.0, 0.0),
        SVIParams(-5.0, 0.1, -0.5, 0.0, 0.1),
    ],
    ids=["negative-b", "rho-outside-unit-interval", "zero-sigma", "negative-variance"],
)
def test_svi_params_when_inadmissible_raise(params):
    with pytest.raises(ValueError):
        params.validate()


def test_wing_slope_when_computed_matches_the_asymptotic_gradient(true_smile):
    far = true_smile.d_total_variance(np.array([200.0]))[0]
    assert far == pytest.approx(TRUE.b * (1.0 + TRUE.rho), abs=1e-6)
    assert TRUE.wing_slope == pytest.approx(TRUE.b * (1.0 + abs(TRUE.rho)))


def test_as_tuple_when_called_returns_parameters_in_declaration_order():
    assert TRUE.as_tuple() == (TRUE.a, TRUE.b, TRUE.rho, TRUE.m, TRUE.sigma)


def test_total_variance_when_at_the_minimum_equals_the_analytic_floor(true_smile):
    k = np.linspace(-2.0, 2.0, 4001)
    floor = TRUE.a + TRUE.b * TRUE.sigma * np.sqrt(1.0 - TRUE.rho**2)
    assert float(np.min(true_smile.total_variance(k))) == pytest.approx(floor, abs=1e-6)


def test_derivatives_when_compared_with_finite_differences_agree(true_smile):
    k = np.array([-0.4, -0.1, 0.0, 0.25])
    numeric_first = (true_smile.total_variance(k + STEP)
                     - true_smile.total_variance(k - STEP)) / (2 * STEP)
    numeric_second = (true_smile.total_variance(k + STEP)
                      - 2 * true_smile.total_variance(k)
                      + true_smile.total_variance(k - STEP)) / STEP**2
    np.testing.assert_allclose(true_smile.d_total_variance(k), numeric_first, atol=1e-8)
    np.testing.assert_allclose(true_smile.d2_total_variance(k), numeric_second, atol=1e-5)


def test_fit_svi_when_the_data_is_exactly_svi_recovers_the_parameters(quoted):
    k, total_variance = quoted
    fitted = fit_svi(k, total_variance, EXPIRY, FORWARD)
    np.testing.assert_allclose(fitted.params.as_tuple(), TRUE.as_tuple(), atol=1e-6)


def test_fit_svi_when_quotes_are_noisy_stays_free_of_butterfly_arbitrage(quoted):
    k, total_variance = quoted
    rng = np.random.default_rng(11)
    noisy = total_variance * (1.0 + rng.normal(scale=0.02, size=total_variance.shape))
    fitted = fit_svi(k, noisy, EXPIRY, FORWARD)
    assert fitted.butterfly_violation(np.linspace(-3.0, 3.0, 2001)) == 0.0


def test_fit_svi_when_enforcing_no_arbitrage_caps_the_wing_slope(quoted):
    k, total_variance = quoted
    steep = total_variance + 3.0 * np.abs(k)
    fitted = fit_svi(k, steep, EXPIRY, FORWARD)
    assert fitted.params.b <= LEE_SLOPE_BOUND + 1e-9


def test_fit_svi_when_weights_favour_one_side_tilts_the_fit(quoted):
    k, total_variance = quoted
    perturbed = total_variance.copy()
    perturbed[:5] += 0.02
    weights = np.where(k < -0.3, 1000.0, 1.0)
    weighted = fit_svi(k, perturbed, EXPIRY, FORWARD, weights=weights)
    unweighted = fit_svi(k, perturbed, EXPIRY, FORWARD)
    left = k[:5]
    weighted_gap = np.abs(weighted.total_variance(left) - perturbed[:5]).mean()
    unweighted_gap = np.abs(unweighted.total_variance(left) - perturbed[:5]).mean()
    assert weighted_gap < unweighted_gap


def test_fit_svi_when_given_fewer_than_five_quotes_raises():
    with pytest.raises(ValueError, match="at least five"):
        fit_svi([0.0, 0.1, 0.2, 0.3], [0.04] * 4, EXPIRY, FORWARD)


def test_svi_smile_when_constructed_with_bad_parameters_raises():
    with pytest.raises(ValueError):
        SVISmile(SVIParams(0.02, -1.0, 0.0, 0.0, 0.1), EXPIRY, FORWARD)
