"""End-to-end recovery of a distribution whose exact answer is known.

These are the tests that matter. A density estimator can look plausible on any
chart; the only real question is whether it returns the distribution the prices
were generated from. Here the prices come from a lognormal mixture with a
closed-form density, so the estimate can be scored against the truth in L1, in
its moments, and in the tail probabilities that motivate the exercise.
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import DISCOUNT, EXPIRY, FORWARD
from scipy.integrate import trapezoid

from rnd.data.synthetic import synthetic_chain
from rnd.pipeline import estimate_density

L1_TOLERANCE = 0.05
MOMENT_TOLERANCE = 0.05


def _l1_error(result, mixture) -> float:
    truth = mixture.pdf(result.density.strikes)
    return float(trapezoid(np.abs(result.density.density - truth), result.density.strikes))


def _true_moments(mixture, strikes):
    density = mixture.pdf(strikes)
    mass = trapezoid(density, strikes)
    mean = trapezoid(strikes * density, strikes) / mass
    centred = strikes - mean
    variance = trapezoid(centred**2 * density, strikes) / mass
    skew = trapezoid(centred**3 * density, strikes) / mass / variance**1.5
    return mean, np.sqrt(variance), skew


@pytest.mark.parametrize("method", ["svi", "spline"])
def test_estimate_density_when_prices_are_clean_recovers_the_true_density(
    clean_chain, mixture, method
):
    result = estimate_density(clean_chain, method=method)
    assert _l1_error(result, mixture) < L1_TOLERANCE


@pytest.mark.parametrize("method", ["svi", "spline"])
def test_estimate_density_when_prices_are_clean_recovers_the_true_moments(
    clean_chain, mixture, method
):
    result = estimate_density(clean_chain, method=method)
    true_mean, true_std, true_skew = _true_moments(mixture, result.density.strikes)
    moments = result.density.moments()
    assert moments.mean == pytest.approx(true_mean, rel=1e-3)
    assert moments.std == pytest.approx(true_std, rel=MOMENT_TOLERANCE)
    assert moments.skewness == pytest.approx(true_skew, abs=0.25)


@pytest.mark.parametrize("method", ["svi", "spline"])
def test_estimate_density_when_fitted_reprices_the_forward(clean_chain, method):
    result = estimate_density(clean_chain, method=method)
    assert abs(result.density.martingale_error()) < 1e-3


@pytest.mark.parametrize("method", ["svi", "spline"])
def test_estimate_density_when_fitted_integrates_to_one(clean_chain, method):
    result = estimate_density(clean_chain, method=method)
    assert result.density.total_mass == pytest.approx(1.0, abs=5e-3)


@pytest.mark.parametrize("method", ["svi", "spline"])
def test_estimate_density_when_fitted_stays_non_negative(clean_chain, method):
    assert estimate_density(clean_chain, method=method).density.min_density >= 0.0


@pytest.mark.parametrize("method", ["svi", "spline"])
def test_estimate_density_when_quotes_are_noisy_still_returns_a_valid_density(
    noisy_chain, method
):
    """Noise is what breaks the naive second difference. It must not break this."""
    result = estimate_density(noisy_chain, method=method)
    assert result.density.min_density >= 0.0
    assert result.smile.butterfly_violation() == 0.0
    assert result.density.total_mass == pytest.approx(1.0, abs=1e-2)


def test_estimate_density_when_run_recovers_the_forward_from_parity(svi_result):
    assert svi_result.forward.forward == pytest.approx(FORWARD, rel=1e-3)


def test_estimate_density_when_run_recovers_the_discount_factor(svi_result):
    assert svi_result.forward.discount == pytest.approx(DISCOUNT, rel=2e-3)


def test_estimate_density_when_run_fits_the_smile_to_within_a_vol_point(svi_result):
    assert svi_result.fit_rmse_vol < 0.01


def test_estimate_density_when_the_smile_is_skewed_prices_a_fatter_left_tail(svi_result):
    """The whole point: the market's downside probability exceeds the lognormal."""
    tails = svi_result.tail_metrics()
    assert tails["ratio(-20%)"] > 1.5
    assert tails["p(-20%)"] > tails["p(-20%) lognormal"]


def test_lognormal_benchmark_when_built_shares_the_grid_and_the_forward(svi_result):
    benchmark = svi_result.lognormal_benchmark()
    np.testing.assert_allclose(benchmark.strikes, svi_result.density.strikes)
    assert benchmark.martingale_error() == pytest.approx(0.0, abs=1e-4)


def test_lognormal_benchmark_when_compared_is_less_skewed_than_the_fit(svi_result):
    assert svi_result.density.moments().skewness < svi_result.lognormal_benchmark(
    ).moments().skewness


def test_atm_vol_when_read_off_the_fit_is_a_plausible_index_volatility(svi_result):
    assert 0.10 < svi_result.atm_vol < 0.30


def test_diagnostics_when_collected_carry_the_checks_a_reviewer_needs(svi_result):
    diagnostics = svi_result.diagnostics()
    assert {"total_mass", "martingale_error_bps", "min_density", "worst_durrleman_g",
            "atm_vol", "q05", "q50", "q95"} <= set(diagnostics)
    assert all(np.isfinite(value) for value in diagnostics.values())


def test_summary_when_printed_names_the_method_and_the_forward(svi_result):
    text = svi_result.summary()
    assert "svi" in text
    assert "forward" in text
    assert "martingale error" in text


def test_quotes_when_returned_carry_the_fitted_volatility(svi_result):
    assert {"implied_vol", "fitted_vol", "log_moneyness", "total_variance"} <= set(
        svi_result.quotes.columns
    )


def test_arbitrage_report_when_the_chain_is_synthetic_covers_every_call(svi_result):
    assert svi_result.arbitrage.n_strikes > 10


def test_svi_and_spline_when_run_on_the_same_chain_agree_on_the_tails(
    svi_result, spline_result
):
    for level in (0.8 * FORWARD, FORWARD, 1.2 * FORWARD):
        assert svi_result.density.prob_below(level) == pytest.approx(
            spline_result.density.prob_below(level), abs=0.02
        )


def test_estimate_density_when_the_method_is_unknown_raises(clean_chain):
    with pytest.raises(ValueError, match="unknown method"):
        estimate_density(clean_chain, method="kernel")


def test_estimate_density_when_the_chain_has_no_pairs_raises(mixture):
    chain = synthetic_chain(mixture, EXPIRY, strikes=[FORWARD], discount=DISCOUNT)
    with pytest.raises(ValueError, match="matched call-put pairs"):
        estimate_density(chain)


def test_estimate_density_when_cleaning_removes_almost_everything_raises(mixture):
    strikes = np.linspace(0.97 * FORWARD, 1.03 * FORWARD, 4)
    chain = synthetic_chain(mixture, EXPIRY, strikes=strikes, discount=DISCOUNT)
    with pytest.raises(ValueError, match="survived cleaning"):
        estimate_density(chain)
