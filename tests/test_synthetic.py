"""The synthetic market: its closed forms have to be right, or nothing else is."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import trapezoid

from rnd.data.synthetic import LognormalMixture, synthetic_chain

EXPIRY = 0.5
FORWARD = 100.0
DISCOUNT = 0.98
GRID = np.linspace(1e-3, 600.0, 400_001)


@pytest.fixture(scope="module")
def model() -> LognormalMixture:
    return LognormalMixture.crash_scenario(FORWARD, EXPIRY)


def test_crash_scenario_when_built_prices_to_the_requested_forward(model):
    assert model.forward == pytest.approx(FORWARD, rel=1e-12)


def test_crash_scenario_when_built_puts_the_second_component_below_the_first(model):
    assert model.forwards[1] < model.forwards[0]
    assert model.total_variances[1] > model.total_variances[0]


def test_pdf_when_integrated_over_the_whole_line_has_unit_mass(model):
    assert trapezoid(model.pdf(GRID), GRID) == pytest.approx(1.0, abs=1e-6)


def test_pdf_when_integrated_against_the_price_returns_the_forward(model):
    assert trapezoid(GRID * model.pdf(GRID), GRID) == pytest.approx(FORWARD, rel=1e-6)


def test_pdf_when_the_left_tail_is_priced_is_fatter_than_a_single_lognormal(model):
    single = LognormalMixture(np.array([1.0]), np.array([FORWARD]),
                              np.array([0.16**2 * EXPIRY]))
    assert model.cdf(0.75 * FORWARD) > single.cdf(0.75 * FORWARD)


def test_cdf_when_evaluated_far_out_approaches_zero_and_one(model):
    assert model.cdf(1e-6) == pytest.approx(0.0, abs=1e-9)
    assert model.cdf(1e6) == pytest.approx(1.0, abs=1e-9)


def test_call_price_when_differentiated_twice_returns_the_density(model):
    """Breeden-Litzenberger, checked against the model's own closed form.

    The comparison skips the two end points, where ``np.gradient`` falls back
    to a one-sided difference that is first-order accurate; everywhere else the
    agreement is the identity the whole package rests on.
    """
    strikes = np.linspace(60.0, 160.0, 2001)
    prices = model.call_price(strikes, DISCOUNT)
    second = np.gradient(np.gradient(prices, strikes), strikes) / DISCOUNT
    np.testing.assert_allclose(second[2:-2], model.pdf(strikes)[2:-2], atol=1e-6)


def test_put_price_when_compared_with_the_call_satisfies_parity(model):
    strikes = np.array([70.0, 100.0, 140.0])
    spread = model.call_price(strikes, DISCOUNT) - model.put_price(strikes, DISCOUNT)
    np.testing.assert_allclose(spread, DISCOUNT * (FORWARD - strikes), atol=1e-9)


def test_call_price_when_the_strike_is_zero_equals_the_discounted_forward(model):
    assert float(model.call_price(1e-9, DISCOUNT)) == pytest.approx(DISCOUNT * FORWARD, abs=1e-6)


@pytest.mark.parametrize(
    ("weights", "forwards", "variances"),
    [
        ([0.5], [100.0, 90.0], [0.01, 0.02]),
        ([0.5, 0.4], [100.0, 90.0], [0.01, 0.02]),
        ([0.5, 0.5], [100.0, -90.0], [0.01, 0.02]),
        ([0.5, 0.5], [100.0, 90.0], [0.01, 0.0]),
    ],
    ids=["length-mismatch", "weights-do-not-sum-to-one", "negative-forward", "zero-variance"],
)
def test_lognormal_mixture_when_inputs_are_invalid_raises(weights, forwards, variances):
    with pytest.raises(ValueError):
        LognormalMixture(np.array(weights), np.array(forwards), np.array(variances))


@pytest.mark.parametrize(
    ("probability", "size"),
    [(1.0, 0.15), (-0.1, 0.15), (0.05, 0.0), (0.05, 1.0)],
    ids=["certain-crash", "negative-probability", "zero-size", "total-loss"],
)
def test_crash_scenario_when_arguments_are_out_of_range_raises(probability, size):
    with pytest.raises(ValueError):
        LognormalMixture.crash_scenario(FORWARD, EXPIRY, crash_probability=probability,
                                        crash_size=size)


def test_synthetic_chain_when_generated_quotes_both_rights_at_every_strike(model):
    strikes = np.array([80.0, 100.0, 120.0])
    chain = synthetic_chain(model, EXPIRY, strikes=strikes, discount=DISCOUNT)
    assert len(chain) == 2 * strikes.size
    assert set(chain.calls["strike"]) == set(chain.puts["strike"]) == set(strikes)


def test_synthetic_chain_when_generated_brackets_the_model_price(model):
    strikes = np.array([90.0, 100.0, 110.0])
    chain = synthetic_chain(model, EXPIRY, strikes=strikes, discount=DISCOUNT)
    calls = chain.calls.sort_values("strike")
    exact = model.call_price(strikes, DISCOUNT)
    assert np.all(calls["bid"].to_numpy() <= exact)
    assert np.all(calls["ask"].to_numpy() >= exact)


def test_synthetic_chain_when_a_tick_is_set_rounds_every_quote_to_it(model):
    chain = synthetic_chain(model, EXPIRY, strikes=np.array([100.0]), discount=DISCOUNT,
                            tick=0.05)
    remainders = np.remainder(chain.quotes[["bid", "ask"]].to_numpy(), 0.05)
    np.testing.assert_allclose(np.minimum(remainders, 0.05 - remainders), 0.0, atol=1e-9)


def test_synthetic_chain_when_seeded_is_reproducible(model):
    first = synthetic_chain(model, EXPIRY, discount=DISCOUNT, noise_bps=50.0, seed=42)
    second = synthetic_chain(model, EXPIRY, discount=DISCOUNT, noise_bps=50.0, seed=42)
    np.testing.assert_array_equal(first.quotes["mid"], second.quotes["mid"])


def test_synthetic_chain_when_noise_is_added_moves_the_quotes(model):
    quiet = synthetic_chain(model, EXPIRY, discount=DISCOUNT, seed=42)
    noisy = synthetic_chain(model, EXPIRY, discount=DISCOUNT, noise_bps=200.0, seed=42)
    assert not np.allclose(quiet.quotes["mid"], noisy.quotes["mid"])


def test_synthetic_chain_when_no_strikes_are_given_spans_the_distribution(model):
    chain = synthetic_chain(model, EXPIRY, discount=DISCOUNT)
    strikes = chain.quotes["strike"]
    assert strikes.min() < 0.75 * FORWARD
    assert strikes.max() > 1.25 * FORWARD
