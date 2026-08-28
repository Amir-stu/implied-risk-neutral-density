"""The chain container and the quote-cleaning rules."""

from __future__ import annotations

import pandas as pd
import pytest

from rnd.chain import OptionChain, year_fraction

EXPIRY = 0.25
FORWARD = 100.0

RECORDS = [
    {"strike": 90.0, "option_type": "C", "bid": 11.0, "ask": 11.4, "open_interest": 500},
    {"strike": 100.0, "option_type": "C", "bid": 4.0, "ask": 4.2, "open_interest": 900},
    {"strike": 110.0, "option_type": "C", "bid": 1.0, "ask": 1.1, "open_interest": 50},
    {"strike": 90.0, "option_type": "P", "bid": 1.2, "ask": 1.3, "open_interest": 700},
    {"strike": 100.0, "option_type": "P", "bid": 4.1, "ask": 4.3, "open_interest": 800},
    {"strike": 110.0, "option_type": "P", "bid": 10.8, "ask": 11.2, "open_interest": 20},
]


@pytest.fixture
def chain() -> OptionChain:
    return OptionChain.from_records(RECORDS, EXPIRY, spot=100.0, underlying="TEST")


def test_year_fraction_when_a_full_year_separates_the_dates_returns_one():
    assert year_fraction("2026-01-01", "2027-01-01") == pytest.approx(365 / 365)


def test_year_fraction_when_the_expiry_is_in_the_past_raises():
    with pytest.raises(ValueError, match="after the valuation date"):
        year_fraction("2026-06-01", "2026-05-01")


def test_option_chain_when_built_computes_mid_and_spread(chain):
    row = chain.quotes.iloc[0]
    assert row["mid"] == pytest.approx(0.5 * (row["bid"] + row["ask"]))
    assert row["spread"] == pytest.approx(row["ask"] - row["bid"])


def test_option_chain_when_types_are_spelled_out_normalises_them():
    records = [{"strike": 100.0, "option_type": t, "bid": 1.0, "ask": 1.1}
               for t in ("call", "Put", "c", "P")]
    chain = OptionChain.from_records(records, EXPIRY)
    assert set(chain.quotes["option_type"]) == {"C", "P"}
    assert len(chain.calls) == 2 and len(chain.puts) == 2


def test_option_chain_when_the_type_is_unrecognised_raises():
    with pytest.raises(ValueError, match="unrecognised option types"):
        OptionChain.from_records(
            [{"strike": 100.0, "option_type": "straddle", "bid": 1.0, "ask": 1.1}], EXPIRY
        )


def test_option_chain_when_a_column_is_missing_raises():
    with pytest.raises(ValueError, match="missing required columns"):
        OptionChain(pd.DataFrame({"strike": [100.0]}), EXPIRY)


def test_option_chain_when_expiry_is_not_positive_raises():
    with pytest.raises(ValueError, match="expiry must be positive"):
        OptionChain.from_records(RECORDS, 0.0)


def test_len_when_called_counts_every_quote(chain):
    assert len(chain) == len(RECORDS)


def test_clean_when_a_bid_is_missing_drops_the_quote(chain):
    frame = chain.quotes.copy()
    frame.loc[0, "bid"] = 0.0
    cleaned = OptionChain(frame, EXPIRY).clean()
    assert 90.0 not in set(cleaned.calls["strike"])


def test_clean_when_a_market_is_crossed_drops_the_quote(chain):
    frame = chain.quotes.copy()
    frame.loc[1, ["bid", "ask"]] = [4.5, 4.2]
    assert len(OptionChain(frame, EXPIRY).clean()) == len(RECORDS) - 1


def test_clean_when_the_spread_is_too_wide_drops_the_quote():
    records = RECORDS + [
        {"strike": 130.0, "option_type": "C", "bid": 0.05, "ask": 0.60, "open_interest": 5}
    ]
    cleaned = OptionChain.from_records(records, EXPIRY).clean(max_relative_spread=0.5)
    assert 130.0 not in set(cleaned.calls["strike"])


def test_clean_when_open_interest_is_required_drops_thin_strikes(chain):
    cleaned = chain.clean(min_open_interest=100)
    assert set(cleaned.quotes["strike"]) == {90.0, 100.0}


def test_clean_when_every_quote_is_tradeable_keeps_them_all(chain):
    assert len(chain.clean()) == len(RECORDS)


def test_matched_pairs_when_both_legs_are_quoted_lines_them_up(chain):
    pairs = chain.matched_pairs()
    assert list(pairs["strike"]) == [90.0, 100.0, 110.0]
    assert {"call_mid", "put_mid", "call_spread", "put_spread"} <= set(pairs.columns)


def test_matched_pairs_when_one_leg_is_missing_drops_the_strike():
    records = [record for record in RECORDS if not (record["strike"] == 110.0
                                                    and record["option_type"] == "P")]
    assert list(OptionChain.from_records(records, EXPIRY).matched_pairs()["strike"]) == [90.0,
                                                                                         100.0]


def test_out_of_the_money_when_split_at_the_forward_keeps_the_cheap_side(chain):
    otm = chain.out_of_the_money(FORWARD)
    calls = otm[otm["option_type"] == "C"]
    puts = otm[otm["option_type"] == "P"]
    assert calls["strike"].min() >= FORWARD
    assert puts["strike"].max() < FORWARD


def test_out_of_the_money_when_the_forward_is_not_positive_raises(chain):
    with pytest.raises(ValueError, match="forward must be positive"):
        chain.out_of_the_money(0.0)


def test_csv_round_trip_when_written_and_read_back_preserves_the_quotes(chain, tmp_path):
    path = tmp_path / "chain.csv"
    chain.to_csv(path)
    restored = OptionChain.from_csv(path, EXPIRY)
    pd.testing.assert_series_equal(restored.quotes["mid"], chain.quotes["mid"])
