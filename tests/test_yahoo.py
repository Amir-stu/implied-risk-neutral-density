"""Parsing Yahoo-shaped frames. No network is touched here, by design."""

from __future__ import annotations

import pandas as pd
import pytest

from rnd.data.yahoo import build_chain

EXPIRY = 0.25

CALLS = pd.DataFrame(
    {
        "contractSymbol": ["X260320C00095000", "X260320C00105000"],
        "strike": [95.0, 105.0],
        "bid": [7.1, 2.0],
        "ask": [7.4, 2.2],
        "volume": [120, 340],
        "openInterest": [900, 1500],
        "impliedVolatility": [0.21, 0.19],
        "inTheMoney": [True, False],
    }
)

PUTS = pd.DataFrame(
    {
        "contractSymbol": ["X260320P00095000", "X260320P00105000"],
        "strike": [95.0, 105.0],
        "bid": [2.1, 6.9],
        "ask": [2.3, 7.2],
        "volume": [80, 60],
        "openInterest": [700, 400],
        "impliedVolatility": [0.23, 0.18],
        "inTheMoney": [False, True],
    }
)


@pytest.fixture
def chain():
    return build_chain(CALLS, PUTS, EXPIRY, spot=100.0, underlying="TEST", as_of="2026-01-05")


def test_build_chain_when_given_both_sides_keeps_every_quote(chain):
    assert len(chain) == 4
    assert len(chain.calls) == len(chain.puts) == 2


def test_build_chain_when_parsing_renames_open_interest(chain):
    assert "open_interest" in chain.quotes.columns
    assert chain.quotes["open_interest"].sum() == 3500


def test_build_chain_when_parsing_drops_the_vendor_implied_volatility(chain):
    """The vendor number assumes a rate and a dividend nobody publishes."""
    assert "impliedVolatility" not in chain.quotes.columns


def test_build_chain_when_parsing_carries_the_metadata(chain):
    assert chain.underlying == "TEST"
    assert chain.as_of == "2026-01-05"
    assert chain.spot == 100.0


def test_build_chain_when_parsing_computes_mids_from_the_raw_quotes(chain):
    call = chain.calls.set_index("strike").loc[95.0]
    assert call["mid"] == pytest.approx(7.25)


def test_build_chain_when_optional_columns_are_absent_still_works():
    minimal = CALLS[["strike", "bid", "ask"]]
    chain = build_chain(minimal, PUTS[["strike", "bid", "ask"]], EXPIRY)
    assert len(chain) == 4
    assert "open_interest" not in chain.quotes.columns


def test_build_chain_when_the_strike_column_is_missing_raises():
    with pytest.raises(ValueError, match="no strike column"):
        build_chain(CALLS.drop(columns="strike"), PUTS, EXPIRY)


def test_build_chain_when_parsed_is_ready_for_the_parity_regression(chain):
    pairs = chain.matched_pairs()
    assert list(pairs["strike"]) == [95.0, 105.0]


def test_load_yahoo_chain_when_the_optional_dependency_is_absent_says_so(monkeypatch):
    """The loader is re-exported lazily so the package works without yfinance."""
    import builtins

    import rnd.data

    real_import = builtins.__import__

    def missing(name, *args, **kwargs):
        if name == "yfinance":
            raise ImportError("no module named yfinance")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing)
    with pytest.raises(ImportError, match=r"implied-rnd\[yahoo\]"):
        rnd.data.load_yahoo_chain("SPY")
