"""Live option chains from Yahoo Finance.

Yahoo is free, which is the only argument for it. Its quotes are delayed, its
wings are frequently stale, and the implied volatility column it ships is
computed against an assumed rate and dividend that nobody publishes. This
module therefore takes only the four columns that are raw observations --
strike, right, bid, ask -- plus size, and lets the rest of the package derive
everything else from those.

The parsing is separated from the download so it can be tested without a
network connection, which is also the reason ``yfinance`` is an optional
dependency rather than a hard one.
"""

from __future__ import annotations

import pandas as pd

from ..chain import CALL, PUT, OptionChain, year_fraction

YAHOO_COLUMNS = {"bid": "bid", "ask": "ask", "strike": "strike",
                 "volume": "volume", "openInterest": "open_interest"}


def _normalise(frame: pd.DataFrame, option_type: str) -> pd.DataFrame:
    present = {source: target for source, target in YAHOO_COLUMNS.items()
               if source in frame.columns}
    if "strike" not in present:
        raise ValueError("chain frame has no strike column")
    out = frame[list(present)].rename(columns=present).copy()
    out["option_type"] = option_type
    return out


def build_chain(
    calls: pd.DataFrame,
    puts: pd.DataFrame,
    expiry: float,
    spot: float | None = None,
    underlying: str | None = None,
    as_of: str | None = None,
) -> OptionChain:
    """Assemble an :class:`~rnd.chain.OptionChain` from two Yahoo-shaped frames."""
    combined = pd.concat([_normalise(calls, CALL), _normalise(puts, PUT)], ignore_index=True)
    return OptionChain(combined, expiry, spot=spot, underlying=underlying, as_of=as_of)


def load_yahoo_chain(
    ticker: str,
    expiry_date: str | None = None,
    as_of: str | None = None,
) -> OptionChain:
    """Download one expiry for one underlying.

    With no ``expiry_date`` the nearest listed expiry is used. Time to expiry is
    measured in calendar days from ``as_of`` (today by default), which is the
    convention the rest of the package assumes.
    """
    try:
        import yfinance
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "loading live chains needs the optional dependency: pip install 'implied-rnd[yahoo]'"
        ) from exc

    handle = yfinance.Ticker(ticker)
    expiries = handle.options
    if not expiries:
        raise ValueError(f"no listed options found for {ticker}")
    if expiry_date is None:
        expiry_date = expiries[0]
    elif expiry_date not in expiries:
        raise ValueError(f"{ticker} has no expiry {expiry_date}; listed: {list(expiries)}")

    valuation = pd.Timestamp.today().normalize() if as_of is None else pd.Timestamp(as_of)
    chain = handle.option_chain(expiry_date)
    spot = None
    try:  # pragma: no cover - depends on live metadata
        spot = float(handle.fast_info["last_price"])
    except (KeyError, TypeError, ValueError):
        spot = None

    return build_chain(
        chain.calls,
        chain.puts,
        expiry=year_fraction(valuation, expiry_date),
        spot=spot,
        underlying=ticker.upper(),
        as_of=str(valuation.date()),
    )
