"""Option chain container and the cleaning rules applied before any fitting.

Most of the difficulty in this exercise is not the mathematics, it is that a
listed chain contains quotes that are not prices: zero bids on far wings,
crossed markets left behind by a fast tape, strikes that have not traded in a
week. Fitting through them produces a smooth curve through noise and a density
that is smooth and wrong.

The rules below are conventional on a derivatives desk and each one is here
because of the specific damage it prevents.
"""

from __future__ import annotations

from datetime import date, datetime

import numpy as np
import pandas as pd

QUOTE_COLUMNS = ("strike", "option_type", "bid", "ask")
CALL, PUT = "C", "P"
DAYS_PER_YEAR = 365.0


def year_fraction(as_of: date | datetime | str, expiry: date | datetime | str) -> float:
    """Calendar-day year fraction between two dates, on an ACT/365 basis."""
    start = pd.Timestamp(as_of).normalize()
    end = pd.Timestamp(expiry).normalize()
    days = (end - start).days
    if days <= 0:
        raise ValueError("expiry must fall after the valuation date")
    return days / DAYS_PER_YEAR


def _normalise_option_type(values: pd.Series) -> pd.Series:
    mapped = values.astype(str).str.strip().str[0].str.upper()
    unknown = set(mapped.unique()) - {CALL, PUT}
    if unknown:
        raise ValueError(f"unrecognised option types: {sorted(unknown)}")
    return mapped


class OptionChain:
    """Quotes for a single underlying and a single expiry."""

    def __init__(
        self,
        quotes: pd.DataFrame,
        expiry: float,
        spot: float | None = None,
        underlying: str | None = None,
        as_of: str | None = None,
    ):
        missing = [c for c in QUOTE_COLUMNS if c not in quotes.columns]
        if missing:
            raise ValueError(f"quotes are missing required columns: {missing}")
        if expiry <= 0.0:
            raise ValueError("expiry must be positive")

        frame = quotes.copy()
        frame["option_type"] = _normalise_option_type(frame["option_type"])
        for column in ("strike", "bid", "ask"):
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame["mid"] = 0.5 * (frame["bid"] + frame["ask"])
        frame["spread"] = frame["ask"] - frame["bid"]

        self.quotes = frame.sort_values(["option_type", "strike"]).reset_index(drop=True)
        self.expiry = float(expiry)
        self.spot = None if spot is None else float(spot)
        self.underlying = underlying
        self.as_of = as_of

    def __len__(self) -> int:
        return len(self.quotes)

    def __repr__(self) -> str:  # pragma: no cover - convenience only
        name = self.underlying or "chain"
        return f"OptionChain({name}, T={self.expiry:.4f}y, {len(self)} quotes)"

    @property
    def calls(self) -> pd.DataFrame:
        return self.quotes[self.quotes["option_type"] == CALL]

    @property
    def puts(self) -> pd.DataFrame:
        return self.quotes[self.quotes["option_type"] == PUT]

    def _replace(self, frame: pd.DataFrame) -> OptionChain:
        return OptionChain(frame, self.expiry, self.spot, self.underlying, self.as_of)

    @classmethod
    def from_records(cls, records, expiry: float, **kwargs) -> OptionChain:
        """Build a chain from anything ``pandas`` can turn into a frame."""
        return cls(pd.DataFrame(records), expiry, **kwargs)

    @classmethod
    def from_csv(cls, path, expiry: float, **kwargs) -> OptionChain:
        """Read a chain from a CSV with the four required quote columns."""
        return cls(pd.read_csv(path), expiry, **kwargs)

    def to_csv(self, path) -> None:
        """Write the quote columns back out."""
        self.quotes.to_csv(path, index=False)

    def clean(
        self,
        min_bid: float = 0.01,
        max_relative_spread: float = 0.5,
        min_open_interest: int = 0,
    ) -> OptionChain:
        """Drop quotes that are not tradeable prices.

        A missing or zero bid means nobody is buying, so the mid is fiction. A
        crossed or locked market is stale data. A relative spread above
        ``max_relative_spread`` carries less information than the width of its
        own uncertainty, which is what makes deep wings dangerous rather than
        merely noisy.
        """
        frame = self.quotes
        keep = (
            frame["bid"].notna()
            & frame["ask"].notna()
            & (frame["bid"] >= min_bid)
            & (frame["ask"] > frame["bid"])
            & (frame["strike"] > 0.0)
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            relative = frame["spread"] / frame["mid"]
        keep &= relative <= max_relative_spread
        if min_open_interest > 0 and "open_interest" in frame.columns:
            keep &= frame["open_interest"].fillna(0) >= min_open_interest
        return self._replace(frame[keep])

    def matched_pairs(self) -> pd.DataFrame:
        """Strikes quoted as both a call and a put, side by side.

        This is the input to the put-call parity regression, so it is built
        from mids at strikes where both legs survived cleaning.
        """
        calls = self.calls[["strike", "mid", "spread"]].rename(
            columns={"mid": "call_mid", "spread": "call_spread"}
        )
        puts = self.puts[["strike", "mid", "spread"]].rename(
            columns={"mid": "put_mid", "spread": "put_spread"}
        )
        return calls.merge(puts, on="strike", how="inner").sort_values("strike")

    def out_of_the_money(self, forward: float) -> pd.DataFrame:
        """Keep puts below the forward and calls above it.

        Out-of-the-money options carry the information. Their in-the-money
        mirror images are dominated by intrinsic value, so the same amount of
        quote noise translates into a far larger volatility error, and they
        are usually the stalest quotes on the board.
        """
        if forward <= 0.0:
            raise ValueError("forward must be positive")
        frame = self.quotes
        is_call = frame["option_type"] == CALL
        keep = (is_call & (frame["strike"] >= forward)) | (~is_call & (frame["strike"] < forward))
        return frame[keep].sort_values("strike").reset_index(drop=True)
