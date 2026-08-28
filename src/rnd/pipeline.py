"""End to end: a chain of quotes in, a risk-neutral density and its diagnostics out.

The order of operations is the whole method, and it is not arbitrary.

1. Clean the quotes. Untradeable prices are removed, not down-weighted.
2. Recover the forward and the discount factor from put-call parity, so no
   interest rate or dividend assumption enters anywhere.
3. Keep the out-of-the-money wing of each side, where the information is.
4. Invert to implied volatility. Quotes outside the no-arbitrage bounds return
   ``nan`` here and drop out on their own.
5. Fit a smile to total variance in log-moneyness, weighted by how tightly each
   quote pins its own volatility.
6. Differentiate the smile analytically into a density.
7. Check the result: does it integrate to one, does it price the forward back,
   is it non-negative everywhere.

Step 7 is not decoration. A density that fails it is wrong regardless of how
convincing the plot looks.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .arbitrage import ArbitrageReport, check_call_curve
from .black import implied_vol_vector, vega
from .chain import CALL, OptionChain
from .density import RiskNeutralDensity, density_from_smile
from .forward import ForwardEstimate, implied_forward
from .smile import FlatSmile, Smile, fit_spline_smile
from .svi import fit_svi

METHODS = ("svi", "spline")
MIN_QUOTES_FOR_FIT = 5


def _quote_weights(frame: pd.DataFrame, forward: float, expiry: float, discount: float):
    """Weight each quote by how precisely it determines its own volatility.

    A quote pins the volatility to about ``half_spread / vega``. Fitting total
    variance rather than volatility adds the Jacobian ``2 * sigma * T``, so the
    weight below is one over the implied uncertainty in total variance. In
    practice this stops a two-cent wing quote from dragging the fit around as
    hard as a liquid at-the-money strike.
    """
    vegas = vega(forward, frame["strike"].to_numpy(), frame["total_variance"].to_numpy(),
                 expiry, discount)
    half_spread = np.maximum(0.5 * frame["spread"].to_numpy(), 1e-6)
    vol_uncertainty = np.maximum(half_spread / np.maximum(vegas, 1e-8), 1e-6)
    variance_uncertainty = 2.0 * frame["implied_vol"].to_numpy() * expiry * vol_uncertainty
    return 1.0 / np.maximum(variance_uncertainty, 1e-10)


def _implied_vol_frame(chain: OptionChain, estimate: ForwardEstimate) -> pd.DataFrame:
    """Out-of-the-money quotes with implied volatility and log-moneyness attached."""
    frame = chain.out_of_the_money(estimate.forward).copy()
    frame["implied_vol"] = implied_vol_vector(
        frame["mid"].to_numpy(),
        estimate.forward,
        frame["strike"].to_numpy(),
        estimate.expiry,
        estimate.discount,
        (frame["option_type"] == CALL).to_numpy(),
    )
    frame = frame[np.isfinite(frame["implied_vol"])].reset_index(drop=True)
    frame["log_moneyness"] = np.log(frame["strike"] / estimate.forward)
    frame["total_variance"] = frame["implied_vol"] ** 2 * estimate.expiry
    return frame


def _fit_smile(method: str, frame: pd.DataFrame, estimate: ForwardEstimate,
               weights, smoothing) -> Smile:
    if method == "svi":
        return fit_svi(frame["log_moneyness"], frame["total_variance"],
                       estimate.expiry, estimate.forward, weights)
    if method == "spline":
        return fit_spline_smile(frame["log_moneyness"], frame["total_variance"],
                                estimate.expiry, estimate.forward, weights, smoothing)
    raise ValueError(f"unknown method {method!r}; expected one of {METHODS}")


@dataclass(frozen=True)
class RNDResult:
    """A fitted density together with everything needed to distrust it."""

    density: RiskNeutralDensity
    smile: Smile
    forward: ForwardEstimate
    quotes: pd.DataFrame
    arbitrage: ArbitrageReport
    fit_rmse_vol: float
    method: str
    underlying: str | None = None

    @property
    def atm_vol(self) -> float:
        """Implied volatility at the forward."""
        return float(np.atleast_1d(self.smile.implied_vol(0.0))[0])

    def lognormal_benchmark(self) -> RiskNeutralDensity:
        """The Black-Scholes density with the same forward and at-the-money vol.

        The gap between this and the fitted density is the entire point of the
        exercise: same centre, same at-the-money volatility, different tails.
        """
        flat = FlatSmile(self.atm_vol, self.forward.expiry, self.forward.forward)
        return density_from_smile(flat, strikes=self.density.strikes)

    def tail_metrics(self, moves=(-0.20, -0.10, 0.10)) -> dict[str, float]:
        """Probability of each move, fitted against lognormal.

        A ratio above one on the downside is the crash risk the market is
        paying for and the lognormal is missing.
        """
        benchmark = self.lognormal_benchmark()
        out: dict[str, float] = {}
        for move in moves:
            level = self.forward.forward * (1.0 + move)
            if move < 0:
                fitted, base = self.density.prob_below(level), benchmark.prob_below(level)
            else:
                fitted = 1.0 - self.density.prob_below(level)
                base = 1.0 - benchmark.prob_below(level)
            label = f"{move:+.0%}"
            out[f"p({label})"] = fitted
            out[f"p({label}) lognormal"] = base
            out[f"ratio({label})"] = fitted / base if base > 0 else float("nan")
        return out

    def diagnostics(self) -> dict[str, float]:
        """Every number a reviewer should look at before believing the density."""
        moments = self.density.moments()
        return {
            "n_quotes": float(len(self.quotes)),
            "forward": self.forward.forward,
            "discount": self.forward.discount,
            "implied_rate": self.forward.implied_rate,
            "parity_r_squared": self.forward.r_squared,
            "atm_vol": self.atm_vol,
            "fit_rmse_vol_points": self.fit_rmse_vol * 100.0,
            "total_mass": self.density.total_mass,
            "martingale_error_bps": self.density.martingale_error() * 1e4,
            "min_density": self.density.min_density,
            "worst_durrleman_g": self.smile.butterfly_violation(),
            **moments.as_dict(),
            "q05": self.density.quantile(0.05),
            "q50": self.density.quantile(0.50),
            "q95": self.density.quantile(0.95),
            **self.tail_metrics(),
        }

    def summary(self) -> str:
        """Human-readable diagnostics block."""
        lines = [
            f"{self.underlying or 'chain'}  method={self.method}  "
            f"T={self.forward.expiry:.4f}y  quotes={len(self.quotes)}",
            f"forward {self.forward.forward:,.4f}   discount {self.forward.discount:.6f}   "
            f"implied rate {self.forward.implied_rate:.4%}",
            f"atm vol {self.atm_vol:.4%}   smile fit rmse "
            f"{self.fit_rmse_vol * 100:.3f} vol points",
            f"mass {self.density.total_mass:.6f}   martingale error "
            f"{self.density.martingale_error() * 1e4:+.2f} bp   "
            f"min density {self.density.min_density:.3e}",
            f"quotes: {self.arbitrage.summary()}",
        ]
        moments = self.density.moments()
        lines.append(
            f"mean {moments.mean:,.4f}   sd {moments.std:,.4f}   "
            f"skew {moments.skewness:+.4f}   excess kurtosis {moments.excess_kurtosis:+.4f}"
        )
        tails = self.tail_metrics()
        lines.append(
            "  ".join(
                f"{key} {value:.4f}" for key, value in tails.items() if key.startswith("p(")
            )
        )
        return "\n".join(lines)


def estimate_density(
    chain: OptionChain,
    method: str = "svi",
    n_points: int = 801,
    width: float = 6.0,
    smoothing: float | None = None,
    min_bid: float = 0.01,
    max_relative_spread: float = 0.5,
    band: int = 8,
) -> RNDResult:
    """Fit a risk-neutral density to a cleaned option chain.

    ``method`` selects the smile: ``"svi"`` for the parametric fit with the
    no-butterfly penalty, ``"spline"`` for the weighted smoothing spline. On a
    well-behaved chain they agree closely; where they disagree, the chain is
    telling you something about its own quality.
    """
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; expected one of {METHODS}")

    clean = chain.clean(min_bid=min_bid, max_relative_spread=max_relative_spread)
    pairs = clean.matched_pairs()
    if len(pairs) < 3:
        raise ValueError("not enough matched call-put pairs to imply a forward")
    parity_weights = 1.0 / np.maximum(pairs["call_spread"] + pairs["put_spread"], 1e-6)
    estimate = implied_forward(
        pairs["strike"], pairs["call_mid"], pairs["put_mid"], clean.expiry,
        weights=parity_weights, band=band,
    )

    frame = _implied_vol_frame(clean, estimate)
    if len(frame) < MIN_QUOTES_FOR_FIT:
        raise ValueError(
            f"only {len(frame)} quotes survived cleaning; need at least {MIN_QUOTES_FOR_FIT}"
        )
    weights = _quote_weights(frame, estimate.forward, estimate.expiry, estimate.discount)
    smile = _fit_smile(method, frame, estimate, weights, smoothing)

    fitted_vol = smile.implied_vol(frame["log_moneyness"].to_numpy())
    rmse = float(np.sqrt(np.mean((fitted_vol - frame["implied_vol"].to_numpy()) ** 2)))
    frame["fitted_vol"] = fitted_vol

    density = density_from_smile(smile, n_points=n_points, width=width)
    calls = clean.calls.sort_values("strike")
    report = check_call_curve(
        calls["strike"].to_numpy(), calls["mid"].to_numpy(),
        estimate.forward, estimate.discount,
    )
    return RNDResult(
        density=density,
        smile=smile,
        forward=estimate,
        quotes=frame,
        arbitrage=report,
        fit_rmse_vol=rmse,
        method=method,
        underlying=chain.underlying,
    )
