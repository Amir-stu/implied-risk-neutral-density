"""Charts for a fitted density. ``matplotlib`` is an optional dependency.

Four panels, because four things need to be looked at before a density is
believed: the fit against the quotes it came from, the density itself against
the lognormal it is supposed to beat, the cumulative distribution, and
Durrleman's ``g``. The last panel is the one that catches a bad fit -- it dips
below zero exactly where the density would have gone negative.
"""

from __future__ import annotations

from .density import RiskNeutralDensity
from .pipeline import RNDResult

MARKET_COLOUR = "#2b2b2b"
FIT_COLOUR = "#0b6e99"
BENCHMARK_COLOUR = "#b03a2e"


def _pyplot():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "plotting needs the optional dependency: pip install 'implied-rnd[plot]'"
        ) from exc
    return plt


def plot_smile(result: RNDResult, ax=None):
    """Quoted implied volatilities against the fitted smile."""
    plt = _pyplot()
    ax = ax or plt.subplots()[1]
    quotes = result.quotes
    ax.scatter(quotes["strike"], quotes["implied_vol"] * 100, s=16,
               color=MARKET_COLOUR, label="quoted mid", zorder=3)
    strikes = result.density.strikes
    fitted = result.smile.implied_vol(result.smile.log_moneyness(strikes)) * 100
    inside = (strikes >= quotes["strike"].min()) & (strikes <= quotes["strike"].max())
    ax.plot(strikes[inside], fitted[inside], color=FIT_COLOUR, lw=1.8,
            label=f"{result.method} fit")
    ax.axvline(result.forward.forward, color="grey", lw=0.8, ls="--", label="forward")
    ax.set_xlabel("strike")
    ax.set_ylabel("implied volatility (%)")
    ax.set_title("Smile")
    ax.legend(frameon=False, fontsize=8)
    return ax


def plot_density(result: RNDResult, ax=None, show_benchmark: bool = True):
    """The implied density, with the equivalent lognormal for scale."""
    plt = _pyplot()
    ax = ax or plt.subplots()[1]
    density = result.density
    ax.plot(density.strikes, density.density, color=FIT_COLOUR, lw=1.8, label="implied")
    if show_benchmark:
        benchmark = result.lognormal_benchmark()
        ax.plot(benchmark.strikes, benchmark.density, color=BENCHMARK_COLOUR, lw=1.4,
                ls="--", label="lognormal at the same ATM vol")
    ax.axvline(result.forward.forward, color="grey", lw=0.8, ls="--")
    ax.set_xlim(density.quantile(0.001), density.quantile(0.999))
    ax.set_xlabel("price at expiry")
    ax.set_ylabel("risk-neutral density")
    ax.set_title("Implied distribution")
    ax.legend(frameon=False, fontsize=8)
    return ax


def plot_cdf(density: RiskNeutralDensity, ax=None):
    """Cumulative risk-neutral distribution."""
    plt = _pyplot()
    ax = ax or plt.subplots()[1]
    ax.plot(density.strikes, density.cdf(), color=FIT_COLOUR, lw=1.6)
    ax.set_xlim(density.quantile(0.001), density.quantile(0.999))
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("price at expiry")
    ax.set_ylabel("cumulative probability")
    ax.set_title("Cumulative distribution")
    return ax


def plot_durrleman(result: RNDResult, ax=None):
    """Durrleman's ``g``; negative anywhere means the smile is not arbitrage-free."""
    plt = _pyplot()
    ax = ax or plt.subplots()[1]
    k = result.smile.log_moneyness(result.density.strikes)
    ax.plot(k, result.smile.durrleman_g(k), color=FIT_COLOUR, lw=1.6)
    ax.axhline(0.0, color=BENCHMARK_COLOUR, lw=1.0)
    ax.set_xlabel("log-moneyness")
    ax.set_ylabel("g(k)")
    ax.set_title("Butterfly condition")
    return ax


def plot_report(result: RNDResult, path=None, figsize=(11.0, 8.0)):
    """Assemble all four panels into one figure, optionally saving it."""
    plt = _pyplot()
    figure, axes = plt.subplots(2, 2, figsize=figsize)
    plot_smile(result, axes[0][0])
    plot_density(result, axes[0][1])
    plot_cdf(result.density, axes[1][0])
    plot_durrleman(result, axes[1][1])
    label = result.underlying or "chain"
    figure.suptitle(
        f"{label}   T = {result.forward.expiry:.3f}y   forward "
        f"{result.forward.forward:,.2f}   ATM vol {result.atm_vol:.1%}",
        fontsize=11,
    )
    figure.tight_layout()
    if path is not None:
        figure.savefig(path, dpi=150)
    return figure
