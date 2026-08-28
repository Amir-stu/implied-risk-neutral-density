"""Command line entry point: ``rnd``.

Three subcommands. ``demo`` runs the whole method on a bundled chain with no
network access, ``estimate`` runs it on a CSV of quotes, and ``fetch`` pulls a
live chain from Yahoo. All three print the same diagnostics block, because the
diagnostics are the deliverable as much as the density is.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .chain import OptionChain, year_fraction
from .data.synthetic import LognormalMixture, synthetic_chain
from .pipeline import METHODS, RNDResult, estimate_density

SAMPLE_CHAIN = Path(__file__).parent / "sample_data" / "synthetic_index_chain.csv"
SAMPLE_EXPIRY_YEARS = 0.25
EXIT_USAGE = 2


def _write_outputs(result: RNDResult, out_dir: Path, plot: bool) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []

    density_path = out_dir / "density.csv"
    result.density.to_frame().to_csv(density_path, index=False)
    written.append(density_path)

    quotes_path = out_dir / "fitted_quotes.csv"
    result.quotes.to_csv(quotes_path, index=False)
    written.append(quotes_path)

    diagnostics_path = out_dir / "diagnostics.json"
    payload = {
        "underlying": result.underlying,
        "method": result.method,
        "expiry_years": result.forward.expiry,
        "diagnostics": result.diagnostics(),
        "arbitrage": result.arbitrage.as_dict(),
    }
    diagnostics_path.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
    written.append(diagnostics_path)

    if plot:
        from .plotting import plot_report

        figure_path = out_dir / "report.png"
        plot_report(result, figure_path)
        written.append(figure_path)
    return written


def _report(result: RNDResult, args) -> int:
    print(result.summary())
    if args.out:
        for path in _write_outputs(result, Path(args.out), plot=not args.no_plot):
            print(f"wrote {path}")
    return 0


def _expiry_years(args, chain_expiry: float | None = None) -> float:
    if args.expiry_years is not None:
        return float(args.expiry_years)
    if args.expiry_date is not None:
        return year_fraction(args.as_of or "today", args.expiry_date)
    if chain_expiry is not None:
        return chain_expiry
    raise SystemExit("pass either --expiry-years or --expiry-date")


def _run_demo(args) -> int:
    if SAMPLE_CHAIN.exists():
        chain = OptionChain.from_csv(SAMPLE_CHAIN, SAMPLE_EXPIRY_YEARS, underlying="SAMPLE")
    else:  # pragma: no cover - only when the package data is missing
        model = LognormalMixture.crash_scenario(forward=100.0, expiry=SAMPLE_EXPIRY_YEARS)
        chain = synthetic_chain(model, SAMPLE_EXPIRY_YEARS, discount=0.99, seed=0)
    return _report(estimate_density(chain, method=args.method), args)


def _run_estimate(args) -> int:
    chain = OptionChain.from_csv(
        args.csv, _expiry_years(args), underlying=args.underlying or Path(args.csv).stem
    )
    result = estimate_density(
        chain, method=args.method, min_bid=args.min_bid, max_relative_spread=args.max_spread
    )
    return _report(result, args)


def _run_fetch(args) -> int:
    from .data.yahoo import load_yahoo_chain

    chain = load_yahoo_chain(args.ticker, args.expiry_date, args.as_of)
    result = estimate_density(
        chain, method=args.method, min_bid=args.min_bid, max_relative_spread=args.max_spread
    )
    return _report(result, args)


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--method", choices=METHODS, default="svi", help="smile model")
    parser.add_argument("--out", help="directory for density.csv, diagnostics.json and report.png")
    parser.add_argument("--no-plot", action="store_true", help="skip the PNG report")


def _add_filters(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--min-bid", type=float, default=0.01,
                        help="drop quotes bid below this")
    parser.add_argument("--max-spread", type=float, default=0.5,
                        help="drop quotes whose spread exceeds this fraction of the mid")


def build_parser() -> argparse.ArgumentParser:
    """Assemble the argument parser."""
    parser = argparse.ArgumentParser(
        prog="rnd",
        description="Recover the risk-neutral density implied by listed option prices.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    demo = subparsers.add_parser("demo", help="run on the bundled chain, no network needed")
    _add_common(demo)
    demo.set_defaults(handler=_run_demo)

    estimate = subparsers.add_parser("estimate", help="run on a CSV of quotes")
    estimate.add_argument("csv", help="CSV with strike, option_type, bid, ask")
    estimate.add_argument("--expiry-years", type=float, help="time to expiry in years")
    estimate.add_argument("--expiry-date", help="expiry date, as an alternative to --expiry-years")
    estimate.add_argument("--as-of", help="valuation date used with --expiry-date")
    estimate.add_argument("--underlying", help="label for the output")
    _add_common(estimate)
    _add_filters(estimate)
    estimate.set_defaults(handler=_run_estimate)

    fetch = subparsers.add_parser("fetch", help="download a live chain from Yahoo Finance")
    fetch.add_argument("ticker")
    fetch.add_argument("--expiry-date", help="listed expiry; the nearest one by default")
    fetch.add_argument("--as-of", help="valuation date, today by default")
    _add_common(fetch)
    _add_filters(fetch)
    fetch.set_defaults(handler=_run_fetch)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and dispatch. Returns a process exit code."""
    args = build_parser().parse_args(argv)
    try:
        return args.handler(args)
    except (ValueError, FileNotFoundError, ImportError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_USAGE


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
