"""Run the estimator on the bundled chain and write the report figure.

Reproduces the numbers quoted in the README. No network access needed.
"""

from __future__ import annotations

from pathlib import Path

from rnd import OptionChain, estimate_density
from rnd.cli import SAMPLE_CHAIN, SAMPLE_EXPIRY_YEARS
from rnd.plotting import plot_report

OUTPUT = Path(__file__).resolve().parents[1] / "docs" / "report.png"


def main() -> None:
    chain = OptionChain.from_csv(SAMPLE_CHAIN, SAMPLE_EXPIRY_YEARS, underlying="SAMPLE")
    result = estimate_density(chain, method="svi")
    print(result.summary())

    tails = result.tail_metrics()
    print(
        f"\na 20% drop: {tails['p(-20%)']:.2%} implied against "
        f"{tails['p(-20%) lognormal']:.2%} lognormal, "
        f"a ratio of {tails['ratio(-20%)']:.1f}x"
    )

    OUTPUT.parent.mkdir(exist_ok=True)
    plot_report(result, OUTPUT)
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
