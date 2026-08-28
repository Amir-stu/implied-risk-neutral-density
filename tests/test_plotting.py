"""Charts. Rendered on a headless backend and checked for content, not looks."""

from __future__ import annotations

import matplotlib
import pytest

matplotlib.use("Agg")

from rnd.plotting import (  # noqa: E402  - the backend must be set before pyplot loads
    plot_cdf,
    plot_density,
    plot_durrleman,
    plot_report,
    plot_smile,
)


@pytest.fixture(autouse=True)
def close_figures():
    yield
    matplotlib.pyplot.close("all")


def test_plot_smile_when_drawn_shows_the_quotes_and_the_fit(svi_result):
    axes = plot_smile(svi_result)
    assert axes.get_ylabel() == "implied volatility (%)"
    assert len(axes.collections) == 1
    assert len(axes.lines) >= 1


def test_plot_density_when_drawn_includes_the_lognormal_benchmark(svi_result):
    axes = plot_density(svi_result)
    labels = [line.get_label() for line in axes.lines]
    assert "implied" in labels
    assert any("lognormal" in str(label) for label in labels)


def test_plot_density_when_the_benchmark_is_suppressed_draws_one_curve(svi_result):
    axes = plot_density(svi_result, show_benchmark=False)
    assert len([line for line in axes.lines if line.get_label() == "implied"]) == 1


def test_plot_cdf_when_drawn_spans_the_unit_interval(svi_result):
    assert plot_cdf(svi_result.density).get_ylim() == (0.0, 1.0)


def test_plot_durrleman_when_drawn_labels_the_condition(svi_result):
    assert plot_durrleman(svi_result).get_ylabel() == "g(k)"


def test_plot_report_when_assembled_has_four_panels(svi_result):
    assert len(plot_report(svi_result).axes) == 4


def test_plot_report_when_given_a_path_writes_a_file(svi_result, tmp_path):
    path = tmp_path / "report.png"
    plot_report(svi_result, path)
    assert path.stat().st_size > 0
