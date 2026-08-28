"""The command line: exit codes, written artefacts and error handling."""

from __future__ import annotations

import json

import pytest

from rnd.cli import SAMPLE_CHAIN, build_parser, main


def test_sample_chain_when_packaged_is_present():
    assert SAMPLE_CHAIN.exists()


def test_demo_when_run_prints_a_summary_and_exits_clean(capsys):
    assert main(["demo", "--no-plot"]) == 0
    assert "forward" in capsys.readouterr().out


@pytest.mark.parametrize("method", ["svi", "spline"])
def test_demo_when_a_method_is_chosen_names_it_in_the_summary(capsys, method):
    main(["demo", "--method", method, "--no-plot"])
    assert f"method={method}" in capsys.readouterr().out


def test_demo_when_an_output_directory_is_given_writes_every_artefact(tmp_path, capsys):
    assert main(["demo", "--no-plot", "--out", str(tmp_path)]) == 0
    assert (tmp_path / "density.csv").exists()
    assert (tmp_path / "fitted_quotes.csv").exists()
    assert (tmp_path / "diagnostics.json").exists()
    assert "wrote" in capsys.readouterr().out


def test_demo_when_plotting_is_enabled_writes_the_report(tmp_path):
    main(["demo", "--out", str(tmp_path)])
    assert (tmp_path / "report.png").stat().st_size > 0


def test_diagnostics_json_when_written_is_valid_and_complete(tmp_path):
    main(["demo", "--no-plot", "--out", str(tmp_path)])
    payload = json.loads((tmp_path / "diagnostics.json").read_text(encoding="utf-8"))
    assert payload["method"] == "svi"
    assert {"total_mass", "martingale_error_bps"} <= set(payload["diagnostics"])
    assert "worst_butterfly" in payload["arbitrage"]


def test_estimate_when_given_a_csv_and_an_expiry_runs(tmp_path, clean_chain, capsys):
    path = tmp_path / "quotes.csv"
    clean_chain.to_csv(path)
    assert main(["estimate", str(path), "--expiry-years", "0.25", "--no-plot"]) == 0
    assert "atm vol" in capsys.readouterr().out


def test_estimate_when_given_an_expiry_date_derives_the_year_fraction(tmp_path, clean_chain):
    path = tmp_path / "quotes.csv"
    clean_chain.to_csv(path)
    exit_code = main([
        "estimate", str(path), "--as-of", "2026-01-01", "--expiry-date", "2026-04-02",
        "--no-plot",
    ])
    assert exit_code == 0


def test_estimate_when_an_underlying_is_given_labels_the_output(tmp_path, clean_chain, capsys):
    path = tmp_path / "quotes.csv"
    clean_chain.to_csv(path)
    main(["estimate", str(path), "--expiry-years", "0.25", "--underlying", "ACME", "--no-plot"])
    assert "ACME" in capsys.readouterr().out


def test_estimate_when_neither_expiry_argument_is_given_exits(tmp_path, clean_chain):
    path = tmp_path / "quotes.csv"
    clean_chain.to_csv(path)
    with pytest.raises(SystemExit):
        main(["estimate", str(path), "--no-plot"])


def test_estimate_when_the_file_is_missing_returns_a_usage_code(tmp_path, capsys):
    exit_code = main(["estimate", str(tmp_path / "absent.csv"), "--expiry-years", "0.25"])
    assert exit_code == 2
    assert "error:" in capsys.readouterr().err


def test_estimate_when_the_chain_is_unusable_returns_a_usage_code(tmp_path, capsys):
    path = tmp_path / "thin.csv"
    path.write_text("strike,option_type,bid,ask\n100,C,1.0,1.1\n", encoding="utf-8")
    assert main(["estimate", str(path), "--expiry-years", "0.25"]) == 2
    assert "error:" in capsys.readouterr().err


def test_parser_when_no_subcommand_is_given_exits():
    with pytest.raises(SystemExit):
        build_parser().parse_args([])


def test_parser_when_the_method_is_unknown_exits():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["demo", "--method", "kernel"])


def test_parser_when_filters_are_supplied_parses_them():
    args = build_parser().parse_args(
        ["estimate", "quotes.csv", "--expiry-years", "0.5", "--min-bid", "0.05",
         "--max-spread", "0.2"]
    )
    assert args.min_bid == 0.05
    assert args.max_spread == 0.2
