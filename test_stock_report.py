"""Tests for the --report mode functions added to daily_pipeline.py."""

import os
import numpy as np
import pandas as pd
import pytest


def _make_csv(path: str, n: int = 300):
    """Write a minimal OHLCV CSV."""
    rng = np.random.RandomState(42)
    dates = pd.bdate_range("2020-01-01", periods=n)
    close = 10.0 + np.cumsum(rng.randn(n) * 0.3)
    close = np.maximum(close, 1.0)
    df = pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d"),
        "open":   close * (1 + rng.randn(n) * 0.005),
        "high":   close * (1 + np.abs(rng.randn(n) * 0.01)),
        "low":    close * (1 - np.abs(rng.randn(n) * 0.01)),
        "close":  close,
        "volume": rng.randint(1_000_000, 10_000_000, n).astype(float),
    })
    df.to_csv(path, index=False)


@pytest.fixture()
def data_dir(tmp_path):
    _make_csv(str(tmp_path / "000001_20yr.csv"))
    _make_csv(str(tmp_path / "600519_20yr.csv"))
    _make_csv(str(tmp_path / "601933_20yr.csv"))
    # Files that should be ignored
    (tmp_path / "601933_3yr.csv").write_text("junk")
    (tmp_path / "601933_2025.csv").write_text("junk")
    (tmp_path / "random_picks_log.json").write_text("{}")
    return str(tmp_path)


def _make_record(symbol, sharpe, ret, drawdown, win_rate):
    return {
        "symbol": symbol, "label": symbol,
        "avg_sharpe": sharpe, "avg_return": ret,
        "avg_max_drawdown": drawdown, "avg_win_rate": win_rate,
        "bh_return": ret * 0.6,
        "km_return": ret, "km_sharpe": sharpe,
        "lstm_return": ret * 0.9, "lstm_sharpe": sharpe * 0.9,
        "lgbm_return": ret * 1.1, "lgbm_sharpe": sharpe * 1.1,
        "ppo_return": ret * 0.8, "ppo_sharpe": sharpe * 0.8,
        "td3_return": ret * 1.0, "td3_sharpe": sharpe * 1.0,
    }


# ---------------------------------------------------------------------------
# discover_stock_csvs
# ---------------------------------------------------------------------------

class TestDiscoverStockCsvs:
    def test_finds_20yr_csvs(self, data_dir):
        from daily_pipeline import discover_stock_csvs
        paths = discover_stock_csvs(data_dir)
        basenames = [os.path.basename(p) for p in paths]
        assert "000001_20yr.csv" in basenames
        assert "600519_20yr.csv" in basenames
        assert "601933_20yr.csv" in basenames

    def test_excludes_non_20yr(self, data_dir):
        from daily_pipeline import discover_stock_csvs
        paths = discover_stock_csvs(data_dir)
        basenames = [os.path.basename(p) for p in paths]
        assert "601933_3yr.csv" not in basenames
        assert "601933_2025.csv" not in basenames
        assert "random_picks_log.json" not in basenames

    def test_returns_sorted(self, data_dir):
        from daily_pipeline import discover_stock_csvs
        paths = discover_stock_csvs(data_dir)
        basenames = [os.path.basename(p) for p in paths]
        assert basenames == sorted(basenames)

    def test_missing_dir_returns_empty(self, tmp_path):
        from daily_pipeline import discover_stock_csvs
        result = discover_stock_csvs(str(tmp_path / "nonexistent"))
        assert result == []


# ---------------------------------------------------------------------------
# compute_composite_score
# ---------------------------------------------------------------------------

class TestCompositeScore:
    def test_higher_sharpe_wins(self):
        from daily_pipeline import compute_composite_score
        r1 = _make_record("A", sharpe=2.0, ret=10.0, drawdown=-10.0, win_rate=55.0)
        r2 = _make_record("B", sharpe=0.5, ret=10.0, drawdown=-10.0, win_rate=55.0)
        assert compute_composite_score(r1) > compute_composite_score(r2)

    def test_higher_return_wins(self):
        from daily_pipeline import compute_composite_score
        r1 = _make_record("A", sharpe=1.0, ret=50.0, drawdown=-10.0, win_rate=55.0)
        r2 = _make_record("B", sharpe=1.0, ret=5.0,  drawdown=-10.0, win_rate=55.0)
        assert compute_composite_score(r1) > compute_composite_score(r2)

    def test_lower_drawdown_wins(self):
        from daily_pipeline import compute_composite_score
        r1 = _make_record("A", sharpe=1.0, ret=10.0, drawdown=-5.0,  win_rate=55.0)
        r2 = _make_record("B", sharpe=1.0, ret=10.0, drawdown=-40.0, win_rate=55.0)
        assert compute_composite_score(r1) > compute_composite_score(r2)

    def test_returns_float(self):
        from daily_pipeline import compute_composite_score
        r = _make_record("A", sharpe=1.0, ret=10.0, drawdown=-10.0, win_rate=55.0)
        assert isinstance(compute_composite_score(r), float)


# ---------------------------------------------------------------------------
# rank_stocks
# ---------------------------------------------------------------------------

class TestRankStocks:
    def test_best_stock_ranked_first(self):
        from daily_pipeline import rank_stocks
        records = [
            _make_record("A", sharpe=0.5, ret=5.0,  drawdown=-20.0, win_rate=50.0),
            _make_record("B", sharpe=2.0, ret=30.0, drawdown=-5.0,  win_rate=65.0),
            _make_record("C", sharpe=1.0, ret=10.0, drawdown=-15.0, win_rate=55.0),
        ]
        ranked = rank_stocks(records)
        assert ranked[0]["symbol"] == "B"
        assert ranked[-1]["symbol"] == "A"

    def test_adds_composite_score_and_rank(self):
        from daily_pipeline import rank_stocks
        records = [_make_record("X", sharpe=1.0, ret=10.0, drawdown=-10.0, win_rate=55.0)]
        ranked = rank_stocks(records)
        assert "composite_score" in ranked[0]
        assert ranked[0]["rank"] == 1

    def test_ranks_are_sequential(self):
        from daily_pipeline import rank_stocks
        records = [
            _make_record("A", sharpe=0.5, ret=5.0,  drawdown=-20.0, win_rate=50.0),
            _make_record("B", sharpe=2.0, ret=30.0, drawdown=-5.0,  win_rate=65.0),
            _make_record("C", sharpe=1.0, ret=10.0, drawdown=-15.0, win_rate=55.0),
        ]
        ranked = rank_stocks(records)
        assert [r["rank"] for r in ranked] == [1, 2, 3]


# ---------------------------------------------------------------------------
# format_report
# ---------------------------------------------------------------------------

class TestFormatReport:
    def _three_records(self):
        return rank_stocks_helper([
            _make_record("AAA", sharpe=2.0, ret=30.0, drawdown=-5.0,  win_rate=65.0),
            _make_record("BBB", sharpe=1.5, ret=20.0, drawdown=-10.0, win_rate=60.0),
            _make_record("CCC", sharpe=1.0, ret=10.0, drawdown=-15.0, win_rate=55.0),
        ])

    def test_contains_top3_symbols(self):
        from daily_pipeline import format_report, rank_stocks
        records = rank_stocks([
            _make_record("AAA", sharpe=2.0, ret=30.0, drawdown=-5.0,  win_rate=65.0),
            _make_record("BBB", sharpe=1.5, ret=20.0, drawdown=-10.0, win_rate=60.0),
            _make_record("CCC", sharpe=1.0, ret=10.0, drawdown=-15.0, win_rate=55.0),
        ])
        md = format_report(records, top_n=3)
        assert "AAA" in md
        assert "BBB" in md
        assert "CCC" in md

    def test_starts_with_heading(self):
        from daily_pipeline import format_report, rank_stocks
        records = rank_stocks([_make_record("AAA", 2.0, 30.0, -5.0, 65.0)])
        md = format_report(records, top_n=1)
        assert md.startswith("#")

    def test_contains_markdown_table(self):
        from daily_pipeline import format_report, rank_stocks
        records = rank_stocks([_make_record("AAA", 2.0, 30.0, -5.0, 65.0)])
        md = format_report(records, top_n=1)
        assert "|" in md

    def test_top3_section_present(self):
        from daily_pipeline import format_report, rank_stocks
        records = rank_stocks([
            _make_record("AAA", 2.0, 30.0, -5.0, 65.0),
            _make_record("BBB", 1.5, 20.0, -10.0, 60.0),
            _make_record("CCC", 1.0, 10.0, -15.0, 55.0),
        ])
        md = format_report(records, top_n=3)
        assert "Top 3" in md

    def test_full_ranking_table_present(self):
        from daily_pipeline import format_report, rank_stocks
        records = rank_stocks([
            _make_record("AAA", 2.0, 30.0, -5.0, 65.0),
            _make_record("BBB", 1.5, 20.0, -10.0, 60.0),
        ])
        md = format_report(records, top_n=2)
        assert "Full Rankings" in md
