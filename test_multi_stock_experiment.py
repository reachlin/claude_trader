#!/usr/bin/env python3
"""Tests for multi_stock_experiment.py — deeper BiLSTM comparison across stocks."""

import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from multi_stock_experiment import (
    discover_csvs,
    load_stock_df,
    split_train_test,
    build_predictor,
    evaluate_on_stocks,
    run_experiment,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ohlcv(n: int = 200, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 10.0 + np.cumsum(rng.normal(0, 0.1, n))
    close = np.clip(close, 1.0, None)
    spread = rng.uniform(0.02, 0.08, n)
    high = close * (1 + spread)
    low = close * (1 - spread)
    open_ = low + rng.uniform(0, 1, n) * (high - low)
    volume = rng.integers(100_000, 1_000_000, n).astype(float)
    return pd.DataFrame({
        "date": pd.date_range("2020-01-01", periods=n, freq="B").strftime("%Y-%m-%d"),
        "open": open_, "close": close, "high": high, "low": low, "volume": volume,
    })


# ---------------------------------------------------------------------------
# discover_csvs
# ---------------------------------------------------------------------------

class TestDiscoverCsvs:
    def test_finds_csv_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            for name in ("aaa.csv", "bbb.csv"):
                open(os.path.join(tmpdir, name), "w").close()
            open(os.path.join(tmpdir, "ignore.txt"), "w").close()
            result = discover_csvs(tmpdir)
            assert len(result) == 2
            assert all(p.endswith(".csv") for p in result)

    def test_excludes_non_csv(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            open(os.path.join(tmpdir, "data.json"), "w").close()
            open(os.path.join(tmpdir, "stock.csv"), "w").close()
            result = discover_csvs(tmpdir)
            assert len(result) == 1

    def test_returns_sorted(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            for name in ("c.csv", "a.csv", "b.csv"):
                open(os.path.join(tmpdir, name), "w").close()
            result = discover_csvs(tmpdir)
            assert result == sorted(result)

    def test_empty_dir_returns_empty(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            assert discover_csvs(tmpdir) == []


# ---------------------------------------------------------------------------
# load_stock_df
# ---------------------------------------------------------------------------

class TestLoadStockDf:
    def test_loads_and_computes_indicators(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "test.csv")
            _make_ohlcv(200).to_csv(csv_path, index=False)
            df = load_stock_df(csv_path)
            # Should have indicator columns
            from trading_bot import FEATURE_COLS
            for col in FEATURE_COLS:
                assert col in df.columns, f"Missing {col}"

    def test_drops_nan_rows(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "test.csv")
            _make_ohlcv(200).to_csv(csv_path, index=False)
            df = load_stock_df(csv_path)
            from trading_bot import FEATURE_COLS
            assert not df[FEATURE_COLS].isnull().any().any()

    def test_returns_dataframe(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "test.csv")
            _make_ohlcv(200).to_csv(csv_path, index=False)
            df = load_stock_df(csv_path)
            assert isinstance(df, pd.DataFrame)


# ---------------------------------------------------------------------------
# split_train_test
# ---------------------------------------------------------------------------

class TestSplitTrainTest:
    def test_split_sizes(self):
        df = pd.DataFrame({"x": range(100)})
        train, test = split_train_test(df, train_ratio=0.7)
        assert len(train) == 70
        assert len(test) == 30

    def test_no_overlap(self):
        df = pd.DataFrame({"x": range(100)})
        train, test = split_train_test(df, train_ratio=0.8)
        assert len(train) + len(test) == 100

    def test_returns_dataframes(self):
        df = pd.DataFrame({"x": range(100)})
        train, test = split_train_test(df, 0.7)
        assert isinstance(train, pd.DataFrame)
        assert isinstance(test, pd.DataFrame)


# ---------------------------------------------------------------------------
# build_predictor
# ---------------------------------------------------------------------------

class TestBuildPredictor:
    def test_returns_range_predictor(self):
        from range_predictor import RangePredictor
        p = build_predictor("baseline")
        assert isinstance(p, RangePredictor)

    def test_deep_config_has_more_layers(self):
        baseline = build_predictor("baseline")
        deep = build_predictor("deep")
        assert deep.num_layers > baseline.num_layers or deep.fc_sizes != baseline.fc_sizes

    def test_unknown_config_raises(self):
        with pytest.raises(ValueError):
            build_predictor("nonexistent_config")


# ---------------------------------------------------------------------------
# evaluate_on_stocks
# ---------------------------------------------------------------------------

class TestEvaluateOnStocks:
    def test_returns_dict_per_stock(self):
        from range_predictor import RangePredictor
        from trading_bot import FEATURE_COLS, compute_indicators

        # Build small test DataFrames
        dfs = {}
        for i in range(2):
            df = _make_ohlcv(200, seed=i)
            df = compute_indicators(df).dropna(subset=FEATURE_COLS).reset_index(drop=True)
            dfs[f"stock_{i}"] = df

        p = RangePredictor(window_size=20, epochs=2, batch_size=16)
        train_dfs = [df.iloc[:100].reset_index(drop=True) for df in dfs.values()]
        test_dfs = {k: df.iloc[100:].reset_index(drop=True) for k, df in dfs.items()}

        p.fit_multi(train_dfs)
        results = evaluate_on_stocks(p, test_dfs)

        assert isinstance(results, dict)
        for key in dfs:
            assert key in results
            assert "total_score" in results[key]
            assert "n_predictions" in results[key]


# ---------------------------------------------------------------------------
# run_experiment (integration)
# ---------------------------------------------------------------------------

class TestRunExperiment:
    def test_returns_results_for_all_configs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            for i in range(2):
                csv_path = os.path.join(tmpdir, f"stock_{i}.csv")
                _make_ohlcv(200, seed=i).to_csv(csv_path, index=False)

            results = run_experiment(
                data_dir=tmpdir,
                train_ratio=0.7,
                epochs=2,
                batch_size=16,
                configs=["baseline", "deep"],
            )
            assert "baseline" in results
            assert "deep" in results

    def test_results_contain_per_stock_scores(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            for i in range(2):
                csv_path = os.path.join(tmpdir, f"stock_{i}.csv")
                _make_ohlcv(200, seed=i).to_csv(csv_path, index=False)

            results = run_experiment(
                data_dir=tmpdir,
                train_ratio=0.7,
                epochs=2,
                batch_size=16,
                configs=["baseline"],
            )
            for stock_results in results["baseline"].values():
                assert "total_score" in stock_results
                assert "n_predictions" in stock_results
