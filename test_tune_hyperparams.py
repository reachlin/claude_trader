#!/usr/bin/env python3
"""Tests for the hyperparameter tuning script."""

import numpy as np
import pandas as pd
import pytest

from trading_bot import FEATURE_COLS, TradingBot, compute_indicators


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_ohlcv(n=500, seed=42):
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range("2020-01-01", periods=n)
    close = 10.0 + np.cumsum(rng.randn(n) * 0.1)
    close = np.maximum(close, 1.0)
    return pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d"),
        "open": close * (1 + rng.randn(n) * 0.005),
        "high": close * (1 + np.abs(rng.randn(n) * 0.01)),
        "low": close * (1 - np.abs(rng.randn(n) * 0.01)),
        "close": close,
        "volume": rng.randint(1_000_000, 10_000_000, n).astype(float),
    })


# ---------------------------------------------------------------------------
# Grid content tests
# ---------------------------------------------------------------------------
class TestGridContents:
    def test_kmeans_grid_has_valid_keys(self):
        from tune_hyperparams import KMEANS_GRID
        assert "n_clusters" in KMEANS_GRID
        assert "feature_subsets" in KMEANS_GRID
        for n in KMEANS_GRID["n_clusters"]:
            assert isinstance(n, int) and n >= 2

    def test_lstm_phase1_grid_has_valid_keys(self):
        from tune_hyperparams import LSTM_PHASE1_GRID
        assert "window_size" in LSTM_PHASE1_GRID
        assert "lr" in LSTM_PHASE1_GRID
        assert "batch_size" in LSTM_PHASE1_GRID

    def test_lstm_phase2_grid_has_valid_keys(self):
        from tune_hyperparams import LSTM_PHASE2_GRID
        assert "hidden1" in LSTM_PHASE2_GRID
        assert "hidden2" in LSTM_PHASE2_GRID

    def test_feature_subsets_all_valid(self):
        from tune_hyperparams import KMEANS_GRID
        for name, cols in KMEANS_GRID["feature_subsets"]:
            for c in cols:
                assert c in FEATURE_COLS, f"{c} not in FEATURE_COLS"


# ---------------------------------------------------------------------------
# Validation split tests
# ---------------------------------------------------------------------------
class TestValidationSplit:
    def test_inner_split_is_chronological(self):
        from tune_hyperparams import make_inner_split
        df = _make_ohlcv(500)
        df = compute_indicators(df).dropna(subset=FEATURE_COLS).reset_index(drop=True)
        train_df, val_df = make_inner_split(df, val_ratio=0.25)
        # train comes before val chronologically
        assert len(train_df) > 0
        assert len(val_df) > 0
        assert train_df["date"].iloc[-1] <= val_df["date"].iloc[0]

    def test_no_overlap(self):
        from tune_hyperparams import make_inner_split
        df = _make_ohlcv(500)
        df = compute_indicators(df).dropna(subset=FEATURE_COLS).reset_index(drop=True)
        train_df, val_df = make_inner_split(df, val_ratio=0.25)
        train_dates = set(train_df["date"])
        val_dates = set(val_df["date"])
        assert len(train_dates & val_dates) == 0


# ---------------------------------------------------------------------------
# Tune function return tests
# ---------------------------------------------------------------------------
class TestTuneFunctions:
    def test_tune_kmeans_returns_sorted(self):
        from tune_hyperparams import tune_kmeans
        df = _make_ohlcv(300)
        results = tune_kmeans(df, train_ratio=0.6, top_k=3)
        assert len(results) > 0
        # Should be sorted by sharpe descending
        sharpes = [r["sharpe_ratio"] for r in results]
        assert sharpes == sorted(sharpes, reverse=True)

    def test_tune_kmeans_result_keys(self):
        from tune_hyperparams import tune_kmeans
        df = _make_ohlcv(300)
        results = tune_kmeans(df, train_ratio=0.6, top_k=3)
        for r in results:
            assert "params" in r
            assert "sharpe_ratio" in r
            assert "total_return" in r


# ---------------------------------------------------------------------------
# Signal name mapping tests
# ---------------------------------------------------------------------------
class TestSignalNameMapping:
    @pytest.mark.parametrize("n", [3, 4, 5, 6, 7, 8])
    def test_assign_signal_names_length(self, n):
        names = TradingBot._assign_signal_names(n)
        assert len(names) == n

    @pytest.mark.parametrize("n", [3, 4, 5, 6, 7, 8])
    def test_assign_signal_names_valid(self, n):
        valid = {"strong_sell", "mild_sell", "hold", "mild_buy", "strong_buy"}
        names = TradingBot._assign_signal_names(n)
        assert all(s in valid for s in names)

    def test_assign_signal_names_5_is_canonical(self):
        names = TradingBot._assign_signal_names(5)
        assert names == ["strong_sell", "mild_sell", "hold", "mild_buy", "strong_buy"]

    def test_endpoints_are_extreme(self):
        """First should be strong_sell, last should be strong_buy."""
        for n in [3, 4, 5, 6, 7, 8]:
            names = TradingBot._assign_signal_names(n)
            assert names[0] == "strong_sell"
            assert names[-1] == "strong_buy"
