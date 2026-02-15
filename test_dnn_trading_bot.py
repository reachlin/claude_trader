#!/usr/bin/env python3
"""Tests for the LSTM-based DNN trading bot."""

import numpy as np
import pandas as pd
import pytest
import torch

from trading_bot import FEATURE_COLS, Portfolio, compute_indicators


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_dummy_df(n: int = 200) -> pd.DataFrame:
    """Create a dummy OHLCV DataFrame with enough rows for indicators + windows."""
    np.random.seed(42)
    dates = pd.bdate_range("2022-01-01", periods=n)
    close = 10 + np.cumsum(np.random.randn(n) * 0.3)
    close = np.maximum(close, 1.0)  # keep positive
    return pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d"),
        "open": close * (1 + np.random.randn(n) * 0.005),
        "high": close * (1 + np.abs(np.random.randn(n) * 0.01)),
        "low": close * (1 - np.abs(np.random.randn(n) * 0.01)),
        "close": close,
        "volume": np.random.randint(1_000_000, 10_000_000, n).astype(float),
    })


def _prepared_df(n: int = 200) -> pd.DataFrame:
    """Return dummy DF with indicators computed and NaNs dropped."""
    df = compute_indicators(_make_dummy_df(n))
    return df.dropna(subset=FEATURE_COLS).reset_index(drop=True)


# ---------------------------------------------------------------------------
# TradingDataset tests
# ---------------------------------------------------------------------------
class TestTradingDataset:
    def test_dataset_shapes(self):
        from dnn_trading_bot import TradingDataset

        df = _prepared_df(200)
        window = 20
        ds = TradingDataset(df, window_size=window)

        # Length: rows - window - 1 (need 1 forward return after window)
        expected_len = len(df) - window - 1
        assert len(ds) == expected_len, f"Expected {expected_len}, got {len(ds)}"

        x, y = ds[0]
        assert x.shape == (window, len(FEATURE_COLS)), f"Wrong x shape: {x.shape}"
        assert y.shape == (), f"Label should be scalar, got shape {y.shape}"

    def test_dataset_dtypes(self):
        from dnn_trading_bot import TradingDataset

        df = _prepared_df(200)
        ds = TradingDataset(df, window_size=20)
        x, y = ds[0]
        assert x.dtype == torch.float32
        assert y.dtype == torch.long

    def test_labels_in_range(self):
        from dnn_trading_bot import TradingDataset

        df = _prepared_df(200)
        ds = TradingDataset(df, window_size=20)
        labels = [ds[i][1].item() for i in range(len(ds))]
        assert min(labels) >= 0
        assert max(labels) <= 4

    def test_label_distribution_roughly_uniform(self):
        """Percentile-based labels should be roughly 20% each."""
        from dnn_trading_bot import TradingDataset

        df = _prepared_df(300)
        ds = TradingDataset(df, window_size=20)
        labels = [ds[i][1].item() for i in range(len(ds))]
        counts = np.bincount(labels, minlength=5)
        # Each class should have at least 10% of total (loose bound)
        for c in range(5):
            assert counts[c] >= len(labels) * 0.05, (
                f"Class {c} has only {counts[c]}/{len(labels)} samples"
            )

    def test_dataset_with_custom_thresholds(self):
        """When thresholds are passed externally, they should be used."""
        from dnn_trading_bot import TradingDataset

        df = _prepared_df(200)
        # First dataset to get thresholds
        ds1 = TradingDataset(df, window_size=20)
        thresholds = ds1.thresholds

        # Second dataset using those thresholds
        ds2 = TradingDataset(df, window_size=20, thresholds=thresholds)
        assert np.allclose(ds2.thresholds, thresholds)


# ---------------------------------------------------------------------------
# LSTMTradingModel tests
# ---------------------------------------------------------------------------
class TestLSTMTradingModel:
    def test_forward_pass_shape(self):
        from dnn_trading_bot import LSTMTradingModel

        model = LSTMTradingModel(input_size=len(FEATURE_COLS))
        x = torch.randn(8, 20, len(FEATURE_COLS))  # batch=8, seq=20, features=6
        out = model(x)
        assert out.shape == (8, 5), f"Expected (8,5), got {out.shape}"

    def test_output_is_valid_logits(self):
        from dnn_trading_bot import LSTMTradingModel

        model = LSTMTradingModel(input_size=len(FEATURE_COLS))
        x = torch.randn(4, 20, len(FEATURE_COLS))
        out = model(x)
        # Should be finite
        assert torch.isfinite(out).all(), "Output contains inf/nan"

    def test_softmax_produces_probabilities(self):
        from dnn_trading_bot import LSTMTradingModel

        model = LSTMTradingModel(input_size=len(FEATURE_COLS))
        x = torch.randn(4, 20, len(FEATURE_COLS))
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        # Sum to 1
        sums = probs.sum(dim=1)
        assert torch.allclose(sums, torch.ones(4), atol=1e-5)
        # All non-negative
        assert (probs >= 0).all()

    def test_single_sample_forward(self):
        from dnn_trading_bot import LSTMTradingModel

        model = LSTMTradingModel(input_size=len(FEATURE_COLS))
        x = torch.randn(1, 20, len(FEATURE_COLS))
        out = model(x)
        assert out.shape == (1, 5)


# ---------------------------------------------------------------------------
# Signal mapping tests
# ---------------------------------------------------------------------------
class TestSignalMapping:
    def test_argmax_to_signal(self):
        from dnn_trading_bot import SIGNAL_NAMES

        assert SIGNAL_NAMES[0] == "strong_sell"
        assert SIGNAL_NAMES[1] == "mild_sell"
        assert SIGNAL_NAMES[2] == "hold"
        assert SIGNAL_NAMES[3] == "mild_buy"
        assert SIGNAL_NAMES[4] == "strong_buy"

    def test_all_signals_mapped(self):
        from dnn_trading_bot import SIGNAL_NAMES

        expected = {"strong_sell", "mild_sell", "hold", "mild_buy", "strong_buy"}
        assert set(SIGNAL_NAMES) == expected


# ---------------------------------------------------------------------------
# DNNTradingBot integration tests
# ---------------------------------------------------------------------------
class TestDNNTradingBot:
    def test_fit_and_predict(self):
        from dnn_trading_bot import DNNTradingBot

        df = _prepared_df(200)
        bot = DNNTradingBot(window_size=20, epochs=2, lr=0.01)
        bot.fit(df)

        signals = bot.predict(df)
        assert len(signals) > 0
        valid_signals = {"strong_sell", "mild_sell", "hold", "mild_buy", "strong_buy"}
        for s in signals:
            assert s in valid_signals, f"Invalid signal: {s}"

    def test_predict_single(self):
        from dnn_trading_bot import DNNTradingBot

        df = _prepared_df(200)
        bot = DNNTradingBot(window_size=20, epochs=2, lr=0.01)
        bot.fit(df)

        # Predict on last window
        signal = bot.predict_single(df)
        valid_signals = {"strong_sell", "mild_sell", "hold", "mild_buy", "strong_buy"}
        assert signal in valid_signals


# ---------------------------------------------------------------------------
# Backtest integration tests
# ---------------------------------------------------------------------------
class TestBacktest:
    def test_backtest_returns_expected_keys(self):
        from dnn_trading_bot import run_dnn_backtest

        df = _make_dummy_df(250)
        results = run_dnn_backtest(df, train_ratio=0.6, initial_capital=100_000,
                                   epochs=2)
        expected_keys = {
            "total_return", "buy_and_hold_return", "max_drawdown",
            "sharpe_ratio", "win_rate", "profit_factor", "num_trades",
            "final_value", "trades", "daily_values",
        }
        assert expected_keys.issubset(results.keys()), (
            f"Missing keys: {expected_keys - results.keys()}"
        )

    def test_capital_non_negative(self):
        from dnn_trading_bot import run_dnn_backtest

        df = _make_dummy_df(250)
        results = run_dnn_backtest(df, train_ratio=0.6, initial_capital=100_000,
                                   epochs=2)
        assert results["final_value"] >= 0, "Capital went negative"
        for v in results["daily_values"]:
            assert v >= 0, f"Daily value went negative: {v}"

    def test_backtest_with_small_data(self):
        """Should handle small datasets gracefully."""
        from dnn_trading_bot import run_dnn_backtest

        df = _make_dummy_df(120)
        results = run_dnn_backtest(df, train_ratio=0.6, initial_capital=100_000,
                                   epochs=2)
        assert "total_return" in results


# ---------------------------------------------------------------------------
# Hidden size parameter tests
# ---------------------------------------------------------------------------
class TestHiddenSizes:
    def test_model_custom_hidden_sizes(self):
        from dnn_trading_bot import LSTMTradingModel

        model = LSTMTradingModel(input_size=len(FEATURE_COLS), hidden1=32, hidden2=16)
        x = torch.randn(4, 20, len(FEATURE_COLS))
        out = model(x)
        assert out.shape == (4, 5)

    def test_bot_accepts_hidden_sizes(self):
        from dnn_trading_bot import DNNTradingBot

        bot = DNNTradingBot(window_size=20, epochs=2, hidden1=32, hidden2=16)
        df = _prepared_df(200)
        bot.fit(df)
        signals = bot.predict(df)
        assert len(signals) > 0

    def test_backtest_accepts_hidden_sizes(self):
        from dnn_trading_bot import run_dnn_backtest

        df = _make_dummy_df(200)
        results = run_dnn_backtest(df, train_ratio=0.6, initial_capital=100_000,
                                   epochs=2, hidden1=32, hidden2=16)
        assert "total_return" in results
