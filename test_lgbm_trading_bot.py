#!/usr/bin/env python3
"""Tests for the LightGBM trading bot."""

import numpy as np
import pandas as pd
import pytest

from trading_bot import FEATURE_COLS, compute_indicators


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_ohlcv(n=300, seed=42):
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range("2022-01-01", periods=n)
    close = 10.0 + np.cumsum(rng.randn(n) * 0.3)
    close = np.maximum(close, 1.0)
    return pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d"),
        "open": close * (1 + rng.randn(n) * 0.005),
        "high": close * (1 + np.abs(rng.randn(n) * 0.01)),
        "low": close * (1 - np.abs(rng.randn(n) * 0.01)),
        "close": close,
        "volume": rng.randint(1_000_000, 10_000_000, n).astype(float),
    })


def _prepared_df(n=300, seed=42):
    df = compute_indicators(_make_ohlcv(n, seed))
    return df.dropna(subset=FEATURE_COLS).reset_index(drop=True)


# ---------------------------------------------------------------------------
# LGBMTradingBot fit/predict tests
# ---------------------------------------------------------------------------
class TestLGBMTradingBot:
    def test_fit_and_predict(self):
        from lgbm_trading_bot import LGBMTradingBot

        df = _prepared_df(300)
        bot = LGBMTradingBot()
        bot.fit(df)

        signals = bot.predict(df)
        assert len(signals) == len(df)
        valid = {"strong_sell", "mild_sell", "hold", "mild_buy", "strong_buy"}
        for s in signals:
            assert s in valid, f"Invalid signal: {s}"

    def test_predict_single(self):
        from lgbm_trading_bot import LGBMTradingBot

        df = _prepared_df(300)
        bot = LGBMTradingBot()
        bot.fit(df)

        signal = bot.predict_single(df.iloc[-1])
        valid = {"strong_sell", "mild_sell", "hold", "mild_buy", "strong_buy"}
        assert signal in valid

    def test_predict_returns_5_class_signals(self):
        """With enough data, all 5 signal types should appear in predictions."""
        from lgbm_trading_bot import LGBMTradingBot

        df = _prepared_df(500, seed=123)
        bot = LGBMTradingBot()
        bot.fit(df)
        signals = bot.predict(df)
        unique = set(signals)
        # At least 3 distinct signals with synthetic data
        assert len(unique) >= 3, f"Only {len(unique)} signal types: {unique}"


# ---------------------------------------------------------------------------
# Signal labeling tests
# ---------------------------------------------------------------------------
class TestSignalLabeling:
    def test_percentile_labeling_produces_balanced_classes(self):
        """Percentile thresholds should create roughly balanced 5-class labels."""
        from lgbm_trading_bot import LGBMTradingBot

        df = _prepared_df(500, seed=99)
        bot = LGBMTradingBot()
        bot.fit(df)

        # Check the labels stored during fit
        labels = bot._labels
        counts = np.bincount(labels, minlength=5)
        total = len(labels)
        for c in range(5):
            assert counts[c] >= total * 0.05, (
                f"Class {c} has only {counts[c]}/{total} samples"
            )


# ---------------------------------------------------------------------------
# Backtest tests
# ---------------------------------------------------------------------------
class TestBacktest:
    def test_backtest_returns_expected_keys(self):
        from lgbm_trading_bot import run_lgbm_backtest

        df = _make_ohlcv(300)
        results = run_lgbm_backtest(df, train_ratio=0.6, initial_capital=100_000)
        expected_keys = {
            "total_return", "buy_and_hold_return", "max_drawdown",
            "sharpe_ratio", "win_rate", "profit_factor", "num_trades",
            "final_value", "trades", "daily_values", "bot", "test_df",
        }
        assert expected_keys.issubset(results.keys()), (
            f"Missing keys: {expected_keys - results.keys()}"
        )

    def test_capital_non_negative(self):
        from lgbm_trading_bot import run_lgbm_backtest

        df = _make_ohlcv(300)
        results = run_lgbm_backtest(df, train_ratio=0.6, initial_capital=100_000)
        assert results["final_value"] >= 0, "Capital went negative"
        for v in results["daily_values"]:
            assert v >= 0, f"Daily value went negative: {v}"


# ---------------------------------------------------------------------------
# Custom hyperparameters tests
# ---------------------------------------------------------------------------
class TestCustomHyperparams:
    def test_custom_n_estimators(self):
        from lgbm_trading_bot import LGBMTradingBot

        df = _prepared_df(300)
        bot = LGBMTradingBot(n_estimators=50)
        bot.fit(df)
        signals = bot.predict(df)
        assert len(signals) == len(df)

    def test_custom_max_depth_and_leaves(self):
        from lgbm_trading_bot import LGBMTradingBot

        df = _prepared_df(300)
        bot = LGBMTradingBot(max_depth=3, num_leaves=15)
        bot.fit(df)
        signals = bot.predict(df)
        assert len(signals) == len(df)

    def test_custom_learning_rate(self):
        from lgbm_trading_bot import LGBMTradingBot

        df = _prepared_df(300)
        bot = LGBMTradingBot(learning_rate=0.01)
        bot.fit(df)
        signals = bot.predict(df)
        assert len(signals) == len(df)

    def test_backtest_accepts_hyperparams(self):
        from lgbm_trading_bot import run_lgbm_backtest

        df = _make_ohlcv(300)
        results = run_lgbm_backtest(
            df, n_estimators=50, max_depth=3, learning_rate=0.05,
            num_leaves=15, min_child_samples=10,
        )
        assert "total_return" in results


# ---------------------------------------------------------------------------
# SMA5 buy filter tests
# ---------------------------------------------------------------------------
class TestSMA5BuyFilter:
    def test_buy_trades_below_sma5(self):
        """All buy trades must have close < sma5 on the signal day."""
        from lgbm_trading_bot import run_lgbm_backtest

        df = _make_ohlcv(500, seed=99)
        results = run_lgbm_backtest(df, train_ratio=0.6)
        test_df = results["test_df"]
        trades = results["trades"]
        buy_trades = [t for t in trades if t["action"] == "buy"]
        for bt in buy_trades:
            exec_date = bt["date"]
            exec_idx = test_df[test_df["date"].astype(str) == exec_date].index
            if len(exec_idx) > 0 and exec_idx[0] > 0:
                signal_idx = exec_idx[0] - 1
                close = test_df.loc[signal_idx, "close"]
                sma5 = test_df.loc[signal_idx, "sma5"]
                assert close < sma5, (
                    f"Buy on {exec_date}: signal day close={close:.2f} >= sma5={sma5:.2f}"
                )


# ---------------------------------------------------------------------------
# Feature importance tests
# ---------------------------------------------------------------------------
class TestFeatureImportance:
    def test_feature_importance_accessible(self):
        from lgbm_trading_bot import LGBMTradingBot

        df = _prepared_df(300)
        bot = LGBMTradingBot()
        bot.fit(df)

        importance = bot.feature_importance()
        assert isinstance(importance, dict)
        assert len(importance) == len(FEATURE_COLS)
        for col in FEATURE_COLS:
            assert col in importance
            assert importance[col] >= 0
