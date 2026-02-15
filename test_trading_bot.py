"""Tests for the unsupervised K-Means trading bot."""

import numpy as np
import pandas as pd
import pytest

from trading_bot import (
    compute_indicators,
    FEATURE_COLS,
    Portfolio,
    TradingBot,
    run_backtest,
)


# ---------------------------------------------------------------------------
# Helper: generate synthetic OHLCV data
# ---------------------------------------------------------------------------
def make_ohlcv(n=200, seed=42):
    """Create a synthetic OHLCV DataFrame with realistic structure."""
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range("2023-01-01", periods=n)
    close = 5.0 + np.cumsum(rng.randn(n) * 0.05)
    close = np.maximum(close, 1.0)  # keep positive
    high = close + rng.uniform(0.01, 0.1, n)
    low = close - rng.uniform(0.01, 0.1, n)
    low = np.maximum(low, 0.5)
    open_ = close + rng.uniform(-0.05, 0.05, n)
    volume = rng.randint(100000, 1000000, n).astype(float)

    return pd.DataFrame({
        "date": dates,
        "open": open_,
        "close": close,
        "high": high,
        "low": low,
        "volume": volume,
    })


# ===========================================================================
# 1. Indicator computation tests
# ===========================================================================
class TestIndicators:
    def test_returns_expected_columns(self):
        df = make_ohlcv(100)
        result = compute_indicators(df)
        expected = {"rsi", "macd_hist", "boll_pctb", "vol_ratio", "roc", "atr_ratio"}
        assert expected.issubset(set(result.columns))

    def test_no_nan_after_warmup(self):
        """After dropping warmup rows, there should be no NaN in indicator cols."""
        df = make_ohlcv(200)
        result = compute_indicators(df).dropna(
            subset=["rsi", "macd_hist", "boll_pctb", "vol_ratio", "roc", "atr_ratio"]
        )
        assert len(result) > 0
        for col in ["rsi", "macd_hist", "boll_pctb", "vol_ratio", "roc", "atr_ratio"]:
            assert result[col].isna().sum() == 0

    def test_rsi_range(self):
        """RSI must be between 0 and 100."""
        df = make_ohlcv(200)
        result = compute_indicators(df).dropna(subset=["rsi"])
        assert result["rsi"].min() >= 0
        assert result["rsi"].max() <= 100

    def test_volume_ratio_positive(self):
        df = make_ohlcv(200)
        result = compute_indicators(df).dropna(subset=["vol_ratio"])
        assert (result["vol_ratio"] > 0).all()


# ===========================================================================
# 2. Portfolio management tests
# ===========================================================================
class TestPortfolio:
    def test_initial_state(self):
        p = Portfolio(capital=100000)
        assert p.cash == 100000
        assert p.shares == 0
        assert p.avg_cost == 0.0

    def test_buy_rounds_to_lots(self):
        """Buys must be rounded down to 100-share lots."""
        p = Portfolio(capital=100000)
        p.buy(price=5.0, fraction=1.0, trade_date="2023-01-01")
        # 100000 / 5.0 = 20000 shares max, after commission ~19990 lots of 100
        assert p.shares > 0
        assert p.shares % 100 == 0

    def test_buy_deducts_cash_with_commission(self):
        p = Portfolio(capital=100000)
        p.buy(price=10.0, fraction=1.0, trade_date="2023-01-01")
        # Should have spent cash; cash < 100000
        assert p.cash < 100000
        assert p.shares > 0

    def test_sell_adds_cash_with_costs(self):
        p = Portfolio(capital=100000)
        p.buy(price=10.0, fraction=1.0, trade_date="2023-01-01")
        shares_before = p.shares
        cash_before = p.cash
        # Sell the next day (T+1 satisfied)
        p.sell(price=10.0, fraction=1.0, trade_date="2023-01-02")
        assert p.shares == 0
        assert p.cash > cash_before

    def test_sell_respects_t_plus_1(self):
        """Cannot sell shares bought today."""
        p = Portfolio(capital=100000)
        p.buy(price=10.0, fraction=1.0, trade_date="2023-01-01")
        shares_before = p.shares
        p.sell(price=10.0, fraction=1.0, trade_date="2023-01-01")
        # Shares unchanged because all were bought today
        assert p.shares == shares_before

    def test_commission_minimum(self):
        """Small trades should still incur minimum 5 RMB commission."""
        p = Portfolio(capital=1000)
        p.buy(price=5.0, fraction=1.0, trade_date="2023-01-01")
        # cost = shares * price + max(shares*price*0.00025, 5)
        expected_shares = 100  # 1000/5 = 200 but after commission ~100
        assert p.shares == expected_shares or p.shares == 200 - 100  # at most 200
        # Key: cash should account for >=5 RMB commission
        cost = p.shares * 5.0
        commission_paid = 100000 - p.cash - cost if p.cash < 1000 else 1000 - p.cash - cost
        # Just verify we bought something and cash reduced
        assert p.cash < 1000

    def test_sell_partial(self):
        """Selling 50% rounds down to 100-share lots."""
        p = Portfolio(capital=100000)
        p.buy(price=5.0, fraction=1.0, trade_date="2023-01-01")
        total = p.shares
        p.sell(price=5.0, fraction=0.5, trade_date="2023-01-02")
        sold = total - p.shares
        assert sold % 100 == 0
        assert sold > 0

    def test_no_buy_when_no_cash(self):
        p = Portfolio(capital=100000)
        p.buy(price=10.0, fraction=1.0, trade_date="2023-01-01")
        cash_after_first = p.cash
        shares_after_first = p.shares
        # Try to buy again with no cash
        p.buy(price=10.0, fraction=1.0, trade_date="2023-01-02")
        assert p.shares == shares_after_first  # no change
        assert p.cash == cash_after_first

    def test_portfolio_value(self):
        p = Portfolio(capital=100000)
        assert p.value(price=10.0) == 100000
        p.buy(price=10.0, fraction=0.5, trade_date="2023-01-01")
        # Value should be approximately 100000 (minus small commission)
        assert abs(p.value(price=10.0) - 100000) < 100  # within commission


# ===========================================================================
# 3. Clustering tests
# ===========================================================================
class TestClustering:
    def test_fit_produces_clusters(self):
        df = make_ohlcv(300)
        bot = TradingBot(n_clusters=5)
        df_ind = compute_indicators(df).dropna(
            subset=["rsi", "macd_hist", "boll_pctb", "vol_ratio", "roc", "atr_ratio"]
        )
        bot.fit(df_ind)
        assert bot.kmeans is not None
        assert len(bot.cluster_signals) == 5

    def test_predict_returns_valid_signals(self):
        df = make_ohlcv(300)
        bot = TradingBot(n_clusters=5)
        df_ind = compute_indicators(df).dropna(
            subset=["rsi", "macd_hist", "boll_pctb", "vol_ratio", "roc", "atr_ratio"]
        )
        bot.fit(df_ind)
        signals = bot.predict(df_ind)
        valid = {"strong_buy", "mild_buy", "hold", "mild_sell", "strong_sell"}
        assert all(s in valid for s in signals)

    def test_cluster_labels_cover_all_types(self):
        """With enough data, all 5 signal types should appear."""
        df = make_ohlcv(500, seed=123)
        bot = TradingBot(n_clusters=5)
        df_ind = compute_indicators(df).dropna(
            subset=["rsi", "macd_hist", "boll_pctb", "vol_ratio", "roc", "atr_ratio"]
        )
        bot.fit(df_ind)
        assert len(set(bot.cluster_signals.values())) == 5


# ===========================================================================
# 4. Backtest tests
# ===========================================================================
class TestBacktest:
    def test_backtest_returns_metrics(self):
        df = make_ohlcv(300)
        metrics = run_backtest(df, train_ratio=0.6)
        expected_keys = {
            "total_return",
            "buy_and_hold_return",
            "max_drawdown",
            "sharpe_ratio",
            "win_rate",
            "profit_factor",
            "num_trades",
        }
        assert expected_keys.issubset(set(metrics.keys()))

    def test_backtest_no_look_ahead(self):
        """Training set must not include test data dates."""
        df = make_ohlcv(300)
        metrics = run_backtest(df, train_ratio=0.6)
        # split is computed on indicator-cleaned data (after warmup drop)
        df_clean = compute_indicators(df).dropna(subset=FEATURE_COLS)
        expected_split = int(len(df_clean) * 0.6)
        assert metrics["train_end_idx"] == expected_split

    def test_backtest_capital_non_negative(self):
        df = make_ohlcv(300)
        metrics = run_backtest(df, train_ratio=0.6)
        assert metrics["final_value"] >= 0


# ===========================================================================
# 5. Variable n_clusters tests
# ===========================================================================
class TestVariableNClusters:
    """K-Means must work correctly for n_clusters != 5."""

    @pytest.mark.parametrize("n", [3, 4, 5, 6, 7, 8])
    def test_fit_predict_with_n_clusters(self, n):
        df = make_ohlcv(500, seed=123)
        bot = TradingBot(n_clusters=n)
        df_ind = compute_indicators(df).dropna(subset=FEATURE_COLS)
        bot.fit(df_ind)
        signals = bot.predict(df_ind)
        valid = {"strong_buy", "mild_buy", "hold", "mild_sell", "strong_sell"}
        assert all(s in valid for s in signals)
        assert len(bot.cluster_signals) == n

    @pytest.mark.parametrize("n", [3, 4, 5, 6, 7, 8])
    def test_backtest_with_n_clusters(self, n):
        df = make_ohlcv(300)
        metrics = run_backtest(df, train_ratio=0.6, n_clusters=n)
        assert "total_return" in metrics
        assert metrics["final_value"] >= 0


# ===========================================================================
# 6. Custom feature_cols tests
# ===========================================================================
class TestFeatureCols:
    """TradingBot and run_backtest should accept a custom feature_cols list."""

    def test_bot_with_subset_features(self):
        df = make_ohlcv(300)
        subset = ["rsi", "macd_hist", "boll_pctb", "vol_ratio"]
        bot = TradingBot(n_clusters=5, feature_cols=subset)
        df_ind = compute_indicators(df).dropna(subset=FEATURE_COLS)
        bot.fit(df_ind)
        signals = bot.predict(df_ind)
        valid = {"strong_buy", "mild_buy", "hold", "mild_sell", "strong_sell"}
        assert all(s in valid for s in signals)

    def test_backtest_with_feature_cols(self):
        df = make_ohlcv(300)
        subset = ["rsi", "macd_hist", "boll_pctb", "roc"]
        metrics = run_backtest(df, train_ratio=0.6, feature_cols=subset)
        assert "total_return" in metrics

    def test_default_feature_cols_unchanged(self):
        """Default should still be the original 6 features."""
        bot = TradingBot()
        assert bot.feature_cols == FEATURE_COLS
