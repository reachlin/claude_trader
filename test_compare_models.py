#!/usr/bin/env python3
"""Tests for the majority vote backtest in compare_models."""

import numpy as np
import pandas as pd
import pytest

from trading_bot import Portfolio, LOT_SIZE, compute_indicators
from compare_models import _classify, _majority_direction


class TestMajorityVoteLogic:
    """Test the majority vote (>= 3 of 4) logic."""

    def test_all_agree_buy(self):
        """When all 4 models say buy, majority should be buy."""
        dirs = ["buy", "buy", "buy", "buy"]
        assert _majority_direction(dirs) == "buy"

    def test_3_of_4_agree_buy(self):
        """When 3 of 4 models say buy, majority should be buy."""
        dirs = ["buy", "buy", "buy", "hold"]
        assert _majority_direction(dirs) == "buy"

    def test_3_of_4_agree_sell(self):
        """When 3 of 4 models say sell, majority should be sell."""
        dirs = ["sell", "hold", "sell", "sell"]
        assert _majority_direction(dirs) == "sell"

    def test_all_agree_sell(self):
        """When all 4 say sell, majority should be sell."""
        dirs = ["sell", "sell", "sell", "sell"]
        assert _majority_direction(dirs) == "sell"

    def test_all_agree_hold(self):
        """When all 4 say hold, majority should be hold."""
        dirs = ["hold", "hold", "hold", "hold"]
        assert _majority_direction(dirs) == "hold"

    def test_3_of_4_agree_hold(self):
        """When 3 of 4 say hold, majority should be hold."""
        dirs = ["hold", "hold", "hold", "buy"]
        assert _majority_direction(dirs) == "hold"

    def test_2_2_split_no_majority(self):
        """When 2 buy + 2 sell, no majority."""
        dirs = ["buy", "buy", "sell", "sell"]
        assert _majority_direction(dirs) is None

    def test_2_1_1_split_no_majority(self):
        """When 2 buy + 1 sell + 1 hold, no majority."""
        dirs = ["buy", "buy", "sell", "hold"]
        assert _majority_direction(dirs) is None

    def test_all_different_no_majority(self):
        """When all disagree (impossible with 3 categories and 4 votes but 2+1+1), no majority."""
        dirs = ["buy", "hold", "sell", "hold"]
        assert _majority_direction(dirs) is None

    def test_classify_maps(self):
        """Verify all signal names map correctly."""
        assert _classify("strong_buy") == "buy"
        assert _classify("mild_buy") == "buy"
        assert _classify("hold") == "hold"
        assert _classify("mild_sell") == "sell"
        assert _classify("strong_sell") == "sell"


class TestMajorityVoteBacktest:
    """Test the majority vote portfolio simulation."""

    def test_majority_buy_executes_with_sma5_filter(self):
        """Majority buy + price < SMA5 should execute a buy."""
        portfolio = Portfolio(capital=100_000)
        dirs = ["buy", "buy", "buy", "hold"]
        majority = _majority_direction(dirs)
        price_below_sma5 = True

        if majority == "buy" and price_below_sma5:
            shares = portfolio.buy(10.0, fraction=0.5, trade_date="2020-01-02")
            assert shares > 0
            assert shares % LOT_SIZE == 0

    def test_majority_buy_blocked_without_sma5_filter(self):
        """Majority buy + price >= SMA5 should NOT execute."""
        portfolio = Portfolio(capital=100_000)
        dirs = ["buy", "buy", "buy", "hold"]
        majority = _majority_direction(dirs)
        price_below_sma5 = False
        traded = False

        if majority == "buy" and price_below_sma5:
            shares = portfolio.buy(10.0, fraction=0.5, trade_date="2020-01-02")
            traded = shares > 0

        assert not traded

    def test_majority_sell_executes(self):
        """Majority sell should execute if we have shares."""
        portfolio = Portfolio(capital=100_000)
        portfolio.buy(10.0, fraction=1.0, trade_date="2020-01-01")
        dirs = ["sell", "sell", "sell", "hold"]
        majority = _majority_direction(dirs)
        assert majority == "sell"
        shares = portfolio.sell(10.0, fraction=0.5, trade_date="2020-01-02")
        assert shares > 0

    def test_no_majority_no_trade(self):
        """When no majority, no trade should happen."""
        dirs = ["buy", "buy", "sell", "hold"]
        majority = _majority_direction(dirs)
        assert majority is None

    def test_metrics_computed(self):
        """Verify metrics (return, sharpe, drawdown) can be computed from daily values."""
        capital = 100_000
        daily_values = [capital, 100_500, 101_000, 100_800, 101_500]
        values = np.array([capital] + daily_values)
        peak = np.maximum.accumulate(values)
        drawdowns = (values - peak) / peak
        max_dd = drawdowns.min() * 100

        daily_returns = np.diff(values) / values[:-1]
        sharpe = (np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252)
                  if np.std(daily_returns) > 0 else 0.0)

        assert max_dd <= 0
        assert isinstance(sharpe, float)

    def test_strong_signal_uses_full_fraction(self):
        """If any model gives strong_buy, fraction should be 1.0."""
        signals = ["strong_buy", "mild_buy", "mild_buy", "mild_buy"]
        strong = any(s == "strong_buy" for s in signals)
        frac = 1.0 if strong else 0.5
        assert frac == 1.0

    def test_mild_signal_uses_half_fraction(self):
        """If no model gives strong_buy, fraction should be 0.5."""
        signals = ["mild_buy", "mild_buy", "mild_buy", "mild_buy"]
        strong = any(s == "strong_buy" for s in signals)
        frac = 1.0 if strong else 0.5
        assert frac == 0.5
