#!/usr/bin/env python3
"""Tests for the PPO reinforcement learning trading bot."""

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
# TradingEnv tests
# ---------------------------------------------------------------------------
class TestTradingEnv:
    def test_reset_returns_obs_shape(self):
        from ppo_trading_bot import TradingEnv

        df = _prepared_df(200)
        env = TradingEnv(df)
        obs, info = env.reset()
        assert obs.shape == (10,), f"Expected (10,), got {obs.shape}"

    def test_step_returns_valid_tuple(self):
        from ppo_trading_bot import TradingEnv

        df = _prepared_df(200)
        env = TradingEnv(df)
        env.reset()
        obs, reward, terminated, truncated, info = env.step(2)  # hold
        assert obs.shape == (10,)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)

    def test_action_space_is_discrete_5(self):
        from ppo_trading_bot import TradingEnv
        import gymnasium as gym

        df = _prepared_df(200)
        env = TradingEnv(df)
        assert isinstance(env.action_space, gym.spaces.Discrete)
        assert env.action_space.n == 5

    def test_observation_space_is_box_10(self):
        from ppo_trading_bot import TradingEnv
        import gymnasium as gym

        df = _prepared_df(200)
        env = TradingEnv(df)
        assert isinstance(env.observation_space, gym.spaces.Box)
        assert env.observation_space.shape == (10,)

    def test_episode_terminates(self):
        from ppo_trading_bot import TradingEnv

        df = _prepared_df(100)
        env = TradingEnv(df)
        env.reset()
        done = False
        steps = 0
        while not done:
            obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
            done = terminated or truncated
            steps += 1
        assert steps > 0
        assert steps <= len(df)


# ---------------------------------------------------------------------------
# PPOTradingBot fit/predict tests
# ---------------------------------------------------------------------------
class TestPPOTradingBot:
    def test_fit_and_predict(self):
        from ppo_trading_bot import PPOTradingBot

        df = _prepared_df(300)
        bot = PPOTradingBot(total_timesteps=1000)
        bot.fit(df)

        signals = bot.predict(df)
        assert len(signals) == len(df)
        valid = {"strong_sell", "mild_sell", "hold", "mild_buy", "strong_buy"}
        for s in signals:
            assert s in valid, f"Invalid signal: {s}"

    def test_predict_single(self):
        from ppo_trading_bot import PPOTradingBot

        df = _prepared_df(300)
        bot = PPOTradingBot(total_timesteps=1000)
        bot.fit(df)

        signal = bot.predict_single(df.iloc[-1])
        valid = {"strong_sell", "mild_sell", "hold", "mild_buy", "strong_buy"}
        assert signal in valid

    def test_multiple_signal_types(self):
        """PPO should produce at least 2 distinct signal types."""
        from ppo_trading_bot import PPOTradingBot

        df = _prepared_df(500, seed=123)
        bot = PPOTradingBot(total_timesteps=2000)
        bot.fit(df)
        signals = bot.predict(df)
        unique = set(signals)
        assert len(unique) >= 2, f"Only {len(unique)} signal types: {unique}"


# ---------------------------------------------------------------------------
# Backtest tests
# ---------------------------------------------------------------------------
class TestBacktest:
    def test_backtest_returns_expected_keys(self):
        from ppo_trading_bot import run_ppo_backtest

        df = _make_ohlcv(300)
        results = run_ppo_backtest(df, train_ratio=0.6, initial_capital=100_000,
                                   total_timesteps=1000)
        expected_keys = {
            "total_return", "buy_and_hold_return", "max_drawdown",
            "sharpe_ratio", "win_rate", "profit_factor", "num_trades",
            "final_value", "trades", "daily_values", "bot", "test_df",
        }
        assert expected_keys.issubset(results.keys()), (
            f"Missing keys: {expected_keys - results.keys()}"
        )

    def test_capital_non_negative(self):
        from ppo_trading_bot import run_ppo_backtest

        df = _make_ohlcv(300)
        results = run_ppo_backtest(df, train_ratio=0.6, initial_capital=100_000,
                                   total_timesteps=1000)
        assert results["final_value"] >= 0, "Capital went negative"
        for v in results["daily_values"]:
            assert v >= 0, f"Daily value went negative: {v}"


# ---------------------------------------------------------------------------
# Custom hyperparameters tests
# ---------------------------------------------------------------------------
class TestCustomHyperparams:
    def test_custom_total_timesteps(self):
        from ppo_trading_bot import PPOTradingBot

        df = _prepared_df(300)
        bot = PPOTradingBot(total_timesteps=500)
        bot.fit(df)
        signals = bot.predict(df)
        assert len(signals) == len(df)

    def test_custom_learning_rate(self):
        from ppo_trading_bot import PPOTradingBot

        df = _prepared_df(300)
        bot = PPOTradingBot(total_timesteps=1000, learning_rate=1e-4)
        bot.fit(df)
        signals = bot.predict(df)
        assert len(signals) == len(df)

    def test_backtest_accepts_hyperparams(self):
        from ppo_trading_bot import run_ppo_backtest

        df = _make_ohlcv(300)
        results = run_ppo_backtest(
            df, total_timesteps=1000, learning_rate=1e-4,
        )
        assert "total_return" in results


# ---------------------------------------------------------------------------
# SMA5 buy filter tests
# ---------------------------------------------------------------------------
class TestSMA5BuyFilter:
    def test_buy_trades_below_sma5(self):
        """All buy trades must have close < sma5 on the signal day."""
        from ppo_trading_bot import run_ppo_backtest

        df = _make_ohlcv(500, seed=99)
        results = run_ppo_backtest(df, train_ratio=0.6, total_timesteps=2000)
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
