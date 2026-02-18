#!/usr/bin/env python3
"""Tests for the TD3 meta-judge trading bot."""

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


def _make_augmented_df(n=200, seed=42):
    """DataFrame with indicators + 4 model signal columns."""
    df = _prepared_df(n, seed)
    rng = np.random.RandomState(seed)
    choices = ["strong_sell", "mild_sell", "hold", "mild_buy", "strong_buy"]
    for col in ["km_signal", "lstm_signal", "lgbm_signal", "ppo_signal"]:
        df[col] = rng.choice(choices, len(df))
    return df


# ---------------------------------------------------------------------------
# action_to_signal tests
# ---------------------------------------------------------------------------
class TestActionToSignal:
    def test_strong_buy(self):
        from td3_trading_bot import action_to_signal
        assert action_to_signal(0.9) == "strong_buy"
        assert action_to_signal(0.51) == "strong_buy"

    def test_mild_buy(self):
        from td3_trading_bot import action_to_signal
        assert action_to_signal(0.4) == "mild_buy"
        assert action_to_signal(0.21) == "mild_buy"

    def test_hold_range(self):
        from td3_trading_bot import action_to_signal
        assert action_to_signal(0.0) == "hold"
        assert action_to_signal(0.19) == "hold"
        assert action_to_signal(-0.19) == "hold"

    def test_mild_sell(self):
        from td3_trading_bot import action_to_signal
        assert action_to_signal(-0.4) == "mild_sell"
        assert action_to_signal(-0.21) == "mild_sell"

    def test_strong_sell(self):
        from td3_trading_bot import action_to_signal
        assert action_to_signal(-0.9) == "strong_sell"
        assert action_to_signal(-0.51) == "strong_sell"

    def test_boundary_exactly_05(self):
        from td3_trading_bot import action_to_signal
        # val > 0.5 → strong_buy; val == 0.5 falls to mild_buy
        assert action_to_signal(0.5) == "mild_buy"
        # val > -0.5 → mild_sell; val == -0.5 falls through to strong_sell
        assert action_to_signal(-0.5) == "strong_sell"


# ---------------------------------------------------------------------------
# SIGNAL_TO_FLOAT tests
# ---------------------------------------------------------------------------
class TestSignalToFloat:
    def test_all_signals_present(self):
        from td3_trading_bot import SIGNAL_TO_FLOAT
        expected = {"strong_sell", "mild_sell", "hold", "mild_buy", "strong_buy"}
        assert set(SIGNAL_TO_FLOAT.keys()) == expected

    def test_values_in_range(self):
        from td3_trading_bot import SIGNAL_TO_FLOAT
        for v in SIGNAL_TO_FLOAT.values():
            assert -1.0 <= v <= 1.0

    def test_ordering(self):
        from td3_trading_bot import SIGNAL_TO_FLOAT
        assert (SIGNAL_TO_FLOAT["strong_sell"] < SIGNAL_TO_FLOAT["mild_sell"]
                < SIGNAL_TO_FLOAT["hold"]
                < SIGNAL_TO_FLOAT["mild_buy"]
                < SIGNAL_TO_FLOAT["strong_buy"])


# ---------------------------------------------------------------------------
# TD3MetaEnv tests
# ---------------------------------------------------------------------------
class TestTD3MetaEnv:
    def test_observation_space_shape(self):
        from td3_trading_bot import TD3MetaEnv, OBS_DIM
        import gymnasium as gym

        df = _make_augmented_df(200)
        env = TD3MetaEnv(df)
        assert isinstance(env.observation_space, gym.spaces.Box)
        assert env.observation_space.shape == (OBS_DIM,)

    def test_action_space_is_continuous_box(self):
        from td3_trading_bot import TD3MetaEnv
        import gymnasium as gym

        df = _make_augmented_df(200)
        env = TD3MetaEnv(df)
        assert isinstance(env.action_space, gym.spaces.Box)
        assert env.action_space.shape == (1,)
        assert env.action_space.low[0] == -1.0
        assert env.action_space.high[0] == 1.0

    def test_reset_returns_correct_obs_shape(self):
        from td3_trading_bot import TD3MetaEnv, OBS_DIM

        df = _make_augmented_df(200)
        env = TD3MetaEnv(df)
        obs, info = env.reset()
        assert obs.shape == (OBS_DIM,)
        assert obs.dtype == np.float32
        assert isinstance(info, dict)

    def test_step_returns_valid_tuple(self):
        from td3_trading_bot import TD3MetaEnv, OBS_DIM

        df = _make_augmented_df(200)
        env = TD3MetaEnv(df)
        env.reset()
        action = np.array([0.0], dtype=np.float32)  # hold
        obs, reward, terminated, truncated, info = env.step(action)
        assert obs.shape == (OBS_DIM,)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)

    def test_episode_terminates(self):
        from td3_trading_bot import TD3MetaEnv

        df = _make_augmented_df(100)
        env = TD3MetaEnv(df)
        env.reset()
        done = False
        steps = 0
        while not done:
            action = env.action_space.sample()
            _, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            steps += 1
        assert steps > 0
        assert steps <= len(df)

    def test_obs_encodes_signal_columns(self):
        """First 4 elements of obs should reflect the 4 signal columns."""
        from td3_trading_bot import TD3MetaEnv, SIGNAL_TO_FLOAT

        df = _make_augmented_df(50)
        # Force a known signal on first row
        df.loc[0, "km_signal"] = "strong_buy"
        df.loc[0, "lstm_signal"] = "hold"
        df.loc[0, "lgbm_signal"] = "mild_sell"
        df.loc[0, "ppo_signal"] = "strong_sell"
        env = TD3MetaEnv(df)
        obs, _ = env.reset()
        assert obs[0] == pytest.approx(SIGNAL_TO_FLOAT["strong_buy"])
        assert obs[1] == pytest.approx(SIGNAL_TO_FLOAT["hold"])
        assert obs[2] == pytest.approx(SIGNAL_TO_FLOAT["mild_sell"])
        assert obs[3] == pytest.approx(SIGNAL_TO_FLOAT["strong_sell"])


# ---------------------------------------------------------------------------
# TD3TradingBot tests
# ---------------------------------------------------------------------------
class TestTD3TradingBot:
    def test_fit_and_predict(self):
        from td3_trading_bot import TD3TradingBot

        df = _make_augmented_df(200)
        bot = TD3TradingBot(total_timesteps=500)
        bot.fit(df)
        signals = bot.predict(df)
        assert len(signals) == len(df)
        valid = {"strong_sell", "mild_sell", "hold", "mild_buy", "strong_buy"}
        for s in signals:
            assert s in valid, f"Invalid signal: {s}"

    def test_predict_single(self):
        from td3_trading_bot import TD3TradingBot

        df = _make_augmented_df(200)
        bot = TD3TradingBot(total_timesteps=500)
        bot.fit(df)
        signal = bot.predict_single(df.iloc[-1])
        valid = {"strong_sell", "mild_sell", "hold", "mild_buy", "strong_buy"}
        assert signal in valid

    def test_custom_learning_rate(self):
        from td3_trading_bot import TD3TradingBot

        df = _make_augmented_df(200)
        bot = TD3TradingBot(total_timesteps=500, learning_rate=1e-4)
        bot.fit(df)
        signals = bot.predict(df)
        assert len(signals) == len(df)


# ---------------------------------------------------------------------------
# run_td3_backtest tests
# ---------------------------------------------------------------------------
class TestRunTD3Backtest:
    def test_returns_expected_keys(self):
        from td3_trading_bot import run_td3_backtest

        df = _make_ohlcv(300)
        results = run_td3_backtest(
            df, train_ratio=0.6, initial_capital=100_000,
            total_timesteps=500, base_timesteps=500, lstm_epochs=2,
        )
        expected_keys = {
            "total_return", "buy_and_hold_return", "max_drawdown",
            "sharpe_ratio", "win_rate", "profit_factor", "num_trades",
            "final_value", "trades", "daily_values", "bot", "test_df",
            "train_end_idx",
        }
        assert expected_keys.issubset(results.keys()), (
            f"Missing keys: {expected_keys - results.keys()}"
        )

    def test_capital_non_negative(self):
        from td3_trading_bot import run_td3_backtest

        df = _make_ohlcv(300)
        results = run_td3_backtest(
            df, train_ratio=0.6, initial_capital=100_000,
            total_timesteps=500, base_timesteps=500, lstm_epochs=2,
        )
        assert results["final_value"] >= 0
        for v in results["daily_values"]:
            assert v >= 0

    def test_train_test_split(self):
        from td3_trading_bot import run_td3_backtest

        df = _make_ohlcv(300)
        results = run_td3_backtest(
            df, train_ratio=0.6, initial_capital=100_000,
            total_timesteps=500, base_timesteps=500, lstm_epochs=2,
        )
        # test_df should be ~40% of the indicator-cleaned df
        n_test = len(results["test_df"])
        assert n_test > 0
        split = results["train_end_idx"]
        assert split > 0

    def test_daily_values_length(self):
        from td3_trading_bot import run_td3_backtest

        df = _make_ohlcv(300)
        results = run_td3_backtest(
            df, train_ratio=0.6, initial_capital=100_000,
            total_timesteps=500, base_timesteps=500, lstm_epochs=2,
        )
        # daily_values has one entry per test day (minus last, which is terminal)
        assert len(results["daily_values"]) == len(results["test_df"]) - 1
