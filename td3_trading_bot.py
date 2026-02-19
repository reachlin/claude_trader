#!/usr/bin/env python3
"""TD3 meta-judge trading bot for China A-shares.

Uses Twin Delayed DDPG (TD3) as a meta-model that learns when to trust
(or override) the 4 base model signals (K-Means, LSTM, LightGBM, PPO).

The observation space includes the 4 base model signals encoded as floats,
6 z-scored technical indicators, and 4 portfolio state features. The action
space is continuous Box(1,) in [-1, 1], discretized to the same 5-level
signal scheme used by all other bots.

Workflow:
  1. Train K-Means, LSTM, LightGBM, PPO on the training set (first 60%).
  2. Collect their predictions on both train and test sets (no lookahead).
  3. Augment each set with signal columns and train TD3 on the augmented
     training environment.
  4. Evaluate TD3 on the augmented test set with identical backtest rules.
"""

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from sklearn.preprocessing import StandardScaler
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise

from trading_bot import (
    FEATURE_COLS,
    Portfolio,
    TradingBot,
    compute_indicators,
    LOT_SIZE,
)
from dnn_trading_bot import DNNTradingBot
from lgbm_trading_bot import LGBMTradingBot
from ppo_trading_bot import PPOTradingBot

SIGNAL_NAMES = ["strong_sell", "mild_sell", "hold", "mild_buy", "strong_buy"]

# Encode each signal as a float in [-1, 1] for the observation vector
SIGNAL_TO_FLOAT = {
    "strong_sell": -1.0,
    "mild_sell":   -0.5,
    "hold":         0.0,
    "mild_buy":     0.5,
    "strong_buy":   1.0,
}

# Columns injected by the 4 base models
META_SIGNAL_COLS = ["km_signal", "lstm_signal", "lgbm_signal", "ppo_signal"]

# Observation: 4 signals + 6 indicators + 4 portfolio state = 14
OBS_DIM = len(META_SIGNAL_COLS) + len(FEATURE_COLS) + 4


# ---------------------------------------------------------------------------
# Action discretisation
# ---------------------------------------------------------------------------
def action_to_signal(val: float) -> str:
    """Map a continuous TD3 action in [-1, 1] to a 5-level signal name.

    Thresholds (symmetric):
        val > 0.5   → strong_buy
        val > 0.2   → mild_buy
        val > -0.2  → hold
        val > -0.5  → mild_sell
        else        → strong_sell
    """
    if val > 0.5:
        return "strong_buy"
    elif val > 0.2:
        return "mild_buy"
    elif val > -0.2:
        return "hold"
    elif val > -0.5:
        return "mild_sell"
    else:
        return "strong_sell"


# ---------------------------------------------------------------------------
# Gymnasium Environment
# ---------------------------------------------------------------------------
class TD3MetaEnv(gym.Env):
    """Simulated trading environment for the TD3 meta-model.

    Observation (OBS_DIM = 14,):
        [0:4]   4 base model signals encoded as floats via SIGNAL_TO_FLOAT
        [4:10]  6 z-scored technical indicators (FEATURE_COLS)
        [10:14] 4 portfolio state features:
                - cash_fraction:  cash / initial_capital
                - shares_fraction: shares * close / initial_capital
                - pnl_fraction:   (portfolio_value - initial_capital) / initial_capital
                - days_since_normalized: min(days_since_last_trade, 30) / 30

    Action: Box(1,) in [-1.0, 1.0] — continuous, discretised by action_to_signal.

    Reward: daily_return * 100 + 0.1 * rolling_30d_sharpe_bonus
    """

    metadata = {"render_modes": []}

    def __init__(self, df: pd.DataFrame, initial_capital: float = 100_000):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.initial_capital = initial_capital

        self.scaler = StandardScaler()
        self.scaler.fit(self.df[FEATURE_COLS])

        self.action_space = spaces.Box(
            low=np.array([-1.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32,
        )

        self._current_step = 0
        self._portfolio = None
        self._last_trade_step = 0
        self._daily_returns: list[float] = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._current_step = 0
        self._portfolio = Portfolio(capital=self.initial_capital)
        self._last_trade_step = 0
        self._daily_returns = []
        return self._get_obs(), {}

    def _get_obs(self) -> np.ndarray:
        row = self.df.iloc[self._current_step]

        # 4 base model signals → floats
        signal_feats = np.array(
            [SIGNAL_TO_FLOAT.get(str(row[c]), 0.0) for c in META_SIGNAL_COLS],
            dtype=np.float32,
        )

        # 6 z-scored technical indicators
        indicators = self.scaler.transform(
            [row[FEATURE_COLS].values]
        )[0].astype(np.float32)

        # 4 portfolio state features
        close = float(row["close"])
        value = self._portfolio.value(close)
        cash_frac = self._portfolio.cash / self.initial_capital
        shares_frac = self._portfolio.shares * close / self.initial_capital
        pnl_frac = (value - self.initial_capital) / self.initial_capital
        days_since = min(self._current_step - self._last_trade_step, 30) / 30.0
        portfolio_state = np.array(
            [cash_frac, shares_frac, pnl_frac, days_since], dtype=np.float32
        )

        return np.concatenate([signal_feats, indicators, portfolio_state])

    def step(self, action):
        action_val = float(action[0])
        signal = action_to_signal(action_val)

        close_before = float(self.df.loc[self._current_step, "close"])
        value_before = self._portfolio.value(close_before)

        if self._current_step + 1 < len(self.df):
            exec_price = float(self.df.loc[self._current_step + 1, "open"])
            trade_date = str(self.df.loc[self._current_step + 1, "date"])

            shares_traded = 0
            if signal == "strong_buy":
                shares_traded = self._portfolio.buy(
                    exec_price, fraction=1.0, trade_date=trade_date
                )
            elif signal == "mild_buy":
                shares_traded = self._portfolio.buy(
                    exec_price, fraction=0.5, trade_date=trade_date
                )
            elif signal == "strong_sell":
                shares_traded = self._portfolio.sell(
                    exec_price, fraction=1.0, trade_date=trade_date
                )
            elif signal == "mild_sell":
                shares_traded = self._portfolio.sell(
                    exec_price, fraction=0.5, trade_date=trade_date
                )

            if shares_traded > 0:
                self._last_trade_step = self._current_step

        self._current_step += 1

        terminated = self._current_step >= len(self.df) - 1
        truncated = False

        if not terminated:
            close_after = float(self.df.loc[self._current_step, "close"])
        else:
            close_after = float(self.df.iloc[-1]["close"])
        value_after = self._portfolio.value(close_after)

        daily_return = (
            (value_after - value_before) / value_before if value_before > 0 else 0.0
        )
        self._daily_returns.append(daily_return)

        sharpe_bonus = 0.0
        if len(self._daily_returns) >= 30:
            recent = np.array(self._daily_returns[-30:])
            std = np.std(recent)
            if std > 0:
                sharpe_bonus = 0.1 * (np.mean(recent) / std)

        reward = float(daily_return * 100 + sharpe_bonus)
        return self._get_obs(), reward, terminated, truncated, {}


# ---------------------------------------------------------------------------
# TD3 Bot
# ---------------------------------------------------------------------------
class TD3TradingBot:
    """TD3 meta-model that uses 4 base model signals + indicators to trade."""

    def __init__(
        self,
        total_timesteps: int = 50_000,
        learning_rate: float = 1e-3,
        buffer_size: int = 10_000,
        learning_starts: int = 100,
        batch_size: int = 256,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: int = 1,
        gradient_steps: int = 1,
        policy_delay: int = 2,
        target_policy_noise: float = 0.2,
        target_noise_clip: float = 0.5,
        action_noise_sigma: float = 0.1,
    ):
        self.total_timesteps = total_timesteps
        self.learning_rate = learning_rate
        self.buffer_size = buffer_size
        self.learning_starts = learning_starts
        self.batch_size = batch_size
        self.tau = tau
        self.gamma = gamma
        self.train_freq = train_freq
        self.gradient_steps = gradient_steps
        self.policy_delay = policy_delay
        self.target_policy_noise = target_policy_noise
        self.target_noise_clip = target_noise_clip
        self.action_noise_sigma = action_noise_sigma
        self.model = None
        self.scaler = None

    def fit(self, df: pd.DataFrame):
        """Train TD3 on an augmented DataFrame that includes 4 signal columns."""
        env = TD3MetaEnv(df)
        self.scaler = env.scaler

        n_actions = env.action_space.shape[0]
        action_noise = NormalActionNoise(
            mean=np.zeros(n_actions),
            sigma=self.action_noise_sigma * np.ones(n_actions),
        )

        self.model = TD3(
            "MlpPolicy",
            env,
            learning_rate=self.learning_rate,
            buffer_size=self.buffer_size,
            learning_starts=self.learning_starts,
            batch_size=self.batch_size,
            tau=self.tau,
            gamma=self.gamma,
            train_freq=self.train_freq,
            gradient_steps=self.gradient_steps,
            policy_delay=self.policy_delay,
            target_policy_noise=self.target_policy_noise,
            target_noise_clip=self.target_noise_clip,
            action_noise=action_noise,
            verbose=0,
        )
        self.model.learn(total_timesteps=self.total_timesteps)

    def predict(self, df: pd.DataFrame) -> list[str]:
        """Predict signal for each row of an augmented DataFrame (stateless)."""
        neutral_portfolio = np.array([1.0, 0.0, 0.0, 1.0], dtype=np.float32)

        indicators = self.scaler.transform(df[FEATURE_COLS].values)

        signals = []
        for i in range(len(df)):
            row = df.iloc[i]
            signal_feats = np.array(
                [SIGNAL_TO_FLOAT.get(str(row[c]), 0.0) for c in META_SIGNAL_COLS],
                dtype=np.float32,
            )
            obs = np.concatenate([
                signal_feats,
                indicators[i].astype(np.float32),
                neutral_portfolio,
            ])
            action, _ = self.model.predict(obs, deterministic=True)
            signals.append(action_to_signal(float(action[0])))
        return signals

    def predict_single(self, row: pd.Series) -> str:
        """Predict signal for a single row (must include 4 signal columns)."""
        signal_feats = np.array(
            [SIGNAL_TO_FLOAT.get(str(row[c]), 0.0) for c in META_SIGNAL_COLS],
            dtype=np.float32,
        )
        indicator_vals = self.scaler.transform(
            [row[FEATURE_COLS].values]
        )[0].astype(np.float32)
        neutral_portfolio = np.array([1.0, 0.0, 0.0, 1.0], dtype=np.float32)
        obs = np.concatenate([signal_feats, indicator_vals, neutral_portfolio])
        action, _ = self.model.predict(obs, deterministic=True)
        return action_to_signal(float(action[0]))


# ---------------------------------------------------------------------------
# Signal generation helpers
# ---------------------------------------------------------------------------
def _get_base_signals(
    df: pd.DataFrame,
    km_bot: TradingBot,
    lstm_bot: DNNTradingBot,
    lgbm_bot: LGBMTradingBot,
    ppo_bot: PPOTradingBot,
) -> pd.DataFrame:
    """Return df with 4 signal columns appended (no lookahead — bots already trained).

    LSTM predict() returns len(df)-window_size signals aligned to rows
    [window_size .. len(df)-1]; rows before that receive 'hold'.
    """
    aug = df.copy().reset_index(drop=True)

    aug["km_signal"] = km_bot.predict(aug)

    lstm_raw = lstm_bot.predict(aug)
    lstm_signals = ["hold"] * lstm_bot.window_size + list(lstm_raw)
    aug["lstm_signal"] = lstm_signals[: len(aug)]

    aug["lgbm_signal"] = lgbm_bot.predict(aug)
    aug["ppo_signal"] = ppo_bot.predict(aug)

    return aug


# ---------------------------------------------------------------------------
# Backtest
# ---------------------------------------------------------------------------
def run_td3_backtest(
    df: pd.DataFrame,
    train_ratio: float = 0.6,
    initial_capital: float = 100_000,
    # TD3 hyperparameters
    total_timesteps: int = 50_000,
    learning_rate: float = 1e-3,
    buffer_size: int = 10_000,
    learning_starts: int = 100,
    batch_size: int = 256,
    # Base model training budgets (kept small so backtest is fast by default)
    base_timesteps: int = 50_000,   # PPO / K-Means timesteps
    lstm_epochs: int = 30,
) -> dict:
    """Full walk-forward backtest for the TD3 meta-judge.

    Steps:
      1. Compute indicators; split 60 / 40.
      2. Train K-Means, LSTM, LightGBM, PPO on the 60% train set.
      3. Generate their signals on both train and test (no lookahead).
      4. Augment both DataFrames with the 4 signal columns.
      5. Train TD3 on the augmented train environment.
      6. Evaluate TD3 on the augmented test set with SMA5 buy filter.
    """
    df = compute_indicators(df)
    df = df.dropna(subset=FEATURE_COLS).reset_index(drop=True)

    split = int(len(df) * train_ratio)
    train_df = df.iloc[:split].copy().reset_index(drop=True)
    test_df = df.iloc[split:].copy().reset_index(drop=True)

    # ---- 1. Train 4 base models on train_df --------------------------------
    print("  [TD3] Training K-Means base model...")
    km_bot = TradingBot()
    km_bot.fit(train_df)

    print("  [TD3] Training LSTM base model...")
    lstm_bot = DNNTradingBot(epochs=lstm_epochs)
    lstm_bot.fit(train_df)

    print("  [TD3] Training LightGBM base model...")
    lgbm_bot = LGBMTradingBot()
    lgbm_bot.fit(train_df)

    print("  [TD3] Training PPO base model...")
    ppo_bot = PPOTradingBot(total_timesteps=base_timesteps)
    ppo_bot.fit(train_df)

    # ---- 2. Augment train and test DataFrames with base model signals -------
    print("  [TD3] Generating base model signals on train set...")
    train_aug = _get_base_signals(train_df, km_bot, lstm_bot, lgbm_bot, ppo_bot)

    print("  [TD3] Generating base model signals on test set...")
    test_aug = _get_base_signals(test_df, km_bot, lstm_bot, lgbm_bot, ppo_bot)

    # ---- 3. Train TD3 meta-model on augmented train data -------------------
    print(f"  [TD3] Training TD3 meta-model ({total_timesteps:,} timesteps)...")
    td3_bot = TD3TradingBot(
        total_timesteps=total_timesteps,
        learning_rate=learning_rate,
        buffer_size=buffer_size,
        learning_starts=learning_starts,
        batch_size=batch_size,
    )
    td3_bot.fit(train_aug)

    # ---- 4. Generate TD3 signals on test set --------------------------------
    td3_signals = td3_bot.predict(test_aug)
    test_aug["signal"] = td3_signals

    # ---- 5. Simulate trading (same rules as other bots) ----------------------
    portfolio = Portfolio(capital=initial_capital)
    trades = []
    daily_values = []

    for i in range(len(test_aug) - 1):
        signal = test_aug.loc[i, "signal"]
        exec_price = float(test_aug.loc[i + 1, "open"])
        trade_date = str(test_aug.loc[i + 1, "date"])
        price_below_sma5 = (
            float(test_aug.loc[i, "close"]) < float(test_aug.loc[i, "sma5"])
        )

        shares_traded = 0
        action = "hold"

        if signal == "strong_buy" and price_below_sma5:
            shares_traded = portfolio.buy(exec_price, fraction=1.0,
                                          trade_date=trade_date)
            if shares_traded > 0:
                action = "buy"
        elif signal == "mild_buy" and price_below_sma5:
            shares_traded = portfolio.buy(exec_price, fraction=0.5,
                                          trade_date=trade_date)
            if shares_traded > 0:
                action = "buy"
        elif signal == "strong_sell":
            shares_traded = portfolio.sell(exec_price, fraction=1.0,
                                           trade_date=trade_date)
            if shares_traded > 0:
                action = "sell"
        elif signal == "mild_sell":
            shares_traded = portfolio.sell(exec_price, fraction=0.5,
                                           trade_date=trade_date)
            if shares_traded > 0:
                action = "sell"

        if shares_traded > 0:
            trades.append({
                "date": trade_date,
                "action": action,
                "price": exec_price,
                "shares": shares_traded,
                "signal": signal,
            })

        close_price = float(test_aug.loc[i + 1, "close"])
        daily_values.append(portfolio.value(close_price))

    # ---- 6. Metrics ----------------------------------------------------------
    final_price = float(test_aug.iloc[-1]["close"])
    final_value = portfolio.value(final_price)
    total_return = (final_value - initial_capital) / initial_capital * 100

    values = np.array([initial_capital] + daily_values)
    peak = np.maximum.accumulate(values)
    drawdowns = (values - peak) / peak
    max_drawdown = drawdowns.min() * 100

    daily_returns = np.diff(values) / values[:-1]
    sharpe = (
        np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252)
        if np.std(daily_returns) > 0
        else 0.0
    )

    buy_trades = [t for t in trades if t["action"] == "buy"]
    sell_trades = [t for t in trades if t["action"] == "sell"]
    trade_pnl = []
    for j in range(min(len(buy_trades), len(sell_trades))):
        pnl = (
            (sell_trades[j]["price"] - buy_trades[j]["price"])
            * buy_trades[j]["shares"]
        )
        trade_pnl.append(pnl)

    wins = [p for p in trade_pnl if p > 0]
    losses = [p for p in trade_pnl if p <= 0]
    win_rate = len(wins) / len(trade_pnl) * 100 if trade_pnl else 0.0
    profit_factor = (
        sum(wins) / abs(sum(losses))
        if losses and sum(losses) != 0
        else float("inf")
    )

    # Buy and hold
    bh_shares = int(initial_capital / test_aug.iloc[0]["open"] // LOT_SIZE) * LOT_SIZE
    bh_cost = bh_shares * float(test_aug.iloc[0]["open"])
    bh_value = bh_shares * final_price + (initial_capital - bh_cost)
    bh_return = (bh_value - initial_capital) / initial_capital * 100

    return {
        "total_return": total_return,
        "buy_and_hold_return": bh_return,
        "max_drawdown": max_drawdown,
        "sharpe_ratio": sharpe,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "num_trades": len(trades),
        "final_value": final_value,
        "trades": trades,
        "daily_values": daily_values,
        "bot": td3_bot,
        "test_df": test_aug,
        "train_end_idx": split,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    import argparse

    parser = argparse.ArgumentParser(description="TD3 meta-judge trading bot backtest")
    parser.add_argument("--csv", default="data/601933_10yr.csv", help="CSV file path")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    print(f"Loaded {len(df)} rows from {args.csv}")
    print(f"Date range: {df['date'].iloc[0]} to {df['date'].iloc[-1]}\n")

    print("=" * 60)
    print("TD3 META-JUDGE BACKTEST")
    print("=" * 60)
    results = run_td3_backtest(df, train_ratio=0.6, initial_capital=100_000)

    print(f"\n{'=' * 60}")
    print("TD3 RESULTS")
    print("=" * 60)
    rows = [
        ("Total Return",  f"{results['total_return']:+.2f}%",
         f"{results['buy_and_hold_return']:+.2f}%"),
        ("Max Drawdown",  f"{results['max_drawdown']:.2f}%", "N/A"),
        ("Sharpe Ratio",  f"{results['sharpe_ratio']:.3f}", "N/A"),
        ("Win Rate",      f"{results['win_rate']:.1f}%", "N/A"),
        ("Profit Factor", f"{results['profit_factor']:.2f}", "N/A"),
        ("Num Trades",    f"{results['num_trades']}", "1"),
        ("Final Value",   f"{results['final_value']:,.0f}", "N/A"),
    ]
    header = f"  {'Metric':<22s} {'TD3':>12s} {'Buy&Hold':>12s}"
    print(header)
    print("  " + "-" * 46)
    for label, td3_v, bh_v in rows:
        print(f"  {label:<22s} {td3_v:>12s} {bh_v:>12s}")

    # Recent trades
    if results["trades"]:
        print(f"\n{'=' * 60}")
        print("RECENT TRADES (last 10)")
        print("=" * 60)
        for t in results["trades"][-10:]:
            print(f"  {t['date']}  {t['action']:4s}  {t['shares']:6d} shares @ "
                  f"{t['price']:.2f}  ({t['signal']})")

    # Latest signal
    print(f"\n{'=' * 60}")
    print("TODAY'S SIGNAL")
    print("=" * 60)
    bot = results["bot"]
    last_row = results["test_df"].iloc[-1]
    signal = bot.predict_single(last_row)
    print(f"  Date:   {last_row['date']}")
    print(f"  Close:  {last_row['close']:.2f}")
    print(f"  Signal: {signal}")
    if signal in ("strong_buy", "mild_buy"):
        frac = 1.0 if signal == "strong_buy" else 0.5
        print(f"  Advice: BUY at next open, deploy {frac*100:.0f}% of cash")
    elif signal in ("strong_sell", "mild_sell"):
        frac = 1.0 if signal == "strong_sell" else 0.5
        print(f"  Advice: SELL at next open, sell {frac*100:.0f}% of holdings")
    else:
        print("  Advice: HOLD — no action")


if __name__ == "__main__":
    main()
