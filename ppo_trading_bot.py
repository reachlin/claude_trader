#!/usr/bin/env python3
"""PPO reinforcement learning trading bot for China A-shares.

Uses Stable-Baselines3 PPO with a Gymnasium environment. The agent learns a
trading policy by interacting with a simulated market, complementing the
supervised K-Means, LSTM, and LightGBM models.

Action space: Discrete(5) -> SIGNAL_NAMES [strong_sell..strong_buy]
Observation:  Box(10,) = 6 z-scored indicators + 4 portfolio state features
"""

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from sklearn.preprocessing import StandardScaler
from stable_baselines3 import PPO

from trading_bot import (
    FEATURE_COLS,
    Portfolio,
    compute_indicators,
    LOT_SIZE,
)

SIGNAL_NAMES = ["strong_sell", "mild_sell", "hold", "mild_buy", "strong_buy"]


# ---------------------------------------------------------------------------
# Gymnasium Environment
# ---------------------------------------------------------------------------
class TradingEnv(gym.Env):
    """Simulated trading environment for PPO.

    Observation (10,):
        6 z-scored technical indicators + 4 portfolio state features:
        - cash_fraction:   cash / initial_capital
        - shares_fraction: shares * close / initial_capital
        - pnl_fraction:    (portfolio_value - initial_capital) / initial_capital
        - days_since_trade_normalized: min(days, 30) / 30

    Action: Discrete(5) -> maps to SIGNAL_NAMES

    Reward: daily_return * 100 + 0.1 * sharpe_30d
    """

    metadata = {"render_modes": []}

    def __init__(self, df: pd.DataFrame, initial_capital: float = 100_000):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.initial_capital = initial_capital

        # Fit scaler on indicators
        self.scaler = StandardScaler()
        self.scaler.fit(self.df[FEATURE_COLS])

        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32,
        )

        self._current_step = 0
        self._portfolio = None
        self._last_trade_step = 0
        self._daily_returns = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._current_step = 0
        self._portfolio = Portfolio(capital=self.initial_capital)
        self._last_trade_step = 0
        self._daily_returns = []
        return self._get_obs(), {}

    def _get_obs(self):
        row = self.df.iloc[self._current_step]
        indicators = self.scaler.transform([row[FEATURE_COLS].values])[0]

        close = row["close"]
        value = self._portfolio.value(close)
        cash_frac = self._portfolio.cash / self.initial_capital
        shares_frac = self._portfolio.shares * close / self.initial_capital
        pnl_frac = (value - self.initial_capital) / self.initial_capital
        days_since = min(self._current_step - self._last_trade_step, 30) / 30.0

        portfolio_state = np.array([cash_frac, shares_frac, pnl_frac, days_since],
                                   dtype=np.float32)
        obs = np.concatenate([indicators.astype(np.float32), portfolio_state])
        return obs

    def step(self, action):
        signal = SIGNAL_NAMES[action]
        close_before = self.df.loc[self._current_step, "close"]
        value_before = self._portfolio.value(close_before)

        # Execute trade at next day's open
        if self._current_step + 1 < len(self.df):
            exec_price = self.df.loc[self._current_step + 1, "open"]
            trade_date = str(self.df.loc[self._current_step + 1, "date"])

            shares_traded = 0
            if signal == "strong_buy":
                shares_traded = self._portfolio.buy(exec_price, fraction=1.0,
                                                     trade_date=trade_date)
            elif signal == "mild_buy":
                shares_traded = self._portfolio.buy(exec_price, fraction=0.5,
                                                     trade_date=trade_date)
            elif signal == "strong_sell":
                shares_traded = self._portfolio.sell(exec_price, fraction=1.0,
                                                      trade_date=trade_date)
            elif signal == "mild_sell":
                shares_traded = self._portfolio.sell(exec_price, fraction=0.5,
                                                      trade_date=trade_date)

            if shares_traded > 0:
                self._last_trade_step = self._current_step

        self._current_step += 1

        terminated = self._current_step >= len(self.df) - 1
        truncated = False

        if not terminated:
            close_after = self.df.loc[self._current_step, "close"]
            value_after = self._portfolio.value(close_after)
        else:
            close_after = self.df.iloc[-1]["close"]
            value_after = self._portfolio.value(close_after)

        # Reward: daily return * 100 + small Sharpe bonus
        daily_return = (value_after - value_before) / value_before if value_before > 0 else 0.0
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
# Bot
# ---------------------------------------------------------------------------
class PPOTradingBot:
    """PPO-based trading signal generator using Stable-Baselines3."""

    def __init__(
        self,
        total_timesteps: int = 100_000,
        learning_rate: float = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        clip_range: float = 0.2,
        ent_coef: float = 0.01,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ):
        self.total_timesteps = total_timesteps
        self.learning_rate = learning_rate
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.clip_range = clip_range
        self.ent_coef = ent_coef
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.model = None
        self.scaler = None

    def fit(self, df: pd.DataFrame):
        """Create TradingEnv and train SB3 PPO agent."""
        env = TradingEnv(df)
        self.scaler = env.scaler

        # Clamp n_steps to avoid SB3 assertion error on small datasets
        n_steps = min(self.n_steps, max(len(df) - 2, 1))
        batch_size = min(self.batch_size, n_steps)

        self.model = PPO(
            "MlpPolicy",
            env,
            learning_rate=self.learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=self.n_epochs,
            clip_range=self.clip_range,
            ent_coef=self.ent_coef,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            verbose=0,
        )
        self.model.learn(total_timesteps=self.total_timesteps)

    def predict(self, df: pd.DataFrame) -> list[str]:
        """Predict signal for each row. Returns one signal per row (like LightGBM).

        Uses neutral portfolio state [1.0, 0.0, 0.0, 1.0] for the 4 portfolio
        features, making predictions purely indicator-driven (stateless).
        """
        features = df[FEATURE_COLS]
        X = self.scaler.transform(features)

        signals = []
        neutral_state = np.array([1.0, 0.0, 0.0, 1.0], dtype=np.float32)
        for i in range(len(X)):
            obs = np.concatenate([X[i].astype(np.float32), neutral_state])
            action, _ = self.model.predict(obs, deterministic=True)
            signals.append(SIGNAL_NAMES[int(action)])
        return signals

    def predict_single(self, row: pd.Series) -> str:
        """Predict signal for a single row (like LightGBM)."""
        features = row[FEATURE_COLS].values.reshape(1, -1)
        X = self.scaler.transform(features)[0]
        neutral_state = np.array([1.0, 0.0, 0.0, 1.0], dtype=np.float32)
        obs = np.concatenate([X.astype(np.float32), neutral_state])
        action, _ = self.model.predict(obs, deterministic=True)
        return SIGNAL_NAMES[int(action)]


# ---------------------------------------------------------------------------
# Backtest
# ---------------------------------------------------------------------------
def run_ppo_backtest(
    df: pd.DataFrame,
    train_ratio: float = 0.6,
    initial_capital: float = 100_000,
    total_timesteps: int = 100_000,
    learning_rate: float = 3e-4,
    n_steps: int = 2048,
    batch_size: int = 64,
    n_epochs: int = 10,
    clip_range: float = 0.2,
    ent_coef: float = 0.01,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> dict:
    """Walk-forward backtest: train PPO on first portion, test on rest."""
    df = compute_indicators(df)
    df = df.dropna(subset=FEATURE_COLS).reset_index(drop=True)

    split = int(len(df) * train_ratio)
    train_df = df.iloc[:split].copy().reset_index(drop=True)
    test_df = df.iloc[split:].copy().reset_index(drop=True)

    # Train
    bot = PPOTradingBot(
        total_timesteps=total_timesteps,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        clip_range=clip_range,
        ent_coef=ent_coef,
        gamma=gamma,
        gae_lambda=gae_lambda,
    )
    bot.fit(train_df)

    # Generate signals on test data
    signals = bot.predict(test_df)
    test_df["signal"] = signals

    # Simulate trading (with SMA5 buy filter)
    portfolio = Portfolio(capital=initial_capital)
    trades = []
    daily_values = []

    for i in range(len(test_df) - 1):
        signal = test_df.loc[i, "signal"]
        exec_price = test_df.loc[i + 1, "open"]
        trade_date = str(test_df.loc[i + 1, "date"])
        price_below_sma5 = test_df.loc[i, "close"] < test_df.loc[i, "sma5"]

        shares_traded = 0
        action = "hold"

        if signal == "strong_buy" and price_below_sma5:
            shares_traded = portfolio.buy(exec_price, fraction=1.0, trade_date=trade_date)
            if shares_traded > 0:
                action = "buy"
        elif signal == "mild_buy" and price_below_sma5:
            shares_traded = portfolio.buy(exec_price, fraction=0.5, trade_date=trade_date)
            if shares_traded > 0:
                action = "buy"
        elif signal == "strong_sell":
            shares_traded = portfolio.sell(exec_price, fraction=1.0, trade_date=trade_date)
            if shares_traded > 0:
                action = "sell"
        elif signal == "mild_sell":
            shares_traded = portfolio.sell(exec_price, fraction=0.5, trade_date=trade_date)
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

        close_price = test_df.loc[i + 1, "close"]
        daily_values.append(portfolio.value(close_price))

    # Final valuation
    final_price = test_df.iloc[-1]["close"]
    final_value = portfolio.value(final_price)

    # Buy and hold
    bh_shares = int(initial_capital / test_df.iloc[0]["open"] // LOT_SIZE) * LOT_SIZE
    bh_cost = bh_shares * test_df.iloc[0]["open"]
    bh_value = bh_shares * final_price + (initial_capital - bh_cost)
    bh_return = (bh_value - initial_capital) / initial_capital * 100

    # Metrics
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

    # Win rate
    trade_pnl = []
    buy_trades = [t for t in trades if t["action"] == "buy"]
    sell_trades = [t for t in trades if t["action"] == "sell"]
    for i in range(min(len(buy_trades), len(sell_trades))):
        pnl = (sell_trades[i]["price"] - buy_trades[i]["price"]) * buy_trades[i]["shares"]
        trade_pnl.append(pnl)

    wins = [p for p in trade_pnl if p > 0]
    losses = [p for p in trade_pnl if p <= 0]
    win_rate = len(wins) / len(trade_pnl) * 100 if trade_pnl else 0
    profit_factor = (
        sum(wins) / abs(sum(losses)) if losses and sum(losses) != 0 else float("inf")
    )

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
        "bot": bot,
        "test_df": test_df,
        "train_end_idx": split,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    import argparse

    parser = argparse.ArgumentParser(description="PPO trading bot backtest")
    parser.add_argument("--csv", default="data/601933_10yr.csv", help="CSV file path")
    args = parser.parse_args()

    csv_path = args.csv
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path}")
    print(f"Date range: {df['date'].iloc[0]} to {df['date'].iloc[-1]}\n")

    # Run PPO backtest
    print("=" * 60)
    print("PPO TRADING BOT")
    print("=" * 60)
    ppo_results = run_ppo_backtest(
        df, train_ratio=0.6, initial_capital=100_000,
    )

    # Run K-Means backtest for comparison
    from trading_bot import run_backtest
    print(f"\n{'=' * 60}")
    print("K-MEANS BASELINE (for comparison)")
    print("=" * 60)
    kmeans_results = run_backtest(df, train_ratio=0.6, initial_capital=100_000)

    # Comparison table
    print(f"\n{'=' * 60}")
    print("COMPARISON TABLE")
    print("=" * 60)
    bh_return = ppo_results["buy_and_hold_return"]
    header = f"  {'Metric':<22s} {'PPO':>12s} {'K-Means':>12s} {'Buy&Hold':>12s}"
    print(header)
    print("  " + "-" * 58)

    rows = [
        ("Total Return",
         f"{ppo_results['total_return']:+.2f}%",
         f"{kmeans_results['total_return']:+.2f}%",
         f"{bh_return:+.2f}%"),
        ("Max Drawdown",
         f"{ppo_results['max_drawdown']:.2f}%",
         f"{kmeans_results['max_drawdown']:.2f}%",
         "N/A"),
        ("Sharpe Ratio",
         f"{ppo_results['sharpe_ratio']:.2f}",
         f"{kmeans_results['sharpe_ratio']:.2f}",
         "N/A"),
        ("Win Rate",
         f"{ppo_results['win_rate']:.1f}%",
         f"{kmeans_results['win_rate']:.1f}%",
         "N/A"),
        ("Profit Factor",
         f"{ppo_results['profit_factor']:.2f}",
         f"{kmeans_results['profit_factor']:.2f}",
         "N/A"),
        ("Num Trades",
         f"{ppo_results['num_trades']}",
         f"{kmeans_results['num_trades']}",
         "1"),
        ("Final Value",
         f"{ppo_results['final_value']:,.0f}",
         f"{kmeans_results['final_value']:,.0f}",
         "N/A"),
    ]
    for label, ppo_v, km_v, bh_v in rows:
        print(f"  {label:<22s} {ppo_v:>12s} {km_v:>12s} {bh_v:>12s}")

    # Recent trades
    trades = ppo_results["trades"]
    if trades:
        print(f"\n{'=' * 60}")
        print("PPO RECENT TRADES (last 10)")
        print("=" * 60)
        for t in trades[-10:]:
            print(f"  {t['date']}  {t['action']:4s}  {t['shares']:6d} shares @ "
                  f"{t['price']:.2f}  ({t['signal']})")

    # Latest signal
    print(f"\n{'=' * 60}")
    print("PPO TODAY'S SIGNAL")
    print("=" * 60)
    df_full = compute_indicators(df).dropna(subset=FEATURE_COLS).reset_index(drop=True)
    latest = df_full.iloc[-1]
    bot = ppo_results["bot"]
    latest_signal = bot.predict_single(latest)
    print(f"  Date:   {latest['date']}")
    print(f"  Close:  {latest['close']:.2f}")
    print(f"  Signal: {latest_signal}")

    if latest_signal in ("strong_buy", "mild_buy"):
        fraction = 1.0 if latest_signal == "strong_buy" else 0.5
        print(f"  Advice: BUY at next open, deploy {fraction*100:.0f}% of cash")
    elif latest_signal in ("strong_sell", "mild_sell"):
        fraction = 1.0 if latest_signal == "strong_sell" else 0.5
        print(f"  Advice: SELL at next open, sell {fraction*100:.0f}% of holdings")
    else:
        print(f"  Advice: HOLD — no action")


if __name__ == "__main__":
    main()
