#!/usr/bin/env python3
"""LightGBM gradient-boosted tree trading bot for China A-shares.

Uses the same 6 technical indicators as the K-Means bot, but classifies
next-day return into 5 categories using percentile-based labeling (same as
LSTM). Each row is an independent sample — no sliding windows needed.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier

from trading_bot import (
    FEATURE_COLS,
    Portfolio,
    compute_indicators,
    LOT_SIZE,
)

SIGNAL_NAMES = ["strong_sell", "mild_sell", "hold", "mild_buy", "strong_buy"]


# ---------------------------------------------------------------------------
# Bot
# ---------------------------------------------------------------------------
class LGBMTradingBot:
    """LightGBM classifier on technical indicators to generate trade signals."""

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 5,
        learning_rate: float = 0.1,
        num_leaves: int = 31,
        min_child_samples: int = 20,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.min_child_samples = min_child_samples
        self.model = None
        self.scaler = StandardScaler()
        self.thresholds = None
        self._labels = None  # stored for test inspection

    def fit(self, df: pd.DataFrame):
        """Fit scaler + LGBMClassifier on indicator features with percentile labels."""
        features = df[FEATURE_COLS].values
        self.scaler.fit(features)
        X = self.scaler.transform(features)

        # Forward 1-day return
        closes = df["close"].values
        fwd_returns = np.empty(len(closes), dtype=np.float64)
        fwd_returns[:-1] = (closes[1:] - closes[:-1]) / closes[:-1]
        fwd_returns[-1] = 0.0

        # Percentile thresholds (exclude last row which has no real forward return)
        self.thresholds = np.percentile(fwd_returns[:-1], [20, 40, 60, 80])

        # Assign labels: 0=strong_sell, 1=mild_sell, 2=hold, 3=mild_buy, 4=strong_buy
        labels = np.digitize(fwd_returns, self.thresholds).astype(np.int64)
        self._labels = labels

        self.model = LGBMClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            num_leaves=self.num_leaves,
            min_child_samples=self.min_child_samples,
            random_state=42,
            verbose=-1,
        )
        self.model.fit(X, labels)

    def predict(self, df: pd.DataFrame) -> list[str]:
        """Predict signal for each row. Returns one signal per row."""
        features = df[FEATURE_COLS].values
        X = self.scaler.transform(features)
        preds = self.model.predict(X)
        return [SIGNAL_NAMES[p] for p in preds]

    def predict_single(self, row: pd.Series) -> str:
        """Predict signal for a single row."""
        features = row[FEATURE_COLS].values.reshape(1, -1)
        X = self.scaler.transform(features)
        pred = self.model.predict(X)[0]
        return SIGNAL_NAMES[pred]

    def feature_importance(self) -> dict[str, int]:
        """Return feature importance dict from the fitted model."""
        importances = self.model.feature_importances_
        return {col: int(imp) for col, imp in zip(FEATURE_COLS, importances)}


# ---------------------------------------------------------------------------
# Backtest
# ---------------------------------------------------------------------------
def run_lgbm_backtest(
    df: pd.DataFrame,
    train_ratio: float = 0.6,
    initial_capital: float = 100_000,
    n_estimators: int = 100,
    max_depth: int = 5,
    learning_rate: float = 0.1,
    num_leaves: int = 31,
    min_child_samples: int = 20,
) -> dict:
    """Walk-forward backtest: train LightGBM on first portion, test on rest."""
    df = compute_indicators(df)
    df = df.dropna(subset=FEATURE_COLS).reset_index(drop=True)

    split = int(len(df) * train_ratio)
    train_df = df.iloc[:split].copy().reset_index(drop=True)
    test_df = df.iloc[split:].copy().reset_index(drop=True)

    # Train
    bot = LGBMTradingBot(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        min_child_samples=min_child_samples,
    )
    bot.fit(train_df)

    # Generate signals on test data
    signals = bot.predict(test_df)
    test_df["signal"] = signals

    # Simulate trading
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

    parser = argparse.ArgumentParser(description="LightGBM trading bot backtest")
    parser.add_argument("--csv", default="601933_10yr.csv", help="CSV file path")
    args = parser.parse_args()

    csv_path = args.csv
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path}")
    print(f"Date range: {df['date'].iloc[0]} to {df['date'].iloc[-1]}\n")

    # Run LightGBM backtest
    print("=" * 60)
    print("LIGHTGBM TRADING BOT")
    print("=" * 60)
    lgbm_results = run_lgbm_backtest(
        df, train_ratio=0.6, initial_capital=100_000,
    )

    # Feature importance
    bot = lgbm_results["bot"]
    importance = bot.feature_importance()
    print("\nFeature Importance:")
    for col, imp in sorted(importance.items(), key=lambda x: x[1], reverse=True):
        print(f"  {col:<12s}  {imp}")

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
    bh_return = lgbm_results["buy_and_hold_return"]
    header = f"  {'Metric':<22s} {'LightGBM':>12s} {'K-Means':>12s} {'Buy&Hold':>12s}"
    print(header)
    print("  " + "-" * 58)

    rows = [
        ("Total Return",
         f"{lgbm_results['total_return']:+.2f}%",
         f"{kmeans_results['total_return']:+.2f}%",
         f"{bh_return:+.2f}%"),
        ("Max Drawdown",
         f"{lgbm_results['max_drawdown']:.2f}%",
         f"{kmeans_results['max_drawdown']:.2f}%",
         "N/A"),
        ("Sharpe Ratio",
         f"{lgbm_results['sharpe_ratio']:.2f}",
         f"{kmeans_results['sharpe_ratio']:.2f}",
         "N/A"),
        ("Win Rate",
         f"{lgbm_results['win_rate']:.1f}%",
         f"{kmeans_results['win_rate']:.1f}%",
         "N/A"),
        ("Profit Factor",
         f"{lgbm_results['profit_factor']:.2f}",
         f"{kmeans_results['profit_factor']:.2f}",
         "N/A"),
        ("Num Trades",
         f"{lgbm_results['num_trades']}",
         f"{kmeans_results['num_trades']}",
         "1"),
        ("Final Value",
         f"{lgbm_results['final_value']:,.0f}",
         f"{kmeans_results['final_value']:,.0f}",
         "N/A"),
    ]
    for label, lgbm_v, km_v, bh_v in rows:
        print(f"  {label:<22s} {lgbm_v:>12s} {km_v:>12s} {bh_v:>12s}")

    # Recent trades
    trades = lgbm_results["trades"]
    if trades:
        print(f"\n{'=' * 60}")
        print("LGBM RECENT TRADES (last 10)")
        print("=" * 60)
        for t in trades[-10:]:
            print(f"  {t['date']}  {t['action']:4s}  {t['shares']:6d} shares @ "
                  f"{t['price']:.2f}  ({t['signal']})")

    # Latest signal
    print(f"\n{'=' * 60}")
    print("LGBM TODAY'S SIGNAL")
    print("=" * 60)
    df_full = compute_indicators(df).dropna(subset=FEATURE_COLS).reset_index(drop=True)
    latest = df_full.iloc[-1]
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
