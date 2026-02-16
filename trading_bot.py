#!/usr/bin/env python3
"""Unsupervised K-Means clustering trading bot for China A-shares.

Learns market state patterns from technical indicators, then generates
daily trading signals: strong_buy, mild_buy, hold, mild_sell, strong_sell.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Technical Indicators
# ---------------------------------------------------------------------------
FEATURE_COLS = ["rsi", "macd_hist", "boll_pctb", "vol_ratio", "roc", "atr_ratio"]


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute 6 technical indicators on OHLCV data.

    Returns a copy of df with added indicator columns.  Rows where any
    indicator is NaN (warmup period) should be dropped by the caller.
    """
    df = df.copy()
    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"].astype(float)

    # 1. RSI(14)
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi"] = 100 - 100 / (1 + rs)

    # 2. MACD Histogram(12, 26, 9)
    ema12 = close.ewm(span=12, min_periods=12, adjust=False).mean()
    ema26 = close.ewm(span=26, min_periods=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, min_periods=9, adjust=False).mean()
    df["macd_hist"] = macd_line - signal_line

    # 3. Bollinger %B(20, 2)
    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    upper = sma20 + 2 * std20
    lower = sma20 - 2 * std20
    df["boll_pctb"] = (close - lower) / (upper - lower).replace(0, np.nan)

    # 4. Volume Ratio (volume / 20-day SMA of volume)
    vol_sma20 = volume.rolling(20).mean()
    df["vol_ratio"] = volume / vol_sma20.replace(0, np.nan)

    # 5. ROC(10) — Rate of Change
    df["roc"] = close.pct_change(10) * 100

    # 6. ATR Ratio (ATR14 / close) — normalized volatility
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    atr14 = tr.ewm(span=14, min_periods=14, adjust=False).mean()
    df["atr_ratio"] = atr14 / close

    # 7. SMA(5) — 5-day simple moving average (used as buy filter, not a model feature)
    df["sma5"] = close.rolling(5).mean()

    return df


# ---------------------------------------------------------------------------
# Portfolio
# ---------------------------------------------------------------------------
COMMISSION_RATE = 0.00025   # 0.025%
COMMISSION_MIN = 5.0        # minimum 5 RMB
STAMP_TAX_RATE = 0.0005     # 0.05% on sells only
LOT_SIZE = 100


class Portfolio:
    """Simulates a China A-share portfolio with T+1, lot sizes, and commissions."""

    def __init__(self, capital: float = 100_000):
        self.cash = capital
        self.shares = 0
        self.avg_cost = 0.0
        self._buy_date = None  # date when last batch was bought (for T+1)
        self._today_bought = 0  # shares bought today (cannot sell same day)

    def buy(self, price: float, fraction: float, trade_date: str) -> int:
        """Buy up to `fraction` of available cash worth of shares.

        Returns number of shares actually bought.
        """
        if fraction <= 0 or self.cash <= 0:
            return 0

        budget = self.cash * fraction
        # Solve: lots * LOT_SIZE * price + commission <= budget
        max_shares = int(budget / price // LOT_SIZE) * LOT_SIZE
        if max_shares <= 0:
            return 0

        cost = max_shares * price
        commission = max(cost * COMMISSION_RATE, COMMISSION_MIN)

        # Check if we can afford it with commission
        while max_shares > 0 and cost + commission > self.cash:
            max_shares -= LOT_SIZE
            cost = max_shares * price
            commission = max(cost * COMMISSION_RATE, COMMISSION_MIN)

        if max_shares <= 0:
            return 0

        total_cost = cost + commission
        self.cash -= total_cost
        old_total = self.shares * self.avg_cost
        self.shares += max_shares
        self.avg_cost = (old_total + cost) / self.shares if self.shares > 0 else 0

        # Track T+1
        if self._buy_date == trade_date:
            self._today_bought += max_shares
        else:
            self._buy_date = trade_date
            self._today_bought = max_shares

        return max_shares

    def sell(self, price: float, fraction: float, trade_date: str) -> int:
        """Sell up to `fraction` of sellable shares.

        Respects T+1: shares bought today cannot be sold.
        Returns number of shares actually sold.
        """
        if fraction <= 0 or self.shares <= 0:
            return 0

        # T+1: exclude shares bought today
        sellable = self.shares
        if self._buy_date == trade_date:
            sellable -= self._today_bought
        if sellable <= 0:
            return 0

        sell_shares = int(sellable * fraction // LOT_SIZE) * LOT_SIZE
        if sell_shares <= 0:
            return 0

        revenue = sell_shares * price
        commission = max(revenue * COMMISSION_RATE, COMMISSION_MIN)
        stamp_tax = revenue * STAMP_TAX_RATE
        net_revenue = revenue - commission - stamp_tax

        self.shares -= sell_shares
        self.cash += net_revenue
        if self.shares == 0:
            self.avg_cost = 0.0

        return sell_shares

    def value(self, price: float) -> float:
        """Total portfolio value at given price."""
        return self.cash + self.shares * price


# ---------------------------------------------------------------------------
# Trading Bot (K-Means)
# ---------------------------------------------------------------------------
class TradingBot:
    """K-Means clustering on technical indicators to generate trade signals."""

    def __init__(self, n_clusters: int = 5, feature_cols: list[str] | None = None):
        self.n_clusters = n_clusters
        self.feature_cols = feature_cols if feature_cols is not None else FEATURE_COLS
        self.kmeans = None
        self.scaler = StandardScaler()
        self.cluster_signals = {}  # cluster_id -> signal name

    @staticmethod
    def _assign_signal_names(n_clusters: int) -> list[str]:
        """Map n clusters to 5 signal levels by interpolation.

        Returns a list of length n_clusters, ordered from most bearish to
        most bullish (matching the sorted-by-return order).
        """
        canonical = ["strong_sell", "mild_sell", "hold", "mild_buy", "strong_buy"]
        if n_clusters == 5:
            return list(canonical)
        names = []
        for i in range(n_clusters):
            idx = int(round(i * 4 / (n_clusters - 1))) if n_clusters > 1 else 2
            names.append(canonical[idx])
        return names

    def fit(self, df: pd.DataFrame):
        """Fit K-Means on indicator features and label clusters by forward return."""
        features = df[self.feature_cols].values
        self.scaler.fit(features)
        X = self.scaler.transform(features)

        self.kmeans = KMeans(n_clusters=self.n_clusters, n_init=20, random_state=42)
        self.kmeans.fit(X)

        # Label clusters by average next-day return
        labels = self.kmeans.labels_
        fwd_return = df["close"].pct_change().shift(-1).values

        cluster_returns = {}
        for c in range(self.n_clusters):
            mask = labels == c
            returns_c = fwd_return[mask]
            returns_c = returns_c[~np.isnan(returns_c)]
            cluster_returns[c] = returns_c.mean() if len(returns_c) > 0 else 0.0

        # Rank clusters by average forward return
        sorted_clusters = sorted(cluster_returns.items(), key=lambda x: x[1])
        signal_names = self._assign_signal_names(self.n_clusters)
        self.cluster_signals = {
            c: signal_names[i] for i, (c, _) in enumerate(sorted_clusters)
        }

    def predict(self, df: pd.DataFrame) -> list[str]:
        """Predict signal for each row."""
        features = df[self.feature_cols].values
        X = self.scaler.transform(features)
        labels = self.kmeans.predict(X)
        return [self.cluster_signals[l] for l in labels]

    def predict_single(self, row: pd.Series) -> str:
        """Predict signal for a single row."""
        features = row[self.feature_cols].values.reshape(1, -1)
        X = self.scaler.transform(features)
        label = self.kmeans.predict(X)[0]
        return self.cluster_signals[label]


# ---------------------------------------------------------------------------
# Backtest engine
# ---------------------------------------------------------------------------
def run_backtest(
    df: pd.DataFrame,
    train_ratio: float = 0.6,
    initial_capital: float = 100_000,
    n_clusters: int = 5,
    feature_cols: list[str] | None = None,
) -> dict:
    """Walk-forward backtest: train on first `train_ratio`, test on rest.

    Execution price: next day's open.
    """
    df = compute_indicators(df)
    df = df.dropna(subset=FEATURE_COLS).reset_index(drop=True)

    split = int(len(df) * train_ratio)
    train_df = df.iloc[:split].copy()
    test_df = df.iloc[split:].copy().reset_index(drop=True)

    # Fit on training data
    bot = TradingBot(n_clusters=n_clusters, feature_cols=feature_cols)
    bot.fit(train_df)

    # Generate signals on test data
    signals = bot.predict(test_df)
    test_df["signal"] = signals

    # Simulate trading on test period
    portfolio = Portfolio(capital=initial_capital)
    trades = []
    daily_values = []

    for i in range(len(test_df) - 1):
        signal = test_df.loc[i, "signal"]
        exec_price = test_df.loc[i + 1, "open"]  # execute at next day's open
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

    # Buy and hold comparison
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

    # Daily returns for Sharpe
    daily_returns = np.diff(values) / values[:-1]
    sharpe = (
        np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252)
        if np.std(daily_returns) > 0 else 0.0
    )

    # Win rate and profit factor from trades
    trade_pnl = []
    buy_trades = [t for t in trades if t["action"] == "buy"]
    sell_trades = [t for t in trades if t["action"] == "sell"]

    # Simplified: pair consecutive buy/sell
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
        "train_end_idx": split,
        "trades": trades,
        "daily_values": daily_values,
        "bot": bot,
        "test_df": test_df,
    }


# ---------------------------------------------------------------------------
# Main: run backtest on 601933 and print results
# ---------------------------------------------------------------------------
def main():
    import argparse

    parser = argparse.ArgumentParser(description="K-Means trading bot backtest")
    parser.add_argument("--csv", default="601933_10yr.csv", help="CSV file path")
    args = parser.parse_args()

    csv_path = args.csv
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path}")
    print(f"Date range: {df['date'].iloc[0]} to {df['date'].iloc[-1]}\n")

    results = run_backtest(df, train_ratio=0.6, initial_capital=100_000)
    bot = results["bot"]
    test_df = results["test_df"]

    # Cluster analysis
    print("=" * 60)
    print("CLUSTER ANALYSIS")
    print("=" * 60)
    for cid, signal in sorted(bot.cluster_signals.items()):
        count = (bot.kmeans.labels_ == cid).sum()
        center = bot.scaler.inverse_transform(
            bot.kmeans.cluster_centers_[cid].reshape(1, -1)
        )[0]
        print(f"  Cluster {cid} -> {signal:12s}  (n={count:3d})  "
              f"RSI={center[0]:.1f}  MACD_H={center[1]:.4f}  "
              f"Boll%B={center[2]:.2f}  VolR={center[3]:.2f}  "
              f"ROC={center[4]:.1f}  ATRR={center[5]:.4f}")

    # Performance metrics
    print(f"\n{'=' * 60}")
    print("BACKTEST RESULTS")
    print("=" * 60)
    print(f"  Initial Capital:    RMB 100,000.00")
    print(f"  Final Value:        RMB {results['final_value']:,.2f}")
    print(f"  Total Return:       {results['total_return']:+.2f}%")
    print(f"  Buy & Hold Return:  {results['buy_and_hold_return']:+.2f}%")
    print(f"  Max Drawdown:       {results['max_drawdown']:.2f}%")
    print(f"  Sharpe Ratio:       {results['sharpe_ratio']:.2f}")
    print(f"  Number of Trades:   {results['num_trades']}")
    print(f"  Win Rate:           {results['win_rate']:.1f}%")
    print(f"  Profit Factor:      {results['profit_factor']:.2f}")

    # Recent trades
    trades = results["trades"]
    if trades:
        print(f"\n{'=' * 60}")
        print("RECENT TRADES (last 10)")
        print("=" * 60)
        for t in trades[-10:]:
            print(f"  {t['date']}  {t['action']:4s}  {t['shares']:6d} shares @ "
                  f"{t['price']:.2f}  ({t['signal']})")

    # Latest signal (today's advice)
    print(f"\n{'=' * 60}")
    print("TODAY'S SIGNAL")
    print("=" * 60)
    # Compute indicators on full dataset and get latest signal
    df_full = compute_indicators(df).dropna(subset=FEATURE_COLS)
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
