#!/usr/bin/env python3
"""LSTM-based DNN trading bot for China A-shares.

Uses sliding windows of technical indicators to classify next-day return
into 5 categories (strong_sell to strong_buy), then executes trades via
the same Portfolio engine used by the K-Means baseline.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from trading_bot import (
    FEATURE_COLS,
    Portfolio,
    compute_indicators,
    LOT_SIZE,
)

SIGNAL_NAMES = ["strong_sell", "mild_sell", "hold", "mild_buy", "strong_buy"]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class TradingDataset(Dataset):
    """Sliding-window dataset over technical indicators.

    Each sample is (window of indicators, label from forward 1-day return).
    Labels are 5-class based on percentile thresholds of forward returns.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        window_size: int = 20,
        thresholds: np.ndarray | None = None,
    ):
        self.window_size = window_size
        features = df[FEATURE_COLS].values.astype(np.float32)

        # Forward 1-day return
        closes = df["close"].values
        fwd_returns = np.empty(len(closes), dtype=np.float32)
        fwd_returns[:-1] = (closes[1:] - closes[:-1]) / closes[:-1]
        fwd_returns[-1] = 0.0  # last row has no forward return

        # Compute percentile thresholds on valid range (window onwards, exclude last)
        valid_returns = fwd_returns[window_size:-1]
        if thresholds is not None:
            self.thresholds = thresholds
        else:
            self.thresholds = np.percentile(valid_returns, [20, 40, 60, 80])

        # Assign labels
        labels = np.digitize(fwd_returns, self.thresholds).astype(np.int64)
        # digitize: <t[0]->0, t[0]<=x<t[1]->1, ..., >=t[3]->4

        # Build sliding windows: sample i uses rows [i : i+window], label at i+window
        self.X = []
        self.y = []
        for i in range(len(features) - window_size - 1):
            self.X.append(features[i : i + window_size])
            self.y.append(labels[i + window_size])

        self.X = np.array(self.X)
        self.y = np.array(self.y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]), torch.tensor(self.y[idx])


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class LSTMTradingModel(nn.Module):
    """LSTM(64) -> LSTM(32) -> FC(16) -> FC(5)."""

    def __init__(self, input_size: int = 6, hidden1: int = 64, hidden2: int = 32):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size, hidden1, batch_first=True)
        self.lstm2 = nn.LSTM(hidden1, hidden2, batch_first=True)
        self.fc1 = nn.Linear(hidden2, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 5)

    def forward(self, x):
        out, _ = self.lstm1(x)
        out, _ = self.lstm2(out)
        out = out[:, -1, :]  # last timestep
        out = self.relu(self.fc1(out))
        return self.fc2(out)


# ---------------------------------------------------------------------------
# Bot wrapper
# ---------------------------------------------------------------------------
class DNNTradingBot:
    """Wraps the LSTM model: fit, predict, predict_single."""

    def __init__(
        self,
        window_size: int = 20,
        epochs: int = 50,
        batch_size: int = 32,
        lr: float = 0.001,
        patience: int = 10,
        hidden1: int = 64,
        hidden2: int = 32,
    ):
        self.window_size = window_size
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.patience = patience
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.model = None
        self.scaler_mean = None
        self.scaler_std = None
        self.thresholds = None

    def fit(self, df: pd.DataFrame):
        """Train the LSTM on the given DataFrame (must have indicators computed)."""
        features = df[FEATURE_COLS].values.astype(np.float32)

        # Fit scaler on training data
        self.scaler_mean = features.mean(axis=0)
        self.scaler_std = features.std(axis=0)
        self.scaler_std[self.scaler_std == 0] = 1.0

        # Normalize
        df_scaled = df.copy()
        for i, col in enumerate(FEATURE_COLS):
            df_scaled[col] = (df_scaled[col] - self.scaler_mean[i]) / self.scaler_std[i]

        # Create dataset (thresholds computed from training data)
        full_ds = TradingDataset(df_scaled, window_size=self.window_size)
        self.thresholds = full_ds.thresholds

        # Split into train/val (90/10)
        n = len(full_ds)
        val_size = max(1, int(n * 0.1))
        train_size = n - val_size
        train_ds = torch.utils.data.Subset(full_ds, range(train_size))
        val_ds = torch.utils.data.Subset(full_ds, range(train_size, n))

        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False)

        # Class weights for imbalance
        labels = full_ds.y[:train_size]
        counts = np.bincount(labels, minlength=5).astype(np.float32)
        counts[counts == 0] = 1.0
        weights = (1.0 / counts) * len(labels) / 5.0
        class_weights = torch.tensor(weights)

        # Model
        self.model = LSTMTradingModel(
            input_size=len(FEATURE_COLS),
            hidden1=self.hidden1,
            hidden2=self.hidden2,
        )
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss(weight=class_weights)

        # Training loop with early stopping
        best_val_loss = float("inf")
        patience_counter = 0
        best_state = None

        for epoch in range(self.epochs):
            # Train
            self.model.train()
            train_loss = 0.0
            for xb, yb in train_loader:
                optimizer.zero_grad()
                logits = self.model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * xb.size(0)
            train_loss /= train_size

            # Validate
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for xb, yb in val_loader:
                    logits = self.model(xb)
                    loss = criterion(logits, yb)
                    val_loss += loss.item() * xb.size(0)
            val_loss /= val_size

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
            else:
                patience_counter += 1

            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"  Epoch {epoch+1:3d}/{self.epochs}  "
                      f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

            if patience_counter >= self.patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.model.eval()

    def _scale_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply stored scaler to a DataFrame."""
        df_scaled = df.copy()
        for i, col in enumerate(FEATURE_COLS):
            df_scaled[col] = (df_scaled[col] - self.scaler_mean[i]) / self.scaler_std[i]
        return df_scaled

    def predict(self, df: pd.DataFrame) -> list[str]:
        """Predict signals for all valid windows in df."""
        df_scaled = self._scale_df(df)
        features = df_scaled[FEATURE_COLS].values.astype(np.float32)

        signals = []
        self.model.eval()
        with torch.no_grad():
            for i in range(len(features) - self.window_size):
                window = torch.tensor(
                    features[i : i + self.window_size]
                ).unsqueeze(0)
                logits = self.model(window)
                pred = logits.argmax(dim=1).item()
                signals.append(SIGNAL_NAMES[pred])
        return signals

    def predict_single(self, df: pd.DataFrame) -> str:
        """Predict signal from the last window_size rows of df."""
        df_scaled = self._scale_df(df)
        features = df_scaled[FEATURE_COLS].values.astype(np.float32)
        window = torch.tensor(features[-self.window_size :]).unsqueeze(0)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(window)
            pred = logits.argmax(dim=1).item()
        return SIGNAL_NAMES[pred]


# ---------------------------------------------------------------------------
# Backtest
# ---------------------------------------------------------------------------
def run_dnn_backtest(
    df: pd.DataFrame,
    train_ratio: float = 0.6,
    initial_capital: float = 100_000,
    window_size: int = 20,
    epochs: int = 50,
    batch_size: int = 32,
    lr: float = 0.001,
    hidden1: int = 64,
    hidden2: int = 32,
) -> dict:
    """Walk-forward backtest: train LSTM on first portion, test on rest."""
    df = compute_indicators(df)
    df = df.dropna(subset=FEATURE_COLS).reset_index(drop=True)

    split = int(len(df) * train_ratio)
    train_df = df.iloc[:split].copy().reset_index(drop=True)
    test_df = df.iloc[split:].copy().reset_index(drop=True)

    # Train
    print(f"Training LSTM on {len(train_df)} rows...")
    bot = DNNTradingBot(
        window_size=window_size, epochs=epochs,
        batch_size=batch_size, lr=lr,
        hidden1=hidden1, hidden2=hidden2,
    )
    bot.fit(train_df)

    # Generate signals on test data
    signals = bot.predict(test_df)
    # signals[i] corresponds to the prediction made using window ending at
    # test_df row (i + window_size - 1), to be executed at the next day.
    # Align: signal index i maps to test_df row (i + window_size)
    signal_start = window_size
    test_signals = {}
    for i, sig in enumerate(signals):
        row_idx = i + signal_start
        if row_idx < len(test_df):
            test_signals[row_idx] = sig

    # Simulate trading
    portfolio = Portfolio(capital=initial_capital)
    trades = []
    daily_values = []

    for i in range(len(test_df) - 1):
        signal = test_signals.get(i, "hold")
        exec_price = test_df.loc[i + 1, "open"]
        trade_date = str(test_df.loc[i + 1, "date"])

        shares_traded = 0
        action = "hold"

        if signal == "strong_buy":
            shares_traded = portfolio.buy(exec_price, fraction=1.0, trade_date=trade_date)
            if shares_traded > 0:
                action = "buy"
        elif signal == "mild_buy":
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

    parser = argparse.ArgumentParser(description="LSTM DNN trading bot backtest")
    parser.add_argument("--csv", default="601933_10yr.csv", help="CSV file path")
    args = parser.parse_args()

    csv_path = args.csv
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path}")
    print(f"Date range: {df['date'].iloc[0]} to {df['date'].iloc[-1]}\n")

    # Run DNN backtest
    print("=" * 60)
    print("DNN (LSTM) TRADING BOT")
    print("=" * 60)
    dnn_results = run_dnn_backtest(
        df, train_ratio=0.6, initial_capital=100_000,
        window_size=20, epochs=50, batch_size=32, lr=0.001,
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
    header = f"  {'Metric':<22s} {'DNN (LSTM)':>12s} {'K-Means':>12s} {'Buy&Hold':>12s}"
    print(header)
    print("  " + "-" * 58)

    bh_return = dnn_results["buy_and_hold_return"]
    rows = [
        ("Total Return",
         f"{dnn_results['total_return']:+.2f}%",
         f"{kmeans_results['total_return']:+.2f}%",
         f"{bh_return:+.2f}%"),
        ("Max Drawdown",
         f"{dnn_results['max_drawdown']:.2f}%",
         f"{kmeans_results['max_drawdown']:.2f}%",
         "N/A"),
        ("Sharpe Ratio",
         f"{dnn_results['sharpe_ratio']:.2f}",
         f"{kmeans_results['sharpe_ratio']:.2f}",
         "N/A"),
        ("Win Rate",
         f"{dnn_results['win_rate']:.1f}%",
         f"{kmeans_results['win_rate']:.1f}%",
         "N/A"),
        ("Profit Factor",
         f"{dnn_results['profit_factor']:.2f}",
         f"{kmeans_results['profit_factor']:.2f}",
         "N/A"),
        ("Num Trades",
         f"{dnn_results['num_trades']}",
         f"{kmeans_results['num_trades']}",
         "1"),
        ("Final Value",
         f"{dnn_results['final_value']:,.0f}",
         f"{kmeans_results['final_value']:,.0f}",
         "N/A"),
    ]
    for label, dnn_v, km_v, bh_v in rows:
        print(f"  {label:<22s} {dnn_v:>12s} {km_v:>12s} {bh_v:>12s}")

    # Recent trades
    trades = dnn_results["trades"]
    if trades:
        print(f"\n{'=' * 60}")
        print("DNN RECENT TRADES (last 10)")
        print("=" * 60)
        for t in trades[-10:]:
            print(f"  {t['date']}  {t['action']:4s}  {t['shares']:6d} shares @ "
                  f"{t['price']:.2f}  ({t['signal']})")

    # Latest signal
    print(f"\n{'=' * 60}")
    print("DNN TODAY'S SIGNAL")
    print("=" * 60)
    df_full = compute_indicators(df).dropna(subset=FEATURE_COLS).reset_index(drop=True)
    bot = dnn_results["bot"]
    latest_signal = bot.predict_single(df_full)
    latest = df_full.iloc[-1]
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
