#!/usr/bin/env python3
"""BiLSTM next-day price range predictor for China A-shares.

Predicts (low, high) of the next trading day as values relative to today's
close.  A custom scoring scheme rewards tight, accurate range estimates:

  +2  — both predicted prices fit inside actual range
  +1  — predicted low >= actual low (high overshoots, but floor is safe)
  -1  — both ends outside actual range (floor missed and ceiling blown)
   0  — everything else (low misses, high is fine)
"""

import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from trading_bot import FEATURE_COLS, compute_indicators


# ---------------------------------------------------------------------------
# Asymmetric loss
# ---------------------------------------------------------------------------

def pinball_loss(pred: torch.Tensor, target: torch.Tensor, tau: float) -> torch.Tensor:
    """Quantile (pinball) loss.

    L(pred, target, tau) = tau * max(target - pred, 0)
                         + (1 - tau) * max(pred - target, 0)

    tau > 0.5  →  underestimates penalised more  (pushes pred up)
    tau < 0.5  →  overestimates penalised more   (pushes pred down)
    tau = 0.5  →  symmetric MAE
    """
    error = target - pred
    loss = torch.where(error >= 0, tau * error, (tau - 1) * error)
    return loss.mean()


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_prediction(
    pred_low: float,
    pred_high: float,
    actual_low: float,
    actual_high: float,
) -> int:
    """Score a single range prediction.

    Returns
    -------
    +2  both predicted prices fit inside the actual range
        (pred_low >= actual_low AND pred_high <= actual_high)
    +1  predicted low is >= actual low, but predicted high overshoots
        (pred_low >= actual_low AND pred_high > actual_high)
    -1  both ends outside the actual range
        (pred_low < actual_low AND pred_high > actual_high)
     0  everything else  (low misses, high is fine)
    """
    low_ok = pred_low >= actual_low
    high_ok = pred_high <= actual_high

    if low_ok and high_ok:
        return 2
    if low_ok and not high_ok:
        return 1
    if not low_ok and not high_ok:
        return -1
    return 0


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class RangeDataset(Dataset):
    """Sliding-window dataset that targets next-day (low, high) relative to
    today's close.

    target_low  = (next_day_low  - today_close) / today_close
    target_high = (next_day_high - today_close) / today_close
    """

    def __init__(self, df: pd.DataFrame, window_size: int = 20):
        self.window_size = window_size

        features = df[FEATURE_COLS].values.astype(np.float32)
        closes = df["close"].values.astype(np.float32)
        lows = df["low"].values.astype(np.float32)
        highs = df["high"].values.astype(np.float32)

        self.X: list[np.ndarray] = []
        self.y: list[np.ndarray] = []

        # window [i : i+window_size] → predict row i+window_size
        # "today" = row i+window_size-1 (last row of the window)
        n = len(features)
        for i in range(n - window_size - 1):
            today_close = closes[i + window_size - 1]
            next_low = lows[i + window_size]
            next_high = highs[i + window_size]

            rel_low = (next_low - today_close) / today_close
            rel_high = (next_high - today_close) / today_close

            self.X.append(features[i : i + window_size])
            self.y.append(np.array([rel_low, rel_high], dtype=np.float32))

        self.X = np.array(self.X)  # (N, window_size, n_features)
        self.y = np.array(self.y)  # (N, 2)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return (
            torch.tensor(self.X[idx]),
            torch.tensor(self.y[idx]),
        )


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class BiLSTMRangeModel(nn.Module):
    """Bidirectional LSTM with dual regression heads for (low, high).

    Architecture
    ------------
    BiLSTM(hidden, num_layers) → last timestep → FC(fc_size) → ReLU
        → head_low (1)
        → head_high (1)
    """

    def __init__(
        self,
        input_size: int = 6,
        hidden: int = 64,
        num_layers: int = 2,
        fc_size: int = 32,
        dropout: float = 0.2,
    ):
        super().__init__()
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.bilstm = nn.LSTM(
            input_size,
            hidden,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=lstm_dropout,
        )
        bilstm_out = hidden * 2  # bidirectional doubles the output
        self.shared = nn.Sequential(
            nn.Linear(bilstm_out, fc_size),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.head_low = nn.Linear(fc_size, 1)
        self.head_high = nn.Linear(fc_size, 1)

    def forward(self, x: torch.Tensor):
        """x: (batch, seq_len, input_size) → (low, high) each (batch,)"""
        out, _ = self.bilstm(x)
        out = out[:, -1, :]  # last timestep
        shared = self.shared(out)
        low = self.head_low(shared).squeeze(-1)
        high = self.head_high(shared).squeeze(-1)
        return low, high


# ---------------------------------------------------------------------------
# Predictor wrapper
# ---------------------------------------------------------------------------

class RangePredictor:
    """Wraps BiLSTMRangeModel: fit, predict, predict_single, evaluate_score."""

    def __init__(
        self,
        window_size: int = 20,
        epochs: int = 100,
        batch_size: int = 32,
        lr: float = 1e-3,
        patience: int = 15,
        hidden: int = 64,
        num_layers: int = 2,
        fc_size: int = 32,
        dropout: float = 0.2,
        tau_low: float = 0.8,
        tau_high: float = 0.2,
    ):
        self.window_size = window_size
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.patience = patience
        self.hidden = hidden
        self.num_layers = num_layers
        self.fc_size = fc_size
        self.dropout = dropout
        self.tau_low = tau_low    # >0.5 → penalise underestimates → pred_low pushed up
        self.tau_high = tau_high  # <0.5 → penalise overestimates  → pred_high pulled down

        self.model: BiLSTMRangeModel | None = None
        self.scaler_mean: np.ndarray | None = None
        self.scaler_std: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _scale(self, df: pd.DataFrame) -> pd.DataFrame:
        df_s = df.copy()
        for i, col in enumerate(FEATURE_COLS):
            df_s[col] = (df_s[col] - self.scaler_mean[i]) / self.scaler_std[i]
        return df_s

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame) -> None:
        """Train the BiLSTM on df (must have indicator columns already)."""
        # Need at least 10 training samples for a meaningful fit
        min_rows = self.window_size + 1 + 10
        if len(df) < min_rows:
            raise ValueError(
                f"Need at least {min_rows} rows (window_size + 11); got {len(df)}"
            )

        features = df[FEATURE_COLS].values.astype(np.float32)
        self.scaler_mean = features.mean(axis=0)
        self.scaler_std = features.std(axis=0)
        self.scaler_std[self.scaler_std == 0] = 1.0

        df_scaled = self._scale(df)

        full_ds = RangeDataset(df_scaled, window_size=self.window_size)
        n = len(full_ds)
        val_size = max(1, int(n * 0.1))
        train_size = n - val_size

        train_ds = torch.utils.data.Subset(full_ds, range(train_size))
        val_ds = torch.utils.data.Subset(full_ds, range(train_size, n))

        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False)

        self.model = BiLSTMRangeModel(
            input_size=len(FEATURE_COLS),
            hidden=self.hidden,
            num_layers=self.num_layers,
            fc_size=self.fc_size,
            dropout=self.dropout,
        )
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        best_val_loss = float("inf")
        patience_counter = 0
        best_state = None

        for epoch in range(self.epochs):
            # Train
            self.model.train()
            train_loss = 0.0
            for xb, yb in train_loader:
                optimizer.zero_grad()
                pred_low, pred_high = self.model(xb)
                loss = (
                    pinball_loss(pred_low,  yb[:, 0], self.tau_low) +
                    pinball_loss(pred_high, yb[:, 1], self.tau_high)
                )
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * xb.size(0)
            train_loss /= train_size

            # Validate
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for xb, yb in val_loader:
                    pred_low, pred_high = self.model(xb)
                    loss = (
                        pinball_loss(pred_low,  yb[:, 0], self.tau_low) +
                        pinball_loss(pred_high, yb[:, 1], self.tau_high)
                    )
                    val_loss += loss.item() * xb.size(0)
            val_loss /= val_size

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
            else:
                patience_counter += 1

            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(
                    f"  Epoch {epoch+1:3d}/{self.epochs}  "
                    f"train={train_loss:.6f}  val={val_loss:.6f}"
                )

            if patience_counter >= self.patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.model.eval()

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def predict(self, df: pd.DataFrame) -> list[tuple[float, float]]:
        """Return (pred_low_abs, pred_high_abs) for every valid window in df.

        Predictions are in absolute price (not relative), reconstructed from
        today's close.
        """
        df_scaled = self._scale(df)
        features = df_scaled[FEATURE_COLS].values.astype(np.float32)
        closes = df["close"].values.astype(np.float32)

        self.model.eval()
        results = []
        with torch.no_grad():
            n = len(features)
            for i in range(n - self.window_size - 1):
                window = torch.tensor(
                    features[i : i + self.window_size]
                ).unsqueeze(0)
                rel_low, rel_high = self.model(window)
                today_close = float(closes[i + self.window_size - 1])
                pred_low = float(rel_low.item()) * today_close + today_close
                pred_high = float(rel_high.item()) * today_close + today_close
                results.append((pred_low, pred_high))
        return results

    def predict_single(self, df: pd.DataFrame) -> tuple[float, float]:
        """Predict next trading day's (low, high) from the last window_size rows."""
        df_scaled = self._scale(df)
        features = df_scaled[FEATURE_COLS].values.astype(np.float32)
        closes = df["close"].values.astype(np.float32)

        window = torch.tensor(features[-self.window_size :]).unsqueeze(0)
        today_close = float(closes[-1])

        self.model.eval()
        with torch.no_grad():
            rel_low, rel_high = self.model(window)
        pred_low = float(rel_low.item()) * today_close + today_close
        pred_high = float(rel_high.item()) * today_close + today_close
        return pred_low, pred_high

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate_score(self, df: pd.DataFrame) -> dict:
        """Score all predictions against actual next-day (low, high).

        Returns a dict with total_score, plus_one, minus_one, zero, n_predictions.
        """
        predictions = self.predict(df)
        closes = df["close"].values.astype(np.float32)
        lows = df["low"].values.astype(np.float32)
        highs = df["high"].values.astype(np.float32)

        plus_two = plus_one = minus_one = zero = 0
        for i, (pred_low, pred_high) in enumerate(predictions):
            # target row is i + window_size (same alignment as predict())
            target_row = i + self.window_size
            actual_low = float(lows[target_row])
            actual_high = float(highs[target_row])
            s = score_prediction(pred_low, pred_high, actual_low, actual_high)
            if s == 2:
                plus_two += 1
            elif s == 1:
                plus_one += 1
            elif s == -1:
                minus_one += 1
            else:
                zero += 1

        n = plus_two + plus_one + minus_one + zero
        total = plus_two * 2 + plus_one * 1 + minus_one * (-1)
        return {
            "total_score": total,
            "plus_two": plus_two,
            "plus_one": plus_one,
            "minus_one": minus_one,
            "zero": zero,
            "n_predictions": n,
        }


# ---------------------------------------------------------------------------
# Backtest
# ---------------------------------------------------------------------------

def run_range_backtest(
    df: pd.DataFrame,
    train_ratio: float = 0.7,
    window_size: int = 20,
    epochs: int = 100,
    batch_size: int = 32,
    lr: float = 1e-3,
    patience: int = 15,
    hidden: int = 64,
    num_layers: int = 2,
    tau_low: float = 0.8,
    tau_high: float = 0.2,
) -> dict:
    """Walk-forward backtest: train on first portion, evaluate scoring on rest."""
    df = compute_indicators(df)
    df = df.dropna(subset=FEATURE_COLS).reset_index(drop=True)

    split = int(len(df) * train_ratio)
    train_df = df.iloc[:split].copy().reset_index(drop=True)
    test_df = df.iloc[split:].copy().reset_index(drop=True)

    print(f"Training BiLSTM on {len(train_df)} rows...")
    predictor = RangePredictor(
        window_size=window_size,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        patience=patience,
        hidden=hidden,
        num_layers=num_layers,
        tau_low=tau_low,
        tau_high=tau_high,
    )
    predictor.fit(train_df)

    print(f"Evaluating on {len(test_df)} test rows...")
    scores = predictor.evaluate_score(test_df)

    return {
        **scores,
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "predictor": predictor,
        "test_df": test_df,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="BiLSTM next-day price range predictor")
    parser.add_argument("--csv", default="data/601328_20yr.csv", help="CSV file path")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--window", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--tau-low", type=float, default=0.8,
                        help="Pinball tau for low head (>0.5 penalises underestimates)")
    parser.add_argument("--tau-high", type=float, default=0.2,
                        help="Pinball tau for high head (<0.5 penalises overestimates)")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    print(f"Loaded {len(df)} rows from {args.csv}")
    print(f"Date range: {df['date'].iloc[0]} → {df['date'].iloc[-1]}\n")

    print("=" * 60)
    print("BiLSTM RANGE PREDICTOR BACKTEST")
    print("=" * 60)
    result = run_range_backtest(
        df,
        train_ratio=args.train_ratio,
        window_size=args.window,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
        hidden=args.hidden,
        num_layers=args.num_layers,
        tau_low=args.tau_low,
        tau_high=args.tau_high,
    )

    n = result["n_predictions"]
    print(f"\n{'=' * 60}")
    print("SCORING RESULTS (test set)")
    print("=" * 60)
    print(f"  Predictions   : {n}")
    print(f"  +2 (both in)  : {result['plus_two']:5d}  ({result['plus_two']/n*100:.1f}%)")
    print(f"  +1 (low ok)   : {result['plus_one']:5d}  ({result['plus_one']/n*100:.1f}%)")
    print(f"   0 (rest)     : {result['zero']:5d}  ({result['zero']/n*100:.1f}%)")
    print(f"  -1 (both out) : {result['minus_one']:5d}  ({result['minus_one']/n*100:.1f}%)")
    print(f"  Total score   : {result['total_score']:+d}")
    print(f"  Score / pred  : {result['total_score']/n:+.3f}")

    # Next-day prediction
    print(f"\n{'=' * 60}")
    print("NEXT TRADING DAY PREDICTION")
    print("=" * 60)
    predictor = result["predictor"]
    df_full = compute_indicators(df).dropna(subset=FEATURE_COLS).reset_index(drop=True)
    pred_low, pred_high = predictor.predict_single(df_full)
    last_row = df_full.iloc[-1]
    print(f"  Last date  : {last_row['date']}")
    print(f"  Last close : {last_row['close']:.4f}")
    print(f"  Pred low   : {pred_low:.4f}")
    print(f"  Pred high  : {pred_high:.4f}")
    print(f"  Pred range : {pred_high - pred_low:.4f}")


if __name__ == "__main__":
    main()
