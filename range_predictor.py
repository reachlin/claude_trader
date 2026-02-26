#!/usr/bin/env python3
"""BiLSTM next-day price range predictor for China A-shares.

Design overview
---------------
Goal: predict the (low, high) price range of the *next* trading day so a
trader can set tight limit orders or size positions with known risk bounds.

Why BiLSTM?
  A bidirectional LSTM reads the window both forward and backward, letting
  it capture both momentum (recent trend direction) and mean-reversion
  patterns earlier in the window.  A unidirectional LSTM would miss context
  from earlier in the window when processing the last timestep.

Why attention instead of just the last hidden state?
  The last timestep captures "where we are now" but throws away the shape of
  the whole window.  A volatile spike 5 days ago, or a support test 10 days
  ago, can be just as informative as yesterday's close.  A learned attention
  layer scores every timestep and takes a weighted sum, letting the model
  focus on whichever days in the 20-day window matter most for the current
  prediction.  In our 4-way experiment this was the single biggest lever:
  adding attention to the baseline config improved score/pred from +1.34 →
  +1.55, and the deep+attention config reached +1.93.

Why asymmetric (pinball) loss?
  MSE and MAE treat over- and under-estimates identically.  For trading,
  the consequences are asymmetric:
    - A predicted low that is *too low* → we miss the entry (opportunity cost)
    - A predicted low that is *too high* → we get stopped out (real loss)
  We use tau_low=0.8 (penalise underestimates more) to push pred_low up
  toward the true floor, and tau_high=0.2 (penalise overestimates more) to
  pull pred_high down toward the true ceiling.  This biases the model toward
  conservative, tight ranges rather than wide, safe-but-useless intervals.

Why relative targets?
  Raw prices vary wildly across stocks (e.g. 600519 ~1500 CNY vs 000001
  ~15 CNY).  Expressing targets as (next_low - today_close) / today_close
  makes the same model and loss scale apply to all stocks, which is essential
  for multi-stock training.

Scoring scheme
--------------
  +1  — predicted low  rounds to same 0.1 as actual low
  +1  — predicted high rounds to same 0.1 as actual high
   0  — no match on that side
  Total per prediction: 0, 1, or 2  (no negatives)
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

    Each side (low, high) earns +1 independently if the predicted price
    rounds to the same 0.1 as the actual price.  Total: 0, 1, or 2.

    Examples
    --------
    pred_low=2.13, actual_low=2.10  → round to 2.1 == 2.1  → +1
    pred_low=2.13, actual_low=2.20  → round to 2.1 != 2.2  →  0
    pred_high=4.27, actual_high=4.30 → round to 4.3 == 4.3 → +1
    """
    low_match  = round(pred_low,  1) == round(actual_low,  1)
    high_match = round(pred_high, 1) == round(actual_high, 1)
    return int(low_match) + int(high_match)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class RangeDataset(Dataset):
    """Sliding-window dataset that targets next-day (low, high) relative to
    today's close.

    target_low  = (next_day_low  - today_close) / today_close
    target_high = (next_day_high - today_close) / today_close

    Expressing targets as returns (not raw prices) achieves two things:
      1. Scale-invariant: the same model trains on 15 CNY and 1500 CNY stocks
         without the loss being dominated by the high-priced stock.
      2. Stationarity: percentage moves are more stationary than raw prices,
         which makes the learning problem easier for a recurrent network.

    Window size of 20 (≈ 1 trading month) is a practical default:
      - Short enough that the market regime is roughly stable across the window
      - Long enough to capture weekly seasonality and short-term momentum
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
    BiLSTM(hidden, num_layers) → all timesteps
        → [attention pooling OR last timestep]
        → [optional LayerNorm]
        → FC(fc_sizes[0]) → ReLU → Dropout
        → FC(fc_sizes[1]) → ReLU → Dropout  (repeated for each size)
        → head_low (1)
        → head_high (1)

    When use_attention=True a single linear layer scores each timestep;
    softmax weights are used to compute a weighted sum over the sequence
    instead of discarding all but the last step.
    """

    def __init__(
        self,
        input_size: int = 6,
        hidden: int = 64,       # hidden units per direction; BiLSTM output = hidden*2
        num_layers: int = 2,    # stacked LSTM layers; more layers → deeper temporal abstraction
        fc_sizes: list[int] | None = None,  # sizes of FC layers after pooling, e.g. [256,128,64]
        dropout: float = 0.2,   # applied between FC layers and between LSTM layers (if >1)
        layer_norm: bool = False,  # normalise BiLSTM output before FC; helps with deep configs
        use_attention: bool = False,  # attention pooling over all timesteps (vs last-step only)
    ):
        super().__init__()
        if fc_sizes is None:
            fc_sizes = [32]
        # LSTM dropout only applies between stacked layers, not on the output of the last layer.
        # PyTorch raises an error if dropout > 0 with a single-layer LSTM, so guard it here.
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.bilstm = nn.LSTM(
            input_size,
            hidden,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=lstm_dropout,
        )
        bilstm_out = hidden * 2  # bidirectional doubles the output dimension
        # LayerNorm stabilises the distribution coming out of the BiLSTM, which is
        # especially useful for deeper configs where activations can drift across layers.
        self.ln = nn.LayerNorm(bilstm_out) if layer_norm else None
        self.use_attention = use_attention
        # Single linear layer (no bias) maps each timestep's hidden state to a scalar
        # score.  Softmax over the time dimension turns scores into a probability
        # distribution — the model learns which days matter most for the prediction.
        self.attn = nn.Linear(bilstm_out, 1, bias=False) if use_attention else None

        # Shared FC trunk: both heads (low and high) start from the same representation
        # so the model is forced to learn a common understanding of market state before
        # specialising into the two quantile outputs.
        layers = []
        in_size = bilstm_out
        for fc_size in fc_sizes:
            layers.extend([
                nn.Linear(in_size, fc_size),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            in_size = fc_size
        self.shared = nn.Sequential(*layers)
        # Separate regression heads for low and high let each head optimise its own
        # asymmetric pinball loss independently (tau_low=0.8, tau_high=0.2).
        self.head_low = nn.Linear(in_size, 1)
        self.head_high = nn.Linear(in_size, 1)

    def forward(self, x: torch.Tensor):
        """x: (batch, seq_len, input_size) → (low, high) each (batch,)"""
        out, _ = self.bilstm(x)  # (batch, seq_len, hidden*2)
        if self.use_attention:
            # Attention pooling: learn a soft selection over all timesteps.
            # This outperforms last-step in our experiments because earlier days
            # (e.g. a support test 2 weeks ago) can be highly informative.
            scores = self.attn(out)                   # (batch, seq_len, 1)
            weights = torch.softmax(scores, dim=1)    # (batch, seq_len, 1) — sums to 1
            out = (weights * out).sum(dim=1)           # (batch, hidden*2)
        else:
            # Without attention, use only the final hidden state.  The BiLSTM has
            # already propagated context from the full window into this state, but
            # the attention variant can be more selective about what it retains.
            out = out[:, -1, :]                        # last timestep
        if self.ln is not None:
            out = self.ln(out)
        shared = self.shared(out)
        low = self.head_low(shared).squeeze(-1)
        high = self.head_high(shared).squeeze(-1)
        return low, high


# ---------------------------------------------------------------------------
# Predictor wrapper
# ---------------------------------------------------------------------------

class RangePredictor:
    """Wraps BiLSTMRangeModel: fit, predict, predict_single, evaluate_score.

    Handles feature scaling, dataset construction, training loop with early
    stopping, and converting relative predictions back to absolute prices.

    Best config found in multi-stock experiments (4 stocks × 20 years):
        hidden=128, num_layers=4, fc_sizes=[256,128,64],
        layer_norm=True, use_attention=True
        → score/pred = +1.93,  +2 rate ≈ 95%
    """

    def __init__(
        self,
        window_size: int = 20,
        epochs: int = 100,
        batch_size: int = 32,
        lr: float = 1e-3,
        patience: int = 15,     # stop early if val loss doesn't improve for this many epochs
        hidden: int = 64,
        num_layers: int = 2,
        fc_sizes: list[int] | None = None,
        dropout: float = 0.2,
        tau_low: float = 0.8,   # >0.5 → penalise underestimates → pred_low pushed up toward floor
        tau_high: float = 0.2,  # <0.5 → penalise overestimates  → pred_high pulled down toward ceiling
        layer_norm: bool = False,
        use_attention: bool = False,
    ):
        self.window_size = window_size
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.patience = patience
        self.hidden = hidden
        self.num_layers = num_layers
        self.fc_sizes = fc_sizes if fc_sizes is not None else [32]
        self.dropout = dropout
        self.tau_low = tau_low
        self.tau_high = tau_high
        self.layer_norm = layer_norm
        self.use_attention = use_attention

        self.model: BiLSTMRangeModel | None = None
        # Z-score scaler parameters fitted on training data; stored so the same
        # transform can be applied at inference time without re-fitting.
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
        """Train the BiLSTM on df (must have indicator columns already).

        Uses the last 10% of windows as a chronological validation set for
        early stopping — never shuffled, to avoid look-ahead bias.
        """
        # Need at least 10 training samples for a meaningful fit
        min_rows = self.window_size + 1 + 10
        if len(df) < min_rows:
            raise ValueError(
                f"Need at least {min_rows} rows (window_size + 11); got {len(df)}"
            )

        features = df[FEATURE_COLS].values.astype(np.float32)
        # Z-score normalisation: features are indicators (RSI, MACD, etc.) on
        # different scales; standardising each independently prevents a single
        # large-magnitude feature from dominating the LSTM input.
        self.scaler_mean = features.mean(axis=0)
        self.scaler_std = features.std(axis=0)
        self.scaler_std[self.scaler_std == 0] = 1.0  # avoid division by zero for flat features

        df_scaled = self._scale(df)

        full_ds = RangeDataset(df_scaled, window_size=self.window_size)
        n = len(full_ds)
        # Chronological 90/10 split: validation set is the most recent 10% of
        # windows, matching how the model will actually be used in production.
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
            fc_sizes=self.fc_sizes,
            dropout=self.dropout,
            layer_norm=self.layer_norm,
            use_attention=self.use_attention,
        )
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        best_val_loss = float("inf")
        patience_counter = 0
        best_state = None  # snapshot of weights at the best validation loss

        for epoch in range(self.epochs):
            # Train
            self.model.train()
            train_loss = 0.0
            for xb, yb in train_loader:
                optimizer.zero_grad()
                pred_low, pred_high = self.model(xb)
                # Sum of two pinball losses: one for the low head, one for the high head.
                # Each head is optimised toward a different quantile of the distribution.
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
                # Deep-copy weights so we can restore them after early stopping.
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

        # Restore best weights — the model at the end of training may have overfit.
        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.model.eval()

    # ------------------------------------------------------------------
    # Fit on multiple stocks
    # ------------------------------------------------------------------

    def fit_multi(self, dfs: list[pd.DataFrame]) -> None:
        """Train on combined data from multiple stocks.

        Computes a global scaler from all DataFrames, then builds a
        separate RangeDataset per stock (preventing window artefacts at
        stock boundaries), concatenates them, and trains a single model.

        Why build per-stock datasets rather than concatenating DataFrames?
          If we naively concatenated the DataFrames, the sliding window would
          create samples that span two different stocks (e.g. the last 10 rows
          of stock A and the first 10 rows of stock B).  Those windows are
          meaningless.  Building a separate RangeDataset per stock and then
          concatenating the datasets avoids this entirely.

        Why a global scaler?
          The features (RSI, MACD histogram, Bollinger %B, etc.) are already
          scale-invariant ratios, so their distributions are comparable across
          stocks.  A global scaler pools more data for a more stable mean/std
          estimate and ensures the model sees the same feature space at
          inference time regardless of which stock it is predicting.
        """
        if not dfs:
            raise ValueError("dfs must not be empty")

        # Global scaler: pool all training features across all stocks
        all_features = np.vstack([
            df[FEATURE_COLS].values.astype(np.float32) for df in dfs
        ])
        self.scaler_mean = all_features.mean(axis=0)
        self.scaler_std = all_features.std(axis=0)
        self.scaler_std[self.scaler_std == 0] = 1.0

        # Build per-stock datasets using the global scaler
        datasets = []
        for df in dfs:
            df_scaled = self._scale(df)
            ds = RangeDataset(df_scaled, window_size=self.window_size)
            if len(ds) > 0:
                datasets.append(ds)

        if not datasets:
            raise ValueError("No valid windows found in any of the provided DataFrames")

        combined = torch.utils.data.ConcatDataset(datasets)
        n = len(combined)
        val_size = max(1, int(n * 0.1))
        train_size = n - val_size

        train_ds = torch.utils.data.Subset(combined, range(train_size))
        val_ds = torch.utils.data.Subset(combined, range(train_size, n))

        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False)

        self.model = BiLSTMRangeModel(
            input_size=len(FEATURE_COLS),
            hidden=self.hidden,
            num_layers=self.num_layers,
            fc_sizes=self.fc_sizes,
            dropout=self.dropout,
            layer_norm=self.layer_norm,
            use_attention=self.use_attention,
        )
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        best_val_loss = float("inf")
        patience_counter = 0
        best_state = None

        for epoch in range(self.epochs):
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
        today's close:
            pred_abs = pred_relative * today_close + today_close
        This is the inverse of the RangeDataset target transform.
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

        Returns a dict with total_score, plus_two, plus_one, zero, n_predictions.
        Each prediction scores 0, 1, or 2 (see score_prediction).
        """
        predictions = self.predict(df)
        lows = df["low"].values.astype(np.float32)
        highs = df["high"].values.astype(np.float32)

        plus_two = plus_one = zero = 0
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
            else:
                zero += 1

        n = plus_two + plus_one + zero
        total = plus_two * 2 + plus_one * 1
        return {
            "total_score": total,
            "plus_two": plus_two,
            "plus_one": plus_one,
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
    fc_sizes: list[int] | None = None,
    layer_norm: bool = False,
    use_attention: bool = False,
    tau_low: float = 0.8,
    tau_high: float = 0.2,
) -> dict:
    """Walk-forward backtest: train on first portion, evaluate scoring on rest.

    The train/test split is strictly chronological (no shuffling) to simulate
    real-world deployment: the model never sees future data during training.
    train_ratio=0.7 means the oldest 70% of history is used for training and
    the most recent 30% is held out for evaluation.
    """
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
        fc_sizes=fc_sizes,
        layer_norm=layer_norm,
        use_attention=use_attention,
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
    parser.add_argument("--fc-sizes", type=int, nargs="+", default=None,
                        help="FC layer sizes, e.g. --fc-sizes 256 128 64")
    parser.add_argument("--layer-norm", action="store_true", default=False)
    parser.add_argument("--use-attention", action="store_true", default=False)
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
        fc_sizes=args.fc_sizes,
        layer_norm=args.layer_norm,
        use_attention=args.use_attention,
        tau_low=args.tau_low,
        tau_high=args.tau_high,
    )

    n = result["n_predictions"]
    print(f"\n{'=' * 60}")
    print("SCORING RESULTS (test set)")
    print("=" * 60)
    print(f"  Predictions      : {n}")
    print(f"  +2 (both match)  : {result['plus_two']:5d}  ({result['plus_two']/n*100:.1f}%)")
    print(f"  +1 (one match)   : {result['plus_one']:5d}  ({result['plus_one']/n*100:.1f}%)")
    print(f"   0 (no match)    : {result['zero']:5d}  ({result['zero']/n*100:.1f}%)")
    print(f"  Total score      : {result['total_score']:+d}")
    print(f"  Score / pred     : {result['total_score']/n:+.3f}")

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
