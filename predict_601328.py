#!/usr/bin/env python3
"""Fetch latest data for 601328, append to historical CSV, predict next trading day."""

import pandas as pd
from fetch_china_stock import fetch_stock_daily
from range_predictor import RangePredictor, run_range_backtest
from trading_bot import FEATURE_COLS, compute_indicators

CSV = "data/601328_20yr.csv"

# 1. Load historical data
hist = pd.read_csv(CSV)
print(f"Historical data: {len(hist)} rows, last date: {hist['date'].iloc[-1]}")

# 2. Fetch any new rows up to today
print("Fetching latest data from akshare...")
latest = fetch_stock_daily("601328", start_date="20260101", end_date="20260225")
latest = latest[["date", "open", "close", "high", "low", "volume"]].copy()
latest["date"] = latest["date"].astype(str)

# 3. Merge: keep only rows newer than what we have
last_date = hist["date"].iloc[-1]
new_rows = latest[latest["date"] > last_date]
if new_rows.empty:
    print("No new rows to append.")
    df = hist
else:
    print(f"Appending {len(new_rows)} new row(s): {new_rows['date'].tolist()}")
    df = pd.concat([hist[["date","open","close","high","low","volume"]], new_rows],
                   ignore_index=True)

print(f"Using {len(df)} rows, last date: {df['date'].iloc[-1]}\n")

# 4. Train deep_attn model and predict
print("Training deep_attn model (hidden=128, 4 layers, attention)...")
result = run_range_backtest(
    df,
    train_ratio=0.7,
    epochs=150,
    patience=20,
    batch_size=64,
    hidden=128,
    num_layers=4,
    fc_sizes=[256, 128, 64],
    layer_norm=True,
    use_attention=True,
)

n = result["n_predictions"]
print(f"\n{'='*50}")
print("BACKTEST SCORING (test set)")
print(f"{'='*50}")
print(f"  Predictions   : {n}")
print(f"  +2 (both in)  : {result['plus_two']:5d}  ({result['plus_two']/n*100:.1f}%)")
print(f"  +1 (low ok)   : {result['plus_one']:5d}  ({result['plus_one']/n*100:.1f}%)")
print(f"   0 (rest)     : {result['zero']:5d}  ({result['zero']/n*100:.1f}%)")
print(f"  -1 (both out) : {result['minus_one']:5d}  ({result['minus_one']/n*100:.1f}%)")
print(f"  Total score   : {result['total_score']:+d}")
print(f"  Score / pred  : {result['total_score']/n:+.3f}")

# 5. Next-day prediction
predictor = result["predictor"]
df_full = compute_indicators(df).dropna(subset=FEATURE_COLS).reset_index(drop=True)
pred_low, pred_high = predictor.predict_single(df_full)
last_row = df_full.iloc[-1]

print(f"\n{'='*50}")
print("NEXT TRADING DAY PREDICTION")
print(f"{'='*50}")
print(f"  Last date   : {last_row['date']}  (today)")
print(f"  Last close  : {last_row['close']:.4f}")
print(f"  Pred low    : {pred_low:.4f}")
print(f"  Pred high   : {pred_high:.4f}")
print(f"  Pred range  : {pred_high - pred_low:.4f}")
chg_low  = (pred_low  - last_row['close']) / last_row['close'] * 100
chg_high = (pred_high - last_row['close']) / last_row['close'] * 100
print(f"  vs close    : {chg_low:+.2f}% ~ {chg_high:+.2f}%")
