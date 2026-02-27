#!/usr/bin/env python3
"""Train one deep_attn model on all stocks combined, then backtest and predict each."""

import os
import pandas as pd
from multi_stock_experiment import discover_csvs, load_stock_df, split_train_test
from range_predictor import RangePredictor
from trading_bot import FEATURE_COLS, compute_indicators

DATA_DIR = "data"

# Load all stocks
csv_paths = discover_csvs(DATA_DIR)
print(f"Found {len(csv_paths)} CSVs: {[os.path.basename(p) for p in csv_paths]}\n")

train_dfs, test_dfs, full_dfs = [], {}, {}
for path in csv_paths:
    name = os.path.splitext(os.path.basename(path))[0]
    df = load_stock_df(path)
    train, test = split_train_test(df, train_ratio=0.7)
    train_dfs.append(train)
    test_dfs[name] = test
    full_dfs[name] = df
    print(f"  {name}: {len(df)} rows  ({len(train)} train / {len(test)} test)")

total_train = sum(len(d) for d in train_dfs)
print(f"\nCombined training rows: {total_train}")

# Train one model on all stocks
print("\nTraining deep_attn on combined data...")
predictor = RangePredictor(
    hidden=128, num_layers=4, fc_sizes=[256, 128, 64],
    layer_norm=True, use_attention=True,
    epochs=150, patience=20, batch_size=64,
)
predictor.fit_multi(train_dfs)

# Backtest and next-day prediction per stock
print(f"\n{'='*60}")
for name, test_df in test_dfs.items():
    scores = predictor.evaluate_score(test_df)
    n = scores["n_predictions"]
    print(f"\n{name}")
    if n == 0:
        print(f"  Backtest: skipped (too few rows for evaluation)")
    else:
        print(f"  Backtest ({n} predictions)")
        print(f"    +2 (both match)  : {scores['plus_two']:4d}  ({scores['plus_two']/n*100:.1f}%)")
        print(f"    +1 (one match)   : {scores['plus_one']:4d}  ({scores['plus_one']/n*100:.1f}%)")
        print(f"     0 (no match)    : {scores['zero']:4d}  ({scores['zero']/n*100:.1f}%)")
        print(f"    -1 (wrong side)  : {scores['minus_one']:4d}  ({scores['minus_one']/n*100:.1f}%)")
        print(f"    Score / pred     : {scores['total_score']/n:+.3f}")

    df_full = full_dfs[name]
    if len(df_full) < predictor.window_size + 1:
        print(f"  Next-day prediction: skipped (too few rows)")
        continue
    pred_low, pred_high = predictor.predict_single(df_full)
    last = df_full.iloc[-1]
    chg_low  = (pred_low  - last["close"]) / last["close"] * 100
    chg_high = (pred_high - last["close"]) / last["close"] * 100
    print(f"  Next-day prediction (from {last['date']})")
    print(f"    Close : {last['close']:.4f}")
    print(f"    Low   : {pred_low:.4f}  ({chg_low:+.2f}%)")
    print(f"    High  : {pred_high:.4f}  ({chg_high:+.2f}%)")
    print(f"    Range : {pred_high - pred_low:.4f}")
