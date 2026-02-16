#!/usr/bin/env python3
"""Run K-Means, LSTM, and LightGBM backtests, compare results, and save a
combined daily trading log with signals and actions for each model."""

import argparse

import numpy as np
import pandas as pd

from trading_bot import (
    FEATURE_COLS,
    compute_indicators,
    run_backtest,
)
from dnn_trading_bot import run_dnn_backtest
from lgbm_trading_bot import run_lgbm_backtest


def main():
    parser = argparse.ArgumentParser(description="Compare K-Means vs LSTM vs LightGBM")
    parser.add_argument("--csv", default="601933_10yr.csv", help="Input CSV")
    parser.add_argument("--output", default="trade_log.csv", help="Output trade log")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    print(f"Loaded {len(df)} rows from {args.csv}")
    print(f"Date range: {df['date'].iloc[0]} to {df['date'].iloc[-1]}\n")

    # --- K-Means backtest ---
    print("=" * 60)
    print("K-MEANS BACKTEST")
    print("=" * 60)
    km_results = run_backtest(df, train_ratio=0.6, initial_capital=100_000)

    # --- LSTM backtest ---
    print(f"\n{'=' * 60}")
    print("LSTM BACKTEST")
    print("=" * 60)
    lstm_results = run_dnn_backtest(
        df, train_ratio=0.6, initial_capital=100_000,
        window_size=20, epochs=50, batch_size=32, lr=0.001,
    )

    # --- LightGBM backtest ---
    print(f"\n{'=' * 60}")
    print("LIGHTGBM BACKTEST")
    print("=" * 60)
    lgbm_results = run_lgbm_backtest(
        df, train_ratio=0.6, initial_capital=100_000,
    )

    # --- Comparison table ---
    print(f"\n{'=' * 72}")
    print("COMPARISON (with SMA5 buy filter)")
    print("=" * 72)

    bh_return = km_results["buy_and_hold_return"]
    header = (
        f"  {'Metric':<22s} {'K-Means':>12s} {'LSTM':>12s}"
        f" {'LightGBM':>12s} {'Buy&Hold':>12s}"
    )
    print(header)
    print("  " + "-" * 70)

    rows = [
        ("Total Return",
         f"{km_results['total_return']:+.2f}%",
         f"{lstm_results['total_return']:+.2f}%",
         f"{lgbm_results['total_return']:+.2f}%",
         f"{bh_return:+.2f}%"),
        ("Max Drawdown",
         f"{km_results['max_drawdown']:.2f}%",
         f"{lstm_results['max_drawdown']:.2f}%",
         f"{lgbm_results['max_drawdown']:.2f}%",
         "N/A"),
        ("Sharpe Ratio",
         f"{km_results['sharpe_ratio']:.3f}",
         f"{lstm_results['sharpe_ratio']:.3f}",
         f"{lgbm_results['sharpe_ratio']:.3f}",
         "N/A"),
        ("Win Rate",
         f"{km_results['win_rate']:.1f}%",
         f"{lstm_results['win_rate']:.1f}%",
         f"{lgbm_results['win_rate']:.1f}%",
         "N/A"),
        ("Profit Factor",
         f"{km_results['profit_factor']:.2f}",
         f"{lstm_results['profit_factor']:.2f}",
         f"{lgbm_results['profit_factor']:.2f}",
         "N/A"),
        ("Num Trades",
         f"{km_results['num_trades']}",
         f"{lstm_results['num_trades']}",
         f"{lgbm_results['num_trades']}",
         "1"),
        ("Final Value",
         f"{km_results['final_value']:,.0f}",
         f"{lstm_results['final_value']:,.0f}",
         f"{lgbm_results['final_value']:,.0f}",
         "N/A"),
    ]
    for label, km_v, ls_v, lg_v, bh_v in rows:
        print(f"  {label:<22s} {km_v:>12s} {ls_v:>12s} {lg_v:>12s} {bh_v:>12s}")

    # --- Build combined daily trade log ---
    km_test = km_results["test_df"].copy()
    lstm_test = lstm_results["test_df"].copy()
    lgbm_test = lgbm_results["test_df"].copy()

    # Reconstruct LSTM signals on the test set
    lstm_bot = lstm_results["bot"]
    lstm_raw_signals = lstm_bot.predict(lstm_test)
    lstm_signal_map = {}
    ws = lstm_bot.window_size
    for i, sig in enumerate(lstm_raw_signals):
        row_idx = i + ws
        if row_idx < len(lstm_test):
            lstm_signal_map[row_idx] = sig

    # Reconstruct LGBM signals on the test set (one signal per row, like K-Means)
    lgbm_bot = lgbm_results["bot"]
    lgbm_raw_signals = lgbm_bot.predict(lgbm_test)

    # Build trade lookup: date -> trade dict
    def _trade_lookup(trades):
        lookup = {}
        for t in trades:
            lookup[t["date"]] = t
        return lookup

    km_trades = _trade_lookup(km_results["trades"])
    lstm_trades = _trade_lookup(lstm_results["trades"])
    lgbm_trades = _trade_lookup(lgbm_results["trades"])

    log_rows = []
    for i in range(len(km_test)):
        date = str(km_test.iloc[i]["date"])
        close = km_test.iloc[i]["close"]
        open_ = km_test.iloc[i]["open"]
        sma5 = km_test.iloc[i]["sma5"]

        km_signal = km_test.iloc[i].get("signal", "hold")
        km_t = km_trades.get(date, {})
        km_action = km_t.get("action", "hold")
        km_shares = km_t.get("shares", 0)

        lstm_signal = lstm_signal_map.get(i, "hold")
        lstm_t = lstm_trades.get(date, {})
        lstm_action = lstm_t.get("action", "hold")
        lstm_shares = lstm_t.get("shares", 0)

        lgbm_signal = lgbm_raw_signals[i] if i < len(lgbm_raw_signals) else "hold"
        lgbm_t = lgbm_trades.get(date, {})
        lgbm_action = lgbm_t.get("action", "hold")
        lgbm_shares = lgbm_t.get("shares", 0)

        log_rows.append({
            "date": date,
            "open": round(open_, 2),
            "close": round(close, 2),
            "sma5": round(sma5, 2) if not np.isnan(sma5) else "",
            "km_signal": km_signal,
            "km_action": km_action,
            "km_shares": km_shares if km_shares else "",
            "lstm_signal": lstm_signal,
            "lstm_action": lstm_action,
            "lstm_shares": lstm_shares if lstm_shares else "",
            "lgbm_signal": lgbm_signal,
            "lgbm_action": lgbm_action,
            "lgbm_shares": lgbm_shares if lgbm_shares else "",
        })

    log_df = pd.DataFrame(log_rows)
    log_df.to_csv(args.output, index=False)

    # Count actions
    km_buys = len([r for r in log_rows if r["km_action"] == "buy"])
    km_sells = len([r for r in log_rows if r["km_action"] == "sell"])
    lstm_buys = len([r for r in log_rows if r["lstm_action"] == "buy"])
    lstm_sells = len([r for r in log_rows if r["lstm_action"] == "sell"])
    lgbm_buys = len([r for r in log_rows if r["lgbm_action"] == "buy"])
    lgbm_sells = len([r for r in log_rows if r["lgbm_action"] == "sell"])

    print(f"\n{'=' * 72}")
    print("TRADE LOG SUMMARY")
    print("=" * 72)
    print(f"  K-Means:   {km_buys} buys, {km_sells} sells")
    print(f"  LSTM:      {lstm_buys} buys, {lstm_sells} sells")
    print(f"  LightGBM:  {lgbm_buys} buys, {lgbm_sells} sells")
    print(f"\n  Saved {len(log_df)} rows to {args.output}")


if __name__ == "__main__":
    main()
