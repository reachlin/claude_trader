#!/usr/bin/env python3
"""Run K-Means, LSTM, LightGBM, PPO, Majority Vote, and TD3 backtests,
compare results, and save a combined daily trading log with signals and
actions for each model."""

import argparse

import numpy as np
import pandas as pd

from trading_bot import (
    FEATURE_COLS,
    LOT_SIZE,
    Portfolio,
    compute_indicators,
    run_backtest,
)
from dnn_trading_bot import run_dnn_backtest
from lgbm_trading_bot import run_lgbm_backtest
from ppo_trading_bot import run_ppo_backtest
from td3_trading_bot import run_td3_backtest


def _classify(signal):
    """Classify a signal into buy/sell/hold."""
    if signal in ("strong_buy", "mild_buy"):
        return "buy"
    elif signal in ("strong_sell", "mild_sell"):
        return "sell"
    return "hold"


def _majority_direction(dirs):
    """Return the majority direction if >= 3 out of 4 agree, else None."""
    from collections import Counter
    counts = Counter(dirs)
    for direction, count in counts.items():
        if count >= 3:
            return direction
    return None


def run_majority_backtest(km_results, lstm_results, lgbm_results, ppo_results,
                          initial_capital=100_000):
    """Run majority vote backtest: trade when >= 3 of 4 models agree.

    Re-uses the trained bots and test data from individual backtests.
    Returns a metrics dict compatible with the comparison table.
    """
    test_df = km_results["test_df"].copy().reset_index(drop=True)

    # Get signals from each model on the test set
    km_bot = km_results["bot"]
    km_signals = km_bot.predict(test_df)

    lstm_bot = lstm_results["bot"]
    lstm_raw = lstm_bot.predict(test_df)
    ws = lstm_bot.window_size
    lstm_signal_map = {}
    for i, sig in enumerate(lstm_raw):
        row_idx = i + ws
        if row_idx < len(test_df):
            lstm_signal_map[row_idx] = sig

    lgbm_bot = lgbm_results["bot"]
    lgbm_signals = lgbm_bot.predict(test_df)

    ppo_bot = ppo_results["bot"]
    ppo_signals = ppo_bot.predict(test_df)

    # Simulate portfolio with majority vote (>= 3 of 4 agree)
    portfolio = Portfolio(capital=initial_capital)
    trades = []
    daily_values = []

    for i in range(len(test_df) - 1):
        sigs = [km_signals[i], lstm_signal_map.get(i, "hold"),
                lgbm_signals[i] if i < len(lgbm_signals) else "hold",
                ppo_signals[i] if i < len(ppo_signals) else "hold"]
        dirs = [_classify(s) for s in sigs]
        majority = _majority_direction(dirs)

        exec_price = test_df.loc[i + 1, "open"]
        trade_date = str(test_df.loc[i + 1, "date"])
        price_below_sma5 = test_df.loc[i, "close"] < test_df.loc[i, "sma5"]

        shares_traded = 0
        action = "hold"

        if majority == "buy" and price_below_sma5:
            strong = any(s == "strong_buy" for s in sigs)
            frac = 1.0 if strong else 0.5
            shares_traded = portfolio.buy(exec_price, fraction=frac, trade_date=trade_date)
            if shares_traded > 0:
                action = "buy"
        elif majority == "sell":
            strong = any(s == "strong_sell" for s in sigs)
            frac = 1.0 if strong else 0.5
            shares_traded = portfolio.sell(exec_price, fraction=frac, trade_date=trade_date)
            if shares_traded > 0:
                action = "sell"

        if shares_traded > 0:
            trades.append({"date": trade_date, "action": action,
                           "price": exec_price, "shares": shares_traded})
        daily_values.append(portfolio.value(test_df.loc[i + 1, "close"]))

    # Compute metrics
    final_price = test_df.iloc[-1]["close"]
    final_value = portfolio.value(final_price)
    total_return = (final_value - initial_capital) / initial_capital * 100

    values = np.array([initial_capital] + daily_values)
    peak = np.maximum.accumulate(values)
    drawdowns = (values - peak) / peak
    max_drawdown = drawdowns.min() * 100

    daily_returns = np.diff(values) / values[:-1]
    sharpe = (np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252)
              if np.std(daily_returns) > 0 else 0.0)

    buy_trades = [t for t in trades if t["action"] == "buy"]
    sell_trades = [t for t in trades if t["action"] == "sell"]
    trade_pnl = []
    for j in range(min(len(buy_trades), len(sell_trades))):
        pnl = (sell_trades[j]["price"] - buy_trades[j]["price"]) * buy_trades[j]["shares"]
        trade_pnl.append(pnl)
    wins = [p for p in trade_pnl if p > 0]
    losses = [p for p in trade_pnl if p <= 0]
    win_rate = len(wins) / len(trade_pnl) * 100 if trade_pnl else 0
    profit_factor = sum(wins) / abs(sum(losses)) if losses and sum(losses) != 0 else float("inf")

    return {
        "total_return": total_return,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_drawdown,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "num_trades": len(trades),
        "final_value": final_value,
        "trades": trades,
    }


def main():
    parser = argparse.ArgumentParser(description="Compare K-Means vs LSTM vs LightGBM vs PPO")
    parser.add_argument("--csv", default="data/601933_10yr.csv", help="Input CSV")
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

    # --- PPO backtest ---
    print(f"\n{'=' * 60}")
    print("PPO BACKTEST")
    print("=" * 60)
    ppo_results = run_ppo_backtest(
        df, train_ratio=0.6, initial_capital=100_000,
    )

    # --- Majority vote backtest ---
    print(f"\n{'=' * 60}")
    print("MAJORITY VOTE BACKTEST")
    print("=" * 60)
    uv_results = run_majority_backtest(km_results, lstm_results, lgbm_results,
                                        ppo_results, initial_capital=100_000)

    # --- TD3 meta-judge backtest ---
    print(f"\n{'=' * 60}")
    print("TD3 META-JUDGE BACKTEST")
    print("=" * 60)
    td3_results = run_td3_backtest(df, train_ratio=0.6, initial_capital=100_000)

    # --- Comparison table ---
    print(f"\n{'=' * 115}")
    print("COMPARISON (with SMA5 buy filter)")
    print("=" * 115)

    bh_return = km_results["buy_and_hold_return"]
    header = (
        f"  {'Metric':<22s} {'K-Means':>12s} {'LSTM':>12s}"
        f" {'LightGBM':>12s} {'PPO':>12s} {'Majority':>12s}"
        f" {'TD3':>12s} {'Buy&Hold':>12s}"
    )
    print(header)
    print("  " + "-" * 113)

    rows = [
        ("Total Return",
         f"{km_results['total_return']:+.2f}%",
         f"{lstm_results['total_return']:+.2f}%",
         f"{lgbm_results['total_return']:+.2f}%",
         f"{ppo_results['total_return']:+.2f}%",
         f"{uv_results['total_return']:+.2f}%",
         f"{td3_results['total_return']:+.2f}%",
         f"{bh_return:+.2f}%"),
        ("Max Drawdown",
         f"{km_results['max_drawdown']:.2f}%",
         f"{lstm_results['max_drawdown']:.2f}%",
         f"{lgbm_results['max_drawdown']:.2f}%",
         f"{ppo_results['max_drawdown']:.2f}%",
         f"{uv_results['max_drawdown']:.2f}%",
         f"{td3_results['max_drawdown']:.2f}%",
         "N/A"),
        ("Sharpe Ratio",
         f"{km_results['sharpe_ratio']:.3f}",
         f"{lstm_results['sharpe_ratio']:.3f}",
         f"{lgbm_results['sharpe_ratio']:.3f}",
         f"{ppo_results['sharpe_ratio']:.3f}",
         f"{uv_results['sharpe_ratio']:.3f}",
         f"{td3_results['sharpe_ratio']:.3f}",
         "N/A"),
        ("Win Rate",
         f"{km_results['win_rate']:.1f}%",
         f"{lstm_results['win_rate']:.1f}%",
         f"{lgbm_results['win_rate']:.1f}%",
         f"{ppo_results['win_rate']:.1f}%",
         f"{uv_results['win_rate']:.1f}%",
         f"{td3_results['win_rate']:.1f}%",
         "N/A"),
        ("Profit Factor",
         f"{km_results['profit_factor']:.2f}",
         f"{lstm_results['profit_factor']:.2f}",
         f"{lgbm_results['profit_factor']:.2f}",
         f"{ppo_results['profit_factor']:.2f}",
         f"{uv_results['profit_factor']:.2f}",
         f"{td3_results['profit_factor']:.2f}",
         "N/A"),
        ("Num Trades",
         f"{km_results['num_trades']}",
         f"{lstm_results['num_trades']}",
         f"{lgbm_results['num_trades']}",
         f"{ppo_results['num_trades']}",
         f"{uv_results['num_trades']}",
         f"{td3_results['num_trades']}",
         "1"),
        ("Final Value",
         f"{km_results['final_value']:,.0f}",
         f"{lstm_results['final_value']:,.0f}",
         f"{lgbm_results['final_value']:,.0f}",
         f"{ppo_results['final_value']:,.0f}",
         f"{uv_results['final_value']:,.0f}",
         f"{td3_results['final_value']:,.0f}",
         "N/A"),
    ]
    for label, km_v, ls_v, lg_v, pp_v, uv_v, td_v, bh_v in rows:
        print(f"  {label:<22s} {km_v:>12s} {ls_v:>12s} {lg_v:>12s}"
              f" {pp_v:>12s} {uv_v:>12s} {td_v:>12s} {bh_v:>12s}")

    # --- Build combined daily trade log ---
    km_test = km_results["test_df"].copy()
    lstm_test = lstm_results["test_df"].copy()
    lgbm_test = lgbm_results["test_df"].copy()
    ppo_test = ppo_results["test_df"].copy()

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

    # Reconstruct PPO signals on the test set (one signal per row, like K-Means)
    ppo_bot = ppo_results["bot"]
    ppo_raw_signals = ppo_bot.predict(ppo_test)

    # Build trade lookup: date -> trade dict
    def _trade_lookup(trades):
        lookup = {}
        for t in trades:
            lookup[t["date"]] = t
        return lookup

    km_trades = _trade_lookup(km_results["trades"])
    lstm_trades = _trade_lookup(lstm_results["trades"])
    lgbm_trades = _trade_lookup(lgbm_results["trades"])
    ppo_trades = _trade_lookup(ppo_results["trades"])
    uv_trades = _trade_lookup(uv_results["trades"])
    td3_trades = _trade_lookup(td3_results["trades"])

    # TD3 test_df already has per-row signals from the meta-model
    td3_test = td3_results["test_df"].copy()
    td3_signal_by_date = {
        str(td3_test.iloc[i]["date"]): td3_test.iloc[i].get("signal", "hold")
        for i in range(len(td3_test))
    }

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

        ppo_signal = ppo_raw_signals[i] if i < len(ppo_raw_signals) else "hold"
        ppo_t = ppo_trades.get(date, {})
        ppo_action = ppo_t.get("action", "hold")
        ppo_shares = ppo_t.get("shares", 0)

        # Majority vote: consensus direction (>= 3 of 4)
        dirs = [_classify(km_signal), _classify(lstm_signal),
                _classify(lgbm_signal), _classify(ppo_signal)]
        uv_signal = _majority_direction(dirs) or "disagree"
        uv_t = uv_trades.get(date, {})
        uv_action = uv_t.get("action", "hold")
        uv_shares = uv_t.get("shares", 0)

        # TD3 meta-judge
        td3_signal = td3_signal_by_date.get(date, "hold")
        td3_t = td3_trades.get(date, {})
        td3_action = td3_t.get("action", "hold")
        td3_shares = td3_t.get("shares", 0)

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
            "ppo_signal": ppo_signal,
            "ppo_action": ppo_action,
            "ppo_shares": ppo_shares if ppo_shares else "",
            "uv_signal": uv_signal,
            "uv_action": uv_action,
            "uv_shares": uv_shares if uv_shares else "",
            "td3_signal": td3_signal,
            "td3_action": td3_action,
            "td3_shares": td3_shares if td3_shares else "",
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
    ppo_buys = len([r for r in log_rows if r["ppo_action"] == "buy"])
    ppo_sells = len([r for r in log_rows if r["ppo_action"] == "sell"])
    uv_buys = len([r for r in log_rows if r["uv_action"] == "buy"])
    uv_sells = len([r for r in log_rows if r["uv_action"] == "sell"])
    td3_buys = len([r for r in log_rows if r["td3_action"] == "buy"])
    td3_sells = len([r for r in log_rows if r["td3_action"] == "sell"])

    print(f"\n{'=' * 100}")
    print("TRADE LOG SUMMARY")
    print("=" * 100)
    print(f"  K-Means:    {km_buys} buys, {km_sells} sells")
    print(f"  LSTM:       {lstm_buys} buys, {lstm_sells} sells")
    print(f"  LightGBM:   {lgbm_buys} buys, {lgbm_sells} sells")
    print(f"  PPO:        {ppo_buys} buys, {ppo_sells} sells")
    print(f"  Majority:   {uv_buys} buys, {uv_sells} sells")
    print(f"  TD3:        {td3_buys} buys, {td3_sells} sells")
    print(f"\n  Saved {len(log_df)} rows to {args.output}")


if __name__ == "__main__":
    main()
