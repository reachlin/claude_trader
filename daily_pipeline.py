#!/usr/bin/env python3
"""Daily pipeline: download latest data, train all models, tune hyperparameters,
and run consensus strategy across all tickers.

Usage:
    python daily_pipeline.py              # run all tickers
    python daily_pipeline.py --skip-tune  # skip tuning (faster, ~1 min vs ~10 min)
"""

import argparse
import os
import sys
import time
import warnings
from datetime import datetime, timedelta

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

import numpy as np
import pandas as pd

from fetch_china_stock import fetch_stock_daily
from trading_bot import (
    FEATURE_COLS,
    TradingBot,
    Portfolio,
    LOT_SIZE,
    compute_indicators,
    run_backtest,
)
from dnn_trading_bot import DNNTradingBot, run_dnn_backtest
from lgbm_trading_bot import LGBMTradingBot, run_lgbm_backtest
from ppo_trading_bot import PPOTradingBot, run_ppo_backtest
from td3_trading_bot import run_td3_backtest
from tune_hyperparams import tune_kmeans, tune_lgbm, tune_lstm, tune_ppo
from compare_models import run_majority_backtest
from range_predictor import RangePredictor


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
TICKERS = [
    {"symbol": "601933", "start": "20160101", "csv": "data/601933_10yr.csv",    "capital": 100_000, "label": "601933 Yonghui"},
    {"symbol": "000001.SH", "start": "20060101", "csv": "data/000001SH_20yr.csv", "capital": 1_000_000, "label": "000001.SH Shanghai Composite"},
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _last_trading_day() -> str:
    """Return today (or last weekday) in YYYYMMDD format."""
    today = datetime.now()
    # If weekend, go back to Friday
    while today.weekday() >= 5:
        today -= timedelta(days=1)
    return today.strftime("%Y%m%d")


def _print_comparison_table(results: dict, title: str):
    """Print a comparison table from a results dict."""
    print(f"\n{'=' * 117}")
    print(title)
    print("=" * 117)

    km, lstm, lgbm, ppo = results["km"], results["lstm"], results["lgbm"], results["ppo"]
    uv = results["majority"]
    td3 = results["td3"]
    bh = km["buy_and_hold_return"]

    header = (f"  {'Metric':<22s} {'K-Means':>12s} {'LSTM':>12s} {'LightGBM':>12s}"
              f" {'PPO':>12s} {'Majority':>12s} {'TD3':>12s} {'Buy&Hold':>12s}")
    print(header)
    print("  " + "-" * 115)

    rows = [
        ("Total Return",   f"{km['total_return']:+.2f}%",  f"{lstm['total_return']:+.2f}%",  f"{lgbm['total_return']:+.2f}%",  f"{ppo['total_return']:+.2f}%",  f"{uv['total_return']:+.2f}%",  f"{td3['total_return']:+.2f}%",  f"{bh:+.2f}%"),
        ("Sharpe Ratio",   f"{km['sharpe_ratio']:.3f}",    f"{lstm['sharpe_ratio']:.3f}",    f"{lgbm['sharpe_ratio']:.3f}",    f"{ppo['sharpe_ratio']:.3f}",    f"{uv['sharpe_ratio']:.3f}",    f"{td3['sharpe_ratio']:.3f}",    "N/A"),
        ("Max Drawdown",   f"{km['max_drawdown']:.2f}%",   f"{lstm['max_drawdown']:.2f}%",   f"{lgbm['max_drawdown']:.2f}%",   f"{ppo['max_drawdown']:.2f}%",   f"{uv['max_drawdown']:.2f}%",   f"{td3['max_drawdown']:.2f}%",   "N/A"),
        ("Win Rate",       f"{km['win_rate']:.1f}%",       f"{lstm['win_rate']:.1f}%",       f"{lgbm['win_rate']:.1f}%",       f"{ppo['win_rate']:.1f}%",       f"{uv['win_rate']:.1f}%",       f"{td3['win_rate']:.1f}%",       "N/A"),
        ("Profit Factor",  f"{km['profit_factor']:.2f}",   f"{lstm['profit_factor']:.2f}",   f"{lgbm['profit_factor']:.2f}",   f"{ppo['profit_factor']:.2f}",   f"{uv['profit_factor']:.2f}",   f"{td3['profit_factor']:.2f}",   "N/A"),
        ("Num Trades",     f"{km['num_trades']}",          f"{lstm['num_trades']}",          f"{lgbm['num_trades']}",          f"{ppo['num_trades']}",          f"{uv['num_trades']}",          f"{td3['num_trades']}",          "1"),
        ("Final Value",    f"{km['final_value']:,.0f}",    f"{lstm['final_value']:,.0f}",    f"{lgbm['final_value']:,.0f}",    f"{ppo['final_value']:,.0f}",    f"{uv['final_value']:,.0f}",    f"{td3['final_value']:,.0f}",    "N/A"),
    ]
    for label, *vals in rows:
        print(f"  {label:<22s}" + "".join(f" {v:>12s}" for v in vals))


def _print_tuning_table(results: dict, title: str, best_params: dict):
    """Print tuned vs original comparison."""
    print(f"\n{'=' * 134}")
    print(title)
    print("=" * 134)

    header = (
        f"  {'Metric':<20s}"
        f"  {'KM Orig':>12s}  {'KM Tuned':>12s}"
        f"  {'LGBM Orig':>12s}  {'LGBM Tuned':>12s}"
        f"  {'PPO Orig':>12s}  {'PPO Tuned':>12s}"
        f"  {'Buy&Hold':>10s}"
    )
    print(header)
    print("  " + "-" * 132)

    km_o, km_t = results["km_orig"], results["km_tuned"]
    lg_o, lg_t = results["lgbm_orig"], results["lgbm_tuned"]
    pp_o, pp_t = results["ppo_orig"], results["ppo_tuned"]
    bh = km_o["buy_and_hold_return"]

    rows = [
        ("Total Return",  f"{km_o['total_return']:+.2f}%", f"{km_t['total_return']:+.2f}%",
         f"{lg_o['total_return']:+.2f}%", f"{lg_t['total_return']:+.2f}%",
         f"{pp_o['total_return']:+.2f}%", f"{pp_t['total_return']:+.2f}%", f"{bh:+.2f}%"),
        ("Sharpe Ratio",  f"{km_o['sharpe_ratio']:.3f}", f"{km_t['sharpe_ratio']:.3f}",
         f"{lg_o['sharpe_ratio']:.3f}", f"{lg_t['sharpe_ratio']:.3f}",
         f"{pp_o['sharpe_ratio']:.3f}", f"{pp_t['sharpe_ratio']:.3f}", "N/A"),
        ("Max Drawdown",  f"{km_o['max_drawdown']:.2f}%", f"{km_t['max_drawdown']:.2f}%",
         f"{lg_o['max_drawdown']:.2f}%", f"{lg_t['max_drawdown']:.2f}%",
         f"{pp_o['max_drawdown']:.2f}%", f"{pp_t['max_drawdown']:.2f}%", "N/A"),
        ("Win Rate",      f"{km_o['win_rate']:.1f}%", f"{km_t['win_rate']:.1f}%",
         f"{lg_o['win_rate']:.1f}%", f"{lg_t['win_rate']:.1f}%",
         f"{pp_o['win_rate']:.1f}%", f"{pp_t['win_rate']:.1f}%", "N/A"),
        ("Num Trades",    f"{km_o['num_trades']}", f"{km_t['num_trades']}",
         f"{lg_o['num_trades']}", f"{lg_t['num_trades']}",
         f"{pp_o['num_trades']}", f"{pp_t['num_trades']}", "1"),
        ("Final Value",   f"{km_o['final_value']:,.0f}", f"{km_t['final_value']:,.0f}",
         f"{lg_o['final_value']:,.0f}", f"{lg_t['final_value']:,.0f}",
         f"{pp_o['final_value']:,.0f}", f"{pp_t['final_value']:,.0f}", "N/A"),
    ]
    for label, *vals in rows:
        print(f"  {label:<20s}" + "".join(f"  {v:>12s}" for v in vals))

    print(f"\n  Best K-Means:  n_clusters={best_params['km']['n_clusters']}, "
          f"features={best_params['km'].get('feature_subset', 'all_6')}")
    print(f"  Best LightGBM: n_est={best_params['lgbm']['n_estimators']}, "
          f"md={best_params['lgbm']['max_depth']}, lr={best_params['lgbm']['learning_rate']}")
    if "ppo" in best_params:
        print(f"  Best PPO:      ts={best_params['ppo']['total_timesteps']}, "
              f"lr={best_params['ppo']['learning_rate']}, "
              f"ent={best_params['ppo']['ent_coef']}, ns={best_params['ppo']['n_steps']}")


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------
def _download_with_retry(ticker: dict, end_date: str, retries: int = 3, backoff: int = 15) -> bool:
    """Download ticker data, retrying on transient errors.

    Returns True if data is available (fresh download or existing CSV),
    False if the download failed and no usable CSV exists.
    """
    symbol, start = ticker["symbol"], ticker["start"]
    csv, label = ticker["csv"], ticker["label"]

    os.makedirs(os.path.dirname(csv) if os.path.dirname(csv) else ".", exist_ok=True)

    for attempt in range(1, retries + 1):
        try:
            df = fetch_stock_daily(symbol, start_date=start, end_date=end_date)
            df.to_csv(csv, index=False)
            print(f"  {label:<35s}  {len(df):>5d} rows  "
                  f"({df['date'].iloc[0]} to {df['date'].iloc[-1]})")
            return True
        except Exception as e:
            if attempt < retries:
                wait = backoff * attempt
                print(f"  {label:<35s}  attempt {attempt}/{retries} FAILED: {e}  "
                      f"(retrying in {wait}s)")
                time.sleep(wait)
            else:
                if os.path.exists(csv):
                    print(f"  {label:<35s}  download FAILED after {retries} attempts: {e}")
                    print(f"  {label:<35s}  WARNING: using existing (possibly stale) CSV")
                    return True
                print(f"  {label:<35s}  FAILED after {retries} attempts: {e}")
                return False


def download_all(end_date: str) -> set:
    """Download/update data for all tickers. Returns set of failed symbols."""
    print("=" * 76)
    print(f"DOWNLOADING DATA (up to {end_date})")
    print("=" * 76)

    failed = set()
    for t in TICKERS:
        if not _download_with_retry(t, end_date):
            failed.add(t["symbol"])
    return failed


# ---------------------------------------------------------------------------
# Train all models
# ---------------------------------------------------------------------------
def train_all(ticker: dict) -> dict:
    """Run all 4 backtests for a ticker. Returns results dict."""
    csv, cap, label = ticker["csv"], ticker["capital"], ticker["label"]
    df = pd.read_csv(csv)

    print(f"\n{'=' * 76}")
    print(f"TRAINING: {label} ({len(df)} rows, capital={cap:,})")
    print("=" * 76)

    km = run_backtest(df, train_ratio=0.6, initial_capital=cap)
    print(f"  K-Means:  return={km['total_return']:+.2f}%  sharpe={km['sharpe_ratio']:.3f}  trades={km['num_trades']}")

    lstm = run_dnn_backtest(df, train_ratio=0.6, initial_capital=cap,
                            window_size=20, epochs=50, batch_size=32, lr=0.001)
    print(f"  LSTM:     return={lstm['total_return']:+.2f}%  sharpe={lstm['sharpe_ratio']:.3f}  trades={lstm['num_trades']}")

    lgbm = run_lgbm_backtest(df, train_ratio=0.6, initial_capital=cap)
    print(f"  LightGBM: return={lgbm['total_return']:+.2f}%  sharpe={lgbm['sharpe_ratio']:.3f}  trades={lgbm['num_trades']}")

    ppo = run_ppo_backtest(df, train_ratio=0.6, initial_capital=cap)
    print(f"  PPO:      return={ppo['total_return']:+.2f}%  sharpe={ppo['sharpe_ratio']:.3f}  trades={ppo['num_trades']}")

    uv = run_majority_backtest(km, lstm, lgbm, ppo, initial_capital=cap)
    print(f"  Majority:  return={uv['total_return']:+.2f}%  sharpe={uv['sharpe_ratio']:.3f}  trades={uv['num_trades']}")

    td3 = run_td3_backtest(df, train_ratio=0.6, initial_capital=cap)
    print(f"  TD3:       return={td3['total_return']:+.2f}%  sharpe={td3['sharpe_ratio']:.3f}  trades={td3['num_trades']}")

    return {"km": km, "lstm": lstm, "lgbm": lgbm, "ppo": ppo, "majority": uv, "td3": td3, "df": df, "capital": cap}


# ---------------------------------------------------------------------------
# Tune hyperparameters (K-Means + LightGBM only — LSTM too slow for routine use)
# ---------------------------------------------------------------------------
def tune_all(ticker: dict) -> dict:
    """Tune K-Means, LightGBM, and PPO for a ticker."""
    csv, cap, label = ticker["csv"], ticker["capital"], ticker["label"]
    df = pd.read_csv(csv)

    print(f"\n{'=' * 76}")
    print(f"TUNING: {label}")
    print("=" * 76)

    # Tune K-Means
    km_results = tune_kmeans(df, train_ratio=0.6, top_k=3, initial_capital=cap)
    best_km = km_results[0]["params"]
    print(f"\n  Best K-Means: n={best_km['n_clusters']}, feat={best_km.get('feature_subset','all_6')}"
          f"  val_sharpe={km_results[0]['sharpe_ratio']:.3f}")

    # Tune LightGBM
    lgbm_results = tune_lgbm(df, train_ratio=0.6, top_k=3, initial_capital=cap)
    best_lgbm = lgbm_results[0]["params"]
    print(f"  Best LightGBM: n_est={best_lgbm['n_estimators']}, md={best_lgbm['max_depth']}, "
          f"lr={best_lgbm['learning_rate']}  val_sharpe={lgbm_results[0]['sharpe_ratio']:.3f}")

    # Tune PPO
    ppo_results = tune_ppo(df, train_ratio=0.6, top_k=3, initial_capital=cap)
    best_ppo = ppo_results[0]["params"] if ppo_results else {}
    if best_ppo:
        print(f"  Best PPO: ts={best_ppo['total_timesteps']}, lr={best_ppo['learning_rate']}, "
              f"ent={best_ppo['ent_coef']}, ns={best_ppo['n_steps']}"
              f"  val_sharpe={ppo_results[0]['sharpe_ratio']:.3f}")

    # Final eval: original vs tuned on test set
    km_orig = run_backtest(df, train_ratio=0.6, initial_capital=cap)
    km_tuned = run_backtest(df, train_ratio=0.6, initial_capital=cap,
                            n_clusters=best_km["n_clusters"],
                            feature_cols=best_km.get("feature_cols"))
    lgbm_orig = run_lgbm_backtest(df, train_ratio=0.6, initial_capital=cap)
    lgbm_tuned = run_lgbm_backtest(df, train_ratio=0.6, initial_capital=cap,
                                    n_estimators=best_lgbm["n_estimators"],
                                    max_depth=best_lgbm["max_depth"],
                                    learning_rate=best_lgbm["learning_rate"])
    ppo_orig = run_ppo_backtest(df, train_ratio=0.6, initial_capital=cap)
    if best_ppo:
        ppo_tuned = run_ppo_backtest(df, train_ratio=0.6, initial_capital=cap,
                                      total_timesteps=best_ppo["total_timesteps"],
                                      learning_rate=best_ppo["learning_rate"],
                                      ent_coef=best_ppo["ent_coef"],
                                      n_steps=best_ppo["n_steps"])
    else:
        ppo_tuned = ppo_orig

    return {
        "km_orig": km_orig, "km_tuned": km_tuned,
        "lgbm_orig": lgbm_orig, "lgbm_tuned": lgbm_tuned,
        "ppo_orig": ppo_orig, "ppo_tuned": ppo_tuned,
        "best_params": {"km": best_km, "lgbm": best_lgbm, "ppo": best_ppo},
    }


# ---------------------------------------------------------------------------
# Consensus strategy
# ---------------------------------------------------------------------------
def run_consensus(ticker: dict) -> dict:
    """Run consensus strategy where all 4 models must agree."""
    csv, cap, label = ticker["csv"], ticker["capital"], ticker["label"]
    df_raw = pd.read_csv(csv)

    df = compute_indicators(df_raw)
    df = df.dropna(subset=FEATURE_COLS).reset_index(drop=True)
    split = int(len(df) * 0.6)
    train_df = df.iloc[:split].copy().reset_index(drop=True)
    test_df = df.iloc[split:].copy().reset_index(drop=True)

    # Train all 4 models
    km_bot = TradingBot(n_clusters=5)
    km_bot.fit(train_df)
    km_signals = km_bot.predict(test_df)

    lstm_bot = DNNTradingBot(window_size=20, epochs=50, batch_size=32, lr=0.001)
    lstm_bot.fit(train_df)
    lstm_raw = lstm_bot.predict(test_df)
    lstm_signal_map = {}
    ws = lstm_bot.window_size
    for i, sig in enumerate(lstm_raw):
        row_idx = i + ws
        if row_idx < len(test_df):
            lstm_signal_map[row_idx] = sig

    lgbm_bot = LGBMTradingBot()
    lgbm_bot.fit(train_df)
    lgbm_signals = lgbm_bot.predict(test_df)

    ppo_bot = PPOTradingBot()
    ppo_bot.fit(train_df)
    ppo_signals = ppo_bot.predict(test_df)

    def classify(signal):
        if signal in ("strong_buy", "mild_buy"):
            return "buy"
        elif signal in ("strong_sell", "mild_sell"):
            return "sell"
        return "hold"

    # Simulate
    portfolio = Portfolio(capital=cap)
    trades = []
    daily_values = []
    agree_buy = agree_sell = agree_hold = disagree = 0

    for i in range(len(test_df) - 1):
        km_sig = km_signals[i]
        lstm_sig = lstm_signal_map.get(i, "hold")
        lgbm_sig = lgbm_signals[i]
        ppo_sig = ppo_signals[i]

        km_dir = classify(km_sig)
        lstm_dir = classify(lstm_sig)
        lgbm_dir = classify(lgbm_sig)
        ppo_dir = classify(ppo_sig)

        exec_price = test_df.loc[i + 1, "open"]
        trade_date = str(test_df.loc[i + 1, "date"])
        price_below_sma5 = test_df.loc[i, "close"] < test_df.loc[i, "sma5"]

        shares_traded = 0
        action = "hold"

        if km_dir == lstm_dir == lgbm_dir == ppo_dir:
            if km_dir == "buy":
                agree_buy += 1
                if price_below_sma5:
                    strong = any(s == "strong_buy" for s in [km_sig, lstm_sig, lgbm_sig, ppo_sig])
                    frac = 1.0 if strong else 0.5
                    shares_traded = portfolio.buy(exec_price, fraction=frac, trade_date=trade_date)
                    if shares_traded > 0:
                        action = "buy"
            elif km_dir == "sell":
                agree_sell += 1
                strong = any(s == "strong_sell" for s in [km_sig, lstm_sig, lgbm_sig, ppo_sig])
                frac = 1.0 if strong else 0.5
                shares_traded = portfolio.sell(exec_price, fraction=frac, trade_date=trade_date)
                if shares_traded > 0:
                    action = "sell"
            else:
                agree_hold += 1
        else:
            disagree += 1

        if shares_traded > 0:
            trades.append({
                "date": trade_date, "action": action, "price": exec_price,
                "shares": shares_traded, "km": km_sig, "lstm": lstm_sig,
                "lgbm": lgbm_sig, "ppo": ppo_sig,
            })

        daily_values.append(portfolio.value(test_df.loc[i + 1, "close"]))

    # Metrics
    final_price = test_df.iloc[-1]["close"]
    final_value = portfolio.value(final_price)
    total_return = (final_value - cap) / cap * 100

    bh_shares = int(cap / test_df.iloc[0]["open"] // LOT_SIZE) * LOT_SIZE
    bh_cost = bh_shares * test_df.iloc[0]["open"]
    bh_value = bh_shares * final_price + (cap - bh_cost)
    bh_return = (bh_value - cap) / cap * 100

    values = np.array([cap] + daily_values)
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

    total_days = agree_buy + agree_sell + agree_hold + disagree

    return {
        "total_return": total_return,
        "buy_and_hold_return": bh_return,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_drawdown,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "num_trades": len(trades),
        "final_value": final_value,
        "trades": trades,
        "agree_buy": agree_buy,
        "agree_sell": agree_sell,
        "agree_hold": agree_hold,
        "disagree": disagree,
        "total_days": total_days,
    }


def print_consensus(result: dict, label: str):
    """Print consensus strategy results."""
    print(f"\n{'=' * 76}")
    print(f"CONSENSUS STRATEGY: {label}")
    print("=" * 76)

    td = result["total_days"]
    ag = result["agree_buy"] + result["agree_sell"] + result["agree_hold"]
    print(f"\n  Agreement: {ag}/{td} days ({ag/td*100:.1f}%)")
    print(f"    agree buy:  {result['agree_buy']:4d}    agree sell: {result['agree_sell']:4d}"
          f"    agree hold: {result['agree_hold']:4d}    disagree: {result['disagree']:4d}")

    print(f"\n  {'Metric':<22s} {'Value':>12s}")
    print("  " + "-" * 34)
    print(f"  {'Total Return':<22s} {result['total_return']:+.2f}%".rjust(0))
    print(f"  {'Buy & Hold':<22s} {result['buy_and_hold_return']:+.2f}%")
    print(f"  {'Sharpe Ratio':<22s} {result['sharpe_ratio']:.3f}")
    print(f"  {'Max Drawdown':<22s} {result['max_drawdown']:.2f}%")
    print(f"  {'Win Rate':<22s} {result['win_rate']:.1f}%")
    print(f"  {'Profit Factor':<22s} {result['profit_factor']:.2f}")
    print(f"  {'Num Trades':<22s} {result['num_trades']}")
    print(f"  {'Final Value':<22s} {result['final_value']:,.0f}")

    trades = result["trades"]
    if trades:
        print(f"\n  Recent consensus trades (last 10):")
        for t in trades[-10:]:
            print(f"    {t['date']}  {t['action']:4s}  {t['shares']:>6d} @ {t['price']:.2f}"
                  f"  [km={t['km']}, lstm={t['lstm']}, lgbm={t['lgbm']}, ppo={t['ppo']}]")


# ---------------------------------------------------------------------------
# Price prediction (BiLSTM range predictor)
# ---------------------------------------------------------------------------
_RANGE_PREDICTOR_CONFIG = dict(
    hidden=128, num_layers=4, fc_sizes=[256, 128, 64],
    layer_norm=True, use_attention=True,
    epochs=150, patience=20, batch_size=64,
)


def run_price_prediction(ticker: dict) -> dict:
    """Train RangePredictor on all data and predict next-day (low, high).

    Also runs a 70/30 walk-forward backtest to report prediction quality.
    Returns a dict with prediction and scoring keys.
    """
    csv, label = ticker["csv"], ticker["label"]
    df_raw = pd.read_csv(csv)
    df = compute_indicators(df_raw).dropna(subset=FEATURE_COLS).reset_index(drop=True)

    # --- Backtest: train on 70%, evaluate on 30% ---
    split = int(len(df) * 0.7)
    train_df = df.iloc[:split].copy().reset_index(drop=True)
    test_df = df.iloc[split:].copy().reset_index(drop=True)

    bt_predictor = RangePredictor(**_RANGE_PREDICTOR_CONFIG)
    bt_predictor.fit(train_df)
    scores = bt_predictor.evaluate_score(test_df)

    # --- Full retrain on all data for next-day prediction ---
    full_predictor = RangePredictor(**_RANGE_PREDICTOR_CONFIG)
    full_predictor.fit(df)
    pred_low, pred_high = full_predictor.predict_single(df)

    last = df.iloc[-1]
    n = scores["n_predictions"]
    return {
        "pred_low": pred_low,
        "pred_high": pred_high,
        "last_date": str(last["date"]),
        "last_close": float(last["close"]),
        "score_per_pred": scores["total_score"] / n if n > 0 else 0.0,
        "n_predictions": n,
        "plus_two": scores["plus_two"],
        "plus_one": scores["plus_one"],
        "zero": scores["zero"],
        "minus_one": scores["minus_one"],
    }


def print_price_prediction(result: dict, label: str):
    """Print next-day price range prediction and backtest quality."""
    print(f"\n{'=' * 76}")
    print(f"PRICE PREDICTION: {label}")
    print("=" * 76)

    close = result["last_close"]
    low, high = result["pred_low"], result["pred_high"]
    chg_low = (low - close) / close * 100
    chg_high = (high - close) / close * 100
    n = result["n_predictions"]

    print(f"\n  Last date  : {result['last_date']}")
    print(f"  Last close : {close:.4f}")
    print(f"  Pred low   : {low:.4f}  ({chg_low:+.2f}%)")
    print(f"  Pred high  : {high:.4f}  ({chg_high:+.2f}%)")
    print(f"  Pred range : {high - low:.4f}")

    if n > 0:
        print(f"\n  Backtest quality ({n} predictions, 30% holdout):")
        print(f"    +2 (both match)  : {result['plus_two']:4d}  ({result['plus_two']/n*100:.1f}%)")
        print(f"    +1 (one match)   : {result['plus_one']:4d}  ({result['plus_one']/n*100:.1f}%)")
        print(f"     0 (no match)    : {result['zero']:4d}  ({result['zero']/n*100:.1f}%)")
        print(f"    -1 (wrong side)  : {result['minus_one']:4d}  ({result['minus_one']/n*100:.1f}%)")
        print(f"    Score / pred     : {result['score_per_pred']:+.3f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Re-download, train, tune, and compare all models")
    parser.add_argument("--skip-tune", action="store_true", help="Skip hyperparameter tuning")
    parser.add_argument(
        "--ticker", metavar="SYMBOL",
        help="Run on a single ticker (e.g. 002142). Downloads data to data/<SYMBOL>_20yr.csv. "
             "If omitted, uses the hardcoded TICKERS list.",
    )
    parser.add_argument(
        "--capital", type=int, default=100_000,
        help="Initial capital when using --ticker (default: 100000)",
    )
    args = parser.parse_args()

    t_start = time.time()
    end_date = _last_trading_day()

    # Build the ticker list: custom symbol or hardcoded defaults
    if args.ticker:
        import os
        sym = args.ticker
        csv_path = os.path.join("data", f"{sym}_20yr.csv")
        os.makedirs("data", exist_ok=True)
        tickers = [{
            "symbol": sym,
            "start": "20040101",
            "csv": csv_path,
            "capital": args.capital,
            "label": sym,
        }]
    else:
        tickers = TICKERS

    # --- Step 1: Download ---
    print("=" * 76)
    print(f"DOWNLOADING DATA (up to {end_date})")
    print("=" * 76)
    failed_downloads = set()
    for ticker in tickers:
        if not _download_with_retry(ticker, end_date):
            failed_downloads.add(ticker["symbol"])

    # Skip tickers whose data could not be downloaded
    tickers = [t for t in tickers if t["symbol"] not in failed_downloads]
    if not tickers:
        print("\nNo tickers available after download failures. Exiting.")
        sys.exit(1)

    # --- Step 2: Train all models on each ticker ---
    all_results = {}
    for ticker in tickers:
        results = train_all(ticker)
        all_results[ticker["symbol"]] = results
        _print_comparison_table(results, f"RESULTS: {ticker['label']}")

    # --- Step 3: Tune hyperparameters ---
    tune_results = {}
    if not args.skip_tune:
        for ticker in tickers:
            tr = tune_all(ticker)
            tune_results[ticker["symbol"]] = tr
            _print_tuning_table(tr, f"TUNING: {ticker['label']}", tr["best_params"])
    else:
        print(f"\n{'=' * 76}")
        print("TUNING: Skipped (--skip-tune)")
        print("=" * 76)

    # --- Step 4: Consensus strategy on each ticker ---
    for ticker in tickers:
        print(f"\nTraining consensus for {ticker['label']}...")
        cons = run_consensus(ticker)
        print_consensus(cons, ticker["label"])

    # --- Step 5: Today's signals ---
    print(f"\n{'=' * 76}")
    print("TODAY'S SIGNALS")
    print("=" * 76)

    for ticker in tickers:
        symbol = ticker["symbol"]
        csv, label = ticker["csv"], ticker["label"]
        df_raw = pd.read_csv(csv)
        df = compute_indicators(df_raw).dropna(subset=FEATURE_COLS).reset_index(drop=True)

        # Train on all data for latest signal
        km_bot = TradingBot(n_clusters=5)
        km_bot.fit(df)
        km_sig = km_bot.predict_single(df.iloc[-1])

        lgbm_bot = LGBMTradingBot()
        lgbm_bot.fit(df)
        lgbm_sig = lgbm_bot.predict_single(df.iloc[-1])

        ppo_bot = PPOTradingBot()
        ppo_bot.fit(df)
        ppo_sig = ppo_bot.predict_single(df.iloc[-1])

        # TD3: use the pre-trained bot from train_all(); the test_df already
        # has all 4 base model signal columns baked in for the test period.
        td3_results = all_results[symbol]["td3"]
        td3_bot = td3_results["bot"]
        td3_last_row = td3_results["test_df"].iloc[-1]
        td3_sig = td3_bot.predict_single(td3_last_row)

        latest = df.iloc[-1]
        date = latest["date"]
        close = latest["close"]

        def classify(s):
            if s in ("strong_buy", "mild_buy"):
                return "BUY"
            elif s in ("strong_sell", "mild_sell"):
                return "SELL"
            return "HOLD"

        km_dir = classify(km_sig)
        lgbm_dir = classify(lgbm_sig)
        ppo_dir = classify(ppo_sig)
        td3_dir = classify(td3_sig)
        dirs = [km_dir, lgbm_dir, ppo_dir, td3_dir]

        # Majority vote: >= 3 of 4 agree
        from collections import Counter
        counts = Counter(dirs)
        majority = next((d for d, c in counts.items() if c >= 3), None)
        verdict = majority if majority else "NO CONSENSUS"

        print(f"\n  {label} ({date}, close={close:.2f}):")
        print(f"    K-Means:  {km_sig:<14s} -> {km_dir}")
        print(f"    LightGBM: {lgbm_sig:<14s} -> {lgbm_dir}")
        print(f"    PPO:      {ppo_sig:<14s} -> {ppo_dir}")
        print(f"    TD3:      {td3_sig:<14s} -> {td3_dir}")
        print(f"    Majority vote (>= 3/4): {verdict}")

    # --- Step 6: Price range prediction ---
    for ticker in tickers:
        print(f"\nTraining price predictor for {ticker['label']}...")
        try:
            pred_result = run_price_prediction(ticker)
            print_price_prediction(pred_result, ticker["label"])
        except Exception as e:
            print(f"  Price prediction failed for {ticker['label']}: {e}")

    elapsed = time.time() - t_start
    print(f"\n{'=' * 76}")
    print(f"Total time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print("=" * 76)

    if failed_downloads:
        print(f"\nWARNING: {len(failed_downloads)} ticker(s) failed to download: "
              f"{', '.join(sorted(failed_downloads))}")
        sys.exit(1)


if __name__ == "__main__":
    main()
