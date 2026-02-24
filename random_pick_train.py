#!/usr/bin/env python3
"""Randomly pick N Shanghai A-shares, train all models, and append results to a log.

Results are saved to data/random_picks_log.json — each run is one entry,
so results accumulate across multiple turns.

Usage:
    python random_pick_train.py             # pick 10 random stocks
    python random_pick_train.py --n 5       # pick 5 random stocks
    python random_pick_train.py --seed 42   # reproducible pick
"""

import argparse
import json
import os
import time
import traceback
import warnings
from datetime import datetime

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

import akshare as ak
import pandas as pd

from daily_pipeline import train_all, tune_all
from fetch_china_stock import fetch_stock_daily

LOG_PATH = "data/random_picks_log.json"
DATA_DIR = "data"
DATA_START = "20040101"
CAPITAL = 100_000
SCALAR_KEYS = [
    "total_return", "buy_and_hold_return", "sharpe_ratio",
    "max_drawdown", "win_rate", "profit_factor", "num_trades", "final_value",
]


# ---------------------------------------------------------------------------
# Stock picking
# ---------------------------------------------------------------------------

def pick_random_stocks(n: int, seed: int | None = None) -> list[dict]:
    """Fetch live A-share spot data and randomly sample n Shanghai stocks."""
    print("Fetching A-share spot data...")
    spot = ak.stock_zh_a_spot_em()
    col_map = {"代码": "symbol", "名称": "name", "成交量": "avg_volume"}
    spot.rename(columns=col_map, inplace=True)
    spot = spot[["symbol", "name", "avg_volume"]].copy()

    # Keep Shanghai main-board (starts with '6', not '688')
    sym = spot["symbol"].astype(str)
    spot = spot[sym.str.startswith("6") & ~sym.str.startswith("688")].copy()

    # Exclude ST / suspended (avg_volume == 0)
    spot = spot[~spot["name"].str.contains(r"ST|⭐", na=False)]
    spot = spot[spot["avg_volume"] > 0]
    spot = spot.reset_index(drop=True)

    print(f"  {len(spot)} eligible Shanghai A-shares")
    sample = spot.sample(n=min(n, len(spot)), random_state=seed)
    picks = sample[["symbol", "name"]].to_dict("records")
    print(f"  Randomly selected {len(picks)} stocks:")
    for p in picks:
        print(f"    {p['symbol']}  {p['name']}")
    return picks


# ---------------------------------------------------------------------------
# Log management
# ---------------------------------------------------------------------------

def load_log() -> list:
    if os.path.exists(LOG_PATH):
        with open(LOG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def save_log(log: list) -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    tmp = LOG_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)
    os.replace(tmp, LOG_PATH)


# ---------------------------------------------------------------------------
# Per-stock training
# ---------------------------------------------------------------------------

def _extract(res: dict) -> dict:
    return {k: res.get(k) for k in SCALAR_KEYS}


def run_stock(symbol: str, name: str) -> dict | None:
    csv_path = os.path.join(DATA_DIR, f"{symbol}_20yr.csv")
    print(f"\n  [{symbol}] Downloading...")
    df = fetch_stock_daily(symbol, start_date=DATA_START)
    if df.empty:
        print(f"  [{symbol}] No data — skipping")
        return None
    df.to_csv(csv_path, index=False)
    print(f"  [{symbol}] {len(df)} rows  ({df['date'].iloc[0]} → {df['date'].iloc[-1]})")

    ticker = {"symbol": symbol, "csv": csv_path, "capital": CAPITAL, "label": f"{symbol} {name}"}
    train_res = train_all(ticker)

    print(f"  [{symbol}] Tuning hyperparameters...")
    tune_res = tune_all(ticker)
    best = tune_res.get("best_params", {}) or {}

    return {
        "symbol": symbol,
        "name": name,
        "rows": len(df),
        "date_range": [str(df["date"].iloc[0]), str(df["date"].iloc[-1])],
        "km":       _extract(train_res["km"]),
        "lstm":     _extract(train_res["lstm"]),
        "lgbm":     _extract(train_res["lgbm"]),
        "ppo":      _extract(train_res["ppo"]),
        "majority": _extract(train_res["majority"]),
        "td3":      _extract(train_res["td3"]),
        "km_tuned":   _extract(tune_res["km_tuned"]),
        "lgbm_tuned": _extract(tune_res["lgbm_tuned"]),
        "ppo_tuned":  _extract(tune_res["ppo_tuned"]),
        "best_params": {
            "km":   {k: v for k, v in (best.get("km") or {}).items()},
            "lgbm": {k: v for k, v in (best.get("lgbm") or {}).items()},
            "ppo":  {k: v for k, v in (best.get("ppo") or {}).items()},
        },
    }


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_turn_summary(stocks: list[dict]) -> None:
    print(f"\n{'=' * 130}")
    print("TURN SUMMARY  (Tuned = best hyperparams from grid search)")
    print("=" * 130)
    hdr = (f"  {'Symbol':<8s}  {'Name':<12s}"
           f"  {'KM%':>7s}  {'KMt%':>7s}"
           f"  {'LSTM%':>7s}"
           f"  {'LGBM%':>7s}  {'LGBMt%':>7s}"
           f"  {'PPO%':>7s}  {'PPOt%':>7s}"
           f"  {'Maj%':>7s}  {'TD3%':>7s}  {'TD3♯':>6s}  {'B&H%':>7s}")
    print(hdr)
    print("  " + "-" * 128)
    for r in stocks:
        km = r["km"]; td3 = r["td3"]; lstm = r["lstm"]
        lgbm = r["lgbm"]; ppo = r["ppo"]; maj = r["majority"]
        kmt = r.get("km_tuned", {}); lgbmt = r.get("lgbm_tuned", {}); ppot = r.get("ppo_tuned", {})
        bh = km.get("buy_and_hold_return", float("nan"))
        print(
            f"  {r['symbol']:<8s}  {r['name'][:12]:<12s}"
            f"  {km.get('total_return', float('nan')):+7.1f}%"
            f"  {kmt.get('total_return', float('nan')):+7.1f}%"
            f"  {lstm.get('total_return', float('nan')):+7.1f}%"
            f"  {lgbm.get('total_return', float('nan')):+7.1f}%"
            f"  {lgbmt.get('total_return', float('nan')):+7.1f}%"
            f"  {ppo.get('total_return', float('nan')):+7.1f}%"
            f"  {ppot.get('total_return', float('nan')):+7.1f}%"
            f"  {maj.get('total_return', float('nan')):+7.1f}%"
            f"  {td3.get('total_return', float('nan')):+7.1f}%"
            f"  {td3.get('sharpe_ratio', float('nan')):6.3f}"
            f"  {bh:+7.1f}%"
        )


def print_cumulative_summary(log: list) -> None:
    print(f"\n{'=' * 80}")
    print(f"CUMULATIVE LOG  ({len(log)} turns, {sum(len(t['stocks']) for t in log)} stocks)")
    print("=" * 80)
    all_stocks = [s for t in log for s in t["stocks"]]
    # sort by td3 sharpe desc
    all_stocks.sort(key=lambda r: r.get("td3", {}).get("sharpe_ratio") or float("-inf"), reverse=True)
    print(f"  {'Symbol':<8s}  {'Name':<12s}  {'Turn':<6s}  {'KM%':>7s}  {'TD3%':>7s}  {'TD3♯':>6s}  {'B&H%':>7s}")
    print("  " + "-" * 68)
    for r in all_stocks:
        km = r["km"]; td3 = r["td3"]
        bh = km.get("buy_and_hold_return", float("nan"))
        turn = r.get("turn", "?")
        print(
            f"  {r['symbol']:<8s}  {r['name'][:12]:<12s}  {str(turn):<6s}"
            f"  {km.get('total_return', float('nan')):+7.1f}%"
            f"  {td3.get('total_return', float('nan')):+7.1f}%"
            f"  {td3.get('sharpe_ratio', float('nan')):6.3f}"
            f"  {bh:+7.1f}%"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Randomly pick and train N stocks")
    parser.add_argument("--n", type=int, default=10, help="Number of stocks to pick")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    args = parser.parse_args()

    t_start = time.time()
    log = load_log()
    turn_num = len(log) + 1

    print("=" * 76)
    print(f"RANDOM PICK TRAINING  —  Turn {turn_num}")
    print(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"  Picking {args.n} random Shanghai A-shares (seed={args.seed})")
    print("=" * 76)

    picks = pick_random_stocks(args.n, seed=args.seed)

    turn_stocks = []
    for pick in picks:
        symbol, name = pick["symbol"], pick["name"]
        t0 = time.time()
        try:
            result = run_stock(symbol, name)
            if result is None:
                continue
            result["turn"] = turn_num
            turn_stocks.append(result)
            elapsed = time.time() - t0
            print(f"  [{symbol}] Completed in {elapsed:.0f}s  "
                  f"km_sharpe={result['km'].get('sharpe_ratio', 0):.3f}  "
                  f"td3_sharpe={result['td3'].get('sharpe_ratio', 0):.3f}")
        except Exception as e:
            elapsed = time.time() - t0
            print(f"  [{symbol}] FAILED in {elapsed:.0f}s: {e}")
            traceback.print_exc()

    if turn_stocks:
        print_turn_summary(turn_stocks)

    # Append this turn to the log
    log.append({
        "turn": turn_num,
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "n_picked": args.n,
        "seed": args.seed,
        "stocks": turn_stocks,
    })
    save_log(log)
    print(f"\nResults appended → {LOG_PATH}")

    print_cumulative_summary(log)

    elapsed_total = time.time() - t_start
    print(f"\nTotal time: {elapsed_total:.0f}s ({elapsed_total/60:.1f} min)")


if __name__ == "__main__":
    main()
