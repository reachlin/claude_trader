#!/usr/bin/env python3
"""Bulk train all 5 models on top 1000 Shanghai A-share candidates.

Reads candidates.csv (from find_candidates.py), downloads 20yr price history,
runs the full pipeline (K-Means, LSTM, LightGBM, PPO, TD3), and saves results
to bulk_checkpoint.json with resume capability.

Usage:
    python bulk_train.py                                  # uses candidates.csv
    python bulk_train.py --candidates my_list.csv         # custom candidate file
    python bulk_train.py --checkpoint my_checkpoint.json  # custom checkpoint file
    python bulk_train.py --batch-size 5                   # smaller batches
    python bulk_train.py --start-from 50                  # skip first N stocks
"""

import argparse
import json
import os
import time
import traceback
from datetime import datetime

import pandas as pd

from daily_pipeline import train_all, tune_all
from fetch_china_stock import fetch_stock_daily


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_CANDIDATES = "candidates.csv"
DEFAULT_CHECKPOINT = "bulk_checkpoint.json"
DEFAULT_DATA_DIR = "data"
DEFAULT_BATCH_SIZE = 10
DEFAULT_CAPITAL = 100_000
DATA_START_DATE = "20040101"

# Scalar keys extracted from each model's backtest result dict
SCALAR_METRIC_KEYS = [
    "total_return",
    "buy_and_hold_return",
    "sharpe_ratio",
    "max_drawdown",
    "win_rate",
    "profit_factor",
    "num_trades",
    "final_value",
]


# ---------------------------------------------------------------------------
# Checkpoint management
# ---------------------------------------------------------------------------

def create_checkpoint(total: int) -> dict:
    """Create a fresh checkpoint structure."""
    return {
        "metadata": {
            "total": total,
            "completed": 0,
            "failed": [],
            "started_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
        },
        "results": {},
    }


def load_checkpoint(path: str, total: int) -> dict:
    """Load checkpoint from file; create fresh one if file does not exist."""
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return create_checkpoint(total)


def save_checkpoint(cp: dict, path: str) -> None:
    """Save checkpoint to JSON (atomic write via temp file)."""
    cp["metadata"]["last_updated"] = datetime.now().isoformat()
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(cp, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


def is_completed(cp: dict, symbol: str) -> bool:
    """Return True if symbol already has results in the checkpoint."""
    return symbol in cp["results"]


# ---------------------------------------------------------------------------
# Batch splitting
# ---------------------------------------------------------------------------

def split_batches(symbols: list, batch_size: int) -> list:
    """Split a flat list of symbols into consecutive batches."""
    if not symbols:
        return []
    return [symbols[i: i + batch_size] for i in range(0, len(symbols), batch_size)]


# ---------------------------------------------------------------------------
# Result serialization
# ---------------------------------------------------------------------------

def _extract_metrics(model_result: dict) -> dict:
    """Extract only scalar metric keys from a model backtest result dict."""
    return {k: model_result[k] for k in SCALAR_METRIC_KEYS if k in model_result}


def serialize_results(
    symbol: str,
    name: str,
    sector: str,
    df: pd.DataFrame,
    train_res: dict,
    tune_res: dict,
) -> dict:
    """Serialize training results to a JSON-safe dict (scalars only).

    Strips non-serializable objects such as DataFrames, bot objects, and
    trade log lists from the raw train_all / tune_all return values.

    Args:
        symbol: Stock ticker code
        name: Stock name
        sector: Industry sector
        df: Price DataFrame (used for row count and date range)
        train_res: Return value of train_all()
        tune_res: Return value of tune_all()

    Returns:
        Flat dict of scalars and nested dicts of scalars — JSON-safe.
    """
    best_params = tune_res.get("best_params", {}) or {}
    km_p = best_params.get("km", {}) or {}
    lgbm_p = best_params.get("lgbm", {}) or {}
    ppo_p = best_params.get("ppo", {}) or {}

    return {
        "symbol": symbol,
        "name": name,
        "sector": sector,
        "rows": len(df),
        "date_range": [str(df["date"].iloc[0]), str(df["date"].iloc[-1])],
        "km": _extract_metrics(train_res["km"]),
        "lstm": _extract_metrics(train_res["lstm"]),
        "lgbm": _extract_metrics(train_res["lgbm"]),
        "ppo": _extract_metrics(train_res["ppo"]),
        "majority": _extract_metrics(train_res["majority"]),
        "td3": _extract_metrics(train_res["td3"]),
        "best_params": {
            "km": {
                "n_clusters": km_p.get("n_clusters"),
                "feature_subset": km_p.get("feature_subset", "all_6"),
            },
            "lgbm": {
                "n_estimators": lgbm_p.get("n_estimators"),
                "max_depth": lgbm_p.get("max_depth"),
                "learning_rate": lgbm_p.get("learning_rate"),
            },
            "ppo": {
                "total_timesteps": ppo_p.get("total_timesteps"),
                "learning_rate": ppo_p.get("learning_rate"),
                "ent_coef": ppo_p.get("ent_coef"),
                "n_steps": ppo_p.get("n_steps"),
            } if ppo_p else {},
        },
    }


# ---------------------------------------------------------------------------
# Per-stock processing
# ---------------------------------------------------------------------------

def process_stock(
    symbol: str,
    name: str,
    sector: str,
    cp_path: str,
    cp: dict,
    data_dir: str = DEFAULT_DATA_DIR,
    capital: int = DEFAULT_CAPITAL,
) -> dict:
    """Download data and run full training pipeline for one stock.

    Saves checkpoint after successful completion. Raises on failure.

    Args:
        symbol: Stock ticker code (e.g. "601933")
        name: Stock name
        sector: Industry sector
        cp_path: Path to checkpoint JSON file
        cp: In-memory checkpoint dict (mutated in-place)
        data_dir: Directory to store per-stock CSVs
        capital: Initial capital for backtesting

    Returns:
        Serialized result dict for this stock.
    """
    os.makedirs(data_dir, exist_ok=True)
    csv_path = os.path.join(data_dir, f"{symbol}_20yr.csv")

    # 1. Download 20-year price history
    print(f"  [{symbol}] Downloading price history...")
    df = fetch_stock_daily(symbol, start_date=DATA_START_DATE)
    if df.empty:
        raise ValueError(f"No price data returned for {symbol}")
    df.to_csv(csv_path, index=False)
    print(f"  [{symbol}] {len(df)} rows  "
          f"({df['date'].iloc[0]} → {df['date'].iloc[-1]})")

    # 2. Build ticker dict for daily_pipeline functions
    ticker = {
        "symbol": symbol,
        "csv": csv_path,
        "capital": capital,
        "label": f"{symbol} {name}",
    }

    # 3. Train all 5 models
    print(f"  [{symbol}] Training all models...")
    train_res = train_all(ticker)
    train_df = train_res["df"]

    # 4. Tune hyperparameters (K-Means, LightGBM, PPO)
    print(f"  [{symbol}] Tuning hyperparameters...")
    tune_res = tune_all(ticker)

    # 5. Serialize and checkpoint
    result = serialize_results(symbol, name, sector, train_df, train_res, tune_res)
    cp["results"][symbol] = result
    cp["metadata"]["completed"] = len(cp["results"])
    save_checkpoint(cp, cp_path)

    return result


# ---------------------------------------------------------------------------
# Summary printing
# ---------------------------------------------------------------------------

def _print_batch_summary(batch_results: list) -> None:
    if not batch_results:
        return
    print(f"\n  {'Symbol':<10s}  {'Name':<14s}  {'Sector':<14s}  "
          f"{'KM Ret%':>8s}  {'KM Sharpe':>9s}  {'TD3 Ret%':>8s}  {'TD3 Sharpe':>10s}")
    print("  " + "-" * 82)
    for r in batch_results:
        km = r.get("km", {})
        td3 = r.get("td3", {})
        print(f"  {r['symbol']:<10s}  {r['name'][:14]:<14s}  {r['sector'][:14]:<14s}  "
              f"  {km.get('total_return', float('nan')):+7.1f}%"
              f"  {km.get('sharpe_ratio', float('nan')):>9.3f}"
              f"  {td3.get('total_return', float('nan')):+7.1f}%"
              f"  {td3.get('sharpe_ratio', float('nan')):>10.3f}")


def _print_final_summary(cp: dict) -> None:
    results = list(cp["results"].values())
    if not results:
        print("No results to summarize.")
        return

    # Sort by TD3 Sharpe (descending)
    results.sort(
        key=lambda r: r.get("td3", {}).get("sharpe_ratio", float("-inf")),
        reverse=True,
    )

    print(f"\n{'=' * 90}")
    print("TOP 20 OVERALL  (by TD3 Sharpe Ratio)")
    print("=" * 90)
    print(f"  {'Rank':<5s}  {'Symbol':<10s}  {'Name':<14s}  {'Sector':<14s}  "
          f"{'KM Ret%':>8s}  {'TD3 Ret%':>8s}  {'TD3 Sharpe':>10s}")
    print("  " + "-" * 78)
    for rank, r in enumerate(results[:20], 1):
        km = r.get("km", {})
        td3 = r.get("td3", {})
        print(f"  {rank:<5d}  {r['symbol']:<10s}  {r['name'][:14]:<14s}  "
              f"{r['sector'][:14]:<14s}  "
              f"  {km.get('total_return', float('nan')):+7.1f}%"
              f"  {td3.get('total_return', float('nan')):+7.1f}%"
              f"  {td3.get('sharpe_ratio', float('nan')):>10.3f}")

    # Best per sector (already sorted by Sharpe, first occurrence per sector wins)
    sector_best: dict[str, dict] = {}
    for r in results:
        s = r.get("sector", "Unknown")
        if s not in sector_best:
            sector_best[s] = r

    print(f"\n{'=' * 90}")
    print("BEST PER SECTOR  (by TD3 Sharpe Ratio)")
    print("=" * 90)
    print(f"  {'Sector':<16s}  {'Symbol':<10s}  {'Name':<14s}  "
          f"{'TD3 Ret%':>8s}  {'TD3 Sharpe':>10s}")
    print("  " + "-" * 65)
    for sector in sorted(sector_best):
        r = sector_best[sector]
        td3 = r.get("td3", {})
        print(f"  {sector[:16]:<16s}  {r['symbol']:<10s}  {r['name'][:14]:<14s}  "
              f"  {td3.get('total_return', float('nan')):+7.1f}%"
              f"  {td3.get('sharpe_ratio', float('nan')):>10.3f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bulk train all models on top Shanghai A-share candidates"
    )
    parser.add_argument(
        "--candidates", default=DEFAULT_CANDIDATES,
        help=f"Candidates CSV (default: {DEFAULT_CANDIDATES})",
    )
    parser.add_argument(
        "--checkpoint", default=DEFAULT_CHECKPOINT,
        help=f"Checkpoint JSON (default: {DEFAULT_CHECKPOINT})",
    )
    parser.add_argument(
        "--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
        help=f"Stocks per batch (default: {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--start-from", type=int, default=0,
        help="Skip first N stocks in candidates list (manual resume override)",
    )
    parser.add_argument(
        "--data-dir", default=DEFAULT_DATA_DIR,
        help=f"Directory for per-stock CSVs (default: {DEFAULT_DATA_DIR})",
    )
    parser.add_argument(
        "--capital", type=int, default=DEFAULT_CAPITAL,
        help=f"Initial capital per stock (default: {DEFAULT_CAPITAL:,})",
    )
    args = parser.parse_args()

    # Load candidates
    if not os.path.exists(args.candidates):
        print(f"ERROR: Candidates file not found: {args.candidates}")
        print("Run 'python find_candidates.py' first.")
        return

    candidates_df = pd.read_csv(args.candidates, dtype={"symbol": str})
    all_symbols = list(zip(
        candidates_df["symbol"].tolist(),
        candidates_df["name"].tolist(),
        candidates_df["sector"].tolist(),
    ))
    total = len(all_symbols)

    # Apply manual start-from offset
    if args.start_from > 0:
        print(f"Manual override: skipping first {args.start_from} stocks")
        all_symbols = all_symbols[args.start_from:]

    # Load or create checkpoint
    cp = load_checkpoint(args.checkpoint, total)
    already_done = sum(1 for sym, _, _ in all_symbols if is_completed(cp, sym))

    print("=" * 76)
    print(f"BULK TRAINING  —  {total} candidates")
    print(f"  Checkpoint : {args.checkpoint}")
    print(f"  Candidates : {args.candidates}")
    print(f"  Batch size : {args.batch_size}")
    print(f"  Data dir   : {args.data_dir}")
    print(f"  Capital    : {args.capital:,}")
    if already_done > 0:
        remaining = len(all_symbols) - already_done
        print(f"  RESUMING   : {already_done} already completed, {remaining} remaining")
    print("=" * 76)

    batches = split_batches(all_symbols, batch_size=args.batch_size)
    total_start = time.time()
    grand_total_done = 0

    for batch_idx, batch in enumerate(batches):
        batch_num = batch_idx + 1
        n_batches = len(batches)
        print(f"\n{'=' * 76}")
        print(f"BATCH {batch_num}/{n_batches}  ({len(batch)} stocks)")
        print("=" * 76)

        batch_results = []
        batch_start = time.time()

        for symbol, name, sector in batch:
            if is_completed(cp, symbol):
                print(f"  [{symbol}] Already done — skipping")
                continue

            t0 = time.time()
            try:
                result = process_stock(
                    symbol=symbol,
                    name=name,
                    sector=sector,
                    cp_path=args.checkpoint,
                    cp=cp,
                    data_dir=args.data_dir,
                    capital=args.capital,
                )
                batch_results.append(result)
                elapsed = time.time() - t0
                grand_total_done += 1
                km_s = result["km"].get("sharpe_ratio", 0)
                td3_s = result["td3"].get("sharpe_ratio", 0)
                print(f"  [{symbol}] Done in {elapsed:.0f}s  "
                      f"km_sharpe={km_s:.3f}  td3_sharpe={td3_s:.3f}")

            except Exception as e:
                elapsed = time.time() - t0
                print(f"  [{symbol}] FAILED in {elapsed:.0f}s: {e}")
                traceback.print_exc()
                if symbol not in cp["metadata"]["failed"]:
                    cp["metadata"]["failed"].append(symbol)
                save_checkpoint(cp, args.checkpoint)

        _print_batch_summary(batch_results)

        batch_elapsed = time.time() - batch_start
        total_elapsed = time.time() - total_start
        completed = cp["metadata"]["completed"]
        n_failed = len(cp["metadata"]["failed"])
        print(f"\n  Batch {batch_num} done in {batch_elapsed:.0f}s  |  "
              f"Total elapsed: {total_elapsed / 3600:.1f}h  |  "
              f"Completed: {completed}/{total}  |  Failed: {n_failed}")

    _print_final_summary(cp)

    print(f"\n{'=' * 76}")
    print(f"ALL DONE  "
          f"completed={cp['metadata']['completed']}  "
          f"failed={len(cp['metadata']['failed'])}  "
          f"total={total}")
    print("=" * 76)


if __name__ == "__main__":
    main()
