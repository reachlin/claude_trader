#!/usr/bin/env python3
"""Download 20yr data for 10 randomly selected new A-share stocks."""

import os
import random
import sys
from datetime import datetime, timedelta
from pathlib import Path

import akshare as ak
import pandas as pd

DATA_DIR = Path(__file__).parent / "data"
START_DATE = (datetime.now() - timedelta(days=365 * 20)).strftime("%Y%m%d")
END_DATE = datetime.now().strftime("%Y%m%d")


def get_already_downloaded() -> set[str]:
    """Extract symbols already present in data folder."""
    symbols = set()
    for f in DATA_DIR.glob("*.csv"):
        name = f.stem
        # Strip suffix like _20yr, _10yr, _2025, _3yr, SH
        for suffix in ("_20yr", "_10yr", "_3yr", "_2025", "SH"):
            name = name.replace(suffix, "")
        # Keep only numeric codes
        if name.isdigit():
            symbols.add(name)
    return symbols


def fetch_all_ashare_symbols() -> list[dict]:
    """Get full list of A-share stocks from akshare."""
    df = ak.stock_info_a_code_name()
    return df.to_dict("records")


def download_stock(symbol: str, name: str) -> bool:
    """Download 20yr data for a stock and save to data/."""
    try:
        df = ak.stock_zh_a_hist(
            symbol=symbol,
            period="daily",
            start_date=START_DATE,
            end_date=END_DATE,
            adjust="qfq",
        )
        if df.empty:
            print(f"  [{symbol}] No data returned, skipping.")
            return False

        column_map = {
            "日期": "date",
            "股票代码": "symbol",
            "开盘": "open",
            "收盘": "close",
            "最高": "high",
            "最低": "low",
            "成交量": "volume",
            "成交额": "amount",
            "振幅": "amplitude",
            "涨跌幅": "pct_change",
            "涨跌额": "change",
            "换手率": "turnover_rate",
        }
        df.rename(columns=column_map, inplace=True)

        out_path = DATA_DIR / f"{symbol}_20yr.csv"
        df.to_csv(out_path, index=False)
        print(f"  [{symbol}] {name}: {len(df)} rows -> {out_path.name}")
        return True
    except Exception as e:
        print(f"  [{symbol}] Error: {e}")
        return False


def main():
    random.seed(None)  # True random

    already = get_already_downloaded()
    print(f"Already downloaded: {sorted(already)}\n")

    print("Fetching A-share stock list...")
    all_stocks = fetch_all_ashare_symbols()
    print(f"Total A-share stocks: {len(all_stocks)}")

    candidates = [s for s in all_stocks if s["code"] not in already]
    print(f"Candidates (new): {len(candidates)}")

    picks = random.sample(candidates, 10)
    print(f"\nRandomly selected 10 stocks:")
    for p in picks:
        print(f"  {p['code']} - {p['name']}")

    print(f"\nDownloading 20yr data (from {START_DATE} to {END_DATE})...")
    success = []
    for p in picks:
        ok = download_stock(p["code"], p["name"])
        if ok:
            success.append(p)

    print(f"\nDone: {len(success)}/{len(picks)} downloaded successfully.")
    for s in success:
        print(f"  {s['code']} {s['name']}")


if __name__ == "__main__":
    main()
