#!/usr/bin/env python3
"""Fetch daily price data for China A-share stocks."""

import argparse
import sys
from datetime import datetime, timedelta

import akshare as ak
import pandas as pd


def fetch_stock_daily(
    symbol: str,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    """Fetch daily OHLCV data for a China A-share stock.

    Args:
        symbol: Stock ticker number, e.g. "000001" (Ping An Bank),
                "600519" (Kweichow Moutai).
        start_date: Start date in YYYYMMDD format. Defaults to 1 year ago.
        end_date: End date in YYYYMMDD format. Defaults to today.

    Returns:
        DataFrame with columns: date, open, close, high, low, volume, amount,
        amplitude, pct_change, change, turnover_rate.
    """
    if end_date is None:
        end_date = datetime.now().strftime("%Y%m%d")
    if start_date is None:
        start_date = (datetime.now() - timedelta(days=365)).strftime("%Y%m%d")

    df = ak.stock_zh_a_hist(
        symbol=symbol,
        period="daily",
        start_date=start_date,
        end_date=end_date,
        adjust="qfq",  # forward-adjusted prices
    )

    # Rename columns from Chinese to English
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

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Fetch daily price data for China A-share stocks."
    )
    parser.add_argument("symbol", help='Stock ticker, e.g. "000001", "600519"')
    parser.add_argument(
        "--start", default=None, help="Start date (YYYYMMDD). Default: 1 year ago"
    )
    parser.add_argument(
        "--end", default=None, help="End date (YYYYMMDD). Default: today"
    )
    parser.add_argument(
        "--last", type=int, default=None, help="Show only the last N rows"
    )
    parser.add_argument(
        "--csv", default=None, help="Save output to CSV file at given path"
    )
    args = parser.parse_args()

    try:
        df = fetch_stock_daily(args.symbol, args.start, args.end)
    except Exception as e:
        print(f"Error fetching data for {args.symbol}: {e}", file=sys.stderr)
        sys.exit(1)

    if df.empty:
        print(f"No data found for symbol {args.symbol}.", file=sys.stderr)
        sys.exit(1)

    if args.last:
        df = df.tail(args.last)

    if args.csv:
        df.to_csv(args.csv, index=False)
        print(f"Saved {len(df)} rows to {args.csv}")
    else:
        pd.set_option("display.max_rows", None)
        pd.set_option("display.width", 200)
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()
