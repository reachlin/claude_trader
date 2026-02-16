#!/usr/bin/env python3
"""Fetch daily price data for China A-share stocks."""

import argparse
import sys
from datetime import datetime, timedelta

import akshare as ak
import pandas as pd


def _is_index_symbol(symbol: str) -> bool:
    """Check if the symbol refers to a market index (e.g. 000001.SH, 399001.SZ)."""
    return "." in symbol and symbol.split(".")[-1].upper() in ("SH", "SZ")


def _fetch_index_daily(
    symbol: str,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    """Fetch daily OHLCV data for a China market index.

    Args:
        symbol: Index symbol with exchange suffix, e.g. "000001.SH" (Shanghai
                Composite), "399001.SZ" (Shenzhen Component).
        start_date: Start date in YYYYMMDD format. Defaults to 1 year ago.
        end_date: End date in YYYYMMDD format. Defaults to today.

    Returns:
        DataFrame with columns: date, open, close, high, low, volume.
    """
    if end_date is None:
        end_date = datetime.now().strftime("%Y%m%d")
    if start_date is None:
        start_date = (datetime.now() - timedelta(days=365)).strftime("%Y%m%d")

    # akshare uses "sh000001" / "sz399001" format
    code, exchange = symbol.split(".")
    prefix = exchange.lower()  # sh or sz
    ak_symbol = f"{prefix}{code}"

    df = ak.stock_zh_index_daily(symbol=ak_symbol)

    # Columns are already English: date, open, high, low, close, volume
    # Filter by date range
    df["date"] = pd.to_datetime(df["date"])
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    df = df[(df["date"] >= start_dt) & (df["date"] <= end_dt)].copy()
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    df = df.reset_index(drop=True)

    return df


def fetch_stock_daily(
    symbol: str,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    """Fetch daily OHLCV data for a China A-share stock or index.

    Args:
        symbol: Stock ticker number, e.g. "000001" (Ping An Bank),
                "600519" (Kweichow Moutai), or index with exchange suffix,
                e.g. "000001.SH" (Shanghai Composite), "399001.SZ" (Shenzhen
                Component).
        start_date: Start date in YYYYMMDD format. Defaults to 1 year ago.
        end_date: End date in YYYYMMDD format. Defaults to today.

    Returns:
        DataFrame with columns: date, open, close, high, low, volume (and more
        for individual stocks).
    """
    if _is_index_symbol(symbol):
        return _fetch_index_daily(symbol, start_date, end_date)

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
