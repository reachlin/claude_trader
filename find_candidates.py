#!/usr/bin/env python3
"""Screen all Shanghai A-shares down to top N high-quality candidates.

Outputs candidates.csv with columns:
  symbol, name, sector, listing_date, avg_volume, market_cap, rank

Usage:
    python find_candidates.py                     # → candidates.csv (1000 stocks)
    python find_candidates.py --output my.csv     # custom output
    python find_candidates.py --top 500           # select fewer stocks
"""

import argparse
import time

import akshare as ak
import pandas as pd


# ---------------------------------------------------------------------------
# Pure filtering functions (no API calls — fully testable)
# ---------------------------------------------------------------------------

def filter_shanghai_codes(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to Shanghai main-board A-shares.

    Keeps codes starting with '6' but excludes STAR Market (688xxx) codes.
    """
    sym = df["symbol"].astype(str)
    mask = sym.str.startswith("6") & ~sym.str.startswith("688")
    return df[mask].copy().reset_index(drop=True)


def exclude_st_stocks(df: pd.DataFrame) -> pd.DataFrame:
    """Exclude special-treatment (ST) and STAR-market (⭐) stocks."""
    if df.empty:
        return df.copy()
    mask = ~df["name"].astype(str).str.contains(r"ST|⭐", na=False)
    return df[mask].copy().reset_index(drop=True)


def filter_listing_date(df: pd.DataFrame, cutoff: str = "20050101") -> pd.DataFrame:
    """Keep only stocks listed on or before the cutoff date (YYYYMMDD string)."""
    return df[df["listing_date"] <= cutoff].copy().reset_index(drop=True)


def _allocate_slots(sector_counts: dict, n: int) -> dict:
    """Allocate n slots proportionally across sectors, minimum 1 per sector.

    Args:
        sector_counts: {sector_name: number_of_qualifying_stocks}
        n: total slots to allocate

    Returns:
        {sector_name: allocated_slot_count}
    """
    sectors = list(sector_counts.keys())
    num_sectors = len(sectors)

    if num_sectors == 0:
        return {}

    if num_sectors >= n:
        # More sectors than slots — give 1 slot to the n largest sectors
        sorted_sectors = sorted(sectors, key=lambda s: sector_counts[s], reverse=True)
        return {s: 1 for s in sorted_sectors[:n]}

    total = sum(sector_counts.values())
    slots = {s: max(1, int(n * cnt / total)) for s, cnt in sector_counts.items()}

    # Adjust to exactly n using round-robin on sorted-by-count sectors
    diff = n - sum(slots.values())
    sorted_by_count = sorted(sector_counts, key=lambda s: sector_counts[s], reverse=True)

    if diff > 0:
        i = 0
        while diff > 0:
            slots[sorted_by_count[i % num_sectors]] += 1
            diff -= 1
            i += 1
    elif diff < 0:
        i = 0
        iters = 0
        max_iters = num_sectors * (abs(diff) + 1)
        while diff < 0 and iters < max_iters:
            s = sorted_by_count[i % num_sectors]
            if slots[s] > 1:
                slots[s] -= 1
                diff += 1
            i += 1
            iters += 1

    return slots


def select_sector_diverse(df: pd.DataFrame, n: int = 1000) -> pd.DataFrame:
    """Select top n stocks with sector diversity.

    Groups by sector, allocates slots proportionally to sector size (min 1),
    and within each sector selects the highest avg_volume stocks.

    Args:
        df: DataFrame with columns: symbol, name, sector, listing_date,
            avg_volume, market_cap
        n: target number of stocks to select

    Returns:
        DataFrame sorted by avg_volume descending with an added 'rank' column.
    """
    if df.empty:
        return df.copy()

    sector_counts = df.groupby("sector").size().to_dict()
    actual_n = min(n, len(df))
    slots = _allocate_slots(sector_counts, actual_n)

    parts = []
    for sector, k in slots.items():
        sector_df = df[df["sector"] == sector].sort_values("avg_volume", ascending=False)
        parts.append(sector_df.head(k))

    if not parts:
        return df.head(0)

    result = pd.concat(parts, ignore_index=True)
    result = result.sort_values("avg_volume", ascending=False).reset_index(drop=True)
    result["rank"] = range(1, len(result) + 1)
    return result


# ---------------------------------------------------------------------------
# API fetch helpers
# ---------------------------------------------------------------------------

def _fetch_stock_info(symbol: str) -> dict | None:
    """Fetch listing date and sector for one stock via akshare.

    Passes timeout=10 directly to the akshare call so a single slow/dead
    endpoint can't stall the whole loop.
    Returns None on failure (suspended/delisted/API error/timeout).
    """
    try:
        info_df = ak.stock_individual_info_em(symbol=symbol, timeout=10)
        # Returns DataFrame with two columns: item name and value
        info = dict(zip(info_df.iloc[:, 0], info_df.iloc[:, 1]))
        listing_date = str(info.get("上市时间", "")).replace("-", "").strip()
        sector = str(info.get("行业", "未知")).strip()
        if not listing_date or listing_date in ("nan", "None", ""):
            return None
        return {"listing_date": listing_date, "sector": sector}
    except Exception:
        return None


def fetch_candidates(
    output_path: str = "candidates.csv",
    top_n: int = 1000,
    prefetch: int = 0,
) -> pd.DataFrame:
    """Screen all Shanghai A-shares and return top N diverse candidates.

    Steps:
        1. Fetch all A-share spot data (volume, market cap)
        2. Filter to Shanghai codes (start with '6'), exclude ST/⭐
        2b. (Optional) Pre-filter to top `prefetch` by volume to reduce API calls
        3. Fetch per-stock listing date + sector with 0.3s rate limiting
        4. Filter to stocks listed before 2005 (20yr history available)
        5. Select top N with sector diversity
        6. Save to output_path and return DataFrame

    Args:
        output_path: CSV output file path
        top_n: number of candidates to select
        prefetch: if > 0, only fetch metadata for the top-prefetch stocks by
            volume (speeds things up when top_n << total universe).
            Defaults to 0 (fetch all).

    Returns:
        DataFrame with selected candidates.
    """
    print("=" * 76)
    print("STEP 1: Fetching all A-share spot data...")
    spot = None
    for attempt in range(1, 6):
        try:
            spot = ak.stock_zh_a_spot_em()
            break
        except Exception as e:
            print(f"  Attempt {attempt}/5 failed: {e}")
            if attempt < 5:
                wait = attempt * 10
                print(f"  Retrying in {wait}s...")
                time.sleep(wait)
    if spot is None:
        raise RuntimeError("Failed to fetch spot data after 5 attempts")

    col_map = {
        "代码": "symbol",
        "名称": "name",
        "成交量": "avg_volume",
        "总市值": "market_cap",
    }
    spot.rename(columns=col_map, inplace=True)
    # Keep only needed columns (drop any others)
    keep_cols = [c for c in ["symbol", "name", "avg_volume", "market_cap"] if c in spot.columns]
    spot = spot[keep_cols].copy()
    print(f"  Total A-shares: {len(spot)}")

    # Filter Shanghai codes
    spot = filter_shanghai_codes(spot)
    print(f"  After Shanghai filter (code starts with '6'): {len(spot)}")

    # Exclude ST / STAR market
    spot = exclude_st_stocks(spot)
    print(f"  After ST/⭐ exclusion: {len(spot)}")

    # Optional volume pre-filter to limit expensive per-stock API calls
    if prefetch > 0 and prefetch < len(spot):
        spot = spot.sort_values("avg_volume", ascending=False).head(prefetch).reset_index(drop=True)
        print(f"  Pre-filtered to top {prefetch} by volume (prefetch mode)")

    print(f"\nSTEP 2: Fetching listing date and sector for {len(spot)} stocks...")
    est_min = len(spot) * 0.3 / 60
    print(f"  (Rate-limited to 0.3s/call — estimated {est_min:.0f} min)")

    meta_rows = []
    for i, row in enumerate(spot.itertuples(), 1):
        if i % 10 == 0 or i == 1:
            pct = i / len(spot) * 100
            print(f"  Progress: {i}/{len(spot)} ({pct:.1f}%)  "
                  f"fetched={len(meta_rows)}", flush=True)
        info = _fetch_stock_info(row.symbol)
        if info is not None:
            meta_rows.append({
                "symbol": row.symbol,
                "name": row.name,
                "avg_volume": getattr(row, "avg_volume", 0),
                "market_cap": getattr(row, "market_cap", 0),
                "listing_date": info["listing_date"],
                "sector": info["sector"],
            })
        time.sleep(0.3)

    meta_df = pd.DataFrame(meta_rows)
    print(f"  Got metadata for {len(meta_df)} stocks")

    print("\nSTEP 3: Filtering by listing date (pre-2005)...")
    meta_df = filter_listing_date(meta_df, cutoff="20050101")
    print(f"  Stocks with 20yr history: {len(meta_df)}")

    print(f"\nSTEP 4: Selecting top {top_n} sector-diverse candidates...")
    candidates = select_sector_diverse(meta_df, n=top_n)
    n_sectors = candidates["sector"].nunique()
    print(f"  Selected: {len(candidates)} stocks across {n_sectors} sectors")

    candidates.to_csv(output_path, index=False)
    print(f"\nSaved → {output_path}")
    print("Top 10 by volume:")
    for _, r in candidates.head(10).iterrows():
        print(f"  {r['symbol']}  {r['name']:<12s}  {r['sector']:<14s}  "
              f"vol={r['avg_volume']:>15,.0f}")

    return candidates


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Screen Shanghai A-shares → candidates.csv"
    )
    parser.add_argument(
        "--output", default="candidates.csv", help="Output CSV path"
    )
    parser.add_argument(
        "--top", type=int, default=1000, help="Number of candidates to select"
    )
    parser.add_argument(
        "--prefetch", type=int, default=0,
        help="Only fetch metadata for top N stocks by volume (0 = fetch all). "
             "Set to top*3 for faster runs when top << full universe.",
    )
    args = parser.parse_args()
    fetch_candidates(output_path=args.output, top_n=args.top, prefetch=args.prefetch)


if __name__ == "__main__":
    main()
