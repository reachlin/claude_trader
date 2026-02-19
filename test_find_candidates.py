#!/usr/bin/env python3
"""Tests for find_candidates.py — written before implementation."""

import pandas as pd
import pytest

from find_candidates import (
    _allocate_slots,
    exclude_st_stocks,
    filter_listing_date,
    filter_shanghai_codes,
    select_sector_diverse,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_spot_df(symbols, names, volumes=None, market_caps=None):
    n = len(symbols)
    return pd.DataFrame({
        "symbol": symbols,
        "name": names,
        "avg_volume": volumes or [1_000] * n,
        "market_cap": market_caps or [1e9] * n,
    })


def _make_full_df(sector_sizes: dict) -> pd.DataFrame:
    """Build a DataFrame with multiple sectors; volume decreasing within each."""
    rows = []
    code_base = 600001
    for sector, count in sector_sizes.items():
        for i in range(count):
            rows.append({
                "symbol": str(code_base),
                "name": f"Stock{code_base}",
                "sector": sector,
                "listing_date": "20000101",
                "avg_volume": (count - i) * 1_000,  # decreasing volume
                "market_cap": 1e9,
            })
            code_base += 1
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# filter_shanghai_codes
# ---------------------------------------------------------------------------

class TestFilterShanghaiCodes:
    def test_keeps_codes_starting_with_6(self):
        df = _make_spot_df(["600001", "601333", "603999"], ["A", "B", "C"])
        result = filter_shanghai_codes(df)
        assert list(result["symbol"]) == ["600001", "601333", "603999"]

    def test_removes_non_6_codes(self):
        df = _make_spot_df(["000001", "300001", "688001", "600519"], ["A", "B", "C", "D"])
        result = filter_shanghai_codes(df)
        assert list(result["symbol"]) == ["600519"]

    def test_empty_input(self):
        df = _make_spot_df([], [])
        result = filter_shanghai_codes(df)
        assert len(result) == 0

    def test_preserves_other_columns(self):
        df = _make_spot_df(["600001"], ["Moutai"])
        result = filter_shanghai_codes(df)
        assert "name" in result.columns
        assert "avg_volume" in result.columns


# ---------------------------------------------------------------------------
# exclude_st_stocks
# ---------------------------------------------------------------------------

class TestExcludeStStocks:
    def test_removes_st_prefix(self):
        df = _make_spot_df(["600001", "600002"], ["ST测试", "正常股票"])
        result = exclude_st_stocks(df)
        assert list(result["symbol"]) == ["600002"]

    def test_removes_star_st(self):
        df = _make_spot_df(["600001", "600002"], ["*ST公司", "好股票"])
        result = exclude_st_stocks(df)
        assert list(result["symbol"]) == ["600002"]

    def test_removes_star_emoji(self):
        df = _make_spot_df(["600001", "600002"], ["⭐科创股票", "正常"])
        result = exclude_st_stocks(df)
        assert list(result["symbol"]) == ["600002"]

    def test_keeps_normal_stocks(self):
        df = _make_spot_df(["600519", "601333"], ["贵州茅台", "上海机场"])
        result = exclude_st_stocks(df)
        assert len(result) == 2

    def test_empty_input(self):
        df = _make_spot_df([], [])
        result = exclude_st_stocks(df)
        assert len(result) == 0


# ---------------------------------------------------------------------------
# filter_listing_date
# ---------------------------------------------------------------------------

class TestFilterListingDate:
    def test_keeps_pre_cutoff_stocks(self):
        df = pd.DataFrame({
            "symbol": ["600001", "600002"],
            "listing_date": ["19991201", "20040630"],
        })
        result = filter_listing_date(df, cutoff="20050101")
        assert list(result["symbol"]) == ["600001", "600002"]

    def test_excludes_post_cutoff_stocks(self):
        df = pd.DataFrame({
            "symbol": ["600001", "600002"],
            "listing_date": ["20050102", "20100101"],
        })
        result = filter_listing_date(df, cutoff="20050101")
        assert len(result) == 0

    def test_boundary_date_included(self):
        df = pd.DataFrame({
            "symbol": ["600001"],
            "listing_date": ["20050101"],
        })
        result = filter_listing_date(df, cutoff="20050101")
        assert len(result) == 1

    def test_mixed_dates(self):
        df = pd.DataFrame({
            "symbol": ["600001", "600002", "600003"],
            "listing_date": ["19991201", "20050101", "20050102"],
        })
        result = filter_listing_date(df)  # default cutoff = "20050101"
        assert list(result["symbol"]) == ["600001", "600002"]


# ---------------------------------------------------------------------------
# _allocate_slots
# ---------------------------------------------------------------------------

class TestAllocateSlots:
    def test_total_equals_n(self):
        counts = {"A": 100, "B": 200, "C": 50, "D": 150}
        slots = _allocate_slots(counts, 1000)
        assert sum(slots.values()) == 1000

    def test_every_sector_gets_at_least_one(self):
        counts = {"A": 1, "B": 999}
        slots = _allocate_slots(counts, 100)
        assert slots["A"] >= 1
        assert slots["B"] >= 1

    def test_larger_sector_gets_more_slots(self):
        counts = {"big": 900, "small": 100}
        slots = _allocate_slots(counts, 100)
        assert slots["big"] > slots["small"]

    def test_single_sector(self):
        counts = {"only": 500}
        slots = _allocate_slots(counts, 100)
        assert slots["only"] == 100

    def test_proportional_allocation(self):
        counts = {"A": 300, "B": 200, "C": 500}
        slots = _allocate_slots(counts, 100)
        assert slots["C"] > slots["A"] > slots["B"]

    def test_more_sectors_than_slots_each_gets_one(self):
        counts = {str(i): 10 for i in range(5)}
        slots = _allocate_slots(counts, 3)
        assert sum(slots.values()) == 3
        assert all(v == 1 for v in slots.values())


# ---------------------------------------------------------------------------
# select_sector_diverse
# ---------------------------------------------------------------------------

class TestSelectSectorDiverse:
    def test_all_sectors_represented(self):
        df = _make_full_df({"Tech": 100, "Finance": 200, "Energy": 50})
        result = select_sector_diverse(df, n=100)
        assert set(result["sector"]) == {"Tech", "Finance", "Energy"}

    def test_total_equals_requested_n(self):
        df = _make_full_df({"A": 50, "B": 100, "C": 80})
        result = select_sector_diverse(df, n=50)
        assert len(result) == 50

    def test_larger_sectors_get_more_slots(self):
        df = _make_full_df({"big": 900, "small": 100})
        result = select_sector_diverse(df, n=100)
        big_count = (result["sector"] == "big").sum()
        small_count = (result["sector"] == "small").sum()
        assert big_count > small_count

    def test_rank_by_volume_within_sector(self):
        df = _make_full_df({"OnlySector": 10})
        result = select_sector_diverse(df, n=5)
        volumes = list(result["avg_volume"])
        assert volumes == sorted(volumes, reverse=True)

    def test_output_has_required_columns(self):
        df = _make_full_df({"A": 20, "B": 30})
        result = select_sector_diverse(df, n=10)
        required = {"symbol", "name", "sector", "listing_date", "avg_volume", "market_cap"}
        assert required.issubset(set(result.columns))

    def test_n_larger_than_available_returns_all(self):
        df = _make_full_df({"A": 5, "B": 3})
        result = select_sector_diverse(df, n=100)
        assert len(result) == 8

    def test_empty_input(self):
        df = _make_full_df({})
        result = select_sector_diverse(df, n=10)
        assert len(result) == 0
