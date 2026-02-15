"""Tests for fetch_china_stock.py."""

from unittest.mock import patch

import pandas as pd
import pytest

from fetch_china_stock import fetch_stock_daily


@pytest.fixture
def sample_raw_df():
    """Sample DataFrame mimicking akshare output with Chinese column names."""
    return pd.DataFrame(
        {
            "日期": ["2024-01-02", "2024-01-03"],
            "开盘": [10.50, 10.60],
            "收盘": [10.80, 10.55],
            "最高": [10.90, 10.70],
            "最低": [10.40, 10.45],
            "成交量": [100000, 120000],
            "成交额": [1080000.0, 1266000.0],
            "振幅": [4.76, 2.36],
            "涨跌幅": [2.86, -2.31],
            "涨跌额": [0.30, -0.25],
            "换手率": [0.50, 0.60],
        }
    )


@patch("fetch_china_stock.ak.stock_zh_a_hist")
def test_fetch_returns_dataframe(mock_hist, sample_raw_df):
    mock_hist.return_value = sample_raw_df
    df = fetch_stock_daily("000001", "20240101", "20240103")
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2


@patch("fetch_china_stock.ak.stock_zh_a_hist")
def test_columns_renamed_to_english(mock_hist, sample_raw_df):
    mock_hist.return_value = sample_raw_df
    df = fetch_stock_daily("000001", "20240101", "20240103")

    expected_cols = [
        "date", "open", "close", "high", "low",
        "volume", "amount", "amplitude", "pct_change", "change", "turnover_rate",
    ]
    for col in expected_cols:
        assert col in df.columns, f"Missing column: {col}"


@patch("fetch_china_stock.ak.stock_zh_a_hist")
def test_ohlcv_values_correct(mock_hist, sample_raw_df):
    mock_hist.return_value = sample_raw_df
    df = fetch_stock_daily("000001", "20240101", "20240103")

    row = df.iloc[0]
    assert row["open"] == 10.50
    assert row["close"] == 10.80
    assert row["high"] == 10.90
    assert row["low"] == 10.40
    assert row["volume"] == 100000


@patch("fetch_china_stock.ak.stock_zh_a_hist")
def test_default_dates_used(mock_hist, sample_raw_df):
    mock_hist.return_value = sample_raw_df
    fetch_stock_daily("600519")

    mock_hist.assert_called_once()
    call_kwargs = mock_hist.call_args
    assert call_kwargs.kwargs["symbol"] == "600519"
    assert call_kwargs.kwargs["start_date"] is not None
    assert call_kwargs.kwargs["end_date"] is not None


@patch("fetch_china_stock.ak.stock_zh_a_hist")
def test_empty_dataframe(mock_hist):
    mock_hist.return_value = pd.DataFrame()
    df = fetch_stock_daily("999999", "20240101", "20240103")
    assert df.empty
