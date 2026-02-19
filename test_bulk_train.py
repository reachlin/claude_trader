#!/usr/bin/env python3
"""Tests for bulk_train.py — written before implementation."""

import json
import os

import pandas as pd
import pytest

from bulk_train import (
    create_checkpoint,
    is_completed,
    load_checkpoint,
    save_checkpoint,
    serialize_results,
    split_batches,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_train_res() -> dict:
    """Mock train_all() return value including non-serializable objects."""
    metrics = {
        "total_return": 10.5,
        "buy_and_hold_return": 8.0,
        "sharpe_ratio": 1.2,
        "max_drawdown": -15.0,
        "win_rate": 60.0,
        "profit_factor": 1.5,
        "num_trades": 50,
        "final_value": 110_500.0,
        "trades": [{"date": "2020-01-01", "action": "buy"}],  # not a scalar
    }

    class _MockBot:
        pass

    return {
        "km": metrics.copy(),
        "lstm": metrics.copy(),
        "lgbm": metrics.copy(),
        "ppo": metrics.copy(),
        "majority": metrics.copy(),
        "td3": {
            **metrics.copy(),
            "bot": _MockBot(),           # non-serializable
            "test_df": pd.DataFrame(),   # non-serializable
        },
        "df": pd.DataFrame({
            "date": ["2006-01-04", "2026-02-13"],
            "close": [10.0, 20.0],
        }),
        "capital": 100_000,
    }


def _make_mock_tune_res() -> dict:
    """Mock tune_all() return value."""
    return {
        "km_orig": {},
        "km_tuned": {},
        "lgbm_orig": {},
        "lgbm_tuned": {},
        "ppo_orig": {},
        "ppo_tuned": {},
        "best_params": {
            "km": {"n_clusters": 5, "feature_subset": "all_6"},
            "lgbm": {"n_estimators": 100, "max_depth": 5, "learning_rate": 0.1},
            "ppo": {
                "total_timesteps": 100_000,
                "learning_rate": 3e-4,
                "ent_coef": 0.01,
                "n_steps": 1024,
            },
        },
    }


def _make_df() -> pd.DataFrame:
    return pd.DataFrame({
        "date": ["2006-01-04", "2026-02-13"],
        "close": [10.0, 20.0],
    })


# ---------------------------------------------------------------------------
# create_checkpoint
# ---------------------------------------------------------------------------

class TestCheckpointCreate:
    def test_has_metadata_key(self):
        cp = create_checkpoint(total=1000)
        assert "metadata" in cp

    def test_has_results_key(self):
        cp = create_checkpoint(total=1000)
        assert "results" in cp

    def test_metadata_total(self):
        cp = create_checkpoint(total=500)
        assert cp["metadata"]["total"] == 500

    def test_metadata_completed_zero(self):
        cp = create_checkpoint(total=100)
        assert cp["metadata"]["completed"] == 0

    def test_metadata_failed_empty_list(self):
        cp = create_checkpoint(total=100)
        assert cp["metadata"]["failed"] == []

    def test_metadata_has_started_at(self):
        cp = create_checkpoint(total=100)
        assert "started_at" in cp["metadata"]

    def test_results_empty_dict(self):
        cp = create_checkpoint(total=100)
        assert cp["results"] == {}


# ---------------------------------------------------------------------------
# save_checkpoint / load_checkpoint
# ---------------------------------------------------------------------------

class TestCheckpointResume:
    def test_load_existing_checkpoint(self, tmp_path):
        cp_path = str(tmp_path / "ckpt.json")
        cp = create_checkpoint(total=100)
        cp["results"]["600001"] = {"symbol": "600001", "km": {"total_return": 5.0}}
        save_checkpoint(cp, cp_path)

        loaded = load_checkpoint(cp_path, total=100)
        assert "600001" in loaded["results"]

    def test_load_creates_new_if_not_exists(self, tmp_path):
        cp_path = str(tmp_path / "nonexistent.json")
        loaded = load_checkpoint(cp_path, total=50)
        assert loaded["metadata"]["total"] == 50
        assert loaded["results"] == {}

    def test_already_done_symbols_detected(self, tmp_path):
        cp_path = str(tmp_path / "ckpt.json")
        cp = create_checkpoint(total=10)
        for sym in ["600001", "600002"]:
            cp["results"][sym] = {"symbol": sym}
        save_checkpoint(cp, cp_path)

        loaded = load_checkpoint(cp_path, total=10)
        assert is_completed(loaded, "600001")
        assert is_completed(loaded, "600002")
        assert not is_completed(loaded, "600003")

    def test_save_and_reload_roundtrip(self, tmp_path):
        cp_path = str(tmp_path / "ckpt.json")
        cp = create_checkpoint(total=5)
        cp["metadata"]["completed"] = 2
        cp["results"]["600001"] = {"symbol": "600001", "sharpe": 1.2}
        save_checkpoint(cp, cp_path)

        loaded = load_checkpoint(cp_path, total=5)
        assert loaded["metadata"]["completed"] == 2
        assert loaded["results"]["600001"]["sharpe"] == 1.2


# ---------------------------------------------------------------------------
# is_completed
# ---------------------------------------------------------------------------

class TestIsCompleted:
    def test_returns_true_for_done_symbol(self):
        cp = create_checkpoint(total=10)
        cp["results"]["601933"] = {"symbol": "601933"}
        assert is_completed(cp, "601933") is True

    def test_returns_false_for_undone_symbol(self):
        cp = create_checkpoint(total=10)
        assert is_completed(cp, "601933") is False


# ---------------------------------------------------------------------------
# Checkpoint saved after each stock
# ---------------------------------------------------------------------------

class TestCheckpointSavedAfterEachStock:
    def test_file_updated_after_each_symbol(self, tmp_path):
        cp_path = str(tmp_path / "ckpt.json")
        cp = create_checkpoint(total=3)
        symbols = ["600001", "600002", "600003"]

        for symbol in symbols:
            cp["results"][symbol] = {"symbol": symbol, "km": {"total_return": 1.0}}
            cp["metadata"]["completed"] = len(cp["results"])
            save_checkpoint(cp, cp_path)

            # File must reflect current state immediately after save
            loaded = load_checkpoint(cp_path, total=3)
            assert symbol in loaded["results"]

    def test_partial_completion_survives_reload(self, tmp_path):
        cp_path = str(tmp_path / "ckpt.json")
        cp = create_checkpoint(total=5)

        for sym in ["600001", "600002"]:
            cp["results"][sym] = {"symbol": sym}
            save_checkpoint(cp, cp_path)

        loaded = load_checkpoint(cp_path, total=5)
        assert len(loaded["results"]) == 2
        assert "600001" in loaded["results"]
        assert "600002" in loaded["results"]
        assert "600003" not in loaded["results"]


# ---------------------------------------------------------------------------
# Failed stock doesn't stop batch
# ---------------------------------------------------------------------------

class TestFailedStockContinues:
    def test_exception_on_one_stock_continues_batch(self, tmp_path):
        cp_path = str(tmp_path / "ckpt.json")
        cp = create_checkpoint(total=3)

        symbols = ["600001", "600002", "600003"]
        processed = []
        failed = []

        for symbol in symbols:
            try:
                if symbol == "600002":
                    raise ValueError("Simulated API failure")
                cp["results"][symbol] = {"symbol": symbol}
                cp["metadata"]["completed"] = len(cp["results"])
                processed.append(symbol)
            except Exception:
                failed.append(symbol)
                cp["metadata"]["failed"].append(symbol)
            save_checkpoint(cp, cp_path)

        assert "600001" in processed
        assert "600002" in failed
        assert "600003" in processed
        assert len(cp["results"]) == 2
        assert "600002" in cp["metadata"]["failed"]

    def test_failed_list_persisted_in_checkpoint(self, tmp_path):
        cp_path = str(tmp_path / "ckpt.json")
        cp = create_checkpoint(total=3)
        cp["metadata"]["failed"].append("600001")
        save_checkpoint(cp, cp_path)

        loaded = load_checkpoint(cp_path, total=3)
        assert "600001" in loaded["metadata"]["failed"]


# ---------------------------------------------------------------------------
# serialize_results
# ---------------------------------------------------------------------------

class TestSerializeResults:
    def test_is_json_serializable(self):
        result = serialize_results(
            "601933", "永辉超市", "一般零售",
            _make_df(), _make_mock_train_res(), _make_mock_tune_res(),
        )
        # Must not raise
        json_str = json.dumps(result)
        loaded = json.loads(json_str)
        assert loaded["symbol"] == "601933"

    def test_contains_all_model_keys(self):
        result = serialize_results(
            "601933", "永辉超市", "一般零售",
            _make_df(), _make_mock_train_res(), _make_mock_tune_res(),
        )
        for key in ("km", "lstm", "lgbm", "ppo", "majority", "td3"):
            assert key in result, f"Missing model key: {key}"

    def test_no_dataframes_in_result(self):
        result = serialize_results(
            "601933", "永辉超市", "一般零售",
            _make_df(), _make_mock_train_res(), _make_mock_tune_res(),
        )

        def _check_no_df(obj, path="root"):
            if isinstance(obj, pd.DataFrame):
                pytest.fail(f"Found DataFrame at path: {path}")
            if isinstance(obj, dict):
                for k, v in obj.items():
                    _check_no_df(v, f"{path}.{k}")

        _check_no_df(result)

    def test_no_bot_objects_in_result(self):
        result = serialize_results(
            "601933", "永辉超市", "一般零售",
            _make_df(), _make_mock_train_res(), _make_mock_tune_res(),
        )
        assert "bot" not in result.get("td3", {})
        assert "test_df" not in result.get("td3", {})

    def test_contains_date_range_and_rows(self):
        result = serialize_results(
            "601933", "永辉超市", "一般零售",
            _make_df(), _make_mock_train_res(), _make_mock_tune_res(),
        )
        assert result["rows"] == 2
        assert result["date_range"] == ["2006-01-04", "2026-02-13"]

    def test_contains_scalar_metrics(self):
        result = serialize_results(
            "601933", "永辉超市", "一般零售",
            _make_df(), _make_mock_train_res(), _make_mock_tune_res(),
        )
        km = result["km"]
        assert km["total_return"] == pytest.approx(10.5)
        assert km["sharpe_ratio"] == pytest.approx(1.2)
        assert km["num_trades"] == 50

    def test_contains_best_params(self):
        result = serialize_results(
            "601933", "永辉超市", "一般零售",
            _make_df(), _make_mock_train_res(), _make_mock_tune_res(),
        )
        assert "best_params" in result
        assert result["best_params"]["km"]["n_clusters"] == 5
        assert result["best_params"]["lgbm"]["n_estimators"] == 100

    def test_metadata_fields(self):
        result = serialize_results(
            "601933", "永辉超市", "一般零售",
            _make_df(), _make_mock_train_res(), _make_mock_tune_res(),
        )
        assert result["symbol"] == "601933"
        assert result["name"] == "永辉超市"
        assert result["sector"] == "一般零售"


# ---------------------------------------------------------------------------
# split_batches
# ---------------------------------------------------------------------------

class TestBatchSplit:
    def test_1000_symbols_into_100_batches_of_10(self):
        symbols = [str(i) for i in range(1000)]
        batches = split_batches(symbols, batch_size=10)
        assert len(batches) == 100
        assert all(len(b) == 10 for b in batches)

    def test_last_batch_smaller(self):
        symbols = [str(i) for i in range(25)]
        batches = split_batches(symbols, batch_size=10)
        assert len(batches) == 3
        assert len(batches[-1]) == 5

    def test_all_symbols_covered(self):
        symbols = [str(i) for i in range(53)]
        batches = split_batches(symbols, batch_size=10)
        flat = [s for b in batches for s in b]
        assert flat == symbols

    def test_batch_size_larger_than_symbols(self):
        symbols = ["600001", "600002"]
        batches = split_batches(symbols, batch_size=10)
        assert len(batches) == 1
        assert len(batches[0]) == 2

    def test_empty_input(self):
        batches = split_batches([], batch_size=10)
        assert batches == []

    def test_exact_multiple(self):
        symbols = [str(i) for i in range(20)]
        batches = split_batches(symbols, batch_size=5)
        assert len(batches) == 4
        assert all(len(b) == 5 for b in batches)
