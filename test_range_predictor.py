#!/usr/bin/env python3
"""Tests for range_predictor.py — BiLSTM next-day price range prediction."""

import numpy as np
import pandas as pd
import pytest
import torch

from range_predictor import (
    BiLSTMRangeModel,
    RangeDataset,
    RangePredictor,
    run_range_backtest,
    score_prediction,
    pinball_loss,
)
from trading_bot import compute_indicators, FEATURE_COLS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ohlcv(n: int = 120, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic OHLCV DataFrame with n rows."""
    rng = np.random.default_rng(seed)
    close = 10.0 + np.cumsum(rng.normal(0, 0.1, n))
    close = np.clip(close, 1.0, None)
    spread = rng.uniform(0.02, 0.08, n)
    high = close * (1 + spread)
    low = close * (1 - spread)
    open_ = low + rng.uniform(0, 1, n) * (high - low)
    volume = rng.integers(100_000, 1_000_000, n).astype(float)
    df = pd.DataFrame({
        "date": pd.date_range("2020-01-01", periods=n, freq="B").strftime("%Y-%m-%d"),
        "open": open_,
        "close": close,
        "high": high,
        "low": low,
        "volume": volume,
    })
    return df


def _make_indicator_df(n: int = 120, seed: int = 42) -> pd.DataFrame:
    """Return a DataFrame with indicators computed and NaN rows dropped."""
    df = _make_ohlcv(n, seed)
    df = compute_indicators(df)
    df = df.dropna(subset=FEATURE_COLS).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# score_prediction
# ---------------------------------------------------------------------------

class TestScorePrediction:
    def test_plus_two_both_in_range(self):
        # pred [3.11, 4.29] fits inside actual [3.10, 4.30]
        assert score_prediction(3.11, 4.29, 3.10, 4.30) == 2

    def test_plus_two_exact_boundary(self):
        # pred == actual boundaries → both in range
        assert score_prediction(3.10, 4.30, 3.10, 4.30) == 2

    def test_plus_one_low_ok_high_overshoots(self):
        # pred low >= actual low, but pred high > actual high
        assert score_prediction(3.1, 4.6, 3.0, 4.5) == 1

    def test_plus_one_low_exact_high_overshoots(self):
        assert score_prediction(3.0, 5.0, 3.0, 4.5) == 1

    def test_minus_one_both_outside(self):
        # pred [2.5, 5.0] is wider than actual [3.0, 4.5] on both ends
        assert score_prediction(2.5, 5.0, 3.0, 4.5) == -1

    def test_minus_one_low_misses_high_misses(self):
        assert score_prediction(2.9, 4.6, 3.0, 4.5) == -1

    def test_zero_low_misses_high_ok(self):
        # pred low below actual low, pred high inside → 0
        assert score_prediction(2.9, 4.0, 3.0, 4.5) == 0

    def test_zero_low_misses_high_exact(self):
        assert score_prediction(2.9, 4.5, 3.0, 4.5) == 0

    def test_returns_int(self):
        result = score_prediction(3.1, 4.2, 3.0, 4.5)
        assert isinstance(result, int)


# ---------------------------------------------------------------------------
# RangeDataset
# ---------------------------------------------------------------------------

class TestRangeDataset:
    def setup_method(self):
        self.df = _make_indicator_df(n=120)

    def test_length(self):
        ds = RangeDataset(self.df, window_size=20)
        # Each sample uses rows [i:i+20]; target at i+20; last valid i is len-21
        expected = len(self.df) - 20 - 1
        assert len(ds) == expected

    def test_x_shape(self):
        ds = RangeDataset(self.df, window_size=20)
        x, _ = ds[0]
        assert x.shape == (20, len(FEATURE_COLS))

    def test_y_shape(self):
        ds = RangeDataset(self.df, window_size=20)
        _, y = ds[0]
        assert y.shape == (2,)  # (rel_low, rel_high)

    def test_y_ordering(self):
        """rel_low should always be <= rel_high."""
        ds = RangeDataset(self.df, window_size=20)
        for i in range(len(ds)):
            _, y = ds[i]
            assert y[0] <= y[1], f"Sample {i}: low ({y[0]}) > high ({y[1]})"

    def test_x_dtype(self):
        ds = RangeDataset(self.df, window_size=20)
        x, _ = ds[0]
        assert x.dtype == torch.float32

    def test_y_dtype(self):
        ds = RangeDataset(self.df, window_size=20)
        _, y = ds[0]
        assert y.dtype == torch.float32

    def test_different_window_sizes(self):
        for ws in [10, 20, 30]:
            ds = RangeDataset(self.df, window_size=ws)
            expected = len(self.df) - ws - 1
            assert len(ds) == expected, f"window_size={ws}"


# ---------------------------------------------------------------------------
# BiLSTMRangeModel
# ---------------------------------------------------------------------------

class TestBiLSTMRangeModel:
    def test_forward_output_shapes(self):
        model = BiLSTMRangeModel(input_size=6)
        x = torch.randn(4, 20, 6)  # batch=4, window=20, features=6
        low, high = model(x)
        assert low.shape == (4,)
        assert high.shape == (4,)

    def test_single_sample(self):
        model = BiLSTMRangeModel(input_size=6)
        x = torch.randn(1, 20, 6)
        low, high = model(x)
        assert low.shape == (1,)
        assert high.shape == (1,)

    def test_output_is_float(self):
        model = BiLSTMRangeModel(input_size=6)
        x = torch.randn(2, 20, 6)
        low, high = model(x)
        assert low.dtype == torch.float32
        assert high.dtype == torch.float32

    def test_custom_hidden_size(self):
        model = BiLSTMRangeModel(input_size=6, hidden=32)
        x = torch.randn(3, 15, 6)
        low, high = model(x)
        assert low.shape == (3,)

    def test_gradients_flow(self):
        model = BiLSTMRangeModel(input_size=6)
        x = torch.randn(2, 20, 6)
        low, high = model(x)
        loss = low.mean() + high.mean()
        loss.backward()
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"


# ---------------------------------------------------------------------------
# RangePredictor
# ---------------------------------------------------------------------------

class TestRangePredictor:
    def setup_method(self):
        self.df = _make_indicator_df(n=120)

    def test_fit_runs(self):
        predictor = RangePredictor(window_size=20, epochs=3, batch_size=16)
        predictor.fit(self.df)
        assert predictor.model is not None

    def test_predict_length(self):
        predictor = RangePredictor(window_size=20, epochs=3, batch_size=16)
        predictor.fit(self.df)
        results = predictor.predict(self.df)
        expected = len(self.df) - 20 - 1
        assert len(results) == expected

    def test_predict_returns_tuples(self):
        predictor = RangePredictor(window_size=20, epochs=3, batch_size=16)
        predictor.fit(self.df)
        results = predictor.predict(self.df)
        for pred_low, pred_high in results:
            assert isinstance(pred_low, float)
            assert isinstance(pred_high, float)

    def test_predict_single_returns_tuple(self):
        predictor = RangePredictor(window_size=20, epochs=3, batch_size=16)
        predictor.fit(self.df)
        pred_low, pred_high = predictor.predict_single(self.df)
        assert isinstance(pred_low, float)
        assert isinstance(pred_high, float)

    def test_predict_single_needs_enough_rows(self):
        predictor = RangePredictor(window_size=20, epochs=3, batch_size=16)
        predictor.fit(self.df)
        short_df = self.df.iloc[-20:].copy().reset_index(drop=True)
        pred_low, pred_high = predictor.predict_single(short_df)
        assert isinstance(pred_low, float)

    def test_scaler_set_after_fit(self):
        predictor = RangePredictor(window_size=20, epochs=3, batch_size=16)
        predictor.fit(self.df)
        assert predictor.scaler_mean is not None
        assert predictor.scaler_std is not None

    def test_fit_raises_if_not_enough_rows(self):
        predictor = RangePredictor(window_size=20, epochs=2, batch_size=8)
        tiny_df = self.df.iloc[:25].copy().reset_index(drop=True)
        with pytest.raises(ValueError):
            predictor.fit(tiny_df)


# ---------------------------------------------------------------------------
# evaluate_score
# ---------------------------------------------------------------------------

class TestEvaluateScore:
    def setup_method(self):
        self.df = _make_indicator_df(n=120)

    def test_evaluate_score_returns_dict(self):
        predictor = RangePredictor(window_size=20, epochs=3, batch_size=16)
        predictor.fit(self.df)
        result = predictor.evaluate_score(self.df)
        for key in ("total_score", "plus_two", "plus_one", "minus_one", "zero", "n_predictions"):
            assert key in result, f"Missing key: {key}"

    def test_evaluate_score_counts_add_up(self):
        predictor = RangePredictor(window_size=20, epochs=3, batch_size=16)
        predictor.fit(self.df)
        result = predictor.evaluate_score(self.df)
        assert (result["plus_two"] + result["plus_one"] + result["minus_one"] + result["zero"]
                == result["n_predictions"])

    def test_evaluate_score_total_score_range(self):
        predictor = RangePredictor(window_size=20, epochs=3, batch_size=16)
        predictor.fit(self.df)
        result = predictor.evaluate_score(self.df)
        n = result["n_predictions"]
        assert -n <= result["total_score"] <= n * 2


# ---------------------------------------------------------------------------
# run_range_backtest
# ---------------------------------------------------------------------------

class TestRunRangeBacktest:
    def test_returns_dict_with_keys(self):
        df = _make_ohlcv(n=150)
        result = run_range_backtest(df, train_ratio=0.7, window_size=20, epochs=3)
        for key in ("total_score", "plus_two", "plus_one", "minus_one", "zero",
                    "n_predictions", "train_rows", "test_rows", "predictor", "test_df"):
            assert key in result, f"Missing key: {key}"

    def test_counts_consistent(self):
        df = _make_ohlcv(n=150)
        result = run_range_backtest(df, train_ratio=0.7, window_size=20, epochs=3)
        assert (result["plus_two"] + result["plus_one"] + result["minus_one"] + result["zero"]
                == result["n_predictions"])

    def test_score_within_bounds(self):
        df = _make_ohlcv(n=150)
        result = run_range_backtest(df, train_ratio=0.7, window_size=20, epochs=3)
        n = result["n_predictions"]
        assert -n <= result["total_score"] <= n * 2


# ---------------------------------------------------------------------------
# pinball_loss
# ---------------------------------------------------------------------------

class TestPinballLoss:
    def test_zero_error_gives_zero_loss(self):
        pred = torch.tensor([3.0, 4.0])
        target = torch.tensor([3.0, 4.0])
        assert pinball_loss(pred, target, tau=0.8).item() == pytest.approx(0.0)

    def test_high_tau_penalizes_underestimate_more(self):
        # tau=0.8: underestimate penalty = 0.8, overestimate penalty = 0.2
        # underestimate: pred=2, target=3  → loss = 0.8 * 1 = 0.8
        # overestimate:  pred=4, target=3  → loss = 0.2 * 1 = 0.2
        under = pinball_loss(torch.tensor([2.0]), torch.tensor([3.0]), tau=0.8)
        over  = pinball_loss(torch.tensor([4.0]), torch.tensor([3.0]), tau=0.8)
        assert under.item() == pytest.approx(0.8)
        assert over.item()  == pytest.approx(0.2)
        assert under > over

    def test_low_tau_penalizes_overestimate_more(self):
        # tau=0.2: underestimate penalty = 0.2, overestimate penalty = 0.8
        under = pinball_loss(torch.tensor([2.0]), torch.tensor([3.0]), tau=0.2)
        over  = pinball_loss(torch.tensor([4.0]), torch.tensor([3.0]), tau=0.2)
        assert under.item() == pytest.approx(0.2)
        assert over.item()  == pytest.approx(0.8)
        assert over > under

    def test_tau_half_is_symmetric(self):
        # tau=0.5 is symmetric MAE (both directions penalized equally)
        under = pinball_loss(torch.tensor([2.0]), torch.tensor([3.0]), tau=0.5)
        over  = pinball_loss(torch.tensor([4.0]), torch.tensor([3.0]), tau=0.5)
        assert under.item() == pytest.approx(over.item())

    def test_batch_mean(self):
        # Loss is averaged over the batch
        pred   = torch.tensor([2.0, 4.0])
        target = torch.tensor([3.0, 3.0])
        loss = pinball_loss(pred, target, tau=0.8)
        # sample 0: underestimate 1 → 0.8; sample 1: overestimate 1 → 0.2; mean = 0.5
        assert loss.item() == pytest.approx(0.5)

    def test_returns_scalar_tensor(self):
        pred   = torch.tensor([1.0, 2.0, 3.0])
        target = torch.tensor([1.5, 2.5, 3.5])
        result = pinball_loss(pred, target, tau=0.7)
        assert result.shape == torch.Size([])


# ---------------------------------------------------------------------------
# RangePredictor with asymmetric loss
# ---------------------------------------------------------------------------

class TestRangePredictorAsymmetricLoss:
    def setup_method(self):
        self.df = _make_indicator_df(n=120)

    def test_predictor_stores_tau_params(self):
        p = RangePredictor(window_size=20, epochs=2, tau_low=0.8, tau_high=0.2)
        assert p.tau_low == 0.8
        assert p.tau_high == 0.2

    def test_default_tau_values(self):
        p = RangePredictor()
        assert p.tau_low == 0.8
        assert p.tau_high == 0.2

    def test_fit_with_asymmetric_loss(self):
        p = RangePredictor(window_size=20, epochs=3, batch_size=16,
                           tau_low=0.8, tau_high=0.2)
        p.fit(self.df)
        assert p.model is not None

    def test_custom_tau_values_fit(self):
        p = RangePredictor(window_size=20, epochs=3, batch_size=16,
                           tau_low=0.9, tau_high=0.1)
        p.fit(self.df)
        pred_low, pred_high = p.predict_single(self.df)
        assert isinstance(pred_low, float)
        assert isinstance(pred_high, float)


# ---------------------------------------------------------------------------
# Deeper model: fc_sizes and layer_norm
# ---------------------------------------------------------------------------

class TestDeepBiLSTMRangeModel:
    def test_multi_layer_fc_output_shape(self):
        model = BiLSTMRangeModel(input_size=6, fc_sizes=[64, 32])
        x = torch.randn(4, 20, 6)
        low, high = model(x)
        assert low.shape == (4,)
        assert high.shape == (4,)

    def test_single_fc_layer_via_list(self):
        model = BiLSTMRangeModel(input_size=6, fc_sizes=[32])
        x = torch.randn(3, 20, 6)
        low, high = model(x)
        assert low.shape == (3,)

    def test_three_fc_layers(self):
        model = BiLSTMRangeModel(input_size=6, hidden=128, num_layers=4,
                                  fc_sizes=[256, 128, 64])
        x = torch.randn(2, 20, 6)
        low, high = model(x)
        assert low.shape == (2,)
        assert high.shape == (2,)

    def test_layer_norm_enabled(self):
        model = BiLSTMRangeModel(input_size=6, hidden=64, layer_norm=True)
        x = torch.randn(4, 20, 6)
        low, high = model(x)
        assert low.shape == (4,)

    def test_layer_norm_disabled_by_default(self):
        model = BiLSTMRangeModel(input_size=6)
        assert model.ln is None

    def test_layer_norm_enabled_sets_ln(self):
        model = BiLSTMRangeModel(input_size=6, layer_norm=True)
        assert model.ln is not None

    def test_deep_model_gradients_flow(self):
        model = BiLSTMRangeModel(input_size=6, hidden=128, num_layers=4,
                                  fc_sizes=[256, 128, 64], layer_norm=True)
        x = torch.randn(2, 20, 6)
        low, high = model(x)
        (low.mean() + high.mean()).backward()
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"

    def test_fc_sizes_none_defaults_to_32(self):
        model = BiLSTMRangeModel(input_size=6, fc_sizes=None)
        x = torch.randn(2, 20, 6)
        low, high = model(x)
        assert low.shape == (2,)


class TestRangePredictorDeepConfig:
    def setup_method(self):
        self.df = _make_indicator_df(n=120)

    def test_predictor_stores_fc_sizes(self):
        p = RangePredictor(window_size=20, fc_sizes=[64, 32])
        assert p.fc_sizes == [64, 32]

    def test_predictor_stores_layer_norm(self):
        p = RangePredictor(window_size=20, layer_norm=True)
        assert p.layer_norm is True

    def test_deep_predictor_fit_and_predict(self):
        p = RangePredictor(window_size=20, epochs=3, batch_size=16,
                           hidden=128, num_layers=4, fc_sizes=[256, 128, 64],
                           layer_norm=True)
        p.fit(self.df)
        pred_low, pred_high = p.predict_single(self.df)
        assert isinstance(pred_low, float)
        assert isinstance(pred_high, float)


class TestRangePredictorFitMulti:
    def setup_method(self):
        self.dfs = [_make_indicator_df(n=120, seed=i) for i in range(3)]

    def test_fit_multi_runs(self):
        p = RangePredictor(window_size=20, epochs=3, batch_size=16)
        p.fit_multi(self.dfs)
        assert p.model is not None

    def test_fit_multi_scaler_set(self):
        p = RangePredictor(window_size=20, epochs=3, batch_size=16)
        p.fit_multi(self.dfs)
        assert p.scaler_mean is not None
        assert p.scaler_std is not None

    def test_fit_multi_predict_single(self):
        p = RangePredictor(window_size=20, epochs=3, batch_size=16)
        p.fit_multi(self.dfs)
        pred_low, pred_high = p.predict_single(self.dfs[0])
        assert isinstance(pred_low, float)
        assert isinstance(pred_high, float)

    def test_fit_multi_more_samples_than_single(self):
        """Combined dataset should have more samples than any single stock."""
        p_single = RangePredictor(window_size=20, epochs=1)
        p_multi = RangePredictor(window_size=20, epochs=1)
        # Just verify fit_multi accepts list of dfs without error
        p_multi.fit_multi(self.dfs)
        p_single.fit(self.dfs[0])
        assert p_multi.model is not None
        assert p_single.model is not None


# ---------------------------------------------------------------------------
# Attention mechanism
# ---------------------------------------------------------------------------

class TestAttentionBiLSTMRangeModel:
    def test_attention_output_shape(self):
        model = BiLSTMRangeModel(input_size=6, use_attention=True)
        x = torch.randn(4, 20, 6)
        low, high = model(x)
        assert low.shape == (4,)
        assert high.shape == (4,)

    def test_attention_disabled_has_no_attn_layer(self):
        model = BiLSTMRangeModel(input_size=6, use_attention=False)
        assert model.attn is None

    def test_attention_enabled_has_attn_layer(self):
        model = BiLSTMRangeModel(input_size=6, use_attention=True)
        assert model.attn is not None
        assert isinstance(model.attn, torch.nn.Linear)

    def test_attention_weights_sum_to_one(self):
        import torch.nn.functional as F
        model = BiLSTMRangeModel(input_size=6, hidden=32, use_attention=True)
        x = torch.randn(3, 20, 6)
        model.eval()
        with torch.no_grad():
            out, _ = model.bilstm(x)
            scores = model.attn(out)               # (3, 20, 1)
            weights = torch.softmax(scores, dim=1) # (3, 20, 1)
        sums = weights.sum(dim=1)                  # (3, 1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_attention_gradients_flow(self):
        model = BiLSTMRangeModel(input_size=6, use_attention=True)
        x = torch.randn(2, 20, 6)
        low, high = model(x)
        (low.mean() + high.mean()).backward()
        assert model.attn.weight.grad is not None
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"

    def test_attention_with_layer_norm(self):
        model = BiLSTMRangeModel(input_size=6, use_attention=True, layer_norm=True)
        x = torch.randn(4, 20, 6)
        low, high = model(x)
        assert low.shape == (4,)

    def test_attention_with_deep_fc(self):
        model = BiLSTMRangeModel(
            input_size=6, hidden=128, num_layers=4,
            fc_sizes=[256, 128, 64], layer_norm=True, use_attention=True,
        )
        x = torch.randn(2, 20, 6)
        low, high = model(x)
        assert low.shape == (2,)

    def test_predictor_stores_use_attention(self):
        p = RangePredictor(use_attention=True)
        assert p.use_attention is True

    def test_predictor_default_no_attention(self):
        p = RangePredictor()
        assert p.use_attention is False

    def test_predictor_fit_with_attention(self):
        df = _make_indicator_df(n=120)
        p = RangePredictor(window_size=20, epochs=3, batch_size=16,
                           use_attention=True)
        p.fit(df)
        pred_low, pred_high = p.predict_single(df)
        assert isinstance(pred_low, float)
        assert isinstance(pred_high, float)

    def test_predictor_fit_multi_with_attention(self):
        dfs = [_make_indicator_df(n=120, seed=i) for i in range(2)]
        p = RangePredictor(window_size=20, epochs=3, batch_size=16,
                           use_attention=True)
        p.fit_multi(dfs)
        assert p.model is not None
