#!/usr/bin/env python3
"""Hyperparameter tuning for K-Means and LSTM trading bots.

Grid search with inner chronological validation split.
Outer split: 60% train / 40% test (untouched during tuning).
Inner split: 75% train / 25% val within the 60%.
Rank by Sharpe ratio. Final eval: retrain best on full 60%, test on 40%.
"""

import argparse
import itertools
import json
import time

import numpy as np
import pandas as pd

from trading_bot import (
    FEATURE_COLS,
    TradingBot,
    Portfolio,
    LOT_SIZE,
    compute_indicators,
    run_backtest,
)

# ---------------------------------------------------------------------------
# Grids
# ---------------------------------------------------------------------------
KMEANS_GRID = {
    "n_clusters": [3, 4, 5, 6, 7, 8],
    "feature_subsets": [
        ("all_6", FEATURE_COLS),
        ("drop_vol", ["rsi", "macd_hist", "boll_pctb", "roc", "atr_ratio"]),
        ("drop_atr", ["rsi", "macd_hist", "boll_pctb", "vol_ratio", "roc"]),
        ("drop_roc", ["rsi", "macd_hist", "boll_pctb", "vol_ratio", "atr_ratio"]),
        ("core_4", ["rsi", "macd_hist", "boll_pctb", "vol_ratio"]),
    ],
}

LSTM_PHASE1_GRID = {
    "window_size": [10, 20, 30],
    "lr": [0.0001, 0.0005, 0.001, 0.005],
    "batch_size": [16, 32, 64],
}

LSTM_PHASE2_GRID = {
    "hidden1": [32, 64],
    "hidden2": [16, 32],
}


# ---------------------------------------------------------------------------
# Inner split helper
# ---------------------------------------------------------------------------
def make_inner_split(
    df: pd.DataFrame, val_ratio: float = 0.25
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chronological inner split: (1-val_ratio) train, val_ratio val."""
    n = len(df)
    split = int(n * (1 - val_ratio))
    train = df.iloc[:split].copy().reset_index(drop=True)
    val = df.iloc[split:].copy().reset_index(drop=True)
    return train, val


# ---------------------------------------------------------------------------
# Evaluate K-Means on a val set
# ---------------------------------------------------------------------------
def _eval_kmeans(train_df, val_df, n_clusters, feature_cols, initial_capital=100_000):
    """Train K-Means on train_df, evaluate on val_df. Return metrics dict."""
    bot = TradingBot(n_clusters=n_clusters, feature_cols=feature_cols)
    bot.fit(train_df)
    signals = bot.predict(val_df)
    val_df = val_df.copy()
    val_df["signal"] = signals

    portfolio = Portfolio(capital=initial_capital)
    daily_values = []

    for i in range(len(val_df) - 1):
        signal = val_df.iloc[i]["signal"]
        exec_price = val_df.iloc[i + 1]["open"]
        trade_date = str(val_df.iloc[i + 1]["date"])

        if signal == "strong_buy":
            portfolio.buy(exec_price, fraction=1.0, trade_date=trade_date)
        elif signal == "mild_buy":
            portfolio.buy(exec_price, fraction=0.5, trade_date=trade_date)
        elif signal == "strong_sell":
            portfolio.sell(exec_price, fraction=1.0, trade_date=trade_date)
        elif signal == "mild_sell":
            portfolio.sell(exec_price, fraction=0.5, trade_date=trade_date)

        daily_values.append(portfolio.value(val_df.iloc[i + 1]["close"]))

    final_value = portfolio.value(val_df.iloc[-1]["close"])
    total_return = (final_value - initial_capital) / initial_capital * 100

    values = np.array([initial_capital] + daily_values)
    peak = np.maximum.accumulate(values)
    drawdowns = (values - peak) / peak
    max_drawdown = drawdowns.min() * 100

    daily_returns = np.diff(values) / values[:-1]
    sharpe = (
        np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252)
        if len(daily_returns) > 1 and np.std(daily_returns) > 0
        else 0.0
    )

    return {
        "total_return": total_return,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_drawdown,
        "final_value": final_value,
    }


# ---------------------------------------------------------------------------
# Evaluate LSTM on a val set
# ---------------------------------------------------------------------------
def _eval_lstm(
    train_df, val_df, window_size, lr, batch_size, hidden1, hidden2,
    epochs=30, patience=8, initial_capital=100_000,
):
    """Train LSTM on train_df, evaluate on val_df. Return metrics dict."""
    from dnn_trading_bot import DNNTradingBot, SIGNAL_NAMES

    bot = DNNTradingBot(
        window_size=window_size, epochs=epochs, batch_size=batch_size,
        lr=lr, patience=patience, hidden1=hidden1, hidden2=hidden2,
    )
    bot.fit(train_df)

    signals = bot.predict(val_df)
    signal_start = window_size
    test_signals = {}
    for i, sig in enumerate(signals):
        row_idx = i + signal_start
        if row_idx < len(val_df):
            test_signals[row_idx] = sig

    portfolio = Portfolio(capital=initial_capital)
    daily_values = []

    for i in range(len(val_df) - 1):
        signal = test_signals.get(i, "hold")
        exec_price = val_df.iloc[i + 1]["open"]
        trade_date = str(val_df.iloc[i + 1]["date"])

        if signal == "strong_buy":
            portfolio.buy(exec_price, fraction=1.0, trade_date=trade_date)
        elif signal == "mild_buy":
            portfolio.buy(exec_price, fraction=0.5, trade_date=trade_date)
        elif signal == "strong_sell":
            portfolio.sell(exec_price, fraction=1.0, trade_date=trade_date)
        elif signal == "mild_sell":
            portfolio.sell(exec_price, fraction=0.5, trade_date=trade_date)

        daily_values.append(portfolio.value(val_df.iloc[i + 1]["close"]))

    final_value = portfolio.value(val_df.iloc[-1]["close"])
    total_return = (final_value - initial_capital) / initial_capital * 100

    values = np.array([initial_capital] + daily_values)
    peak = np.maximum.accumulate(values)
    drawdowns = (values - peak) / peak
    max_drawdown = drawdowns.min() * 100

    daily_returns = np.diff(values) / values[:-1]
    sharpe = (
        np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252)
        if len(daily_returns) > 1 and np.std(daily_returns) > 0
        else 0.0
    )

    return {
        "total_return": total_return,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_drawdown,
        "final_value": final_value,
    }


# ---------------------------------------------------------------------------
# Tune K-Means
# ---------------------------------------------------------------------------
def tune_kmeans(
    df: pd.DataFrame,
    train_ratio: float = 0.6,
    top_k: int = 5,
    initial_capital: float = 100_000,
) -> list[dict]:
    """Grid search over K-Means hyperparameters with inner validation."""
    df = compute_indicators(df)
    df = df.dropna(subset=FEATURE_COLS).reset_index(drop=True)

    split = int(len(df) * train_ratio)
    outer_train = df.iloc[:split].copy().reset_index(drop=True)

    inner_train, inner_val = make_inner_split(outer_train, val_ratio=0.25)

    results = []
    total = len(KMEANS_GRID["n_clusters"]) * len(KMEANS_GRID["feature_subsets"])
    print(f"\nTuning K-Means: {total} configurations...")

    for n_clusters in KMEANS_GRID["n_clusters"]:
        for subset_name, feature_cols in KMEANS_GRID["feature_subsets"]:
            try:
                metrics = _eval_kmeans(
                    inner_train, inner_val, n_clusters, feature_cols, initial_capital,
                )
                results.append({
                    "params": {
                        "n_clusters": n_clusters,
                        "feature_subset": subset_name,
                        "feature_cols": feature_cols,
                    },
                    **metrics,
                })
            except Exception as e:
                print(f"  SKIP n_clusters={n_clusters}, features={subset_name}: {e}")

    results.sort(key=lambda x: x["sharpe_ratio"], reverse=True)
    return results[:top_k]


# ---------------------------------------------------------------------------
# Tune LSTM (2-phase)
# ---------------------------------------------------------------------------
def tune_lstm(
    df: pd.DataFrame,
    train_ratio: float = 0.6,
    top_k: int = 5,
    initial_capital: float = 100_000,
    epochs: int = 30,
    patience: int = 8,
) -> list[dict]:
    """Two-phase grid search over LSTM hyperparameters."""
    df = compute_indicators(df)
    df = df.dropna(subset=FEATURE_COLS).reset_index(drop=True)

    split = int(len(df) * train_ratio)
    outer_train = df.iloc[:split].copy().reset_index(drop=True)

    inner_train, inner_val = make_inner_split(outer_train, val_ratio=0.25)

    # Phase 1: window_size x lr x batch_size (hidden fixed at 64/32)
    phase1_configs = list(itertools.product(
        LSTM_PHASE1_GRID["window_size"],
        LSTM_PHASE1_GRID["lr"],
        LSTM_PHASE1_GRID["batch_size"],
    ))
    total_p1 = len(phase1_configs)
    print(f"\nTuning LSTM Phase 1: {total_p1} configurations...")

    phase1_results = []
    for idx, (ws, lr, bs) in enumerate(phase1_configs, 1):
        print(f"  [{idx}/{total_p1}] ws={ws}, lr={lr}, bs={bs}", end="", flush=True)
        t0 = time.time()
        try:
            metrics = _eval_lstm(
                inner_train, inner_val, ws, lr, bs, 64, 32,
                epochs=epochs, patience=patience, initial_capital=initial_capital,
            )
            elapsed = time.time() - t0
            print(f"  sharpe={metrics['sharpe_ratio']:.3f}  ({elapsed:.1f}s)")
            phase1_results.append({
                "params": {"window_size": ws, "lr": lr, "batch_size": bs,
                           "hidden1": 64, "hidden2": 32},
                **metrics,
            })
        except Exception as e:
            print(f"  SKIP: {e}")

    phase1_results.sort(key=lambda x: x["sharpe_ratio"], reverse=True)
    if not phase1_results:
        return []

    best_p1 = phase1_results[0]["params"]
    print(f"\nBest Phase 1: {best_p1}")

    # Phase 2: hidden1 x hidden2 (use best ws/lr/bs from Phase 1)
    phase2_configs = list(itertools.product(
        LSTM_PHASE2_GRID["hidden1"],
        LSTM_PHASE2_GRID["hidden2"],
    ))
    total_p2 = len(phase2_configs)
    print(f"\nTuning LSTM Phase 2: {total_p2} configurations...")

    phase2_results = []
    for idx, (h1, h2) in enumerate(phase2_configs, 1):
        print(f"  [{idx}/{total_p2}] h1={h1}, h2={h2}", end="", flush=True)
        t0 = time.time()
        try:
            metrics = _eval_lstm(
                inner_train, inner_val,
                best_p1["window_size"], best_p1["lr"], best_p1["batch_size"],
                h1, h2,
                epochs=epochs, patience=patience, initial_capital=initial_capital,
            )
            elapsed = time.time() - t0
            print(f"  sharpe={metrics['sharpe_ratio']:.3f}  ({elapsed:.1f}s)")
            phase2_results.append({
                "params": {
                    "window_size": best_p1["window_size"],
                    "lr": best_p1["lr"],
                    "batch_size": best_p1["batch_size"],
                    "hidden1": h1, "hidden2": h2,
                },
                **metrics,
            })
        except Exception as e:
            print(f"  SKIP: {e}")

    all_results = phase1_results + phase2_results
    all_results.sort(key=lambda x: x["sharpe_ratio"], reverse=True)
    return all_results[:top_k]


# ---------------------------------------------------------------------------
# Final evaluation
# ---------------------------------------------------------------------------
def final_evaluation(
    df_raw: pd.DataFrame,
    best_kmeans_params: dict,
    best_lstm_params: dict,
    train_ratio: float = 0.6,
    initial_capital: float = 100_000,
) -> dict:
    """Retrain best configs on full outer train set, evaluate on test set."""
    from dnn_trading_bot import run_dnn_backtest

    # K-Means: original defaults
    km_orig = run_backtest(
        df_raw, train_ratio=train_ratio, initial_capital=initial_capital,
        n_clusters=5, feature_cols=None,
    )

    # K-Means: tuned
    km_tuned = run_backtest(
        df_raw, train_ratio=train_ratio, initial_capital=initial_capital,
        n_clusters=best_kmeans_params["n_clusters"],
        feature_cols=best_kmeans_params.get("feature_cols"),
    )

    # LSTM: original defaults
    lstm_orig = run_dnn_backtest(
        df_raw, train_ratio=train_ratio, initial_capital=initial_capital,
        window_size=20, epochs=50, batch_size=32, lr=0.001,
        hidden1=64, hidden2=32,
    )

    # LSTM: tuned
    lstm_tuned = run_dnn_backtest(
        df_raw, train_ratio=train_ratio, initial_capital=initial_capital,
        window_size=best_lstm_params["window_size"],
        epochs=50,
        batch_size=best_lstm_params["batch_size"],
        lr=best_lstm_params["lr"],
        hidden1=best_lstm_params["hidden1"],
        hidden2=best_lstm_params["hidden2"],
    )

    return {
        "km_original": _extract_metrics(km_orig),
        "km_tuned": _extract_metrics(km_tuned),
        "lstm_original": _extract_metrics(lstm_orig),
        "lstm_tuned": _extract_metrics(lstm_tuned),
        "buy_and_hold_return": km_orig["buy_and_hold_return"],
    }


def _extract_metrics(results: dict) -> dict:
    return {
        "total_return": results["total_return"],
        "sharpe_ratio": results["sharpe_ratio"],
        "max_drawdown": results["max_drawdown"],
        "win_rate": results["win_rate"],
        "profit_factor": results["profit_factor"],
        "num_trades": results["num_trades"],
        "final_value": results["final_value"],
    }


# ---------------------------------------------------------------------------
# Pretty print
# ---------------------------------------------------------------------------
def print_comparison(eval_results: dict, best_km: dict, best_lstm: dict):
    """Print comparison table and best params."""
    print(f"\n{'=' * 80}")
    print("COMPARISON TABLE (Final Evaluation on Test Set)")
    print("=" * 80)

    header = (
        f"  {'Metric':<20s}"
        f"  {'KM Orig':>12s}"
        f"  {'KM Tuned':>12s}"
        f"  {'LSTM Orig':>12s}"
        f"  {'LSTM Tuned':>12s}"
        f"  {'Buy&Hold':>12s}"
    )
    print(header)
    print("  " + "-" * 78)

    km_o = eval_results["km_original"]
    km_t = eval_results["km_tuned"]
    ls_o = eval_results["lstm_original"]
    ls_t = eval_results["lstm_tuned"]
    bh = eval_results["buy_and_hold_return"]

    rows = [
        ("Total Return",
         f"{km_o['total_return']:+.2f}%", f"{km_t['total_return']:+.2f}%",
         f"{ls_o['total_return']:+.2f}%", f"{ls_t['total_return']:+.2f}%",
         f"{bh:+.2f}%"),
        ("Sharpe Ratio",
         f"{km_o['sharpe_ratio']:.3f}", f"{km_t['sharpe_ratio']:.3f}",
         f"{ls_o['sharpe_ratio']:.3f}", f"{ls_t['sharpe_ratio']:.3f}",
         "N/A"),
        ("Max Drawdown",
         f"{km_o['max_drawdown']:.2f}%", f"{km_t['max_drawdown']:.2f}%",
         f"{ls_o['max_drawdown']:.2f}%", f"{ls_t['max_drawdown']:.2f}%",
         "N/A"),
        ("Win Rate",
         f"{km_o['win_rate']:.1f}%", f"{km_t['win_rate']:.1f}%",
         f"{ls_o['win_rate']:.1f}%", f"{ls_t['win_rate']:.1f}%",
         "N/A"),
        ("Num Trades",
         f"{km_o['num_trades']}", f"{km_t['num_trades']}",
         f"{ls_o['num_trades']}", f"{ls_t['num_trades']}",
         "1"),
        ("Final Value",
         f"{km_o['final_value']:,.0f}", f"{km_t['final_value']:,.0f}",
         f"{ls_o['final_value']:,.0f}", f"{ls_t['final_value']:,.0f}",
         "N/A"),
    ]
    for label, *vals in rows:
        print(f"  {label:<20s}" + "".join(f"  {v:>12s}" for v in vals))

    print(f"\n{'=' * 80}")
    print("BEST PARAMS")
    print("=" * 80)
    print(f"  K-Means: n_clusters={best_km['n_clusters']}, "
          f"features={best_km.get('feature_subset', 'all_6')}")
    print(f"  LSTM:    ws={best_lstm['window_size']}, lr={best_lstm['lr']}, "
          f"bs={best_lstm['batch_size']}, "
          f"h1={best_lstm['hidden1']}, h2={best_lstm['hidden2']}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Hyperparameter tuning")
    parser.add_argument("--csv", default="601933_10yr.csv", help="CSV file path")
    parser.add_argument("--train-ratio", type=float, default=0.6)
    parser.add_argument("--output", default="tuning_results.json")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    print(f"Loaded {len(df)} rows from {args.csv}")
    print(f"Date range: {df['date'].iloc[0]} to {df['date'].iloc[-1]}")

    t_start = time.time()

    # --- Tune K-Means ---
    km_results = tune_kmeans(df, train_ratio=args.train_ratio, top_k=5)
    print(f"\nTop 5 K-Means configs (by val Sharpe):")
    for i, r in enumerate(km_results, 1):
        p = r["params"]
        print(f"  {i}. n={p['n_clusters']}, feat={p['feature_subset']}"
              f"  sharpe={r['sharpe_ratio']:.3f}  ret={r['total_return']:+.2f}%")

    # --- Tune LSTM ---
    lstm_results = tune_lstm(
        df, train_ratio=args.train_ratio, top_k=5,
        epochs=30, patience=8,
    )
    print(f"\nTop 5 LSTM configs (by val Sharpe):")
    for i, r in enumerate(lstm_results, 1):
        p = r["params"]
        print(f"  {i}. ws={p['window_size']}, lr={p['lr']}, bs={p['batch_size']}, "
              f"h1={p['hidden1']}, h2={p['hidden2']}"
              f"  sharpe={r['sharpe_ratio']:.3f}  ret={r['total_return']:+.2f}%")

    # --- Final Evaluation ---
    best_km = km_results[0]["params"]
    best_lstm = lstm_results[0]["params"]

    print(f"\n{'=' * 80}")
    print("FINAL EVALUATION: Retraining best configs on full outer train set")
    print("=" * 80)

    eval_results = final_evaluation(
        df, best_km, best_lstm,
        train_ratio=args.train_ratio,
    )

    print_comparison(eval_results, best_km, best_lstm)

    elapsed = time.time() - t_start
    print(f"\nTotal tuning time: {elapsed:.1f}s")

    # --- Save results ---
    output = {
        "best_kmeans_params": {k: v for k, v in best_km.items() if k != "feature_cols"},
        "best_lstm_params": best_lstm,
        "kmeans_top5": [
            {k: v for k, v in r.items() if k != "params" or k == "params"}
            for r in km_results
        ],
        "lstm_top5": lstm_results,
        "final_evaluation": eval_results,
    }
    # Convert feature_cols lists (not JSON-serializable as-is with numpy)
    for r in output["kmeans_top5"]:
        if "params" in r and "feature_cols" in r["params"]:
            r["params"]["feature_cols"] = list(r["params"]["feature_cols"])

    with open(args.output, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
