#!/usr/bin/env python3
"""Multi-stock experiment: compare baseline vs deeper BiLSTM on all data/CSVs.

Trains one model per config on the combined training data from all stocks,
then evaluates each model on every stock's held-out test set.

Usage
-----
    python multi_stock_experiment.py [--data-dir data] [--train-ratio 0.7]
                                     [--epochs 150] [--patience 20]
"""

import argparse
import glob
import os

import pandas as pd

from range_predictor import RangePredictor
from trading_bot import FEATURE_COLS, compute_indicators


# ---------------------------------------------------------------------------
# Model configs
# ---------------------------------------------------------------------------

CONFIGS = {
    "baseline": dict(
        hidden=64,
        num_layers=2,
        fc_sizes=[32],
        layer_norm=False,
        dropout=0.2,
        use_attention=False,
    ),
    "deep": dict(
        hidden=128,
        num_layers=4,
        fc_sizes=[256, 128, 64],
        layer_norm=True,
        dropout=0.3,
        use_attention=False,
    ),
    "attn": dict(
        hidden=64,
        num_layers=2,
        fc_sizes=[32],
        layer_norm=False,
        dropout=0.2,
        use_attention=True,
    ),
    "deep_attn": dict(
        hidden=128,
        num_layers=4,
        fc_sizes=[256, 128, 64],
        layer_norm=True,
        dropout=0.3,
        use_attention=True,
    ),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def discover_csvs(data_dir: str) -> list[str]:
    """Return sorted list of CSV file paths in data_dir."""
    pattern = os.path.join(data_dir, "*.csv")
    return sorted(glob.glob(pattern))


def load_stock_df(csv_path: str) -> pd.DataFrame:
    """Load CSV, compute indicators, drop NaN rows."""
    df = pd.read_csv(csv_path)
    df = compute_indicators(df)
    df = df.dropna(subset=FEATURE_COLS).reset_index(drop=True)
    return df


def split_train_test(df: pd.DataFrame, train_ratio: float = 0.7):
    """Chronological train/test split."""
    split = int(len(df) * train_ratio)
    train = df.iloc[:split].copy().reset_index(drop=True)
    test = df.iloc[split:].copy().reset_index(drop=True)
    return train, test


def build_predictor(config_name: str, **kwargs) -> RangePredictor:
    """Build a RangePredictor for a named config, overriding with kwargs."""
    if config_name not in CONFIGS:
        raise ValueError(
            f"Unknown config '{config_name}'. Available: {list(CONFIGS.keys())}"
        )
    params = {**CONFIGS[config_name], **kwargs}
    return RangePredictor(**params)


def evaluate_on_stocks(
    predictor: RangePredictor,
    test_dfs: dict[str, pd.DataFrame],
) -> dict[str, dict]:
    """Evaluate a fitted predictor on each stock's test DataFrame."""
    results = {}
    for stock_name, test_df in test_dfs.items():
        if len(test_df) < predictor.window_size + 2:
            print(f"  [SKIP] {stock_name}: too few rows ({len(test_df)})")
            results[stock_name] = {"total_score": 0, "n_predictions": 0,
                                    "plus_two": 0, "plus_one": 0,
                                    "minus_one": 0, "zero": 0}
            continue
        scores = predictor.evaluate_score(test_df)
        results[stock_name] = scores
    return results


def run_experiment(
    data_dir: str = "data",
    train_ratio: float = 0.7,
    epochs: int = 150,
    batch_size: int = 64,
    lr: float = 1e-3,
    patience: int = 20,
    window_size: int = 20,
    configs: list[str] | None = None,
) -> dict[str, dict[str, dict]]:
    """Train each config on combined data, evaluate per stock.

    Returns
    -------
    { config_name: { stock_name: score_dict } }
    """
    if configs is None:
        configs = list(CONFIGS.keys())

    csv_paths = discover_csvs(data_dir)
    if not csv_paths:
        raise FileNotFoundError(f"No CSV files found in '{data_dir}'")

    # Load all stocks and split
    print(f"\nLoading {len(csv_paths)} stock(s) from '{data_dir}'...")
    train_dfs = []
    test_dfs = {}
    for path in csv_paths:
        name = os.path.splitext(os.path.basename(path))[0]
        df = load_stock_df(path)
        train, test = split_train_test(df, train_ratio)
        train_dfs.append(train)
        test_dfs[name] = test
        print(f"  {name}: {len(df)} rows total  "
              f"({len(train)} train / {len(test)} test)")

    total_train = sum(len(d) for d in train_dfs)
    print(f"\n  Combined training rows: {total_train}")

    # Run each config
    all_results: dict[str, dict[str, dict]] = {}
    for config_name in configs:
        print(f"\n{'=' * 60}")
        print(f"CONFIG: {config_name}  {CONFIGS[config_name]}")
        print("=" * 60)

        predictor = build_predictor(
            config_name,
            window_size=window_size,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            patience=patience,
        )
        print(f"Training on combined data ({total_train} rows)...")
        predictor.fit_multi(train_dfs)

        print("Evaluating on each stock's test set...")
        results = evaluate_on_stocks(predictor, test_dfs)
        all_results[config_name] = results

    return all_results


def _print_comparison(all_results: dict[str, dict[str, dict]]) -> None:
    """Print a side-by-side comparison table."""
    configs = list(all_results.keys())
    stocks = list(next(iter(all_results.values())).keys())

    col_w = 22
    header = f"{'Stock':<20}" + "".join(
        f"  {c:>{col_w}}" for c in configs
    )
    print("\n" + "=" * len(header))
    print("COMPARISON  (total_score / n_predictions = score_per_pred)")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for stock in stocks:
        row = f"{stock:<20}"
        for cfg in configs:
            r = all_results[cfg][stock]
            n = r["n_predictions"]
            ts = r["total_score"]
            spp = ts / n if n > 0 else 0.0
            cell = f"{ts:+d}/{n} ({spp:+.3f})"
            row += f"  {cell:>{col_w}}"
        print(row)

    # Totals
    print("-" * len(header))
    totals_row = f"{'TOTAL':<20}"
    for cfg in configs:
        total_ts = sum(all_results[cfg][s]["total_score"] for s in stocks)
        total_n = sum(all_results[cfg][s]["n_predictions"] for s in stocks)
        spp = total_ts / total_n if total_n > 0 else 0.0
        cell = f"{total_ts:+d}/{total_n} ({spp:+.3f})"
        totals_row += f"  {cell:>{col_w}}"
    print(totals_row)
    print("=" * len(header))

    # Breakdown per config
    for cfg in configs:
        print(f"\n--- {cfg} breakdown ---")
        for stock in stocks:
            r = all_results[cfg][stock]
            n = r["n_predictions"]
            if n == 0:
                print(f"  {stock}: no predictions")
                continue
            print(
                f"  {stock}: "
                f"+2={r['plus_two']} ({r['plus_two']/n*100:.0f}%)  "
                f"+1={r['plus_one']} ({r['plus_one']/n*100:.0f}%)  "
                f"0={r['zero']} ({r['zero']/n*100:.0f}%)  "
                f"-1={r['minus_one']} ({r['minus_one']/n*100:.0f}%)  "
                f"score={r['total_score']:+d}"
            )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compare baseline vs deeper BiLSTM on all stocks"
    )
    parser.add_argument("--data-dir", default="data",
                        help="Directory containing CSV files")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--window", type=int, default=20)
    parser.add_argument("--configs", nargs="+", default=None,
                        choices=list(CONFIGS.keys()),
                        help="Which configs to run (default: all)")
    args = parser.parse_args()

    all_results = run_experiment(
        data_dir=args.data_dir,
        train_ratio=args.train_ratio,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
        window_size=args.window,
        configs=args.configs,
    )

    _print_comparison(all_results)


if __name__ == "__main__":
    main()
