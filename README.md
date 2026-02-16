# Claude Trader

Trading signal bots for China A-shares (601933 Yonghui Superstores). Three approaches — unsupervised K-Means clustering, supervised LSTM, and LightGBM gradient-boosted trees — generate daily signals (strong\_buy, mild\_buy, hold, mild\_sell, strong\_sell) from technical indicators, then simulate trades with realistic A-share constraints (T+1, 100-share lots, commissions, stamp tax).

A hyperparameter tuning pipeline finds optimal configs via grid search with chronological inner validation.

## Project Structure

```
claude_trader/
├── fetch_china_stock.py        # Download daily OHLCV from akshare
├── trading_bot.py              # K-Means clustering bot + backtest engine
├── dnn_trading_bot.py          # LSTM neural network bot + backtest engine
├── lgbm_trading_bot.py         # LightGBM gradient-boosted tree bot + backtest
├── compare_models.py           # 3-model comparison + combined trade log
├── tune_hyperparams.py         # Grid search hyperparameter tuning (all 3 models)
├── test_fetch_china_stock.py   # Tests for data fetcher
├── test_trading_bot.py         # Tests for K-Means bot (37 tests)
├── test_dnn_trading_bot.py     # Tests for LSTM bot (18 tests)
├── test_lgbm_trading_bot.py    # Tests for LightGBM bot (12 tests)
├── test_tune_hyperparams.py    # Tests for tuning pipeline (24 tests)
├── 601933_10yr.csv             # 10-year price data (2016–2025, ~2400 rows)
├── 601933_3yr.csv              # 3-year price data (legacy)
├── tuning_results.json         # Saved tuning output
└── CLAUDE.md                   # AI assistant instructions
```

## Setup

### 1. Create conda environment

```bash
conda create -n trader python=3.12 -y
conda activate trader
```

### 2. Install dependencies

```bash
pip install akshare pandas numpy scikit-learn torch lightgbm pytest
```

Required packages:

| Package      | Purpose                          |
|--------------|----------------------------------|
| akshare      | China A-share market data API    |
| pandas       | Data manipulation                |
| numpy        | Numerical computation            |
| scikit-learn | K-Means clustering, scaling      |
| torch        | LSTM neural network (PyTorch)    |
| lightgbm     | Gradient-boosted tree classifier |
| pytest       | Test runner                      |

### 3. Fetch stock data

```bash
conda activate trader
python fetch_china_stock.py 601933 --start 20160101 --end 20251231 --csv 601933_10yr.csv
```

The 10yr CSV is already included in the repo. To fetch a different stock or date range:

```bash
python fetch_china_stock.py <SYMBOL> --start YYYYMMDD --end YYYYMMDD --csv output.csv
```

## Usage

### Run tests

```bash
conda activate trader
pytest test_trading_bot.py test_dnn_trading_bot.py test_lgbm_trading_bot.py test_tune_hyperparams.py -v
```

80 tests total — all should pass.

### Run K-Means trading bot

```bash
python trading_bot.py                          # uses 601933_10yr.csv
python trading_bot.py --csv 601933_3yr.csv     # use a different file
```

Outputs cluster analysis, backtest metrics, recent trades, and today's signal.

### Run LSTM trading bot

```bash
python dnn_trading_bot.py                      # uses 601933_10yr.csv
python dnn_trading_bot.py --csv 601933_3yr.csv
```

Trains the LSTM, runs backtest, prints a comparison table vs K-Means baseline, and today's signal.

### Run LightGBM trading bot

```bash
python lgbm_trading_bot.py                     # uses 601933_10yr.csv
python lgbm_trading_bot.py --csv 601933_3yr.csv
```

Trains a LightGBM classifier, runs backtest, prints feature importance, comparison vs K-Means, and today's signal.

### Compare all models

```bash
python compare_models.py                       # 3-model comparison + trade_log.csv
```

Runs all three backtests, prints a side-by-side comparison table, and saves a combined daily trade log with signals and actions for each model.

### Run hyperparameter tuning

```bash
python tune_hyperparams.py                     # full tuning, ~3 min
python tune_hyperparams.py --csv 601933_10yr.csv --output tuning_results.json
```

This runs:
1. **K-Means grid search** (30 configs) — varies `n_clusters` [3–8] and feature subsets
2. **LSTM 2-phase grid search** (~40 configs) — Phase 1 varies window/lr/batch, Phase 2 varies hidden sizes
3. **LightGBM grid search** (27 configs) — varies `n_estimators`, `max_depth`, `learning_rate`
4. **Final evaluation** — retrains best configs on full training set, tests on held-out 40%

Results are printed as a comparison table and saved to `tuning_results.json`.

## How It Works

### Technical Indicators (6 features)

| Indicator   | Description                       |
|-------------|-----------------------------------|
| RSI(14)     | Relative Strength Index           |
| MACD Hist   | MACD histogram (12, 26, 9)        |
| Boll %B(20) | Bollinger Band percent position   |
| Vol Ratio   | Volume / 20-day average volume    |
| ROC(10)     | 10-day rate of change             |
| ATR Ratio   | ATR(14) / close (norm. volatility)|

### K-Means Bot

1. Compute indicators on training data
2. Fit K-Means (configurable clusters) on scaled features
3. Rank clusters by average next-day forward return
4. Map clusters to signal levels (strong\_sell → strong\_buy)
5. On test data: predict cluster → signal → execute trade at next day's open

### LSTM Bot

1. Build sliding windows of indicator sequences
2. Label windows by forward return percentile (5-class)
3. Train LSTM(hidden1) → LSTM(hidden2) → FC(16) → FC(5) with early stopping
4. Predict class → signal → execute trade at next day's open

### LightGBM Bot

1. Compute indicators — each row is an independent sample (no sliding windows)
2. Label by forward 1-day return percentile (same 5-class scheme as LSTM)
3. Train LGBMClassifier on scaled features
4. Predict class → signal → execute trade at next day's open

Key advantage over LSTM: trains in seconds (vs minutes), works well with small tabular datasets, and provides feature importance rankings.

### Backtest Rules

- **Walk-forward**: train on first 60%, test on remaining 40%
- **Execution**: signals generated at close, executed at next day's open
- **T+1 settlement**: cannot sell shares bought on the same day
- **Lot size**: all trades rounded to 100-share lots
- **Costs**: 0.025% commission (min 5 RMB) + 0.05% stamp tax on sells

### Tuning Methodology

- **Outer split**: 60% train / 40% test (test untouched during tuning)
- **Inner split**: 75% train / 25% validation within the 60% (chronological, no leakage)
- **Ranking metric**: Sharpe ratio on inner validation set
- **Final eval**: retrain best params on full 60%, evaluate on 40%

## Results (601933, 10yr data)

| Metric       | KM Original | KM Tuned | LSTM Original | LSTM Tuned | LGBM Original | LGBM Tuned | Buy & Hold |
|--------------|-------------|----------|---------------|------------|---------------|------------|------------|
| Total Return | **+44.25%** | +0.62%   | +0.00%        | +2.90%     | +23.67%       | -7.06%     | +10.86%    |
| Sharpe Ratio | **0.478**   | 0.154    | 0.000         | 0.121      | 0.352         | 0.019      | —          |
| Max Drawdown | -37.51%     | -42.27%  | 0.00%         | -15.19%    | **-25.71%**   | -30.30%    | —          |
| Win Rate     | 47.9%       | 47.8%    | 0.0%          | 50.0%      | 54.8%         | **59.5%**  | —          |
| Num Trades   | 216         | 246      | 0             | 86         | 380           | 363        | 1          |

**Best K-Means params**: `n_clusters=8, features=drop_roc`

**Best LSTM params**: `window_size=10, lr=0.001, batch_size=16, hidden1=64, hidden2=32`

**Best LightGBM params**: `n_estimators=50, max_depth=7, learning_rate=0.1`

### Analysis

- **K-Means (default)** is the clear winner with +44.25% return and the highest Sharpe ratio (0.478). It outperforms Buy & Hold by 4x.
- **LightGBM (default)** comes second at +23.67%, more than doubling Buy & Hold. It has the best drawdown control (-25.71%) and highest win rate (54.8%), making it the most conservative profitable strategy.
- **LSTM** struggles on this dataset, producing near-zero returns in most configurations.
- **Tuning hurt both LightGBM and K-Means** on the test set — all LightGBM configs had negative Sharpe on the inner validation set, indicating overfitting to the training period. The default hyperparameters remain the best choice for both.
- **Feature importance** (LightGBM): ATR ratio and MACD histogram are the most important features, followed by volume ratio and ROC.
