# Claude Trader

Trading signal bots for China A-shares. Four ML models — unsupervised K-Means clustering, supervised LSTM, LightGBM gradient-boosted trees, and PPO reinforcement learning — generate daily signals (strong\_buy, mild\_buy, hold, mild\_sell, strong\_sell) from technical indicators, then simulate trades with realistic A-share constraints (T+1, 100-share lots, commissions, stamp tax).

A **majority vote** ensemble strategy trades when >= 3 of 4 models agree on direction, combining the strengths of all models.

A hyperparameter tuning pipeline finds optimal configs via grid search with chronological inner validation. A daily pipeline downloads latest data, trains all models, and generates today's signals across multiple tickers.

## Project Structure

```
claude_trader/
├── fetch_china_stock.py        # Download daily OHLCV from akshare
├── trading_bot.py              # K-Means clustering bot + backtest engine
├── dnn_trading_bot.py          # LSTM neural network bot + backtest engine
├── lgbm_trading_bot.py         # LightGBM gradient-boosted tree bot + backtest
├── ppo_trading_bot.py          # PPO reinforcement learning bot + backtest
├── compare_models.py           # 4-model comparison + majority vote + trade log
├── daily_pipeline.py           # Multi-ticker daily pipeline (download, train, tune, consensus)
├── tune_hyperparams.py         # Grid search hyperparameter tuning (all models)
├── test_fetch_china_stock.py   # Tests for data fetcher
├── test_trading_bot.py         # Tests for K-Means bot (37 tests)
├── test_dnn_trading_bot.py     # Tests for LSTM bot (18 tests)
├── test_lgbm_trading_bot.py    # Tests for LightGBM bot (12 tests)
├── test_ppo_trading_bot.py     # Tests for PPO bot (13 tests)
├── test_compare_models.py      # Tests for majority vote logic (17 tests)
├── test_tune_hyperparams.py    # Tests for tuning pipeline (24 tests)
├── 601933_10yr.csv             # 10-year price data (2016–2026, ~2400 rows)
├── 000001SH_20yr.csv           # 20-year Shanghai Composite index data
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
pip install akshare pandas numpy scikit-learn torch lightgbm gymnasium stable-baselines3 pytest
```

Required packages:

| Package           | Purpose                          |
|-------------------|----------------------------------|
| akshare           | China A-share market data API    |
| pandas            | Data manipulation                |
| numpy             | Numerical computation            |
| scikit-learn      | K-Means clustering, scaling      |
| torch             | LSTM neural network (PyTorch)    |
| lightgbm          | Gradient-boosted tree classifier |
| gymnasium         | RL environment API               |
| stable-baselines3 | PPO reinforcement learning       |
| pytest            | Test runner                      |

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
pytest test_trading_bot.py test_dnn_trading_bot.py test_lgbm_trading_bot.py test_ppo_trading_bot.py test_compare_models.py test_tune_hyperparams.py -v
```

124 tests total — all should pass.

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

### Run PPO trading bot

```bash
python ppo_trading_bot.py                      # uses 601933_10yr.csv
python ppo_trading_bot.py --csv 601933_3yr.csv
```

Trains a PPO reinforcement learning agent in a simulated trading environment, runs backtest, and prints today's signal.

### Compare all models

```bash
python compare_models.py                       # 4-model + majority vote comparison + trade_log.csv
```

Runs all four backtests plus the majority vote ensemble, prints a side-by-side comparison table, and saves a combined daily trade log with signals and actions for each model.

### Run daily pipeline

```bash
python daily_pipeline.py                       # download, train, tune, consensus (~8 min)
python daily_pipeline.py --skip-tune           # skip tuning (~2 min)
```

End-to-end pipeline that downloads latest data for all tickers (601933 Yonghui, 000001.SH Shanghai Composite), trains all 4 models + majority vote, optionally tunes hyperparameters, runs consensus strategy, and generates today's signals.

### Run hyperparameter tuning

```bash
python tune_hyperparams.py                     # full tuning, ~3 min
python tune_hyperparams.py --csv 601933_10yr.csv --output tuning_results.json
```

This runs:
1. **K-Means grid search** (30 configs) — varies `n_clusters` [3–8] and feature subsets
2. **LSTM 2-phase grid search** (~40 configs) — Phase 1 varies window/lr/batch, Phase 2 varies hidden sizes
3. **LightGBM grid search** (27 configs) — varies `n_estimators`, `max_depth`, `learning_rate`
4. **PPO grid search** (36 configs) — varies timesteps, learning rate, entropy coefficient, n\_steps
5. **Final evaluation** — retrains best configs on full training set, tests on held-out 40%

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

### PPO Bot

1. Build a Gymnasium trading environment with observation space = 6 scaled indicators + 4 portfolio state features
2. Action space: Discrete(5) mapping to signal levels
3. Reward: daily return \* 100 + 0.1 \* 30-day Sharpe ratio
4. Train PPO agent (Stable-Baselines3) via interaction with simulated market
5. Predict signal → execute trade at next day's open

Key advantage: learns a trading _policy_ that considers portfolio state, not just market indicators.

### Majority Vote Ensemble

1. Run all 4 models independently to generate signals on the test set
2. Classify each signal into buy/sell/hold direction
3. If >= 3 of 4 models agree on direction, take that action
4. SMA5 buy filter still applies (price must be below SMA5 for buy signals)
5. Strong signals from any agreeing model → full position; all mild → half position

Key advantage: filters out noise from individual models while being less restrictive than unanimous voting, producing more trades and better risk-adjusted returns.

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

## Results

### 601933 Yonghui (10yr data, 100K capital)

| Metric       | K-Means  | LSTM     | LightGBM | PPO      | Majority | Buy & Hold |
|--------------|----------|----------|----------|----------|----------|------------|
| Total Return | +51.65%  | +48.47%  | +1.76%   | +7.36%   | **+80.29%** | +11.65% |
| Sharpe Ratio | 0.510    | 0.458    | 0.138    | 0.224    | **0.597**   | —       |
| Max Drawdown | -38.85%  | -52.65%  | **-35.37%** | -47.81% | -43.67%  | —       |
| Win Rate     | 69.4%    | **76.9%** | 43.7%   | 65.6%    | 45.0%    | —       |
| Profit Factor| **11.56** | 5.22    | 0.45     | 0.85     | 2.11     | —       |
| Num Trades   | 152      | 32       | 367      | 65       | 42       | 1       |

### 000001.SH Shanghai Composite (20yr data, 1M capital)

| Metric       | K-Means  | LSTM     | LightGBM | PPO      | Majority | Buy & Hold |
|--------------|----------|----------|----------|----------|----------|------------|
| Total Return | -12.19%  | +17.84%  | +26.37%  | +28.92%  | **+28.24%** | +28.61% |
| Sharpe Ratio | -0.104   | 0.300    | **0.503** | 0.287   | 0.325    | —       |
| Max Drawdown | -33.49%  | -20.69%  | **-8.67%** | -25.78% | -27.82% | —       |
| Win Rate     | 62.5%    | 0.0%     | 45.9%    | 0.0%     | **33.3%** | —       |
| Num Trades   | 23       | 13       | 329      | 1        | 7        | 1       |

### Analysis

- **Majority vote** is the top performer on 601933 at +80.29% return and 0.597 Sharpe — outperforming every individual model and Buy & Hold by 7x. By requiring >= 3 of 4 models to agree, it filters noise while keeping enough trades (42) to capture major moves.
- **K-Means** is consistently strong with +51.65% on 601933 and the highest profit factor (11.56).
- **LSTM** shows variable performance — strong on 601933 (+48.47%, 76.9% win rate) but moderate on the index.
- **LightGBM** has the best drawdown control on both tickers (-35.37% and -8.67%) and the highest Sharpe on the index (0.503), making it the most conservative strategy.
- **PPO** performs moderately, with +28.92% on the index nearly matching Buy & Hold.
- On the Shanghai Composite, most models perform near Buy & Hold (+28.61%), suggesting the index is harder to beat than individual stocks.
