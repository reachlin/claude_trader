# Claude Trader

Trading signal bots for China A-shares. Five ML models — unsupervised K-Means clustering, supervised LSTM, LightGBM gradient-boosted trees, PPO reinforcement learning, and TD3 meta-judge — generate daily signals (strong\_buy, mild\_buy, hold, mild\_sell, strong\_sell) from technical indicators, then simulate trades with realistic A-share constraints (T+1, 100-share lots, commissions, stamp tax).

A **majority vote** ensemble strategy trades when >= 3 of 4 base models agree on direction. A **TD3 meta-judge** uses Twin Delayed DDPG to learn directly from all four base model signals, treating them as additional inputs alongside market indicators.

A hyperparameter tuning pipeline finds optimal configs via grid search with chronological inner validation. A daily pipeline downloads latest data, trains all models, and generates today's signals across multiple tickers.

## Project Structure

```
claude_trader/
├── fetch_china_stock.py        # Download daily OHLCV from akshare
├── trading_bot.py              # K-Means clustering bot + backtest engine
├── dnn_trading_bot.py          # LSTM neural network bot + backtest engine
├── lgbm_trading_bot.py         # LightGBM gradient-boosted tree bot + backtest
├── ppo_trading_bot.py          # PPO reinforcement learning bot + backtest
├── td3_trading_bot.py          # TD3 meta-judge bot (learns from all 4 base signals)
├── compare_models.py           # 5-model comparison + majority vote + TD3 + trade log
├── daily_pipeline.py           # Multi-ticker daily pipeline (download, train, tune, consensus)
├── tune_hyperparams.py         # Grid search hyperparameter tuning (all models)
├── test_fetch_china_stock.py   # Tests for data fetcher
├── test_trading_bot.py         # Tests for K-Means bot (37 tests)
├── test_dnn_trading_bot.py     # Tests for LSTM bot (18 tests)
├── test_lgbm_trading_bot.py    # Tests for LightGBM bot (12 tests)
├── test_ppo_trading_bot.py     # Tests for PPO bot (13 tests)
├── test_td3_trading_bot.py     # Tests for TD3 meta-judge (22 tests)
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
pytest test_trading_bot.py test_dnn_trading_bot.py test_lgbm_trading_bot.py test_ppo_trading_bot.py test_td3_trading_bot.py test_compare_models.py test_tune_hyperparams.py -v
```

146 tests total — all should pass.

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

### Run TD3 meta-judge

```bash
python td3_trading_bot.py                      # uses 601933_10yr.csv
python td3_trading_bot.py --csv 601933_3yr.csv
```

Trains all 4 base models (K-Means, LSTM, LightGBM, PPO) on the training set, augments the data with their signals, then trains a TD3 (Twin Delayed DDPG) agent that learns to judge and combine those signals. Runs backtest and prints today's signal.

### Compare all models

```bash
python compare_models.py                       # all models + majority vote + TD3 + trade_log.csv
```

Runs all five backtests (K-Means, LSTM, LightGBM, PPO, TD3) plus the majority vote ensemble, prints a side-by-side comparison table, and saves a combined daily trade log with signals and actions for each model.

### Run daily pipeline

```bash
python daily_pipeline.py                       # download, train, tune, consensus (~8 min)
python daily_pipeline.py --skip-tune           # skip tuning (~2 min)
```

End-to-end pipeline that downloads latest data for all tickers (601933 Yonghui, 000001.SH Shanghai Composite), trains all 5 models (including TD3 meta-judge) + majority vote, optionally tunes hyperparameters, runs consensus strategy, and generates today's signals.

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

### TD3 Meta-Judge

1. Train all 4 base models (K-Means, LSTM, LightGBM, PPO) on the 60% training set
2. Generate their signals on both train and test sets (no lookahead — models never see test data)
3. Augment each DataFrame with the 4 signal columns, encoding them as floats (`strong_sell=-1` → `strong_buy=+1`)
4. Build a Gymnasium environment with observation = 4 signal floats + 6 scaled indicators + 4 portfolio state (14 dims total)
5. Action space: continuous `Box(1,)` in `[-1, 1]`, discretised to 5 signal levels
6. Train TD3 agent (Stable-Baselines3) with twin Q-networks and delayed policy updates
7. Predict signal → execute trade at next day's open

Key advantage: TD3 learns _when_ to trust (or override) each base model based on market context and portfolio state, going beyond a hard majority vote rule. Twin Q-networks and delayed actor updates make it more stable than PPO for continuous action spaces.

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

| Metric       | K-Means  | LSTM      | LightGBM | PPO      | Majority    | TD3      | Buy & Hold |
|--------------|----------|-----------|----------|----------|-------------|----------|------------|
| Total Return | +51.65%  | +160.81%  | +1.76%   | -29.08%  | **+113.90%**| +44.03%  | +11.65%    |
| Sharpe Ratio | 0.510    | **1.380** | 0.138    | -0.086   | 0.807       | 0.444    | —          |
| Max Drawdown | -38.85%  | -20.08%   | **-35.37%** | -54.08% | -30.00%  | -43.39%  | —          |
| Win Rate     | **69.4%**| 38.5%     | 43.7%    | 39.2%    | 47.8%       | 37.1%    | —          |
| Profit Factor| **11.56**| 0.35      | 0.45     | 1.25     | 3.31        | 1.11     | —          |
| Num Trades   | 152      | 73        | 367      | 225      | 49          | 236      | 1          |

### Analysis

- **Majority vote (+113.90%, Sharpe 0.807)** is the best risk-adjusted strategy — requiring >= 3 of 4 models to agree filters noise while keeping enough trades (49) to capture major moves.
- **LSTM** had a high-return run (+160.81%) but with only 13 buys vs 60 sells and a profit factor of 0.35, suggesting it got lucky on short timing rather than systematic edge.
- **TD3 (+44.03%, Sharpe 0.444)** performs comparably to K-Means but trades more (236 times), paying more in transaction costs. TD3 has the advantage of being able to learn _when_ to trust each base model, but needs more training time to consistently outperform the naive majority vote.
- **K-Means** is consistently strong with +51.65% and the highest profit factor (11.56).
- **LightGBM** has the best max drawdown control (**-35.37%**), making it the most conservative strategy.
- **PPO** struggled this run (-29.08%), showing high variance across runs.
