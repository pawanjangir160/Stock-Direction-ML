# Stock-Direction-ML

Machine learning pipeline to predict next-day stock price direction (up/down) for Reliance Industries (RELIANCE.NS) using technical indicators, with a backtested trading strategy compared against Buy & Hold.

## Overview

This project pulls 5 years of daily price data, engineers technical indicators (RSI, MACD, moving averages), trains classification models to predict next-day price direction, and backtests a simple trading strategy based on those predictions — evaluating it against a passive Buy & Hold approach.

## Data & Features

- **Source**: Yahoo Finance (`yfinance`), 5 years of daily OHLC data for RELIANCE.NS
- **Features used**: Price distance from 20-day moving average, RSI (14-day), MACD, Signal Line, MACD Histogram
- **Target**: Binary label — did the stock close higher the next trading day?

## Models

- Logistic Regression
- Random Forest Classifier
- Naive baseline (today's return direction as tomorrow's prediction)

## Results

### Single Train/Test Split (80/20, chronological)

| Model | Accuracy |
|---|---|
| Logistic Regression | 50.00% |
| Random Forest | 48.36% |
| Naive Baseline | 45.49% |

### Walk-Forward Validation (5 folds, time-series split)

| Metric | ML Model | Naive Baseline |
|---|---|---|
| Average Accuracy | **51.33%** | 49.06% |

Walk-forward validation is more reliable than a single split since it tests the model across multiple, non-overlapping time periods.

### Backtest: ML Strategy vs Buy & Hold

| Metric | ML Strategy | Buy & Hold |
|---|---|---|
| Total Return | **17.65%** | 13.36% |
| Sharpe Ratio (annualized) | **0.324** | 0.254 |
| Max Drawdown | **-19.91%** | -27.18% |

The strategy holds the stock only on days the model predicts an upward move, and holds cash otherwise.

![Strategy vs Buy and Hold](strategy_vs_buyhold.png)

## Charts

| Price & Moving Average | RSI | MACD |
|---|---|---|
| ![Price MA](price_ma.png) | ![RSI](rsi_chart.png) | ![MACD](macd_chart.png) |

## Conclusion

Raw classification accuracy for next-day direction prediction was close to chance level (48-51%), which is expected given the difficulty of short-term stock forecasting. However, walk-forward validation showed a modest but consistent edge over the naive baseline (51.33% vs 49.06%), and the resulting backtested trading strategy outperformed Buy & Hold across all three metrics: return, Sharpe ratio, and maximum drawdown. The model showed a directional bias toward predicting upward movement (65% of predictions), consistent with the stock's overall upward trend over the study period. This suggests that even a modest predictive edge can meaningfully improve risk-adjusted returns and reduce drawdown when translated into a trading strategy, even though day-to-day directional accuracy alone is unremarkable.

## Tech Stack

- Python, pandas, scikit-learn, matplotlib, yfinance

## How to Run

```bash
pip install -r requirements.txt
python main.py
```
