import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

stock = yf.download("RELIANCE.NS", period="5y")

stock.columns = stock.columns.get_level_values(0)

stock["Daily_Return"] = stock["Close"].pct_change()
# Calculate price changes
price_change = stock["Close"].diff()

# Separate gains and losses
gain = price_change.clip(lower=0)
loss = -price_change.clip(upper=0)

# Calculate 14-day average gain and loss
average_gain = gain.rolling(window=14).mean()
average_loss = loss.rolling(window=14).mean()

# Calculate RSI
rs = average_gain / average_loss
stock["RSI"] = 100 - (100 / (1 + rs))

daily_volatility = stock["Daily_Return"].std()
highest_price = stock["Close"].max()
lowest_price = stock["Close"].min()
average_return = stock["Daily_Return"].mean()
best_day = stock["Daily_Return"].max()
worst_day = stock["Daily_Return"].min()

print("\n--- Stock Statistics ---")
print("Highest closing price:", highest_price)
print("Lowest closing price:", lowest_price)
print("Average daily return:", average_return * 100, "%")
print("Best trading day:", best_day * 100, "%")
print("Worst trading day:", worst_day * 100, "%")

print("Daily volatility:", daily_volatility)

# Calculate 20-day moving average
stock["MA5"] = stock["Close"].rolling(window=5).mean()
stock["MA20"] = stock["Close"].rolling(window=20).mean()

# Previous day's return
stock["Previous_Return"] = stock["Daily_Return"].shift(1)

# Price distance from 20-day moving average
stock["Price_MA20_Distance"] = (
    stock["Close"] / stock["MA20"] - 1
)

stock["Price_MA5_Distance"] = (
    stock["Close"] / stock["MA5"] - 1
)

# Calculate MACD
ema12 = stock["Close"].ewm(span=12, adjust=False).mean()
ema26 = stock["Close"].ewm(span=26, adjust=False).mean()

stock["MACD"] = ema12 - ema26

stock["Signal_Line"] = stock["MACD"].ewm(
    span=9, adjust=False
).mean()

stock["MACD_Histogram"] = (
    stock["MACD"] - stock["Signal_Line"]
)

# Previous day's RSI and MACD
stock["Previous_RSI"] = stock["RSI"].shift(1)
stock["Previous_MACD"] = stock["MACD"].shift(1)

# Create prediction target
stock["Tomorrow_Close"] = stock["Close"].shift(-1)

stock["Target"] = (
    stock["Tomorrow_Close"] > stock["Close"]
).astype(int)

# Remove the final row because tomorrow's price is not available
stock = stock.dropna(subset=["Tomorrow_Close"])

# Save processed data

print("\n--- Prediction Target ---")
print(stock[["Close", "Tomorrow_Close", "Target"]].tail(10))

print("\n--- RSI ---")
print(stock[["Close", "RSI"]].tail(10))

# Select features for machine learning
features = [
    "Price_MA20_Distance",
    "RSI",
    "MACD",
    "Signal_Line",
    "MACD_Histogram"
]

print("\n--- ML Features ---")
print(stock[features].tail(10))

print("\n--- MACD ---")
print(stock[["Close", "MACD", "Signal_Line", "MACD_Histogram"]].tail(10))

# Price and moving average
plt.figure(figsize=(12, 6))

plt.plot(stock.index, stock["Close"], label="Closing Price")
plt.plot(stock.index, stock["MA20"], label="20-Day Moving Average")

plt.title("Reliance Industries - Price & Moving Average")
plt.xlabel("Date")
plt.ylabel("Price (₹)")

plt.legend()
plt.grid()
plt.show()


# Daily returns
plt.figure(figsize=(12, 6))

plt.plot(stock.index, stock["Daily_Return"] * 100)

plt.title("Reliance Industries - Daily Returns")
plt.xlabel("Date")
plt.ylabel("Daily Return (%)")

plt.grid()
plt.show()

# RSI chart
plt.figure(figsize=(12, 6))

plt.plot(stock.index, stock["RSI"], label="RSI")

plt.axhline(70, linestyle="--", label="Overbought (70)")
plt.axhline(30, linestyle="--", label="Oversold (30)")

plt.title("Reliance Industries - RSI")
plt.xlabel("Date")
plt.ylabel("RSI")

plt.legend()
plt.grid()
plt.show()

# MACD chart
plt.figure(figsize=(12, 6))

plt.plot(stock.index, stock["MACD"], label="MACD")
plt.plot(stock.index, stock["Signal_Line"], label="Signal Line")

plt.axhline(0, linestyle="--")

plt.title("Reliance Industries - MACD")
plt.xlabel("Date")
plt.ylabel("MACD")

plt.legend()
plt.grid()
plt.show()

# Prepare data for machine learning

ml_data = stock[features + ["Target"]].dropna()

X = ml_data[features]
y = ml_data["Target"]

print("\n--- ML Dataset ---")
print("Total rows:", len(ml_data))
print("Features:", X.shape)
print("Target:", y.shape)


# Split data chronologically
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    shuffle=False
)

print("\n--- Train/Test Split ---")
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# Scale features using training data only
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Train Logistic Regression
model = LogisticRegression(max_iter=1000)

model.fit(X_train_scaled, y_train)

print("\n--- Model Training ---")
print("Model trained successfully!")


# Make predictions
y_pred = model.predict(X_test_scaled)


# Evaluate model
accuracy = accuracy_score(y_test, y_pred)

print("\n--- Model Performance ---")
print("Accuracy:", round(accuracy * 100, 2), "%")


# Confusion matrix
cm = confusion_matrix(y_test, y_pred)

print("\n--- Confusion Matrix ---")
print(cm)


# Classification report
print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred))

# Train Random Forest
rf_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

rf_model.fit(X_train, y_train)

# Make predictions
rf_pred = rf_model.predict(X_test)

# Evaluate Random Forest
rf_accuracy = accuracy_score(y_test, rf_pred)

print("\n--- Random Forest Performance ---")
print("Accuracy:", round(rf_accuracy * 100, 2), "%")

print("\n--- Random Forest Classification Report ---")
print(classification_report(y_test, rf_pred))

# Naive baseline
# Use today's return as the naive baseline
test_returns = stock.loc[X_test.index, "Daily_Return"]

baseline_pred = (test_returns > 0).astype(int)

baseline_accuracy = accuracy_score(y_test, baseline_pred)

print("\n--- Naive Baseline Performance ---")
print(
    "Accuracy:",
    round(baseline_accuracy * 100, 2),
    "%"
)

# Inspect Logistic Regression feature coefficients

feature_importance = pd.DataFrame({
    "Feature": features,
    "Coefficient": model.coef_[0]
})

feature_importance["Absolute_Coefficient"] = (
    feature_importance["Coefficient"].abs()
)

feature_importance = feature_importance.sort_values(
    "Absolute_Coefficient",
    ascending=False
)

print("\n--- Feature Importance ---")
print(feature_importance)

# ==========================================
# Experiment 6 - Walk-Forward Evaluation
# ==========================================

print("\n--- Walk-Forward Evaluation ---")

tscv = TimeSeriesSplit(n_splits=5)

fold_accuracies = []
baseline_accuracies = []

for fold, (train_idx, test_idx) in enumerate(tscv.split(X), start=1):

    # Get training and testing data
    X_train_fold = X.iloc[train_idx]
    X_test_fold = X.iloc[test_idx]

    y_train_fold = y.iloc[train_idx]
    y_test_fold = y.iloc[test_idx]

    # Scale using training data only
    fold_scaler = StandardScaler()

    X_train_fold_scaled = fold_scaler.fit_transform(
        X_train_fold
    )

    X_test_fold_scaled = fold_scaler.transform(
        X_test_fold
    )

    # Train Logistic Regression
    fold_model = LogisticRegression(max_iter=1000)

    fold_model.fit(
        X_train_fold_scaled,
        y_train_fold
    )

    # Make predictions
    fold_pred = fold_model.predict(
        X_test_fold_scaled
    )

    # Calculate ML accuracy
    fold_accuracy = accuracy_score(
        y_test_fold,
        fold_pred
    )

    # Naive baseline
    test_dates = X_test_fold.index

    test_returns = stock.loc[
        test_dates,
        "Daily_Return"
    ]

    baseline_pred = (
        test_returns > 0
    ).astype(int)

    baseline_accuracy = accuracy_score(
        y_test_fold,
        baseline_pred
    )

    fold_accuracies.append(fold_accuracy)
    baseline_accuracies.append(baseline_accuracy)

    print(
        f"Fold {fold}: "
        f"ML = {fold_accuracy * 100:.2f}% | "
        f"Baseline = {baseline_accuracy * 100:.2f}%"
    )


# Average results
average_ml_accuracy = sum(fold_accuracies) / len(
    fold_accuracies
)

average_baseline_accuracy = sum(baseline_accuracies) / len(
    baseline_accuracies
)

print("\n--- Walk-Forward Summary ---")

print(
    "Average ML Accuracy:",
    round(average_ml_accuracy * 100, 2),
    "%"
)

print(
    "Average Baseline Accuracy:",
    round(average_baseline_accuracy * 100, 2),
    "%"
)

# ==========================================
# Stage 7 - Backtesting
# ==========================================

print("\n--- Backtesting ---")

# Store predictions and actual returns
backtest_results = []

tscv = TimeSeriesSplit(n_splits=5)

for fold, (train_idx, test_idx) in enumerate(
    tscv.split(X),
    start=1
):

    X_train_fold = X.iloc[train_idx]
    X_test_fold = X.iloc[test_idx]

    y_train_fold = y.iloc[train_idx]
    y_test_fold = y.iloc[test_idx]

    # Scale using training data only
    fold_scaler = StandardScaler()

    X_train_fold_scaled = fold_scaler.fit_transform(
        X_train_fold
    )

    X_test_fold_scaled = fold_scaler.transform(
        X_test_fold
    )

    # Train model
    fold_model = LogisticRegression(
        max_iter=1000
    )

    fold_model.fit(
        X_train_fold_scaled,
        y_train_fold
    )

    # Predict direction
    fold_pred = fold_model.predict(
        X_test_fold_scaled
    )

    # Get the NEXT trading day's return
    test_dates = X_test_fold.index

    next_day_returns = stock["Daily_Return"].shift(-1).reindex(
        test_dates
    )

    # Strategy:
    # Hold when model predicts UP (1)
    # Remove rows where the next-day return is unavailable
    valid = next_day_returns.notna()

    valid_predictions = fold_pred[valid.to_numpy()]
    valid_returns = next_day_returns[valid]

    strategy_returns = (
        valid_predictions * valid_returns.to_numpy()
    )

    valid_dates = test_dates[valid]

    for date, prediction, actual_return, strategy_return in zip(
        valid_dates,
        valid_predictions,
        valid_returns,
        strategy_returns
    ):
        backtest_results.append({
            "Date": date,
            "Prediction": prediction,
            "Actual_Return": actual_return,
            "Strategy_Return": strategy_return
        })

# Convert results to DataFrame
backtest_df = pd.DataFrame(backtest_results)

backtest_df = backtest_df.sort_values(
    "Date"
)

# Calculate cumulative returns
backtest_df["Strategy_Growth"] = (
    1 + backtest_df["Strategy_Return"]
).cumprod()

backtest_df["Buy_Hold_Growth"] = (
    1 + backtest_df["Actual_Return"]
).cumprod()

# ==========================================
# Diagnostics - Class Balance
# ==========================================

print("\n--- Class Balance (Actual Targets) ---")
print(y.value_counts())

print("\n--- Class Balance (Backtest Predictions) ---")
print(backtest_df["Prediction"].value_counts())

# ==========================================
# Risk-Adjusted Metrics - Sharpe Ratio & Max Drawdown
# ==========================================

# Sharpe Ratio (annualized, assuming 252 trading days)
strategy_sharpe = (
    backtest_df["Strategy_Return"].mean()
    / backtest_df["Strategy_Return"].std()
    * (252 ** 0.5)
)

buy_hold_sharpe = (
    backtest_df["Actual_Return"].mean()
    / backtest_df["Actual_Return"].std()
    * (252 ** 0.5)
)

print("\n--- Sharpe Ratios (Annualized) ---")
print("ML Strategy Sharpe:", round(strategy_sharpe, 3))
print("Buy & Hold Sharpe:", round(buy_hold_sharpe, 3))

# Max Drawdown
strategy_drawdown = (
    backtest_df["Strategy_Growth"] / backtest_df["Strategy_Growth"].cummax() - 1
)

buy_hold_drawdown = (
    backtest_df["Buy_Hold_Growth"] / backtest_df["Buy_Hold_Growth"].cummax() - 1
)

print("\n--- Max Drawdown ---")
print("ML Strategy Max Drawdown:", round(strategy_drawdown.min() * 100, 2), "%")
print("Buy & Hold Max Drawdown:", round(buy_hold_drawdown.min() * 100, 2), "%")

# Calculate final returns
strategy_return = (
    backtest_df["Strategy_Growth"].iloc[-1] - 1
)

buy_hold_return = (
    backtest_df["Buy_Hold_Growth"].iloc[-1] - 1
)


print("\n--- Backtest Results ---")

print(
    "Strategy Return:",
    round(strategy_return * 100, 2),
    "%"
)

print(
    "Buy & Hold Return:",
    round(buy_hold_return * 100, 2),
    "%"
)

# Plot strategy vs buy-and-hold

plt.figure(figsize=(12, 6))

plt.plot(
    backtest_df["Date"],
    backtest_df["Strategy_Growth"],
    label="ML Strategy"
)

plt.plot(
    backtest_df["Date"],
    backtest_df["Buy_Hold_Growth"],
    label="Buy & Hold"
)

plt.title(
    "Reliance Industries - Strategy vs Buy & Hold"
)

plt.xlabel("Date")
plt.ylabel("Growth of ₹1")

plt.legend()
plt.grid()

plt.show()

print("\n--- Backtest Sample ---")
print(
    backtest_df[
        [
            "Date",
            "Prediction",
            "Actual_Return",
            "Strategy_Return"
        ]
    ].head(15)
)