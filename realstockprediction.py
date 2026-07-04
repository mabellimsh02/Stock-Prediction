# %% Imports & config
# This is a step up from stockregression.py: instead of fitting a curve to
# 30 days of raw prices and extrapolating one point, we predict tomorrow's
# DIRECTION (up/down) using engineered features and honestly validate the
# model on data it never trained on. This is closer to how real quant
# workflows are structured - it is still not a system you should trade
# real money on, see the caveats at the bottom.
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import TimeSeriesSplit

TICKER = "AAPL"
HISTORY = "3y"       # need years of data, not weeks, so the model sees
                      # multiple market regimes (rallies, drawdowns, chop)
TEST_FRACTION = 0.2  # last 20% of days (chronologically) held out as test

# %% Fetch data
# yfinance returns MultiIndex columns (Price, Ticker) - drop the ticker
# level since we only ever ask for one symbol at a time.
raw = yf.download(TICKER, period=HISTORY, interval="1d", progress=False)
raw.columns = raw.columns.droplevel(1)
raw = raw[["Open", "High", "Low", "Close", "Volume"]].dropna()

# %% Feature engineering
# The old script's only "feature" was a day index, which encodes nothing
# about the stock itself. Real models instead describe the CURRENT STATE
# of the price series - each row below is something a trader might
# actually look at, expressed as a number the model can use.
df = raw.copy()

# Daily return: today's % change - the basic unit of "how did the stock
# move," and the building block for several features below.
df["return_1d"] = df["Close"].pct_change()

# Lagged returns: yesterday's and the last few days' moves. The model
# can't see "today" when predicting "today," so recent past returns are
# the most direct signal of short-term momentum or reversal.
for lag in (1, 2, 3, 5):
    df[f"return_lag_{lag}"] = df["return_1d"].shift(lag)

# Moving averages, expressed as "price relative to its own average"
# rather than raw price - this normalizes across time so the model isn't
# just learning "AAPL is usually between $X and $Y."
for window in (5, 10, 20):
    sma = df["Close"].rolling(window).mean()
    df[f"price_vs_sma_{window}"] = df["Close"] / sma - 1

# Rolling volatility: how much the stock has been swinging lately.
# High recent volatility often changes how reliable other signals are.
df["volatility_10"] = df["return_1d"].rolling(10).std()

# RSI (Relative Strength Index): a classic momentum indicator. It compares
# the size of recent up-moves to recent down-moves on a 0-100 scale;
# conventionally >70 = "overbought," <30 = "oversold." Computed manually
# here so we don't need an extra dependency.
delta = df["Close"].diff()
gain = delta.clip(lower=0).rolling(14).mean()
loss = (-delta.clip(upper=0)).rolling(14).mean()
df["rsi_14"] = 100 - (100 / (1 + gain / loss))

# Volume change: unusually high volume often accompanies real trend
# changes, vs. low-volume drift that's more likely noise.
df["volume_change"] = df["Volume"].pct_change()

# Target: did price go UP the next trading day? Framing this as
# classification (up/down) rather than regression (exact price) is a
# deliberate choice - predicting the exact next price is much harder and
# the extra precision isn't actually useful for a "should I buy" decision.
df["target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)

FEATURE_COLS = [c for c in df.columns if c.startswith((
    "return_lag_", "price_vs_sma_", "volatility_", "rsi_", "volume_change"
))]

# Rolling windows/lags create NaNs at the start of the series, and
# shift(-1) creates one at the very end (no "tomorrow" for the last row) -
# drop those before training.
model_df = df.dropna(subset=FEATURE_COLS + ["target"])

# %% Chronological train/test split
# THIS is the most important difference from the toy script. A random
# train/test split would let the model "see the future" indirectly
# (adjacent days are correlated), which inflates accuracy in a way that
# won't hold up in real use. Instead we split by TIME: train only on the
# earlier period, test only on the later period the model never touched -
# this mimics actually deploying the model going forward.
split_idx = int(len(model_df) * (1 - TEST_FRACTION))
train_df = model_df.iloc[:split_idx]
test_df = model_df.iloc[split_idx:]

X_train, y_train = train_df[FEATURE_COLS], train_df["target"]
X_test, y_test = test_df[FEATURE_COLS], test_df["target"]

# %% Train model
# Random Forest: builds many decision trees on random subsets of the
# data/features and averages their votes. Chosen over a single model like
# linear/logistic regression because stock-feature relationships are
# rarely simple straight lines, and over a neural net because with only a
# few thousand rows of data, a forest is far less likely to overfit.
# max_depth is kept shallow deliberately - deep trees would memorize noise
# in a market this random rather than learning generalizable patterns.
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=4,
    min_samples_leaf=20,
    random_state=42,
)
model.fit(X_train, y_train)

# %% Evaluate against honest baselines
# Accuracy alone is meaningless without something to compare it to - if
# the stock went up 55% of days in the test period, a model that always
# predicts "up" scores 55% while learning nothing.
preds = model.predict(X_test)

model_acc = accuracy_score(y_test, preds)
majority_baseline = max(y_test.mean(), 1 - y_test.mean())
# "Naive persistence": predict tomorrow repeats today's direction - the
# simplest possible time-series baseline, and a common one to beat.
naive_preds = (test_df["return_1d"] > 0).astype(int)
naive_persistence = accuracy_score(y_test, naive_preds)

print(f"Test period: {test_df.index[0].date()} to {test_df.index[-1].date()} ({len(test_df)} days)")
print(f"Model accuracy:            {model_acc:.1%}")
print(f"Always-predict-majority:   {majority_baseline:.1%}")
print(f"Naive persistence (repeat yesterday's direction): {naive_persistence:.1%}")
print()
print(classification_report(y_test, preds, target_names=["Down", "Up"]))

# %% Feature importance
# Which signals is the model actually relying on? This is a basic form of
# interpretability - if a feature that shouldn't matter (or shouldn't even
# be predictive, like volume_change) dominates, that's a red flag for a
# spurious/overfit pattern rather than a real signal.
importances = pd.Series(model.feature_importances_, index=FEATURE_COLS).sort_values()
plt.figure(figsize=(8, 5))
importances.plot(kind="barh")
plt.title(f"{TICKER} - Feature Importance")
plt.xlabel("Importance")
plt.tight_layout()
plt.show()

# %% Backtest: does the model's signal translate into better returns?
# Simulate a simple strategy: hold the stock only on days the model
# predicted "up," otherwise sit in cash (0 return). Compare its cumulative
# return to plain buy-and-hold over the same test period. This is a sanity
# check separate from accuracy - a model can be directionally right more
# often than not and still lose to buy-and-hold after costs, or vice versa.
strategy_returns = test_df["return_1d"].where(pd.Series(preds, index=test_df.index) == 1, 0)
strategy_curve = (1 + strategy_returns).cumprod()
buy_hold_curve = (1 + test_df["return_1d"]).cumprod()

plt.figure(figsize=(10, 5))
plt.plot(strategy_curve.index, strategy_curve, label="Model strategy")
plt.plot(buy_hold_curve.index, buy_hold_curve, label="Buy & hold")
plt.title(f"{TICKER} - Backtest (test period only, no transaction costs)")
plt.ylabel("Growth of $1")
plt.legend()
plt.tight_layout()
plt.show()

# %% Walk-forward validation
# The split above gives ONE accuracy number from ONE stretch of market
# history - good or bad, it could just be luck of which period got picked.
# Walk-forward validation repeats train-then-test across several
# sequential windows, sliding forward in time, so the result is an average
# over multiple market conditions instead of a single roll of the dice.
#
# TimeSeriesSplit implements this with an *expanding* training window:
# fold 1 trains on the earliest chunk and tests on the chunk right after
# it; fold 2 trains on everything up through that point and tests on the
# next chunk; and so on. Every test fold is always strictly after its own
# training data - never shuffled, never peeking into the future.
N_FOLDS = 5
tscv = TimeSeriesSplit(n_splits=N_FOLDS)

X_all, y_all = model_df[FEATURE_COLS], model_df["target"]
# Out-of-fold predictions: each day gets predicted exactly once, by a
# model that was trained only on data before it - stitching these together
# below gives one continuous, fully honest backtest.
oof_preds = pd.Series(index=model_df.index, dtype=float)
fold_results = []

for fold, (train_idx, test_idx) in enumerate(tscv.split(X_all), start=1):
    X_tr, y_tr = X_all.iloc[train_idx], y_all.iloc[train_idx]
    X_te, y_te = X_all.iloc[test_idx], y_all.iloc[test_idx]

    # Same hyperparameters as before, retrained from scratch on each
    # fold's own training window - this is what "retrain as new data
    # arrives" looks like in practice.
    fold_model = RandomForestClassifier(
        n_estimators=300, max_depth=4, min_samples_leaf=20, random_state=42,
    )
    fold_model.fit(X_tr, y_tr)
    fold_preds = fold_model.predict(X_te)
    oof_preds.iloc[test_idx] = fold_preds

    naive_preds_fold = (model_df["return_1d"].iloc[test_idx] > 0).astype(int)
    fold_results.append({
        "fold": fold,
        "test_start": model_df.index[test_idx[0]].date(),
        "test_end": model_df.index[test_idx[-1]].date(),
        "n_days": len(test_idx),
        "model_acc": accuracy_score(y_te, fold_preds),
        "naive_acc": accuracy_score(y_te, naive_preds_fold),
    })

results_df = pd.DataFrame(fold_results)
print(results_df.to_string(index=False))
print(f"\nMean model accuracy across folds:  {results_df['model_acc'].mean():.1%} "
      f"(+/- {results_df['model_acc'].std():.1%})")
print(f"Mean naive-persistence accuracy:   {results_df['naive_acc'].mean():.1%}")

# %% Walk-forward backtest
# Stitch each fold's out-of-sample predictions into one continuous curve.
# This is the most trustworthy performance chart in the file: every single
# point was predicted by a model that had never seen that day - or
# anything after it - during training.
wf_region = oof_preds.dropna().index
wf_returns = model_df.loc[wf_region, "return_1d"]
wf_strategy_returns = wf_returns.where(oof_preds.loc[wf_region] == 1, 0)

wf_strategy_curve = (1 + wf_strategy_returns).cumprod()
wf_buy_hold_curve = (1 + wf_returns).cumprod()

plt.figure(figsize=(10, 5))
plt.plot(wf_strategy_curve.index, wf_strategy_curve, label="Walk-forward model strategy")
plt.plot(wf_buy_hold_curve.index, wf_buy_hold_curve, label="Buy & hold")
plt.title(f"{TICKER} - Walk-Forward Backtest (all folds combined, no transaction costs)")
plt.ylabel("Growth of $1")
plt.legend()
plt.tight_layout()
plt.show()

# %% Predict tomorrow (informational only, not investment advice)
latest_features = df[FEATURE_COLS].iloc[[-1]]
if latest_features.notna().all(axis=1).iloc[0]:
    prob_up = model.predict_proba(latest_features)[0][1]
    print(f"\nModel's estimated probability {TICKER} closes UP next session: {prob_up:.1%}")
else:
    print("\nNot enough recent data to compute all features for a live prediction.")

# %% Caveats (read this before trusting any of the numbers above)
# - Test accuracy in the 50-55% range is typical and NOT a failure - stock
#   direction is close to a coin flip because if a signal were easy and
#   reliable, other market participants would already be exploiting it
#   away (this is the "efficient market" intuition).
# - This backtest ignores transaction costs, slippage, taxes, and the bid
#   ask spread - all of which eat into or erase small edges.
# - One ticker, one time period: a model that "worked" for AAPL over this
#   window could easily fail on another stock or the next 3 years. Real
#   validation would test across many tickers and time periods.
# - The final "predict tomorrow" cell still uses the single 80/20 model for
#   simplicity - the walk-forward section above is for validating the
#   approach, not for producing the live prediction.
