# %% Imports & config
# This file is split into "# %%" cells. With the VSCode Python extension,
# each cell gets a "Run Cell" / "Run Below" link above it - clicking one
# runs that block in the Interactive Window (a Jupyter-style panel) instead
# of a separate popup, so plots render inline right next to the code.
#
# numpy: turns our day numbers into the array shape sklearn expects
# yfinance: free live stock data (replaces the now-defunct quandl API
#   the original tutorial used)
# matplotlib: for visually comparing the fitted curves against real prices
# SVR: Support Vector Regression - fits a curve to data while tolerating
#   a margin of error, and can bend into different shapes via "kernels"
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.svm import SVR

TICKER = "AAPL"
LOOKBACK_DAYS = 30

# %% Fetch data
def get_recent_prices(ticker, days):
    # We ask for a few extra days beyond LOOKBACK_DAYS because yfinance's
    # calendar includes weekends/holidays where markets are closed - padding
    # the request guarantees we still end up with `days` actual trading days
    # after we .tail() it below.
    data = yf.download(ticker, period=f"{days + 10}d", interval="1d", progress=False)
    return data.tail(days)


data = get_recent_prices(TICKER, LOOKBACK_DAYS)
# We only use the closing price - the single number that summarizes
# "what the stock was worth" at the end of each trading day.
prices = data["Close"].values.ravel()
# Key simplification: instead of using real calendar dates (which SVR
# can't meaningfully compare/scale), we relabel each trading day as a
# plain integer 1, 2, 3... This throws away calendar gaps (weekends,
# holidays) and treats "distance between days" as just "distance between
# trading sessions" - simpler for the model, but also why this is a toy
# demo rather than a real forecasting system.
dates = np.arange(1, len(prices) + 1).reshape(-1, 1)

# %% Train models
def train_models(dates, prices):
    # Three kernels = three different assumptions about the shape of the
    # underlying trend. We train all three so they can be compared side by
    # side rather than betting on one shape being "correct" up front.
    # C controls how strictly the curve must hug the data points (higher C
    # = less tolerance for error = tighter fit, more overfitting risk).

    # Linear kernel: assumes the price moves in a straight-line trend.
    # Simplest model, least flexible, least prone to chasing noise.
    svr_lin = SVR(kernel="linear", C=1e3)

    # Polynomial kernel (degree 2): allows one curve/bend in the trend line,
    # so it can capture a simple acceleration or slowdown in price movement.
    svr_poly = SVR(kernel="poly", C=1e3, degree=2)

    # RBF (Radial Basis Function) kernel: the most flexible - it can wiggle
    # to fit local ups and downs in the data. gamma controls how "tight"
    # each wiggle is; lower gamma = smoother, higher gamma = more overfitting.
    svr_rbf = SVR(kernel="rbf", C=1e3, gamma=0.1)

    # Fitting = the model learns the mapping from day-number -> price by
    # minimizing error (within the margin defined by C) on this data.
    svr_lin.fit(dates, prices)
    svr_poly.fit(dates, prices)
    svr_rbf.fit(dates, prices)

    return svr_lin, svr_poly, svr_rbf


models = train_models(dates, prices)

# %% Plot - run this cell to see the graph inline in the Interactive Window
def plot_models(dates, prices, models):
    svr_lin, svr_poly, svr_rbf = models

    # Scatter = the ground truth: actual closing price on each day.
    plt.scatter(dates, prices, color="black", label="Actual Price")
    # Each line = that model's learned curve evaluated on the same days,
    # so you can visually judge which kernel tracks the real data best
    # without overreacting to noise.
    plt.plot(dates, svr_lin.predict(dates), color="green", label="Linear model")
    plt.plot(dates, svr_poly.predict(dates), color="blue", label="Polynomial model")
    plt.plot(dates, svr_rbf.predict(dates), color="red", label="RBF model")
    plt.xlabel("Day")
    plt.ylabel("Price ($)")
    plt.title(f"{TICKER} Stock Price Prediction")
    plt.legend()
    plt.show()


plot_models(dates, prices, models)

# %% Predict next day
# Extrapolation: ask each fitted curve what price it predicts for the very
# next day-index (one past the last day we trained on). This is the
# "prediction" - note none of the models were tested on unseen data, so
# this measures curve-fitting, not real forecasting accuracy.
next_day = [[len(prices) + 1]]
names = ["Linear", "Polynomial", "RBF"]
for name, model in zip(names, models):
    print(f"{name} prediction: ${model.predict(next_day)[0]:.2f}")
