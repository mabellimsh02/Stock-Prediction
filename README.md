# Stock Prediction

Small experiments in predicting stock price movement with Python, using free `yfinance` data and `scikit-learn` models. This repo contains two scripts that represent different (and increasingly rigorous) approaches to the problem — it's a learning project, **not** a trading system.

## Scripts

### `stockregression.py`
A simple curve-fitting demo. It pulls the last 30 trading days of closing prices for a ticker, relabels each day as an integer index, and fits three Support Vector Regression models (linear, polynomial, RBF kernels) to the price curve. It then extrapolates one day forward. Good for visualizing how different SVR kernels behave, but it has no train/test split, so it isn't a real measure of predictive accuracy.

### `realstockprediction.py`
A more realistic approach. Instead of predicting an exact price, it engineers features from price history (lagged returns, moving averages, volatility, RSI, volume change) and trains a Random Forest classifier to predict whether the stock will close **up or down** the next trading day. It evaluates the model honestly with a chronological train/test split and a walk-forward (time-series cross-validation) backtest, comparing model accuracy against a naive baseline. It finishes with a live "probability of tomorrow being up" estimate.

## Requirements

- Python 3.9+
- - numpy
  - - pandas
    - - yfinance
      - - matplotlib
        - - scikit-learn
         
          - Install with:
         
          - ```bash
            pip install numpy pandas yfinance matplotlib scikit-learn
            ```

            ## Usage

            Run either script directly:

            ```bash
            python stockregression.py
            python realstockprediction.py
            ```

            Both default to the `AAPL` ticker; change the `TICKER` variable at the top of each file to try another symbol.

            ## Disclaimer

            These scripts are for learning and experimentation only. Test accuracy near 50-55% in `realstockprediction.py` is expected, not a bug — next-day stock direction is close to a coin flip. Neither script accounts for transaction costs, slippage, or taxes, and neither should be used to make real investment decisions.
