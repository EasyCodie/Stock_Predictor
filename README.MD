# Stock Price Predictor

A machine learning project that predicts stock price direction using **Python**, **pandas**, **scikit-learn**, and **yfinance**.  
The model compares its performance against a simple "Buy & Hold" benchmark.

---

## Features

- 📈 Fetches historical stock price data via `yfinance`
- 🔨 Feature engineering with moving averages, RSI, and relative distances
- 🤖 Logistic Regression model for binary classification (up/down)
- 📊 Evaluation metrics: Accuracy, Precision, Recall, ROC AUC
- 💰 Benchmark comparison vs. "Always Up" and Buy & Hold strategies
- 🧪 Cross-validation using `TimeSeriesSplit`

---

## Installation

Clone the repository and set up a virtual environment:

```bash
git clone https://github.com/yourusername/stock-predictor.git
cd stock-predictor

# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate      # Windows
# source .venv/bin/activate # Mac/Linux

# Install dependencies
pip install -r requirements.txt
