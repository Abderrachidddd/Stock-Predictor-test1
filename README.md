# Stock Direction Predictor (MVP)

A tiny Streamlit app that trains a logistic regression on daily features (RSI, moving averages, volatility, momentum) and predicts next-day up/down for a ticker.

## Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
