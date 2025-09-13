from typing import Tuple, List
import numpy as np   # numerical arrays & math
import pandas as pd  # dataframes & timeseries tools
from sklearn.pipeline import Pipeline  # easy ML workflow
from sklearn.preprocessing import StandardScaler  # standardize feautures
from sklearn.linear_model import LogisticRegression  # simple classifier
from sklearn.metrics import accuracy_score  # evaluation metric

def rsi(close: pd.Series, period: int = 14, eps: float = 1e-8) -> pd.Series:

    delta = close.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    gain = up.rolling(period).mean()
    loss = down.rolling(period).mean() + eps
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def make_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series, pd.DataFrame, List[str]]:

    df = df.copy()

    df["ret1"] = df["Close"].pct_change()
    df["sma5"] = df["Close"].rolling(5).mean()
    df["sma10"] = df["Close"].rolling(10).mean()
    df["sma_ratio"] = df["sma5"] / df["sma10"] - 1
    df["vol5"] = df["ret1"].rolling(5).std()
    df["rsi14"] = rsi(df["Close"], 14)
    df["mom5"] = df["Close"].pct_change(5)

    df["next_ret"] = df["Close"].pct_change().shift(-1)
    df["y"] = (df["next_ret"] > 0).astype(int)

    features = ["sma_ratio", "vol5", "rsi14", "mom5", "ret1"]

    df = df.dropna()

    x = df[features]
    y = df["y"]
    next_features = x.iloc[[-1]]

    return x, y, df["next_ret"], next_features, features

def train_eval(x: pd.DataFrame, y: pd.Series, next_ret: pd.Series):

    split = int(len(x) * 0.8)
    xtr, xte = x.iloc[:split], x.iloc[split:]
    ytr, yte = y.iloc[:split], y.iloc[split:]
    ret_te = next_ret.iloc[split:]

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000))
    ])

    model.fit(xtr, ytr)
    preds = model.predict(xte)
    acc = accuracy_score(yte, preds)

    signal = pd.Series(preds, index = ret_te.index)
    strat_ret = ret_te * signal

    eq_buy_hold = (1 + ret_te).cumprod()
    eq_model = (1 + strat_ret).cumprod()

    return model, acc, eq_buy_hold.iloc[-1], eq_model.iloc[-1]


