import streamlit as st
import yfinance as yf
import pandas as pd
from model import make_features, train_eval

st.set_page_config(page_title="Stock Direction Predictor", page_icon="📈", layout="centered")

st.title("📈 Stock Direction Predictor (MVP)")
st.caption("Enter a ticker, chose a period, train a quick model, and get a next-day up/down call. For learning, not live trading.")


with st.sidebar:
    st.header("Settings")
    ticker = st.text_input("Ticker", value="AAPL")
    period = st.selectbox("History Period", ["2y", "5y", "10y", "max"], index=1)
    run_button: bool = st.sidebar.button("Train & Predict")


if run_button:

    with st.status("Downloading data and training model...", expanded=False) as status:

        df = yf.download(ticker, period=period, interval="1d", auto_adjust=True, progress=False)
        if df.empty:
            st.error("No data returned. Check the ticker or period.")
            status.update(label="Failed", state="error")
        else:
            x, y, next_ret, next_x, feats = make_features(df)

            if len(x) < 200:
                st.warning("Not enough data after feature creation. Try a longer period.")
                status.update(label="Warning", state="warning")
            else:
                model, acc, bh_final, mdl_final = train_eval(x, y, next_ret)

                proba_up = float(model.predict_proba(next_x)[0, 1])
                pred_up = proba_up >= 0.5

                status.update(label="Done", state="complete")

    if not df.empty and len(x) >= 200:
        st.subheader("Results")
        col1, col2, col3 = st.columns(3)
        col1.metric("Test Accuracy (direction)", f"{acc:.3f}")
        col2.metric("Buy & Hold (test)", f"{bh_final:.3f}×")
        col3.metric("Model Long/Cash (test)", f"{mdl_final:.3f}×")

        st.write(f"**Next-day prediction** for `{ticker}`: **{'UP' if pred_up else 'DOWN'}**  (p(up) = {proba_up:.3f})")
        st.caption(f"Features: {feats}")

        # Show recent price chart
        st.subheader("Recent Price")
        st.line_chart(df["Close"].tail(250))  # show last ~1 year

        # Show feature table (last 10 rows) to understand inputs
        st.subheader("Latest Engineered Features")
        preview = x.tail(10).copy()
        preview["target_next_day_up"] = y.tail(10).values
        st.dataframe(preview, use_container_width=True)

with st.expander("Notes & Next Steps"):
    st.markdown(
        """
- This is a **toy** model: it ignores transaction costs, slippage, regime changes, and risk management.
- To extend:
  - Add more indicators (MACD, Bollinger Bands, sector spreads, VIX).
  - Use **walk-forward** validation (rolling windows) instead of a single 80/20 split.
  - Try other models (RandomForest, XGBoost) or predict **magnitude** (regression) then trade only if abs(pred) > threshold.
  - Add portfolio/risk rules: max exposure, stop losses, position sizing.
- Export charts/tables and save model artifacts for a more complete demo.
        """
    )