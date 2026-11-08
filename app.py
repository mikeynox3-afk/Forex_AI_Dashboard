import os
import zipfile
import glob
import joblib
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(
    page_title="XAUUSD AI Hybrid Dashboard",
    page_icon="📊",
    layout="wide"
)

BASE = os.path.dirname(os.path.abspath(__file__))

archive = os.path.join(BASE, "XAUUSD_ALL_FILES_COMPLETE.zip")
if os.path.exists(archive):
    with zipfile.ZipFile(archive, "r") as z:
        for name in z.namelist():
            if name.startswith("XAUUSD_HYBRID_PROJECT/"):
                z.extract(name, BASE)

DATA_DIR = os.path.join(BASE, "XAUUSD_HYBRID_PROJECT")

st.title("XAUUSD AI HYBRID BOT")
st.caption("Multi-Timeframe Historical Market Research Dashboard")

st.info(
    "শিক্ষামূলক বিশ্লেষণ ও সিমুলেশন। "
    "এটি স্বয়ংক্রিয়ভাবে কোনো আসল ট্রেড করে না।"
)

@st.cache_data
def load_csv(path):
    return pd.read_csv(path)

@st.cache_resource
def load_model(path):
    return joblib.load(path)

# সব timeframe-এর CSV খোঁজা
csv_files = glob.glob(
    os.path.join(DATA_DIR, "XAUUSD_*_Analysis.csv")
)

market_data = {}

for path in csv_files:
    filename = os.path.basename(path)
    timeframe = filename.replace(
        "XAUUSD_", ""
    ).replace("_Analysis.csv", "")

    market_data[timeframe] = load_csv(path)

if not market_data:
    st.error("XAUUSD Analysis CSV পাওয়া যায়নি।")
    st.stop()

# Dashboard
st.subheader("Multi-Timeframe Dashboard")

preferred = ["M5", "M15", "H1", "H4", "D1"]
available = [tf for tf in preferred if tf in market_data]

columns = st.columns(len(available))

for col, tf in zip(columns, available):
    df = market_data[tf]
    last = df.iloc[-1]

    close = last.get("close", last.get("Close", 0))
    rsi = last.get("RSI", 0)

    with col:
        st.metric(tf + " Close", f"{close:.2f}")
        st.caption(f"RSI: {rsi:.2f}")

# Timeframe selection
st.subheader("Market Chart")

selected_tf = st.selectbox(
    "Select Timeframe",
    available
)

df = market_data[selected_tf].copy()

if all(c in df.columns for c in ["open", "high", "low", "close"]):

    x_values = (
        df["timestamp"]
        if "timestamp" in df.columns
        else df.index
    )

    fig = go.Figure(data=[
        go.Candlestick(
            x=x_values,
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name=selected_tf
        )
    ])

    fig.update_layout(
        height=500,
        xaxis_rangeslider_visible=False,
        title=f"XAUUSD {selected_tf}"
    )

    st.plotly_chart(fig, use_container_width=True)

else:
    st.warning("এই CSV-তে OHLC কলাম পাওয়া যায়নি।")

# Technical indicators
st.subheader("Technical Indicators")

latest = df.iloc[-1]

metric_names = ["EMA_20", "EMA_50", "EMA_200", "RSI", "MACD", "ATR"]

metric_cols = st.columns(3)

for i, name in enumerate(metric_names):
    value = latest.get(name)

    with metric_cols[i % 3]:
        if value is not None and pd.notna(value):
            st.metric(name, f"{float(value):.4f}")
        else:
            st.metric(name, "N/A")

# AI Model
st.subheader("AI Model Analysis")

model_candidates = [
    os.path.join(DATA_DIR, "XAUUSD_AI_Model.pkl"),
    os.path.join(DATA_DIR, "forex_ai_model.pkl")
]

model_path = next(
    (p for p in model_candidates if os.path.exists(p)),
    None
)

features_path = os.path.join(DATA_DIR, "AI_Features.txt")

if model_path and os.path.exists(features_path):

    try:
        model = load_model(model_path)

        with open(features_path, "r") as f:
            features = [line.strip() for line in f if line.strip()]

        missing_features = [
            name for name in features if name not in df.columns
        ]

        if missing_features:
            st.warning(
                "AI বিশ্লেষণের জন্য কিছু Feature নেই: "
                + ", ".join(missing_features)
            )
        else:
            X = df[features].tail(1)

            probabilities = model.predict_proba(X)[0]
            classes = list(model.classes_)

            if 1 in classes and 0 in classes:
                bullish = float(probabilities[classes.index(1)])
                bearish = float(probabilities[classes.index(0)])

                a, b = st.columns(2)
                a.metric("Model bullish score", f"{bullish:.1%}")
                b.metric("Model bearish score", f"{bearish:.1%}")

                if bullish >= 0.60:
                    st.success("Model classification: BULLISH")
                elif bearish >= 0.60:
                    st.warning("Model classification: BEARISH")
                else:
                    st.info("Model classification: NO CLEAR SETUP")

                st.caption(
                    "এগুলো মডেলের শ্রেণিবিভাগের স্কোর; "
                    "লাভ বা সফল ট্রেডের সম্ভাবনা নয়।"
                )

            else:
                st.warning("Model-এর class format প্রত্যাশিত নয়।")

    except Exception as e:
        st.error(f"AI Model load/prediction error: {e}")

else:
    st.warning("AI Model অথবা AI_Features.txt পাওয়া যায়নি।")

# Market structure and psychology
st.subheader("Market Structure & Psychology")

for name in [
    "Market_Structure",
    "Order_Flow_State",
    "Psychology_State"
]:
    if name in df.columns:
        st.write(f"**{name}:** {latest[name]}")

# Dashboard CSV
dashboard_path = os.path.join(DATA_DIR, "XAUUSD_Dashboard.csv")

if os.path.exists(dashboard_path):
    st.subheader("Full Analysis Dashboard")
    st.dataframe(
        load_csv(dashboard_path),
        use_container_width=True
    )

# Backtest
backtest_path = os.path.join(DATA_DIR, "XAUUSD_Backtest.csv")

if os.path.exists(backtest_path):
    st.subheader("Historical Backtest Report")
    st.dataframe(
        load_csv(backtest_path).tail(50),
        use_container_width=True
    )

# Raw data
with st.expander("View Historical Data"):
    st.dataframe(df.tail(100), use_container_width=True)

st.caption(
    "XAUUSD Research Simulator | Historical data only | "
    "No live order execution | Not financial advice"
)
