import os
import glob
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(
    page_title="XAUUSD AI Hybrid Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("XAUUSD AI HYBRID BOT")
st.caption("Historical Market Research & Analysis Dashboard")

PROJECT_FOLDER = "XAUUSD_HYBRID_PROJECT"

st.info(
    "এটি শিক্ষামূলক মার্কেট বিশ্লেষণ ও সিমুলেশন। "
    "এটি কোনো ট্রেড সম্পাদন করে না।"
)

@st.cache_data
def load_data():
    files = glob.glob(
        os.path.join(PROJECT_FOLDER, "XAUUSD_*_Analysis.csv")
    )
    data = {}
    for file in files:
        name = os.path.basename(file)
        tf = name.replace("XAUUSD_", "").replace("_Analysis.csv", "")
        data[tf] = pd.read_csv(file)
    return data

data = load_data()

if not data:
    st.warning(
        "Analysis CSV পাওয়া যায়নি। "
        "প্রথমে Colab-এ আগের প্রজেক্টের ফাইল তৈরি করতে হবে।"
    )
    st.stop()

timeframes = [tf for tf in ["M5", "M15", "H1", "H4", "D1"] if tf in data]

selected_tf = st.selectbox("Select Timeframe", timeframes)

df = data[selected_tf].copy()

st.subheader(f"{selected_tf} Market Chart")

if {"open", "high", "low", "close"}.issubset(df.columns):
    fig = go.Figure(data=[
        go.Candlestick(
            x=df.index,
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name=selected_tf
        )
    ])
    fig.update_layout(
        xaxis_rangeslider_visible=False,
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("এই CSV-তে OHLC কলাম পাওয়া যায়নি।")

st.subheader("Market Analysis")

latest = df.iloc[-1]

c1, c2, c3 = st.columns(3)

with c1:
    st.metric("Latest Close", f"{latest.get('close', 0):.2f}")

with c2:
    st.metric("RSI", f"{latest.get('RSI', 0):.2f}")

with c3:
    st.metric("Volume Ratio", f"{latest.get('Volume_Ratio', 0):.2f}")

st.subheader("Market Structure & Psychology")

for column in ["Market_Structure", "Order_Flow_State"]:
    if column in df.columns:
        st.write(f"**{column}:** {latest[column]}")

if "AI_Features.txt" in os.listdir(PROJECT_FOLDER):
    st.success("AI feature configuration পাওয়া গেছে।")

model_path = os.path.join(PROJECT_FOLDER, "XAUUSD_AI_Model.pkl")

if os.path.exists(model_path):
    st.success("Saved AI model পাওয়া গেছে।")
else:
    st.warning("Saved AI model পাওয়া যায়নি।")

st.subheader("Historical Data")
st.dataframe(df.tail(30), use_container_width=True)

st.caption("Research dashboard | No live order execution")
