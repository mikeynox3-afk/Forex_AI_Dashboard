import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta
import feedparser
import os
import joblib
import warnings

warnings.filterwarnings("ignore")

# =====================================================
# XAUUSD AI HYBRID PRO - EDUCATIONAL RESEARCH DASHBOARD
# =====================================================

st.set_page_config(
    page_title="XAUUSD AI HYBRID PRO",
    page_icon="📊",
    layout="wide"
)

st.title("🟡 XAUUSD AI HYBRID PRO")
st.caption("AI-assisted market research dashboard | Educational use only")

# ---------------- SIDEBAR ----------------

st.sidebar.header("⚙️ Dashboard Settings")

period = st.sidebar.selectbox(
    "Historical Data",
    ["5d", "1mo", "3mo", "6mo", "1y", "2y"],
    index=2
)

interval = st.sidebar.selectbox(
    "Chart Interval",
    ["5m", "15m", "30m", "1h", "4h", "1d"],
    index=1
)

refresh = st.sidebar.button("🔄 Refresh Data")

st.sidebar.info(
    "এই অ্যাপটি শুধুমাত্র শিক্ষামূলক গবেষণা ও "
    "ঐতিহাসিক ডেটা বিশ্লেষণের জন্য।"
)

# ---------------- DATA DOWNLOAD ----------------

@st.cache_data(ttl=60)
def download_data(period, interval):

    symbols = ["XAUUSD=X", "GC=F"]

    for symbol in symbols:
        try:
            data = yf.download(
                symbol,
                period=period,
                interval=interval,
                progress=False,
                auto_adjust=True
            )

            if data is not None and not data.empty:
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = data.columns.get_level_values(0)

                data = data.copy()
                data.columns = [
                    str(col).lower().replace(" ", "_")
                    for col in data.columns
                ]

                required = ["open", "high", "low", "close"]

                if all(col in data.columns for col in required):
                    if "volume" not in data.columns:
                        data["volume"] = 0

                    return data.dropna(subset=required), symbol

        except Exception:
            continue

    return pd.DataFrame(), None


with st.spinner("Market data loading..."):
    data, selected_symbol = download_data(period, interval)

if refresh:
    st.cache_data.clear()
    st.rerun()

if data.empty:
    st.error("ডেটা পাওয়া যায়নি। কিছুক্ষণ পর Refresh Data চাপুন।")
    st.stop()

st.success(f"Data loaded: {selected_symbol}")

# ---------------- INDICATORS ----------------

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / period,
        min_periods=period,
        adjust=False
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / period,
        min_periods=period,
        adjust=False
    ).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def add_indicators(df):

    df = df.copy()

    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    df["EMA_20"] = close.ewm(span=20, adjust=False).mean()
    df["EMA_50"] = close.ewm(span=50, adjust=False).mean()
    df["EMA_200"] = close.ewm(span=200, adjust=False).mean()

    df["RSI"] = calculate_rsi(close)

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()

    df["MACD"] = ema12 - ema26
    df["MACD_Signal"] = df["MACD"].ewm(
        span=9,
        adjust=False
    ).mean()

    previous_close = close.shift(1)

    tr = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs()
        ],
        axis=1
    ).max(axis=1)

    df["ATR"] = tr.rolling(14).mean()

    df["Volatility"] = close.pct_change().rolling(20).std()

    average_volume = volume.rolling(20).mean()
    df["Volume_Ratio"] = (
        volume / average_volume.replace(0, np.nan)
    )

    candle_range = (high - low).replace(0, np.nan)

    df["Estimated_Buy_Pressure"] = (
        (close - low) / candle_range
    ) * 100

    df["Estimated_Sell_Pressure"] = (
        (high - close) / candle_range
    ) * 100

    df["VWAP"] = (
        (close * volume).rolling(20).sum()
        / volume.rolling(20).sum().replace(0, np.nan)
    )

    df["BB_Middle"] = close.rolling(20).mean()
    bb_std = close.rolling(20).std()

    df["BB_Upper"] = df["BB_Middle"] + 2 * bb_std
    df["BB_Lower"] = df["BB_Middle"] - 2 * bb_std

    return df


data = add_indicators(data)
data = data.replace([np.inf, -np.inf], np.nan)

latest = data.iloc[-1]
previous = data.iloc[-2] if len(data) > 1 else latest

# ---------------- MARKET OVERVIEW ----------------

st.subheader("📌 Market Overview")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Latest Price",
    f"{latest['close']:.2f}",
    f"{latest['close'] - previous['close']:.2f}"
)

col2.metric(
    "RSI",
    f"{latest['RSI']:.2f}" if pd.notna(latest["RSI"]) else "N/A"
)

col3.metric(
    "ATR",
    f"{latest['ATR']:.2f}" if pd.notna(latest["ATR"]) else "N/A"
)

col4.metric(
    "Volume",
    f"{latest['volume']:,.0f}"
)

# ---------------- CHART ----------------

st.subheader("📈 XAUUSD Candlestick Chart")

fig = go.Figure()

fig.add_trace(
    go.Candlestick(
        x=data.index,
        open=data["open"],
        high=data["high"],
        low=data["low"],
        close=data["close"],
        name="Price"
    )
)

fig.add_trace(
    go.Scatter(
        x=data.index,
        y=data["EMA_20"],
        name="EMA 20",
        line=dict(width=1)
    )
)

fig.add_trace(
    go.Scatter(
        x=data.index,
        y=data["EMA_50"],
        name="EMA 50",
        line=dict(width=1)
    )
)

fig.add_trace(
    go.Scatter(
        x=data.index,
        y=data["EMA_200"],
        name="EMA 200",
        line=dict(width=1)
    )
)

fig.update_layout(
    height=600,
    xaxis_rangeslider_visible=False,
    template="plotly_dark",
    margin=dict(l=10, r=10, t=30, b=10)
)

st.plotly_chart(fig, use_container_width=True)

# ---------------- MARKET STRUCTURE ----------------

st.subheader("🧭 Market Structure")

if latest["EMA_20"] > latest["EMA_50"] > latest["EMA_200"]:
    trend = "Bullish EMA Alignment"
elif latest["EMA_20"] < latest["EMA_50"] < latest["EMA_200"]:
    trend = "Bearish EMA Alignment"
else:
    trend = "Mixed / Sideways Structure"

st.info(f"EMA Structure: {trend}")

# ---------------- RSI & MACD ----------------

col1, col2 = st.columns(2)

with col1:
    st.subheader("RSI Analysis")

    rsi_value = latest["RSI"]

    if pd.isna(rsi_value):
        st.write("RSI data unavailable.")
    elif rsi_value >= 70:
        st.warning("RSI is in the higher range.")
    elif rsi_value <= 30:
        st.warning("RSI is in the lower range.")
    else:
        st.info("RSI is in the middle range.")

    st.metric(
        "RSI Value",
        f"{rsi_value:.2f}" if pd.notna(rsi_value) else "N/A"
    )

with col2:
    st.subheader("MACD Analysis")

    if latest["MACD"] > latest["MACD_Signal"]:
        st.success("MACD is above Signal")
    else:
        st.warning("MACD is below Signal")

    st.metric("MACD", f"{latest['MACD']:.4f}")
    st.metric("Signal", f"{latest['MACD_Signal']:.4f}")

# ---------------- SUPPORT & RESISTANCE ----------------

st.subheader("📍 Support & Resistance")

lookback = min(50, len(data))
recent = data.tail(lookback)

support = recent["low"].min()
resistance = recent["high"].max()

col1, col2 = st.columns(2)

col1.metric("Recent Support", f"{support:.2f}")
col2.metric("Recent Resistance", f"{resistance:.2f}")

# ---------------- ICT HEURISTICS ----------------

st.subheader("🧩 ICT-Style Market Observations")

def market_observations(df):

    observations = []

    if len(df) < 5:
        return ["Not enough candles for observations."]

    current = df.iloc[-1]
    prior = df.iloc[-2]
    recent = df.iloc[-5:-1]

    if current["close"] > recent["high"].max():
        observations.append("Recent high break detected.")

    if current["close"] < recent["low"].min():
        observations.append("Recent low break detected.")

    if current["high"] > recent["high"].max():
        observations.append("Possible buy-side liquidity sweep.")

    if current["low"] < recent["low"].min():
        observations.append("Possible sell-side liquidity sweep.")

    if current["low"] > df.iloc[-3]["high"]:
        observations.append("Possible bullish Fair Value Gap.")

    if current["high"] < df.iloc[-3]["low"]:
        observations.append("Possible bearish Fair Value Gap.")

    if current["close"] > current["open"]:
        observations.append("Latest candle closed bullish.")
    elif current["close"] < current["open"]:
        observations.append("Latest candle closed bearish.")

    if not observations:
        observations.append("No notable heuristic pattern detected.")

    return observations


for item in market_observations(data):
    st.write("•", item)

st.caption(
    "ICT-style observations are simplified candle-based heuristics, "
    "not verified institutional activity."
)

# ---------------- MULTI TIMEFRAME ----------------

st.subheader("⏱️ Multi-Timeframe Overview")

def resample_data(df, rule):

    result = df.resample(rule).agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum"
    })

    return result.dropna()


timeframes = {
    "M5": "5min",
    "M15": "15min",
    "M30": "30min",
    "H1": "1h",
    "H4": "4h",
    "D1": "1D"
}

tf_rows = []

for name, rule in timeframes.items():

    try:
        tf_data = resample_data(data, rule)

        if len(tf_data) < 2:
            continue

        tf_last = tf_data.iloc[-1]
        tf_prev = tf_data.iloc[-2]

        direction = (
            "Bullish"
            if tf_last["close"] > tf_prev["close"]
            else "Bearish"
        )

        tf_rows.append({
            "Timeframe": name,
            "Close": round(tf_last["close"], 2),
            "Previous Close": round(tf_prev["close"], 2),
            "Candle Direction": direction
        })

    except Exception:
        continue

if tf_rows:
    st.dataframe(
        pd.DataFrame(tf_rows),
        use_container_width=True,
        hide_index=True
    )
else:
    st.info("Not enough data for multi-timeframe analysis.")

# ---------------- AI MODEL ----------------

st.subheader("🤖 Optional AI Model")

feature_names = [
    "EMA_20",
    "EMA_50",
    "EMA_200",
    "RSI",
    "MACD",
    "MACD_Signal",
    "ATR",
    "Volatility",
    "Volume_Ratio",
    "Estimated_Buy_Pressure",
    "Estimated_Sell_Pressure"
]

model_paths = [
    "XAUUSD_HYBRID_PROJECT/XAUUSD_AI_Model.pkl",
    "XAUUSD_AI_Model.pkl",
    "forex_ai_model.pkl"
]

model = None
model_path_used = None

for path in model_paths:
    if os.path.exists(path):
        try:
            model = joblib.load(path)
            model_path_used = path
            break
        except Exception as error:
            st.warning(f"Could not load model: {error}")

if model is not None:

    st.caption(f"Loaded model: {model_path_used}")

    try:
        model_input = data[feature_names].iloc[[-1]]

        if model_input.isna().any().any():
            st.warning(
                "AI prediction unavailable: indicator values are incomplete."
            )
        else:
            prediction = model.predict(model_input)[0]

            st.write("Model output:", prediction)

            if hasattr(model, "predict_proba"):
                scores = model.predict_proba(model_input)[0]

                score_df = pd.DataFrame({
                    "Class": model.classes_,
                    "Model Score": scores
                })

                st.dataframe(
                    score_df,
                    use_container_width=True,
                    hide_index=True
                )

            st.caption(
                "Model scores are not win probabilities or guarantees. "
                "Prediction labels depend on the model's original training labels."
            )

    except Exception as error:
        st.warning(f"Model prediction unavailable: {error}")

else:
    st.info(
        "No compatible AI model found. Dashboard indicators "
        "and market observations are still available."
    )

# ---------------- NEWS ----------------

st.subheader("📰 Gold Market News")

@st.cache_data(ttl=900)
def get_news():

    feed_url = (
        "https://news.google.com/rss/search?"
        "q=gold+XAUUSD+market&hl=en-US&gl=US&ceid=US:en"
    )

    feed = feedparser.parse(feed_url)
    return feed.entries[:10]


try:
    news_items = get_news()

    if news_items:
        for item in news_items:
            st.markdown(f"**[{item.title}]({item.link})**")
            if hasattr(item, "published"):
                st.caption(item.published)
    else:
        st.info("No news items available.")

except Exception:
    st.info("News feed is currently unavailable.")

# ---------------- DATA TABLE ----------------

st.subheader("📋 Latest Indicator Data")

display_columns = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "EMA_20",
    "EMA_50",
    "EMA_200",
    "RSI",
    "MACD",
    "MACD_Signal",
    "ATR",
    "Volume_Ratio"
]

available_columns = [
    col for col in display_columns if col in data.columns
]

st.dataframe(
    data[available_columns].tail(20).round(4),
    use_container_width=True
)

# ---------------- DOWNLOAD CSV ----------------

st.subheader("📥 Export Historical Analysis")

csv_data = data.to_csv().encode("utf-8")

st.download_button(
    label="Download Analysis CSV",
    data=csv_data,
    file_name="XAUUSD_Analysis.csv",
    mime="text/csv"
)

# ---------------- FOOTER ----------------

st.divider()

st.caption(
    "XAUUSD AI HYBRID PRO | Educational market research dashboard"
)

st.warning(
    "এই অ্যাপ কোনো আর্থিক পরামর্শ দেয় না, "
    "কোনো ট্রেডিং নির্দেশনা দেয় না এবং কোনো বাস্তব ট্রেড সম্পাদন করে না। "
    "ডেটা ও বিশ্লেষণে ভুল বা বিলম্ব থাকতে পারে।"
)

st.caption(
    f"Last dashboard update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
)
