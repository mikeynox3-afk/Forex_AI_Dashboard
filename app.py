import os
import warnings
from datetime import datetime

import feedparser
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

warnings.filterwarnings("ignore")

# =========================================================
# XAUUSD AI HYBRID PRO
# Educational market research dashboard
# No broker connection / No live trade execution
# =========================================================

st.set_page_config(
    page_title="XAUUSD AI HYBRID PRO",
    page_icon="📊",
    layout="wide"
)

st.title("🟡 XAUUSD AI HYBRID PRO")
st.caption("AI-assisted market research dashboard | Educational use only")

# ---------------- SETTINGS ----------------

st.sidebar.header("⚙️ Dashboard Settings")

timeframe = st.sidebar.selectbox(
    "Select Timeframe",
    ["M5", "M15", "M30", "H1", "H4", "D1"],
    index=1
)

period = st.sidebar.selectbox(
    "Historical Period",
    ["5d", "1mo", "3mo", "6mo", "1y"],
    index=2
)

st.sidebar.caption(
    "Yahoo Finance-এর ডেটা সীমাবদ্ধতার কারণে "
    "কিছু timeframe-এ সর্বোচ্চ historical period কম হতে পারে।"
)

# ---------------- DATA NORMALIZATION ----------------

def normalize_columns(df):
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.columns = [
        str(c).strip().lower().replace(" ", "_")
        for c in df.columns
    ]

    aliases = {
        "adj_close": "close",
        "datetime": "date",
        "timestamp": "date",
        "time": "date",
    }

    df = df.rename(columns=aliases)

    required = ["open", "high", "low", "close"]

    if not all(c in df.columns for c in required):
        return pd.DataFrame()

    if "volume" not in df.columns:
        df["volume"] = 0

    for col in required + ["volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if "date" in df.columns:
        dates = pd.to_datetime(df["date"], errors="coerce", utc=True)
        df = df.drop(columns=["date"])
        df.index = dates

    elif not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index, errors="coerce", utc=True)
        except Exception:
            return pd.DataFrame()

    elif df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")

    df = df[~df.index.isna()]
    df = df.dropna(subset=required)
    df = df[~df.index.duplicated(keep="last")]
    df = df.sort_index()

    return df


# ---------------- YAHOO FINANCE DATA ----------------

@st.cache_data(ttl=120, show_spinner=False)
def get_market_data(selected_period, selected_timeframe):

    interval_map = {
        "M5": "5m",
        "M15": "15m",
        "M30": "30m",
        "H1": "1h",
        "H4": "1h",
        "D1": "1d",
    }

    download_interval = interval_map[selected_timeframe]

    # Yahoo intraday data has period limitations.
    allowed_periods = {
        "5m": ["5d", "1mo"],
        "15m": ["5d", "1mo", "3mo"],
        "30m": ["5d", "1mo", "3mo", "6mo"],
        "1h": ["5d", "1mo", "3mo", "6mo", "1y"],
        "1d": ["5d", "1mo", "3mo", "6mo", "1y"],
    }

    if selected_period not in allowed_periods[download_interval]:
        selected_period = allowed_periods[download_interval][-1]

    errors = []

    for ticker in ["XAUUSD=X", "GC=F"]:
        try:
            raw = yf.Ticker(ticker).history(
                period=selected_period,
                interval=download_interval,
                auto_adjust=True,
                timeout=20
            )

            clean = normalize_columns(raw)

            if not clean.empty and len(clean) >= 2:
                return clean, ticker, errors

            errors.append(f"{ticker}: No usable data returned.")

        except Exception as exc:
            errors.append(f"{ticker}: {str(exc)[:250]}")

    return pd.DataFrame(), None, errors


# ---------------- CSV FALLBACK ----------------

st.sidebar.divider()
st.sidebar.subheader("📂 Alternative Data Source")

uploaded_file = st.sidebar.file_uploader(
    "Upload your OHLC CSV file",
    type=["csv"]
)

csv_data = pd.DataFrame()

if uploaded_file is not None:
    try:
        uploaded_file.seek(0)
        raw_csv = pd.read_csv(uploaded_file)
        csv_data = normalize_columns(raw_csv)

        if csv_data.empty:
            st.sidebar.error(
                "CSV-তে Open, High, Low, Close কলাম পাওয়া যায়নি।"
            )
        else:
            st.sidebar.success(
                f"CSV loaded: {len(csv_data)} rows"
            )

    except Exception as exc:
        st.sidebar.error(f"CSV error: {str(exc)[:200]}")


# ---------------- LOAD DATA ----------------

with st.spinner("Loading market data..."):

    if not csv_data.empty:
        data = csv_data
        data_source = "Uploaded CSV"
        data_errors = []

    else:
        data, data_source, data_errors = get_market_data(
            period,
            timeframe
        )

if data.empty:

    st.error("Market data could not be loaded.")

    st.warning(
        "Yahoo Finance থেকে ডেটা পাওয়া যায়নি। "
        "কিছুক্ষণ পরে আবার চেষ্টা করো অথবা OHLC CSV আপলোড করো।"
    )

    if data_errors:
        with st.expander("Technical error details"):
            for error in data_errors:
                st.write(error)

    st.info(
        "CSV ফাইলে Date, Open, High, Low, Close এবং "
        "ঐচ্ছিক Volume কলাম রাখো।"
    )

    st.stop()

st.success(f"Data source: {data_source}")

# ---------------- TIMEFRAME RESAMPLING ----------------

def resample_ohlc(df, rule):

    result = df.resample(rule).agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum"
    })

    return result.dropna(subset=["open", "high", "low", "close"])


if data_source != "Uploaded CSV" and timeframe == "H4":
    data = resample_ohlc(data, "4h")

if data.empty:
    st.error("Selected timeframe contains insufficient data.")
    st.stop()

# ---------------- INDICATORS ----------------

def calculate_indicators(df):

    df = df.copy()

    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    df["EMA_20"] = close.ewm(
        span=20, adjust=False
    ).mean()

    df["EMA_50"] = close.ewm(
        span=50, adjust=False
    ).mean()

    df["EMA_200"] = close.ewm(
        span=200, adjust=False
    ).mean()

    delta = close.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / 14,
        min_periods=14,
        adjust=False
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / 14,
        min_periods=14,
        adjust=False
    ).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

    df["RSI"] = 100 - (100 / (1 + rs))

    ema12 = close.ewm(
        span=12, adjust=False
    ).mean()

    ema26 = close.ewm(
        span=26, adjust=False
    ).mean()

    df["MACD"] = ema12 - ema26

    df["MACD_Signal"] = df["MACD"].ewm(
        span=9, adjust=False
    ).mean()

    previous_close = close.shift(1)

    true_range = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs()
        ],
        axis=1
    ).max(axis=1)

    df["ATR"] = true_range.rolling(14).mean()

    df["Volatility"] = close.pct_change().rolling(20).std()

    avg_volume = volume.rolling(20).mean()

    df["Volume_Ratio"] = (
        volume / avg_volume.replace(0, np.nan)
    )

    candle_range = (high - low).replace(0, np.nan)

    df["Estimated_Buy_Pressure"] = (
        (close - low) / candle_range
    ) * 100

    df["Estimated_Sell_Pressure"] = (
        (high - close) / candle_range
    ) * 100

    df["BB_Middle"] = close.rolling(20).mean()

    bb_std = close.rolling(20).std()

    df["BB_Upper"] = df["BB_Middle"] + 2 * bb_std
    df["BB_Lower"] = df["BB_Middle"] - 2 * bb_std

    return df.replace([np.inf, -np.inf], np.nan)


data = calculate_indicators(data)

if len(data) < 2:
    st.error("বিশ্লেষণের জন্য পর্যাপ্ত candle নেই।")
    st.stop()

latest = data.iloc[-1]
previous = data.iloc[-2]

# ---------------- MARKET OVERVIEW ----------------

st.subheader("📌 Market Overview")

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Latest Price",
    f"{latest['close']:.2f}",
    f"{latest['close'] - previous['close']:.2f}"
)

c2.metric(
    "RSI",
    f"{latest['RSI']:.2f}"
    if pd.notna(latest["RSI"]) else "N/A"
)

c3.metric(
    "ATR",
    f"{latest['ATR']:.2f}"
    if pd.notna(latest["ATR"]) else "N/A"
)

c4.metric(
    "Volume",
    f"{latest['volume']:,.0f}"
)

st.caption(
    f"Last candle: {data.index[-1]}"
)

# ---------------- PRICE CHART ----------------

st.subheader("📈 Candlestick Chart")

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

for column, label in [
    ("EMA_20", "EMA 20"),
    ("EMA_50", "EMA 50"),
    ("EMA_200", "EMA 200")
]:
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data[column],
            name=label,
            mode="lines"
        )
    )

fig.update_layout(
    template="plotly_dark",
    height=600,
    xaxis_rangeslider_visible=False,
    hovermode="x unified"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# ---------------- TREND OBSERVATION ----------------

st.subheader("🧭 EMA Structure")

if (
    pd.notna(latest["EMA_20"])
    and pd.notna(latest["EMA_50"])
    and pd.notna(latest["EMA_200"])
):

    if (
        latest["EMA_20"] >
        latest["EMA_50"] >
        latest["EMA_200"]
    ):
        st.info("EMA alignment: Bullish structure")

    elif (
        latest["EMA_20"] <
        latest["EMA_50"] <
        latest["EMA_200"]
    ):
        st.info("EMA alignment: Bearish structure")

    else:
        st.info("EMA alignment: Mixed structure")

else:
    st.info("Not enough candles for full EMA analysis.")

# ---------------- RSI AND MACD ----------------

left, right = st.columns(2)

with left:

    st.subheader("RSI Analysis")

    if pd.isna(latest["RSI"]):
        st.write("RSI unavailable.")

    else:
        st.metric("RSI Value", f"{latest['RSI']:.2f}")

        if latest["RSI"] >= 70:
            st.warning("RSI is in the upper range.")

        elif latest["RSI"] <= 30:
            st.warning("RSI is in the lower range.")

        else:
            st.info("RSI is in the middle range.")

with right:

    st.subheader("MACD Analysis")

    st.metric("MACD", f"{latest['MACD']:.4f}")
    st.metric("Signal", f"{latest['MACD_Signal']:.4f}")

    if latest["MACD"] > latest["MACD_Signal"]:
        st.info("MACD is above its signal line.")
    else:
        st.info("MACD is below its signal line.")

# ---------------- SUPPORT AND RESISTANCE ----------------

st.subheader("📍 Recent Price Range")

lookback = min(50, len(data))
recent = data.tail(lookback)

support = recent["low"].min()
resistance = recent["high"].max()

c1, c2 = st.columns(2)

c1.metric("Recent Range Low", f"{support:.2f}")
c2.metric("Recent Range High", f"{resistance:.2f}")

st.caption(
    "These are recent historical range levels, "
    "not guaranteed support or resistance."
)

# ---------------- ICT-STYLE OBSERVATIONS ----------------

st.subheader("🧩 ICT-Style Candle Observations")

observations = []

if len(data) >= 5:

    current = data.iloc[-1]
    prior = data.iloc[-2]
    reference = data.iloc[-5:-1]

    if current["close"] > reference["high"].max():
        observations.append("Recent high breakout observation.")

    if current["close"] < reference["low"].min():
        observations.append("Recent low breakdown observation.")

    if (
        current["high"] > reference["high"].max()
        and current["close"] < reference["high"].max()
    ):
        observations.append("Possible upper liquidity sweep pattern.")

    if (
        current["low"] < reference["low"].min()
        and current["close"] > reference["low"].min()
    ):
        observations.append("Possible lower liquidity sweep pattern.")

    if current["low"] > data.iloc[-3]["high"]:
        observations.append("Possible bullish FVG pattern.")

    if current["high"] < data.iloc[-3]["low"]:
        observations.append("Possible bearish FVG pattern.")

if not observations:
    observations.append("No notable pattern detected in the selected candles.")

for item in observations:
    st.write("•", item)

st.caption(
    "These are simplified candle-based observations. "
    "They do not verify institutional orders or actual liquidity."
)

# ---------------- MULTI TIMEFRAME ----------------

st.subheader("⏱️ Multi-Timeframe Overview")

tf_rules = {
    "M5": "5min",
    "M15": "15min",
    "M30": "30min",
    "H1": "1h",
    "H4": "4h",
    "D1": "1D"
}

if data_source == "Uploaded CSV":

    tf_base = data.copy()

else:

    # Use available data to construct higher timeframes.
    tf_base = data.copy()

tf_results = []

for tf_name, rule in tf_rules.items():

    try:

        if tf_name == timeframe:
            tf_df = tf_base.copy()

        elif tf_name == "H4" and timeframe in ["M5", "M15", "M30", "H1"]:
            tf_df = resample_ohlc(tf_base, "4h")

        elif tf_name == "D1" and timeframe != "D1":
            tf_df = resample_ohlc(tf_base, "1D")

        else:
            continue

        if len(tf_df) < 2:
            continue

        last = tf_df.iloc[-1]
        prev = tf_df.iloc[-2]

        tf_results.append({
            "Timeframe": tf_name,
            "Latest Close": round(last["close"], 2),
            "Previous Close": round(prev["close"], 2),
            "Candle Direction": (
                "Up" if last["close"] > prev["close"] else "Down"
            )
        })

    except Exception:
        continue

if tf_results:

    st.dataframe(
        pd.DataFrame(tf_results),
        use_container_width=True,
        hide_index=True
    )

else:
    st.info("Not enough data for additional timeframes.")

st.caption(
    "Multi-timeframe results are limited by the source data's interval "
    "and available history."
)

# ---------------- OPTIONAL AI MODEL ----------------

st.subheader("🤖 Existing AI Model Status")

model_paths = [
    "XAUUSD_HYBRID_PROJECT/XAUUSD_AI_Model.pkl",
    "XAUUSD_AI_Model.pkl",
    "forex_ai_model.pkl"
]

model_found = False

for model_path in model_paths:

    if os.path.exists(model_path):

        try:

            loaded_model = joblib.load(model_path)
            model_found = True

            st.success(f"AI model file loaded: {model_path}")

            st.info(
                "Model পাওয়া গেছে। তবে মূল training data, "
                "feature calculation এবং class labels যাচাই না করে "
                "কোনো prediction দেখানো হচ্ছে না।"
            )

            break

        except Exception as exc:

            st.warning(
                f"Model file exists but could not be loaded: "
                f"{str(exc)[:250]}"
            )

if not model_found:

    st.info(
        "Compatible AI model file পাওয়া যায়নি। "
        "বাকি dashboard features চালু আছে।"
    )

# ---------------- NEWS ----------------

st.subheader("📰 Gold Market News")

@st.cache_data(ttl=900, show_spinner=False)
def load_news():

    url = (
        "https://news.google.com/rss/search?"
        "q=gold+market+XAUUSD&hl=en-US&gl=US&ceid=US:en"
    )

    feed = feedparser.parse(url)

    return [
        {
            "title": item.get("title", "Untitled"),
            "link": item.get("link", ""),
            "published": item.get("published", "")
        }
        for item in feed.entries[:10]
    ]


try:

    news = load_news()

    if news:

        for item in news:

            if item["link"]:
                st.markdown(
                    f"[{item['title']}]({item['link']})"
                )
            else:
                st.write(item["title"])

            if item["published"]:
                st.caption(item["published"])

    else:
        st.info("News feed returned no articles.")

except Exception as exc:

    st.warning(
        f"News feed unavailable: {str(exc)[:200]}"
    )

# ---------------- INDICATOR TABLE ----------------

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
    "Volatility",
    "Volume_Ratio",
    "Estimated_Buy_Pressure",
    "Estimated_Sell_Pressure"
]

available_columns = [
    col for col in display_columns
    if col in data.columns
]

st.dataframe(
    data[available_columns].tail(30).round(4),
    use_container_width=True
)

# ---------------- CSV EXPORT ----------------

st.subheader("📥 Export Analysis Data")

csv_bytes = data.to_csv().encode("utf-8")

st.download_button(
    label="Download Analysis CSV",
    data=csv_bytes,
    file_name="XAUUSD_Analysis.csv",
    mime="text/csv"
)

# ---------------- DIAGNOSTICS ----------------

with st.expander("🔧 Data Diagnostics"):

    st.write("Data source:", data_source)
    st.write("Selected timeframe:", timeframe)
    st.write("Rows loaded:", len(data))
    st.write("First candle:", str(data.index[0]))
    st.write("Last candle:", str(data.index[-1]))
    st.write("Missing values:")

    st.dataframe(
        data.isna().sum().rename("Missing Count").to_frame(),
        use_container_width=True
    )

# ---------------- FOOTER ----------------

st.divider()

st.caption(
    "XAUUSD AI HYBRID PRO | Educational research dashboard"
)

st.warning(
    "এই অ্যাপ শিক্ষামূলক তথ্য ও ঐতিহাসিক বাজার বিশ্লেষণের জন্য। "
    "এটি আর্থিক পরামর্শ, ভবিষ্যৎ ফলাফলের নিশ্চয়তা বা "
    "বাস্তব ট্রেডিং নির্দেশনা দেয় না।"
)

st.caption(
    "Dashboard updated: "
    + datetime.now().strftime("%Y-%m-%d %H:%M:%S")
)
