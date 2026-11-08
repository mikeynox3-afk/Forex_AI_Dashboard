
import os
import glob
import pickle
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import feedparser

from streamlit_autorefresh import st_autorefresh

warnings.filterwarnings("ignore")

# =====================================================
# 1. PAGE SETTINGS
# =====================================================

st.set_page_config(
    page_title="XAUUSD AI HYBRID PRO",
    page_icon="📊",
    layout="wide"
)

st_autorefresh(interval=60000, key="automatic_refresh")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.join(BASE_DIR, "XAUUSD_HYBRID_PROJECT")

FEATURES = [
    "EMA_20", "EMA_50", "EMA_200",
    "RSI", "MACD", "MACD_Signal",
    "ATR", "Volatility", "Volume_Ratio",
    "Estimated_Buy_Pressure",
    "Estimated_Sell_Pressure"
]

st.title("🏆 XAUUSD AI HYBRID PRO")
st.caption(
    "Multi-Timeframe | ICT | Smart Money | AI | News | Volume Research"
)
st.warning(
    "Educational research and simulation only. "
    "No broker connection or real order execution."
)

# =====================================================
# 2. MARKET DATA
# =====================================================

@st.cache_data(ttl=60)
def download_market():
    symbols = ["XAUUSD=X", "GC=F"]

    for symbol in symbols:
        try:
            data = yf.download(
                symbol,
                period="7d",
                interval="5m",
                progress=False,
                auto_adjust=False
            )

            if data is None or data.empty:
                continue

            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)

            data.columns = [str(c).title() for c in data.columns]

            required = ["Open", "High", "Low", "Close"]

            if not all(c in data.columns for c in required):
                continue

            if "Volume" not in data.columns:
                data["Volume"] = 0

            data = data[required + ["Volume"]].copy()
            data = data.apply(pd.to_numeric, errors="coerce")
            data = data.dropna(subset=required)

            if len(data) > 50:
                return data, symbol

        except Exception:
            continue

    return pd.DataFrame(), None


# =====================================================
# 3. TECHNICAL INDICATORS
# =====================================================

def calculate_indicators(df):

    df = df.copy()

    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"].fillna(0)

    df["EMA_20"] = close.ewm(span=20).mean()
    df["EMA_50"] = close.ewm(span=50).mean()
    df["EMA_200"] = close.ewm(span=200).mean()

    delta = close.diff()

    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()

    rs = gain / loss.replace(0, np.nan)
    df["RSI"] = 100 - (100 / (1 + rs))

    ema12 = close.ewm(span=12).mean()
    ema26 = close.ewm(span=26).mean()

    df["MACD"] = ema12 - ema26
    df["MACD_Signal"] = df["MACD"].ewm(span=9).mean()

    previous_close = close.shift(1)

    tr = pd.concat([
        high - low,
        (high - previous_close).abs(),
        (low - previous_close).abs()
    ], axis=1).max(axis=1)

    df["ATR"] = tr.rolling(14).mean()

    df["Volatility"] = close.pct_change().rolling(20).std()

    avg_volume = volume.rolling(20).mean()

    df["Volume_Ratio"] = (
        volume / avg_volume.replace(0, np.nan)
    )

    candle_range = (high - low).replace(0, np.nan)

    body = close - df["Open"]

    df["Estimated_Buy_Pressure"] = (
        ((body / candle_range) + 1) / 2
    ).clip(0, 1).fillna(0.5)

    df["Estimated_Sell_Pressure"] = (
        1 - df["Estimated_Buy_Pressure"]
    )

    df["Upper_Wick"] = (
        high - df[["Open", "Close"]].max(axis=1)
    )

    df["Lower_Wick"] = (
        df[["Open", "Close"]].min(axis=1) - low
    )

    df["VWAP"] = (
        ((high + low + close) / 3 * volume).cumsum()
        / volume.replace(0, np.nan).cumsum()
    )

    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()

    df["BB_Upper"] = bb_mid + 2 * bb_std
    df["BB_Lower"] = bb_mid - 2 * bb_std

    return df.replace([np.inf, -np.inf], np.nan)


# =====================================================
# 4. MULTI-TIMEFRAME
# =====================================================

def resample_market(df, timeframe):

    rules = {
        "M5": "5min",
        "M15": "15min",
        "M30": "30min",
        "H1": "1h",
        "H4": "4h",
        "D1": "1D",
        "W1": "1W"
    }

    result = df.resample(rules[timeframe]).agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum"
    }).dropna()

    return calculate_indicators(result)


# =====================================================
# 5. ICT AND MARKET STRUCTURE
# =====================================================

def ict_analysis(df):

    recent = df.tail(50)

    if len(recent) < 5:
        return {}

    last = recent.iloc[-1]

    previous = recent.iloc[:-1]

    swing_high = previous["High"].max()
    swing_low = previous["Low"].min()

    bullish_sweep = (
        last["Low"] < swing_low
        and last["Close"] > swing_low
    )

    bearish_sweep = (
        last["High"] > swing_high
        and last["Close"] < swing_high
    )

    bullish_bos = last["Close"] > swing_high
    bearish_bos = last["Close"] < swing_low

    bullish_fvg = False
    bearish_fvg = False

    if len(recent) >= 3:
        first = recent.iloc[-3]

        bullish_fvg = last["Low"] > first["High"]
        bearish_fvg = last["High"] < first["Low"]

    equal_high = (
        recent["High"].tail(10).max()
        - recent["High"].tail(10).min()
    ) < max(float(last["ATR"]) * 0.15, 0.01)

    equal_low = (
        recent["Low"].tail(10).max()
        - recent["Low"].tail(10).min()
    ) < max(float(last["ATR"]) * 0.15, 0.01)

    if bullish_bos:
        structure = "BULLISH BOS"
    elif bearish_bos:
        structure = "BEARISH BOS"
    else:
        structure = "NO CLEAR BREAK"

    high_zone = recent["High"].tail(30).max()
    low_zone = recent["Low"].tail(30).min()

    midpoint = (high_zone + low_zone) / 2

    if last["Close"] > midpoint:
        premium_discount = "PREMIUM"
    elif last["Close"] < midpoint:
        premium_discount = "DISCOUNT"
    else:
        premium_discount = "EQUILIBRIUM"

    return {
        "Structure": structure,
        "Bullish Sweep": bullish_sweep,
        "Bearish Sweep": bearish_sweep,
        "Bullish BOS": bullish_bos,
        "Bearish BOS": bearish_bos,
        "Bullish FVG": bullish_fvg,
        "Bearish FVG": bearish_fvg,
        "Equal High": equal_high,
        "Equal Low": equal_low,
        "Premium/Discount": premium_discount,
        "Swing High": swing_high,
        "Swing Low": swing_low,
        "Midpoint": midpoint
    }


# =====================================================
# 6. SUPPORT AND RESISTANCE
# =====================================================

def support_resistance(df):

    recent = df.tail(100)

    return {
        "Resistance": float(recent["High"].max()),
        "Support": float(recent["Low"].min())
    }


# =====================================================
# 7. VOLUME ANALYSIS
# =====================================================

def volume_analysis(df):

    last = df.iloc[-1]

    ratio = last["Volume_Ratio"]

    if pd.isna(ratio):
        state = "UNAVAILABLE"
    elif ratio >= 2:
        state = "HIGH VOLUME"
    elif ratio >= 1.3:
        state = "ABOVE AVERAGE"
    else:
        state = "NORMAL"

    return {
        "Volume Ratio": ratio,
        "Volume State": state,
        "Buy Pressure Estimate": last["Estimated_Buy_Pressure"],
        "Sell Pressure Estimate": last["Estimated_Sell_Pressure"]
    }


# =====================================================
# 8. AI MODEL
# =====================================================

@st.cache_resource
def load_ai_model():

    paths = [
        os.path.join(PROJECT_DIR, "XAUUSD_AI_Model.pkl"),
        os.path.join(BASE_DIR, "XAUUSD_AI_Model.pkl"),
        os.path.join(PROJECT_DIR, "forex_ai_model.pkl"),
        os.path.join(BASE_DIR, "forex_ai_model.pkl")
    ]

    for path in paths:
        if os.path.exists(path):
            try:
                with open(path, "rb") as f:
                    model = pickle.load(f)
                return model, path
            except Exception:
                continue

    return None, None


def model_analysis(df, model):

    if model is None:
        return "MODEL NOT FOUND", None, None

    try:
        row = df[FEATURES].iloc[[-1]].copy()

        if row.isna().any().any():
            return "INSUFFICIENT FEATURES", None, None

        prediction = model.predict(row)[0]

        probabilities = None

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(row)[0]

        if probabilities is not None:
            classes = list(model.classes_)
            prob_map = dict(zip(classes, probabilities))

            bullish_score = float(prob_map.get(1, 0)) * 100
            bearish_score = float(prob_map.get(0, 0)) * 100
        else:
            bullish_score = None
            bearish_score = None

        label = "BULLISH" if prediction == 1 else "BEARISH"

        return label, bullish_score, bearish_score

    except Exception as e:
        return "MODEL FEATURE ERROR", None, None


# =====================================================
# 9. NEWS ANALYSIS
# =====================================================

@st.cache_data(ttl=300)
def get_news():

    urls = [
        "https://news.google.com/rss/search?q=gold+XAUUSD&hl=en-US&gl=US&ceid=US:en",
        "https://news.google.com/rss/search?q=US+dollar+Federal+Reserve&hl=en-US&gl=US&ceid=US:en",
        "https://news.google.com/rss/search?q=US+inflation+CPI+NFP+FOMC&hl=en-US&gl=US&ceid=US:en"
    ]

    results = {}

    for url in urls:
        try:
            feed = feedparser.parse(url)

            for entry in feed.entries[:10]:

                title = entry.get("title", "News")
                results[title] = {
                    "title": title,
                    "link": entry.get("link", ""),
                    "published": entry.get("published", "Unknown")
                }

        except Exception:
            continue

    return list(results.values())[:20]


def analyze_news(news):

    positive = [
        "gold rises", "gold gains", "weak dollar",
        "rate cut", "safe haven", "dollar falls"
    ]

    negative = [
        "gold falls", "gold drops", "strong dollar",
        "rate hike", "higher yields", "dollar rises"
    ]

    score = 0

    for item in news:

        title = item["title"].lower()

        for word in positive:
            if word in title:
                score += 1

        for word in negative:
            if word in title:
                score -= 1

    if score > 0:
        sentiment = "POSITIVE"
    elif score < 0:
        sentiment = "NEGATIVE"
    else:
        sentiment = "NEUTRAL"

    return sentiment, score


# =====================================================
# 10. MAIN DASHBOARD
# =====================================================

with st.spinner("Loading market data..."):

    market, symbol = download_market()

if market.empty:

    st.error(
        "Market data unavailable. Data provider or connection may be unavailable."
    )

    st.stop()

market = calculate_indicators(market)

latest = market.iloc[-1]

price = float(latest["Close"])

st.caption(f"Data Source: Yahoo Finance | Symbol: {symbol}")
st.caption(f"Latest available data: {market.index[-1]}")

# Main metrics

c1, c2, c3, c4, c5 = st.columns(5)

c1.metric("XAUUSD Price", f"{price:.2f}")

c2.metric(
    "RSI",
    f"{latest['RSI']:.2f}" if pd.notna(latest["RSI"]) else "N/A"
)

c3.metric("EMA 20", f"{latest['EMA_20']:.2f}")

c4.metric("EMA 50", f"{latest['EMA_50']:.2f}")

c5.metric(
    "ATR",
    f"{latest['ATR']:.2f}" if pd.notna(latest["ATR"]) else "N/A"
)

# =====================================================
# 11. CHART
# =====================================================

st.subheader("📈 Advanced Market Chart")

timeframe = st.selectbox(
    "Select Timeframe",
    ["M5", "M15", "M30", "H1", "H4", "D1", "W1"]
)

chart_data = resample_market(market, timeframe)

fig = go.Figure()

fig.add_trace(go.Candlestick(
    x=chart_data.index,
    open=chart_data["Open"],
    high=chart_data["High"],
    low=chart_data["Low"],
    close=chart_data["Close"],
    name="Gold"
))

for column in ["EMA_20", "EMA_50", "EMA_200"]:

    fig.add_trace(go.Scatter(
        x=chart_data.index,
        y=chart_data[column],
        name=column
    ))

fig.update_layout(
    height=600,
    xaxis_rangeslider_visible=False,
    template="plotly_dark"
)

st.plotly_chart(fig, use_container_width=True)

# =====================================================
# 12. MULTI-TIMEFRAME
# =====================================================

st.subheader("🕒 Multi-Timeframe Dashboard")

timeframes = ["M5", "M15", "M30", "H1", "H4", "D1", "W1"]

columns = st.columns(4)

for i, tf in enumerate(timeframes):

    data = resample_market(market, tf)

    if not data.empty:

        row = data.iloc[-1]

        with columns[i % 4]:

            st.metric(
                f"{tf} Close",
                f"{row['Close']:.2f}"
            )

            if pd.notna(row["RSI"]):
                st.caption(f"RSI: {row['RSI']:.2f}")


# =====================================================
# 13. ICT ANALYSIS
# =====================================================

st.subheader("🧠 ICT / Smart Money Analysis")

ict = ict_analysis(market)

if ict:

    a, b, c = st.columns(3)

    a.metric("Market Structure", ict["Structure"])

    b.metric("Premium / Discount", ict["Premium/Discount"])

    c.metric("Swing High", f"{ict['Swing High']:.2f}")

    a, b, c = st.columns(3)

    a.write(f"Bullish Liquidity Sweep: {ict.get('Bullish Sweep', 'N/A')}")
    b.write(f"Bearish Liquidity Sweep: {ict.get('Bearish Sweep', 'N/A')}")
    c.write(f"Bullish BOS: {ict.get('Bullish BOS', 'N/A')}")

    a, b, c = st.columns(3)

    a.write(f"Bearish BOS: {ict.get('Bearish BOS', 'N/A')}")
    b.write(f"Bullish FVG: {ict.get('Bullish FVG', 'N/A')}")
    c.write(f"Bearish FVG: {ict.get('Bearish FVG', 'N/A')}")

    a, b = st.columns(2)

    a.write("Equal High:", ict["Equal High"])
    b.write("Equal Low:", ict["Equal Low"])

    st.caption(
        "ICT, liquidity and institutional activity labels are simplified "
        "price-action heuristics. They do not reveal actual institutional orders."
    )


# =====================================================
# 14. SUPPORT AND RESISTANCE
# =====================================================

st.subheader("📍 Support / Resistance")

sr = support_resistance(market)

a, b = st.columns(2)

a.metric("Support Zone", f"{sr['Support']:.2f}")
b.metric("Resistance Zone", f"{sr['Resistance']:.2f}")


# =====================================================
# 15. VOLUME AND ORDER FLOW PROXY
# =====================================================

st.subheader("📊 Volume & Order Flow Proxy")

vol = volume_analysis(market)

a, b, c = st.columns(3)

a.metric(
    "Volume Ratio",
    f"{vol['Volume Ratio']:.2f}"
    if pd.notna(vol["Volume Ratio"]) else "N/A"
)

b.metric("Volume State", vol["Volume State"])

c.metric(
    "Estimated Buy Pressure",
    f"{vol['Buy Pressure Estimate'] * 100:.1f}%"
)

st.write(
    "Estimated Sell Pressure:",
    f"{vol['Sell Pressure Estimate'] * 100:.1f}%"
)

st.caption(
    "Yahoo Finance volume is a data-provider volume measure, not verified "
    "centralized XAUUSD order flow. Buy/sell pressure is only a candle-based estimate."
)


# =====================================================
# 16. NEWS
# =====================================================

st.subheader("🌍 Gold / USD News Analysis")

news = get_news()

news_sentiment, news_score = analyze_news(news)

a, b = st.columns(2)

a.metric("News Sentiment", news_sentiment)
b.metric("Keyword Score", news_score)

if news:

    for item in news:

        st.markdown(f"**[{item['title']}]({item['link']})**")
        st.caption(item["published"])

else:

    st.info("News feed currently unavailable.")


# =====================================================
# 17. AI MODEL
# =====================================================

st.subheader("🤖 Previous AI Model Analysis")

model, model_path = load_ai_model()

model_label, bullish_score, bearish_score = model_analysis(
    market,
    model
)

if model is not None:

    st.success(f"AI Model Loaded: {os.path.basename(model_path)}")

else:

    st.warning(
        "Previous AI model could not be loaded. Rule-based research remains available."
    )

if bullish_score is not None:

    a, b = st.columns(2)

    a.metric("Model Bullish Score", f"{bullish_score:.1f}%")
    b.metric("Model Bearish Score", f"{bearish_score:.1f}%")

st.write("Model Classification:", model_label)

st.caption(
    "Model classification scores are not probabilities of profit or trade success. "
    "The model is not automatically validated for every timeframe."
)


# =====================================================
# 18. RESEARCH CLASSIFICATION
# =====================================================

st.subheader("🎯 Combined Research Classification")

bull = 0
bear = 0

if latest["EMA_20"] > latest["EMA_50"]:
    bull += 1
else:
    bear += 1

if latest["MACD"] > latest["MACD_Signal"]:
    bull += 1
else:
    bear += 1

if pd.notna(latest["RSI"]):

    if latest["RSI"] > 55:
        bull += 1

    elif latest["RSI"] < 45:
        bear += 1

if ict.get("Bullish Sweep"):
    bull += 1

if ict.get("Bearish Sweep"):
    bear += 1

if ict.get("Bullish BOS"):
    bull += 1

if ict.get("Bearish BOS"):
    bear += 1

if news_sentiment == "POSITIVE":
    bull += 1

elif news_sentiment == "NEGATIVE":
    bear += 1

if bull > bear:
    classification = "BULLISH"
elif bear > bull:
    classification = "BEARISH"
else:
    classification = "NEUTRAL"

a, b, c = st.columns(3)

a.metric("Bullish Evidence", bull)
b.metric("Bearish Evidence", bear)
c.metric("Research Result", classification)

if classification == "BULLISH":

    st.success("BUY — SIMULATION / RESEARCH LABEL")

elif classification == "BEARISH":

    st.error("SELL — SIMULATION / RESEARCH LABEL")

else:

    st.warning("HOLD — SIMULATION / RESEARCH LABEL")


# =====================================================
# 19. SIGNAL HISTORY
# =====================================================

st.subheader("📚 Research Signal History")

if "signal_history" not in st.session_state:

    st.session_state.signal_history = []

signal_record = {
    "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "Price": price,
    "Classification": classification,
    "Bullish Evidence": bull,
    "Bearish Evidence": bear
}

if (
    not st.session_state.signal_history
    or st.session_state.signal_history[-1]["Classification"] != classification
    or st.session_state.signal_history[-1]["Price"] != price
):

    st.session_state.signal_history.append(signal_record)

st.dataframe(
    pd.DataFrame(st.session_state.signal_history[::-1]),
    use_container_width=True
)

# =====================================================
# 20. HISTORICAL DATA
# =====================================================

st.subheader("📁 Historical Market Data")

with st.expander("View Market Data"):

    st.dataframe(
        market.tail(200),
        use_container_width=True
    )

# =====================================================
# 21. SYSTEM STATUS
# =====================================================

st.subheader("⚙️ System Status")

st.write("Automatic refresh: Every 60 seconds")
st.write("Market data: Latest available provider data")
st.write("AI model: Previous model file, if compatible")
st.write("News: RSS feeds, when available")
st.write("Real order execution: Disabled")

st.write(
    "Dashboard update:",
    datetime.now().strftime("%Y-%m-%d %H:%M:%S")
)

st.caption(
    "XAUUSD AI HYBRID PRO | Historical and latest available data research | "
    "Educational simulation only | No financial advice"
)
