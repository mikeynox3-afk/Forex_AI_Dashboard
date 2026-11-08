
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from pathlib import Path
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

# Optional packages
try:
    from streamlit_autorefresh import st_autorefresh
    AUTO_REFRESH_AVAILABLE = True
except Exception:
    AUTO_REFRESH_AVAILABLE = False

try:
    import feedparser
    NEWS_AVAILABLE = True
except Exception:
    NEWS_AVAILABLE = False

try:
    import joblib
    MODEL_AVAILABLE = True
except Exception:
    MODEL_AVAILABLE = False


# --------------------------------------------------
# PAGE SETTINGS
# --------------------------------------------------

st.set_page_config(
    page_title="XAUUSD AI Hybrid Research",
    page_icon="📊",
    layout="wide"
)

st.title("📊 XAUUSD AI HYBRID PRO")
st.caption(
    "Educational Market Research Dashboard | "
    "No broker connection | No live order execution"
)

# --------------------------------------------------
# AUTO REFRESH
# --------------------------------------------------

if AUTO_REFRESH_AVAILABLE:
    st_autorefresh(interval=60000, key="market_refresh")

st.sidebar.header("Dashboard Settings")

refresh_seconds = st.sidebar.selectbox(
    "Auto refresh",
    [30, 60, 120, 300],
    index=1
)

st.sidebar.info(
    "ডেটা Provider নতুন তথ্য দিলে Dashboard আপডেট হবে। "
    "Refresh interval মানেই Real-time data guarantee নয়।"
)

# --------------------------------------------------
# CONSTANTS
# --------------------------------------------------

SYMBOLS = ["XAUUSD=X", "GC=F"]

TIMEFRAMES = {
    "M5": "5min",
    "M15": "15min",
    "M30": "30min",
    "H1": "1h",
    "H4": "4h",
    "D1": "1D"
}

FEATURES = [
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

# --------------------------------------------------
# DOWNLOAD MARKET DATA
# --------------------------------------------------

@st.cache_data(ttl=60, show_spinner=False)
def download_market_data():

    for symbol in SYMBOLS:
        try:
            df = yf.download(
                symbol,
                period="7d",
                interval="5m",
                auto_adjust=False,
                progress=False,
                threads=False
            )

            if df is None or df.empty:
                continue

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            df.columns = [
                str(c).strip().title() for c in df.columns
            ]

            required = [
                "Open", "High", "Low", "Close"
            ]

            if not all(c in df.columns for c in required):
                continue

            if "Volume" not in df.columns:
                df["Volume"] = 0

            df = df[[
                "Open", "High", "Low", "Close", "Volume"
            ]].copy()

            df = df.apply(
                pd.to_numeric, errors="coerce"
            )

            df = df.dropna(
                subset=["Open", "High", "Low", "Close"]
            )

            df = df[~df.index.duplicated(keep="last")]
            df = df.sort_index()

            if len(df) >= 50:
                return df, symbol

        except Exception:
            continue

    return pd.DataFrame(), None


# --------------------------------------------------
# TECHNICAL INDICATORS
# --------------------------------------------------

def calculate_indicators(df):

    df = df.copy()

    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]

    df["EMA_20"] = close.ewm(
        span=20, adjust=False
    ).mean()

    df["EMA_50"] = close.ewm(
        span=50, adjust=False
    ).mean()

    df["EMA_200"] = close.ewm(
        span=200, adjust=False
    ).mean()

    # RSI
    delta = close.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1/14, adjust=False
    ).mean()

    avg_loss = loss.ewm(
        alpha=1/14, adjust=False
    ).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

    df["RSI"] = 100 - (100 / (1 + rs))
    df["RSI"] = df["RSI"].fillna(50)

    # MACD
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

    # ATR
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

    # Volatility
    df["Volatility"] = (
        close.pct_change().rolling(20).std()
    )

    # Volume ratio
    volume_average = volume.rolling(20).mean()

    df["Volume_Ratio"] = (
        volume / volume_average.replace(0, np.nan)
    ).replace([np.inf, -np.inf], np.nan).fillna(1)

    # Estimated pressure: candle-based proxy only
    candle_range = (high - low).replace(0, np.nan)

    buy_pressure = (
        (close - low) / candle_range
    ).clip(0, 1).fillna(0.5)

    sell_pressure = (
        (high - close) / candle_range
    ).clip(0, 1).fillna(0.5)

    df["Estimated_Buy_Pressure"] = buy_pressure
    df["Estimated_Sell_Pressure"] = sell_pressure

    # Bollinger Bands
    middle = close.rolling(20).mean()
    std = close.rolling(20).std()

    df["BB_Upper"] = middle + (2 * std)
    df["BB_Lower"] = middle - (2 * std)

    # VWAP approximation
    typical_price = (high + low + close) / 3

    cumulative_volume = volume.cumsum()

    df["VWAP"] = (
        (typical_price * volume).cumsum()
        / cumulative_volume.replace(0, np.nan)
    )

    return df.replace(
        [np.inf, -np.inf], np.nan
    )


# --------------------------------------------------
# MULTI TIMEFRAME DATA
# --------------------------------------------------

def resample_timeframe(df, timeframe):

    rules = {
        "M15": "15min",
        "M30": "30min",
        "H1": "1h",
        "H4": "4h",
        "D1": "1D"
    }

    if timeframe == "M5":
        return df.copy()

    rule = rules.get(timeframe)

    if rule is None:
        return pd.DataFrame()

    result = df.resample(rule).agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum"
    })

    return result.dropna(subset=["Open", "Close"])


# --------------------------------------------------
# ICT / SMART MONEY HEURISTICS
# --------------------------------------------------

def ict_analysis(df):

    result = {
        "Bullish Sweep": False,
        "Bearish Sweep": False,
        "Bullish BOS": False,
        "Bearish BOS": False,
        "Bullish FVG": False,
        "Bearish FVG": False,
        "Equal High": False,
        "Equal Low": False,
        "Market Structure": "Unavailable",
        "Premium/Discount": "Unavailable"
    }

    if df is None or len(df) < 25:
        return result

    last = df.iloc[-1]
    previous = df.iloc[-2]

    recent = df.iloc[-21:-1]

    prior_high = recent["High"].max()
    prior_low = recent["Low"].min()

    # Break of structure (heuristic)
    result["Bullish BOS"] = bool(
        last["Close"] > prior_high
    )

    result["Bearish BOS"] = bool(
        last["Close"] < prior_low
    )

    # Liquidity sweep (heuristic)
    result["Bullish Sweep"] = bool(
        last["Low"] < prior_low
        and last["Close"] > prior_low
    )

    result["Bearish Sweep"] = bool(
        last["High"] > prior_high
        and last["Close"] < prior_high
    )

    # Fair Value Gap (heuristic)
    result["Bullish FVG"] = bool(
        last["Low"] > df["High"].iloc[-3]
    )

    result["Bearish FVG"] = bool(
        last["High"] < df["Low"].iloc[-3]
    )

    # Equal highs / lows approximation
    atr = last.get("ATR", np.nan)

    if pd.notna(atr) and atr > 0:
        tolerance = atr * 0.15

        highs = df["High"].iloc[-10:-1]
        lows = df["Low"].iloc[-10:-1]

        result["Equal High"] = bool(
            (highs.max() - highs.min()) <= tolerance
        )

        result["Equal Low"] = bool(
            (lows.max() - lows.min()) <= tolerance
        )

    if result["Bullish BOS"]:
        result["Market Structure"] = "Bullish BOS"
    elif result["Bearish BOS"]:
        result["Market Structure"] = "Bearish BOS"
    else:
        result["Market Structure"] = "No confirmed break"

    midpoint = (
        recent["High"].max() + recent["Low"].min()
    ) / 2

    result["Premium/Discount"] = (
        "Premium" if last["Close"] > midpoint
        else "Discount"
    )

    return result


# --------------------------------------------------
# AI MODEL LOADING
# --------------------------------------------------

@st.cache_resource
def load_ai_model():

    if not MODEL_AVAILABLE:
        return None, "joblib unavailable"

    candidates = [
        Path("XAUUSD_HYBRID_PROJECT/XAUUSD_AI_Model.pkl"),
        Path("XAUUSD_AI_Model.pkl"),
        Path("forex_ai_model.pkl")
    ]

    for path in candidates:
        if path.exists():
            try:
                model = joblib.load(path)
                return model, str(path)
            except Exception as error:
                return None, str(error)

    return None, "No compatible model file found"


def run_ai_model(model, df):

    if model is None or df is None or df.empty:
        return "Unavailable", "Model not loaded"

    try:
        row = df.iloc[[-1]][FEATURES].copy()

        if row.isnull().any().any():
            return "Unavailable", "Feature data incomplete"

        expected = getattr(
            model, "n_features_in_", len(FEATURES)
        )

        if expected != len(FEATURES):
            return "Unavailable", "Feature count mismatch"

        prediction = model.predict(row)[0]

        # This mapping assumes the original training labels:
        # 0 = bearish and 1 = bullish.
        if prediction == 1:
            label = "Bullish classification"
        elif prediction == 0:
            label = "Bearish classification"
        else:
            label = f"Class: {prediction}"

        score_text = "Classification only; not a probability"

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(row)[0]
            score_text += (
                f" | Model output: "
                f"{np.round(probabilities, 3).tolist()}"
            )

        return label, score_text

    except Exception as error:
        return "Unavailable", str(error)


# --------------------------------------------------
# NEWS
# --------------------------------------------------

@st.cache_data(ttl=900, show_spinner=False)
def get_news():

    if not NEWS_AVAILABLE:
        return []

    url = (
        "https://news.google.com/rss/search?"
        "q=gold+OR+XAUUSD+market&hl=en-US&gl=US&ceid=US:en"
    )

    try:
        feed = feedparser.parse(url)

        articles = []

        for item in feed.entries[:10]:
            articles.append({
                "title": item.get("title", "No title"),
                "link": item.get("link", "")
            })

        return articles

    except Exception:
        return []


# --------------------------------------------------
# MARKET DATA
# --------------------------------------------------

with st.spinner("Downloading market data..."):

    raw_data, used_symbol = download_market_data()

if raw_data.empty:

    st.error(
        "Market data পাওয়া যায়নি। Yahoo Finance-এর "
        "ডেটা সীমাবদ্ধতা বা নেটওয়ার্ক সমস্যার কারণে হতে পারে।"
    )

    st.info(
        "কিছুক্ষণ পর Refresh করো। "
        "এটি কোনো Broker বা Trading account-এর সঙ্গে যুক্ত নয়।"
    )

    st.stop()


df = calculate_indicators(raw_data)

last = df.iloc[-1]

# --------------------------------------------------
# MARKET OVERVIEW
# --------------------------------------------------

st.subheader("Market Overview")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Latest Price",
    f"{last['Close']:.2f}"
)

col2.metric(
    "RSI",
    f"{last['RSI']:.2f}"
)

col3.metric(
    "EMA 20",
    f"{last['EMA_20']:.2f}"
)

col4.metric(
    "ATR",
    f"{last['ATR']:.4f}"
    if pd.notna(last["ATR"]) else "N/A"
)

st.caption(
    f"Data source: Yahoo Finance ({used_symbol}) | "
    f"Last candle: {df.index[-1]}"
)

# --------------------------------------------------
# PRICE CHART
# --------------------------------------------------

st.subheader("Gold Price Chart")

chart = go.Figure()

chart.add_trace(
    go.Candlestick(
        x=df.index,
        open=df["Open"],
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        name="Gold"
    )
)

chart.add_trace(
    go.Scatter(
        x=df.index,
        y=df["EMA_20"],
        name="EMA 20"
    )
)

chart.add_trace(
    go.Scatter(
        x=df.index,
        y=df["EMA_50"],
        name="EMA 50"
    )
)

chart.update_layout(
    height=550,
    xaxis_rangeslider_visible=False,
    template="plotly_dark"
)

st.plotly_chart(chart, use_container_width=True)

# --------------------------------------------------
# MULTI TIMEFRAME ANALYSIS
# --------------------------------------------------

st.subheader("Multi-Timeframe Analysis")

mtf_rows = []

mtf_frames = {}

for timeframe in TIMEFRAMES:

    frame = resample_timeframe(df, timeframe)

    if frame.empty:
        continue

    frame = calculate_indicators(frame)
    mtf_frames[timeframe] = frame

    current = frame.iloc[-1]

    trend = "Neutral"

    if current["Close"] > current["EMA_20"]:
        trend = "Above EMA 20"

    elif current["Close"] < current["EMA_20"]:
        trend = "Below EMA 20"

    mtf_rows.append({
        "Timeframe": timeframe,
        "Close": round(current["Close"], 2),
        "EMA 20": round(current["EMA_20"], 2),
        "RSI": round(current["RSI"], 2),
        "Trend": trend,
        "Candles": len(frame)
    })

if mtf_rows:
    st.dataframe(
        pd.DataFrame(mtf_rows),
        use_container_width=True,
        hide_index=True
    )

st.caption(
    "M15 থেকে D1 পর্যন্ত ডেটা M5 candles থেকে resample করা। "
    "তাই এগুলো আলাদা provider-এর independent timeframe data নয়।"
)

# --------------------------------------------------
# ICT ANALYSIS
# --------------------------------------------------

st.subheader("ICT / Smart Money Research")

selected_tf = st.selectbox(
    "ICT timeframe",
    list(mtf_frames.keys()),
    index=0
)

ict = ict_analysis(mtf_frames[selected_tf])

a, b, c = st.columns(3)

a.write(
    "Bullish Liquidity Sweep:",
    ict.get("Bullish Sweep", False)
)

b.write(
    "Bearish Liquidity Sweep:",
    ict.get("Bearish Sweep", False)
)

c.write(
    "Market Structure:",
    ict.get("Market Structure", "N/A")
)

a, b, c = st.columns(3)

a.write(
    "Bullish BOS:",
    ict.get("Bullish BOS", False)
)

b.write(
    "Bearish BOS:",
    ict.get("Bearish BOS", False)
)

c.write(
    "Premium / Discount:",
    ict.get("Premium/Discount", "N/A")
)

a, b, c = st.columns(3)

a.write(
    "Bullish FVG:",
    ict.get("Bullish FVG", False)
)

b.write(
    "Bearish FVG:",
    ict.get("Bearish FVG", False)
)

c.write(
    "Equal High / Low:",
    f"{ict.get('Equal High', False)} / "
    f"{ict.get('Equal Low', False)}"
)

st.caption(
    "ICT labels are simplified candle-based heuristics, "
    "not verified institutional order-flow data."
)

# --------------------------------------------------
# VOLUME ANALYSIS
# --------------------------------------------------

st.subheader("Volume Research")

volume_col1, volume_col2 = st.columns(2)

volume_col1.metric(
    "Volume Ratio",
    f"{last['Volume_Ratio']:.2f}"
)

volume_col2.metric(
    "Estimated Buy / Sell Pressure",
    f"{last['Estimated_Buy_Pressure']:.2f} / "
    f"{last['Estimated_Sell_Pressure']:.2f}"
)

st.caption(
    "Estimated pressure is calculated from candle position. "
    "It is not actual institutional buying or selling."
)

# --------------------------------------------------
# AI MODEL
# --------------------------------------------------

st.subheader("AI Model Research")

model, model_status = load_ai_model()

st.write("Model status:", model_status)

ai_label, ai_details = run_ai_model(model, df)

st.metric(
    "AI Classification",
    ai_label
)

st.write(ai_details)

st.warning(
    "AI classification is based on the existing model and its "
    "training labels. It is not a trading signal, prediction "
    "guarantee, or probability of profit."
)

# --------------------------------------------------
# SUPPORT / RESISTANCE
# --------------------------------------------------

st.subheader("Recent Support & Resistance")

recent_data = df.tail(100)

support = recent_data["Low"].min()
resistance = recent_data["High"].max()

support_col, resistance_col = st.columns(2)

support_col.metric(
    "Recent Support Area",
    f"{support:.2f}"
)

resistance_col.metric(
    "Recent Resistance Area",
    f"{resistance:.2f}"
)

# --------------------------------------------------
# NEWS
# --------------------------------------------------

st.subheader("Gold Market News")

articles = get_news()

if articles:

    for article in articles:
        title = article.get("title", "News")

        link = article.get("link", "")

        if link:
            st.markdown(f"- [{title}]({link})")
        else:
            st.write(title)

else:
    st.info(
        "News পাওয়া যায়নি। feedparser বা News feed "
        "সাময়িকভাবে unavailable হতে পারে।"
    )

st.caption(
    "News headlines are external content. "
    "Their market impact is not automatically verified."
)

# --------------------------------------------------
# RAW MARKET DATA
# --------------------------------------------------

st.subheader("Latest Market Data")

display_columns = [
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "EMA_20",
    "EMA_50",
    "RSI",
    "MACD",
    "ATR"
]

st.dataframe(
    df[display_columns].tail(50).sort_index(
        ascending=False
    ),
    use_container_width=True
)

# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "XAUUSD AI HYBRID PRO | Educational Research Only"
)

st.caption(
    "No broker API, no deposits, no live order execution. "
    "Market data may be delayed, incomplete, or unavailable."
)

st.caption(
    f"Dashboard generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
)
