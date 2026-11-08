
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import feedparser
from streamlit_autorefresh import st_autorefresh
from datetime import datetime, timezone

st.set_page_config(
    page_title="XAUUSD AI Research Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("XAUUSD AI HYBRID RESEARCH DASHBOARD")
st.caption("Educational market research and simulation only. No real trades.")

st_autorefresh(interval=60000, key="market_refresh")

@st.cache_data(ttl=60)
def get_market_data():
    for symbol in ["XAUUSD=X", "GC=F"]:
        try:
            df = yf.download(
                symbol,
                period="5d",
                interval="5m",
                progress=False
            )
            if df is not None and not df.empty:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                df = df.rename(columns=str.title)
                required = ["Open", "High", "Low", "Close", "Volume"]
                if all(c in df.columns for c in required):
                    return df[required].dropna(), symbol
        except Exception:
            pass
    return pd.DataFrame(), None

def add_indicators(df):
    df = df.copy()
    df["EMA20"] = df["Close"].ewm(span=20).mean()
    df["EMA50"] = df["Close"].ewm(span=50).mean()
    df["EMA200"] = df["Close"].ewm(span=200).mean()

    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["RSI"] = 100 - (100 / (1 + rs))

    fast = df["Close"].ewm(span=12).mean()
    slow = df["Close"].ewm(span=26).mean()
    df["MACD"] = fast - slow
    df["MACD_Signal"] = df["MACD"].ewm(span=9).mean()

    df["ATR"] = (
        df["High"].combine(df["Close"].shift(), max)
        - df["Low"].combine(df["Close"].shift(), min)
    ).rolling(14).mean()

    return df

def resample_data(df, timeframe):
    rule = {
        "M5": "5min",
        "M15": "15min",
        "H1": "1h",
        "H4": "4h"
    }[timeframe]

    result = df.resample(rule).agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum"
    }).dropna()

    return add_indicators(result)

@st.cache_data(ttl=300)
def get_news():
    feeds = [
        "https://news.google.com/rss/search?q=gold+XAUUSD+OR+Federal+Reserve+OR+US+dollar&hl=en-US&gl=US&ceid=US:en",
        "https://news.google.com/rss/search?q=US+inflation+OR+NFP+OR+Federal+Reserve&hl=en-US&gl=US&ceid=US:en"
    ]

    items = []
    for url in feeds:
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:10]:
                items.append({
                    "title": entry.get("title", "Untitled"),
                    "link": entry.get("link", ""),
                    "published": entry.get("published", "Time unavailable")
                })
        except Exception:
            pass

    unique = {}
    for item in items:
        unique[item["title"]] = item
    return list(unique.values())[:15]

def news_sentiment(news):
    positive_words = [
        "gold rises", "gold gains", "weak dollar",
        "rate cut", "safe haven", "eases"
    ]
    negative_words = [
        "gold falls", "gold drops", "strong dollar",
        "rate hike", "higher yields", "inflation rises"
    ]

    score = 0
    for item in news:
        title = item["title"].lower()
        score += sum(word in title for word in positive_words)
        score -= sum(word in title for word in negative_words)

    if score > 0:
        return "POSITIVE", score
    if score < 0:
        return "NEGATIVE", score
    return "NEUTRAL", score

df, symbol = get_market_data()

if df.empty:
    st.error("Market data পাওয়া যাচ্ছে না। কিছুক্ষণ পর আবার চেষ্টা করুন।")
    st.info("Yahoo Finance data availability ও internet connection-এর ওপর এই dashboard নির্ভরশীল।")
    st.stop()

df = add_indicators(df)

latest = df.iloc[-1]
price = float(latest["Close"])

st.caption(f"Data source: Yahoo Finance | Symbol: {symbol}")
st.caption(f"Last data timestamp: {df.index[-1]}")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Latest Price", f"{price:.2f}")
c2.metric("RSI", f"{latest['RSI']:.2f}" if pd.notna(latest["RSI"]) else "N/A")
c3.metric("EMA 20", f"{latest['EMA20']:.2f}")
c4.metric("EMA 50", f"{latest['EMA50']:.2f}")

st.subheader("Market Chart")

fig = go.Figure()
fig.add_trace(go.Candlestick(
    x=df.index,
    open=df["Open"],
    high=df["High"],
    low=df["Low"],
    close=df["Close"],
    name="Gold"
))
fig.add_trace(go.Scatter(
    x=df.index, y=df["EMA20"],
    name="EMA 20"
))
fig.add_trace(go.Scatter(
    x=df.index, y=df["EMA50"],
    name="EMA 50"
))
fig.update_layout(
    height=520,
    xaxis_rangeslider_visible=False
)
st.plotly_chart(fig, use_container_width=True)

st.subheader("Multi-Timeframe Analysis")

timeframes = ["M5", "M15", "H1", "H4"]
cols = st.columns(4)

for col, tf in zip(cols, timeframes):
    tf_df = resample_data(df, tf)
    if not tf_df.empty:
        row = tf_df.iloc[-1]
        col.metric(
            tf + " Close",
            f"{row['Close']:.2f}"
        )
        col.caption(
            f"RSI: {row['RSI']:.1f}"
            if pd.notna(row["RSI"]) else "RSI: N/A"
        )

st.subheader("ICT-Inspired Market Structure")

recent = df.tail(30)
swing_high = recent["High"].iloc[:-1].max()
swing_low = recent["Low"].iloc[:-1].min()

if price > swing_high:
    structure = "Possible Bullish Break"
elif price < swing_low:
    structure = "Possible Bearish Break"
else:
    structure = "No Clear Break"

fvg_bull = False
fvg_bear = False

if len(df) >= 3:
    a = df.iloc[-3]
    c = df.iloc[-1]
    fvg_bull = c["Low"] > a["High"]
    fvg_bear = c["High"] < a["Low"]

st.write("Market Structure:", structure)
st.write("Possible Bullish FVG:", "Yes" if fvg_bull else "No")
st.write("Possible Bearish FVG:", "Yes" if fvg_bear else "No")
st.caption("ICT labels are simplified candle-based heuristics, not confirmed institutional order flow.")

st.subheader("Gold & USD News Analysis")

news = get_news()
sentiment, news_score = news_sentiment(news)

st.write("Headline Sentiment:", sentiment)
st.caption("Automated keyword sentiment is approximate, not a verified market-impact assessment.")

if news:
    for item in news:
        st.markdown(f"- [{item['title']}]({item['link']})")
        st.caption(item["published"])
else:
    st.warning("News feed পাওয়া যায়নি।")

st.subheader("AI Research Classification")

bull = 0
bear = 0

if latest["EMA20"] > latest["EMA50"]:
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

if structure == "Possible Bullish Break":
    bull += 1
elif structure == "Possible Bearish Break":
    bear += 1

if sentiment == "POSITIVE":
    bull += 1
elif sentiment == "NEGATIVE":
    bear += 1

if bull > bear:
    classification = "BULLISH"
elif bear > bull:
    classification = "BEARISH"
else:
    classification = "NEUTRAL"

st.metric("Research Classification", classification)
st.write(f"Bullish evidence points: {bull}")
st.write(f"Bearish evidence points: {bear}")

if classification == "BULLISH":
    st.success("BUY — Educational simulation label only")
elif classification == "BEARISH":
    st.error("SELL — Educational simulation label only")
else:
    st.warning("HOLD — Educational simulation label only")

st.caption(
    "This is a rule-based research classification, not a validated predictive AI model, "
    "not a trade recommendation, and not a probability of profit."
)

st.subheader("System Status")
st.write("Automatic refresh: Every 60 seconds")
st.write("Data mode: Latest available Yahoo Finance data")
st.write("Real order execution: Disabled")
st.write("Last dashboard update:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

st.caption("XAUUSD AI HYBRID RESEARCH | Educational simulation only")
