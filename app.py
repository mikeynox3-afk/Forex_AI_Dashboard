
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="AI Forex Dashboard",
    page_icon="📈",
    layout="wide"
)

st.title("🤖 AI Forex Analysis Dashboard")
st.caption("Educational Project | Simulated Data")

model = joblib.load("forex_ai_model.pkl")
df = pd.read_csv("forex_sample_data.csv")

st.subheader("EUR/USD Sample Price Chart")

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(df["Close"])
ax.set_xlabel("Sample Day")
ax.set_ylabel("Close Price")
ax.grid(True)

st.pyplot(fig)

st.subheader("Market Data")
st.dataframe(df.tail(10))

st.subheader("Model Analysis")

latest = df[["Open", "High", "Low", "Close"]].iloc[[-1]]
prediction = model.predict(latest)[0]

if prediction == 1:
    st.info("Model's sample estimate: UP")
else:
    st.info("Model's sample estimate: DOWN")

st.warning(
    "This is a simulated educational model, "
    "not a real Forex trading signal."
)
