import altair as alt
import pandas as pd
import requests
import streamlit as st

st.title("🚀 Live Crypto Dashboard")
st.write(
    "An example of advanced Streamlit features: live API calls, caching, tabs, and session state."
)

COINS = {
    "Bitcoin (BTC)": "bitcoin",
    "Ethereum (ETH)": "ethereum",
    "Dogecoin (DOGE)": "dogecoin",
}


@st.cache_data(ttl=300, show_spinner=False)
def get_market_chart(coin_id: str, days: int = 1):
    """Fetch historical market data from the CoinGecko API."""
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
    params = {"vs_currency": "usd", "days": days}
    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()
    data = r.json()
    prices = pd.DataFrame(data["prices"], columns=["timestamp", "price"])
    prices["timestamp"] = pd.to_datetime(prices["timestamp"], unit="ms")
    return prices


with st.sidebar:
    st.header("⚙️ Settings")
    coin_name = st.selectbox("Cryptocurrency", list(COINS.keys()))
    days = st.select_slider(
        "Days of history", options=[1, 7, 14, 30, 90, 180, 365], value=7
    )
    refresh = st.button("🔄 Refresh data")

coin_id = COINS[coin_name]

if "prices" not in st.session_state or refresh:
    st.session_state["prices"] = get_market_chart(coin_id, days)

prices_df = st.session_state["prices"]
latest_price = prices_df["price"].iloc[-1]

st.metric(label=f"{coin_name} price (USD)", value=f"${latest_price:,.2f}")

tab1, tab2 = st.tabs(["📈 Chart", "🧮 Raw data"])

with tab1:
    chart = (
        alt.Chart(prices_df)
        .mark_line()
        .encode(
            x="timestamp:T",
            y="price:Q",
        )
        .properties(width="container")
    )
    st.altair_chart(chart, use_container_width=True)

with tab2:
    st.dataframe(prices_df.tail(200), use_container_width=True)

st.caption("Data source: CoinGecko API — Cached for 5 minutes.")
