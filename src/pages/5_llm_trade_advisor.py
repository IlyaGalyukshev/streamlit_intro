import hashlib
import hmac
import json
import os
import time
from decimal import ROUND_DOWN, Decimal, InvalidOperation
from typing import Dict, List, Tuple

import altair as alt
import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.title("🤖📈 LLM Trade Advisor — BTC/USDT (Spot)")

st.write(
    "Minimalistic interface: BTCUSDT Spot, chart, LLM hint, "
    "market BUY/SELL buttons and a table of all your coins.\n\n"
    "⚠️ *Only training — **NOT** investment advice.*"
)


SYMBOL = "BTCUSDT"
ACCOUNT_TYPE = os.getenv("BYBIT_ACCOUNT_TYPE", "UNIFIED")

BYBIT_BASE = os.getenv("BYBIT_BASE_URL", "https://api.bybit.com")
EP_KLINE = f"{BYBIT_BASE}/v5/market/kline"
EP_TICKERS = f"{BYBIT_BASE}/v5/market/tickers"
EP_INSTR = f"{BYBIT_BASE}/v5/market/instruments-info"
EP_ORDER = f"{BYBIT_BASE}/v5/order/create"
EP_WALLET_BAL = f"{BYBIT_BASE}/v5/account/wallet-balance"


def _http_get(url: str, params: dict, timeout: int = 10) -> dict:
    r = requests.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    return r.json()


def _http_post(url: str, data: str, headers: dict, timeout: int = 10) -> dict:
    r = requests.post(url, data=data, headers=headers, timeout=timeout)
    r.raise_for_status()
    return r.json()


def _sign_headers_post(
    api_key: str, api_secret: str, payload_str: str, recv_window: str = "5000"
) -> dict:
    ts = str(int(time.time() * 1000))
    prehash = ts + api_key + recv_window + payload_str
    sign = hmac.new(api_secret.encode(), prehash.encode(), hashlib.sha256).hexdigest()
    return {
        "X-BAPI-API-KEY": api_key,
        "X-BAPI-TIMESTAMP": ts,
        "X-BAPI-RECV-WINDOW": recv_window,
        "X-BAPI-SIGN": sign,
        "Content-Type": "application/json",
    }


def _signed_get(url: str, params: dict, recv_window: str = "5000") -> dict:
    """Signed GET request. Signs exactly the query string that will actually be sent in the URL."""
    api_key = os.getenv("BYBIT_API_KEY")
    api_secret = os.getenv("BYBIT_API_SECRET")
    if not api_key or not api_secret:
        return {"retCode": -1, "retMsg": "BYBIT_API_KEY/SECRET not set"}

    pr = requests.PreparedRequest()
    pr.prepare_url(url, params)
    query_str = pr.url.split("?", 1)[1] if "?" in pr.url else ""

    ts = str(int(time.time() * 1000))
    prehash = ts + api_key + recv_window + query_str
    sign = hmac.new(api_secret.encode(), prehash.encode(), hashlib.sha256).hexdigest()
    headers = {
        "X-BAPI-API-KEY": api_key,
        "X-BAPI-TIMESTAMP": ts,
        "X-BAPI-RECV-WINDOW": recv_window,
        "X-BAPI-SIGN": sign,
    }
    r = requests.get(pr.url, headers=headers, timeout=10)
    r.raise_for_status()
    return r.json()


def _parse_precision_value(s: str) -> Tuple[Decimal, int]:
    """
    Returns a tuple (step, decimals). Handles both Bybit formats:
      - '0.000001'  -> step=0.000001, decimals=6
      - '6'         -> step=10^-6,    decimals=6
    """
    if s is None or s == "":
        return Decimal("0.00000001"), 8
    s = str(s)
    try:
        if "." in s or s.startswith("0"):
            step = Decimal(s)
            decimals = max(0, -step.as_tuple().exponent)
        else:
            decimals = int(s)
            step = Decimal(1).scaleb(-decimals)
        return step, decimals
    except (InvalidOperation, ValueError):
        return Decimal("0.00000001"), 8


def _round_to_step_down(value: Decimal, step: Decimal) -> Decimal:
    if step <= 0:
        return value
    q = (value / step).to_integral_value(rounding=ROUND_DOWN)
    return q * step


def _fmt_to_step(value: Decimal, step: Decimal) -> str:
    decimals = max(0, -step.as_tuple().exponent)
    return f"{value:.{decimals}f}"


@st.cache_data(ttl=60 * 5, show_spinner=False)
def fetch_klines(symbol: str, interval: str, limit: int = 200) -> pd.DataFrame:
    """Bybit V5 Get Kline (Spot)."""
    params = {
        "category": "spot",
        "symbol": symbol,
        "interval": interval,
        "limit": limit,
    }
    payload = _http_get(EP_KLINE, params)
    if payload.get("retCode") != 0:
        raise RuntimeError(payload.get("retMsg", "Bybit kline error"))

    cols = ["timestamp", "open", "high", "low", "close", "volume", "turnover"]
    df = pd.DataFrame(payload["result"]["list"], columns=cols)
    df = df.astype({c: "float" for c in cols[1:]})
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df.sort_values("timestamp")
    return df


@st.cache_data(ttl=30, show_spinner=False)
def fetch_ticker_last(symbol: str) -> Decimal:
    """Bybit V5 Get Tickers (Spot) — lastPrice."""
    params = {"category": "spot", "symbol": symbol}
    payload = _http_get(EP_TICKERS, params)
    if payload.get("retCode") != 0 or "result" not in payload:
        raise RuntimeError(payload.get("retMsg", "Bybit tickers error"))
    lst = payload["result"].get("list", [])
    if not lst:
        raise RuntimeError("Empty tickers list")
    return Decimal(str(lst[0]["lastPrice"]))


@st.cache_data(ttl=60 * 60, show_spinner=False)
def get_spot_filters(symbol: str) -> dict:
    """
    Returns Spot symbol filters:
    {
      'base_step': Decimal,
      'base_decimals': int,
      'quote_step': Decimal,
      'quote_decimals': int,
      'min_order_qty': Decimal,
      'min_order_amt': Decimal
    }
    """
    params = {"category": "spot", "symbol": symbol}
    data = _http_get(EP_INSTR, params)
    if data.get("retCode") != 0 or not data.get("result", {}).get("list"):
        raise RuntimeError(data.get("retMsg", "instruments-info unavailable"))

    inst = data["result"]["list"][0]
    lot = inst.get("lotSizeFilter", {}) or {}

    base_step, base_decimals = _parse_precision_value(
        str(lot.get("basePrecision", "0.000001"))
    )
    quote_step, quote_decimals = _parse_precision_value(
        str(lot.get("quotePrecision", "0.01"))
    )

    min_order_qty = (
        Decimal(str(lot.get("minOrderQty", "0")))
        if lot.get("minOrderQty") not in (None, "")
        else Decimal("0")
    )
    min_order_amt = (
        Decimal(str(lot.get("minOrderAmt", "0")))
        if lot.get("minOrderAmt") not in (None, "")
        else Decimal("0")
    )

    return {
        "base_step": base_step,
        "base_decimals": base_decimals,
        "quote_step": quote_step,
        "quote_decimals": quote_decimals,
        "min_order_qty": min_order_qty,
        "min_order_amt": min_order_amt,
    }


def get_wallet_balances_all() -> List[Dict[str, Decimal]]:
    """
    Returns a list of coins with non-zero balances from /v5/account/wallet-balance
    for the specified ACCOUNT_TYPE (default: UNIFIED).
    """
    params = {"accountType": ACCOUNT_TYPE}
    data = _signed_get(EP_WALLET_BAL, params)
    if data.get("retCode") != 0:
        return []
    coins = []
    for acc in data.get("result", {}).get("list", []):
        for c in acc.get("coin", []):
            coin = c.get("coin")
            wallet = Decimal(str(c.get("walletBalance", "0") or "0"))
            equity = Decimal(str(c.get("equity", "0") or "0"))
            usd = Decimal(str(c.get("usdValue", "0") or "0"))
            coins.append(
                {
                    "coin": coin,
                    "walletBalance": wallet,
                    "equity": equity,
                    "usdValue": usd,
                }
            )
    coins.sort(key=lambda x: x["usdValue"], reverse=True)
    return coins


def ask_openai_for_trade(symbol: str, df: pd.DataFrame) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return "OPENAI_API_KEY not set. Export the environment variable to enable LLM advice."
    try:
        import openai

        openai.api_key = api_key

        recent = df.tail(100)[
            ["timestamp", "open", "high", "low", "close", "volume"]
        ].copy()
        recent["timestamp"] = recent["timestamp"].dt.strftime("%Y-%m-%d %H:%M")
        candles: List[dict] = recent.to_dict(orient="records")

        system_msg = "You are a concise crypto trading assistant. Respond with BUY, SELL, or HOLD and a brief justification (<15 words)."
        user_msg = (
            f"Given the following recent OHLCV candles for {symbol} (most recent last), what is the recommendation?\n"
            f"```json\n{candles}\n```"
        )
        try:
            resp = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.2,
            )
            return resp.choices[0].message.content.strip()
        except Exception:
            from openai import OpenAI

            client = OpenAI(api_key=api_key)
            resp = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.2,
            )
            return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"Error from OpenAI: {e}"


def place_spot_market_buy_usdt(
    usdt_amount: Decimal, quote_step: Decimal
) -> Tuple[bool, str, dict]:
    """Market BUY: qty is the amount in USDT (quote asset). Rounded down to the nearest multiple of quote_step."""
    api_key = os.getenv("BYBIT_API_KEY")
    api_secret = os.getenv("BYBIT_API_SECRET")
    if not api_key or not api_secret:
        return False, "BYBIT_API_KEY/SECRET not set.", {}

    qty = _round_to_step_down(usdt_amount, quote_step)
    qty_str = _fmt_to_step(qty, quote_step)

    payload = {
        "category": "spot",
        "symbol": SYMBOL,
        "side": "Buy",
        "orderType": "Market",
        "timeInForce": "IOC",
        "qty": qty_str,
    }
    payload_str = json.dumps(payload, separators=(",", ":"))
    headers = _sign_headers_post(api_key, api_secret, payload_str)
    try:
        data = _http_post(EP_ORDER, payload_str, headers=headers)
        if data.get("retCode") == 0:
            return (
                True,
                f"✅ Order accepted. Order ID: {data['result']['orderId']}",
                data,
            )
        else:
            return (
                False,
                f"❌ Bybit error {data.get('retCode')}: {data.get('retMsg')}",
                {"payload": payload, "bybit": data},
            )
    except Exception as e:
        return False, f"Request error: {e}", {"payload": payload}


def place_spot_market_sell_btc(
    btc_qty: Decimal, base_step: Decimal
) -> Tuple[bool, str, dict]:
    """Market SELL: qty is the amount in BTC (base asset). Rounded down to the nearest multiple of base_step."""
    api_key = os.getenv("BYBIT_API_KEY")
    api_secret = os.getenv("BYBIT_API_SECRET")
    if not api_key or not api_secret:
        return False, "BYBIT_API_KEY/SECRET not set.", {}

    qty = _round_to_step_down(btc_qty, base_step)
    qty_str = _fmt_to_step(qty, base_step)

    payload = {
        "category": "spot",
        "symbol": SYMBOL,
        "side": "Sell",
        "orderType": "Market",
        "timeInForce": "IOC",
        "qty": qty_str,
    }
    payload_str = json.dumps(payload, separators=(",", ":"))
    headers = _sign_headers_post(api_key, api_secret, payload_str)
    try:
        data = _http_post(EP_ORDER, payload_str, headers=headers)
        if data.get("retCode") == 0:
            return (
                True,
                f"✅ Order accepted. Order ID: {data['result']['orderId']}",
                data,
            )
        else:
            return (
                False,
                f"❌ Bybit error {data.get('retCode')}: {data.get('retMsg')}",
                {"payload": payload, "bybit": data},
            )
    except Exception as e:
        return False, f"Request error: {e}", {"payload": payload}


interval_labels = {
    "1m": "1",
    "5m": "5",
    "15m": "15",
    "1h": "60",
    "4h": "240",
    "1d": "D",
}

top1, top2 = st.columns([1, 1])
with top1:
    interval_label = st.selectbox("Interval", list(interval_labels.keys()), index=3)
with top2:
    limit = st.slider("Periods", 50, 1000, 200, step=50)

btn_fetch = st.button("Fetch / Refresh 📈 Klines", use_container_width=True)
btn_bal_price = st.button("🔄 Refresh balances & price", use_container_width=True)

if btn_fetch:
    with st.spinner("Fetching candles …"):
        df_prices = fetch_klines(SYMBOL, interval_labels[interval_label], limit)
    st.session_state["df_prices"] = df_prices
    st.session_state["suggestion"] = ask_openai_for_trade(SYMBOL, df_prices)
    try:
        st.session_state["spot_filters"] = get_spot_filters(SYMBOL)
    except Exception as e:
        st.session_state["spot_filters"] = {
            "base_step": Decimal("0.00000001"),
            "base_decimals": 8,
            "quote_step": Decimal("0.01"),
            "quote_decimals": 2,
            "min_order_qty": Decimal("0"),
            "min_order_amt": Decimal("5"),
        }
        st.warning(
            f"Symbol filters unavailable ({e}). Using safe steps: base=1e-8, quote=0.01, minAmt=5 USDT."
        )

if btn_bal_price:
    try:
        st.session_state["balances"] = get_wallet_balances_all()
    except Exception as e:
        st.error(f"Could not get balance: {e}")
    try:
        st.session_state["latest_price"] = fetch_ticker_last(SYMBOL)
    except Exception as e:
        st.error(f"Could not get price for {SYMBOL}: {e}")

if "latest_price" not in st.session_state:
    try:
        st.session_state["latest_price"] = fetch_ticker_last(SYMBOL)
    except Exception:
        st.session_state["latest_price"] = Decimal("0")
if "balances" not in st.session_state:
    st.session_state["balances"] = []


if "df_prices" in st.session_state:
    df_prices = st.session_state["df_prices"]
    suggestion = st.session_state.get("suggestion", "—")
    latest_price = Decimal(str(st.session_state.get("latest_price", Decimal("0"))))
    filters = st.session_state.get(
        "spot_filters",
        {
            "base_step": Decimal("0.00000001"),
            "base_decimals": 8,
            "quote_step": Decimal("0.01"),
            "quote_decimals": 2,
            "min_order_qty": Decimal("0"),
            "min_order_amt": Decimal("5"),
        },
    )

    base_step = filters["base_step"]
    quote_step = filters["quote_step"]
    min_order_qty = filters["min_order_qty"]
    min_order_amt = max(filters["min_order_amt"], Decimal("5"))

    st.subheader("Recent price chart (BTCUSDT)")
    chart = (
        alt.Chart(df_prices)
        .mark_line()
        .encode(x="timestamp:T", y="close:Q")
        .properties(height=400)
    )
    st.altair_chart(chart, use_container_width=True)

    st.metric("BTCUSDT — Last price (USDT)", f"{float(latest_price):,.2f}")

    st.subheader("LLM suggestion")
    lower = suggestion.lower()
    if lower.startswith("buy"):
        st.success(suggestion)
    elif lower.startswith("sell"):
        st.error(suggestion)
    else:
        st.info(suggestion)

    st.subheader("💰 Wallet — your coins (all, where balance > 0)")
    balances = st.session_state.get("balances", [])
    if balances:
        df_bal = pd.DataFrame(
            [
                {
                    "Coin": b["coin"],
                    "Wallet": float(b["walletBalance"]),
                    "Equity": float(b["equity"]),
                    "USD Value": float(b["usdValue"]),
                }
                for b in balances
            ]
        )
        total_usd = df_bal["USD Value"].sum()
        st.dataframe(df_bal, use_container_width=True, height=260)
        st.caption(f"Total in wallet (USD): {total_usd:,.2f}")
        wallet_map = {b["coin"]: b for b in balances}
        btc_bal = wallet_map.get("BTC", {"walletBalance": Decimal("0")})[
            "walletBalance"
        ]
        usdt_bal = wallet_map.get("USDT", {"walletBalance": Decimal("0")})[
            "walletBalance"
        ]
    else:
        st.info("Click «🔄 Refresh balances & price» to pull coins and price.")
        btc_bal = Decimal("0")
        usdt_bal = Decimal("0")

    st.subheader("🛒 Market BUY (for USDT) / 💼 Market SELL (BTC)")
    buy_col, sell_col = st.columns(2)

    with buy_col:
        default_buy = max(Decimal("10"), min_order_amt)
        buy_usdt = st.number_input(
            "BUY amount (USDT)",
            min_value=float(min_order_amt),
            value=float(default_buy),
            step=float(filters["quote_step"]),
            format="%.8f",
            key="buy_usdt_amt",
        )
        if st.button("✅ BUY BTC (Market)", use_container_width=True, key="btn_buy"):
            try:
                usdt_amt = Decimal(str(buy_usdt))
                # округляем к шагу quote
                usdt_amt = _round_to_step_down(usdt_amt, quote_step)
                if usdt_amt < min_order_amt:
                    st.error(
                        f"Nominal below minimum: {usdt_amt} < {min_order_amt} USDT."
                    )
                else:
                    ok, msg, resp = place_spot_market_buy_usdt(usdt_amt, quote_step)
                    st.write(msg)
                    if not ok:
                        st.json(resp)
            except Exception as e:
                st.error(f"BUY error: {e}")

    with sell_col:
        default_sell = (
            float(btc_bal) if isinstance(btc_bal, Decimal) and btc_bal > 0 else 0.001
        )
        sell_btc = st.number_input(
            "SELL quantity (BTC)",
            min_value=float(filters["base_step"]),
            value=float(default_sell),
            step=float(filters["base_step"]),
            format="%.8f",
            key="sell_btc_qty",
        )
        if st.button("❌ SELL BTC (Market)", use_container_width=True, key="btn_sell"):
            try:
                btc_qty = Decimal(str(sell_btc))
                btc_qty = _round_to_step_down(btc_qty, base_step)
                if btc_qty <= 0:
                    st.error("BTC quantity must be > 0.")
                elif min_order_qty > 0 and btc_qty < min_order_qty:
                    st.error(
                        f"Quantity below minimum for symbol: {btc_qty} < {min_order_qty} BTC."
                    )
                else:
                    notional = btc_qty * (
                        Decimal("0") if latest_price == 0 else latest_price
                    )
                    if notional < min_order_amt:
                        st.error(
                            f"Nominal below minimum: {float(notional):.6f} < {float(min_order_amt):.6f} USDT."
                        )
                    else:
                        ok, msg, resp = place_spot_market_sell_btc(btc_qty, base_step)
                        st.write(msg)
                        if not ok:
                            st.json(resp)
            except Exception as e:
                st.error(f"SELL error: {e}")

    with st.expander("Raw OHLCV data"):
        st.dataframe(df_prices.tail(200), use_container_width=True)

st.caption(
    "Data: Bybit V5 — 5m cache (klines) · Wallet non-zero assets · Powered by OpenAI. Not financial advice."
)
