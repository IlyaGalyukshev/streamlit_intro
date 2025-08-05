import os, sys, time, json, math, random, csv, datetime
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional

import pandas as pd
import numpy as np
import requests
import yfinance as yf

START_CAPITAL_USD = 100.00
MAX_POSITIONS = 6
REBALANCE_WEEKDAY = 4
UNIVERSE_SAMPLE = 400
MICROCAP_MAX_USD = 3e8
MIN_DOLLAR_VOL = 1e5
HISTORY_DAYS = 90

DATA_DIR = "ai_microcap_data"
os.makedirs(DATA_DIR, exist_ok=True)
PORTF_PATH = os.path.join(DATA_DIR, "portfolio.csv")
TRADES_PATH = os.path.join(DATA_DIR, "trades.csv")
ORDERS_PATH = os.path.join(DATA_DIR, "orders.csv")
PERF_PATH = os.path.join(DATA_DIR, "performance.csv")

MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


def _build_session() -> requests.Session:
    s = requests.Session()
    ua = os.getenv(
        "SEC_USER_AGENT", "AI-MicroCap-Experiment/1.0 (+contact@example.com)"
    )
    s.headers.update(
        {
            "User-Agent": ua,
            "Accept": "*/*",
            "Accept-Encoding": "gzip, deflate",
            "Connection": "keep-alive",
        }
    )
    s.timeout = 20
    return s


SESSION = _build_session()


_NASDAQ_URLS = [
    "https://www.nasdaqtrader.com/dynamic/symdir/nasdaqlisted.txt",
    "https://www.nasdaqtrader.com/dynamic/symdir/otherlisted.txt",
    "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqtraded.txt",
]

_SEC_EXCHANGE_JSON = "https://www.sec.gov/files/company_tickers_exchange.json"


def _fetch_text(url: str) -> Optional[str]:
    try:
        r = SESSION.get(url, timeout=20)
        r.raise_for_status()
        return r.text
    except Exception:
        return None


def _parse_nasdaq_table(text: str) -> List[str]:
    """
    Universal parser for nasdaqlisted.txt / otherlisted.txt / nasdaqtraded.txt files.
    These files are '|' delimited and contain header and trailing metadata lines.
    Returns a clean list of tickers without special symbols.
    """
    tickers = []
    for line in text.splitlines():
        if (
            "File Creation Time" in line
            or "Symbol|" in line
            or line.strip() == ""
            or line.startswith("ACT Symbol|")
        ):
            continue
        parts = line.split("|")
        sym = parts[0].strip()
        if len(sym) == 0 or any(c in sym for c in ("^", "/", ".", "=")):
            continue
        tickers.append(sym)
    return tickers


def _download_from_nasdaqtrader() -> List[str]:
    all_syms: List[str] = []
    for url in _NASDQ_URLS_SAFE():
        txt = _fetch_text(url)
        if not txt:
            continue
        syms = _parse_nasdaq_table(txt)
        all_syms.extend(syms)
    return sorted(set(all_syms))


def _NASDQ_URLS_SAFE() -> List[str]:
    return list(_NASDAQ_URLS)


def _download_from_sec_exchange() -> List[str]:
    """
    Official SEC JSON dataset: company_tickers_exchange.json
    Contains records like [{'ticker': 'AAPL', 'exchange': 'Nasdaq', ...}, ...]
    """
    try:
        r = SESSION.get(_SEC_EXCHANGE_JSON, timeout=30)
        r.raise_for_status()
        data = r.json()
        tickers = [
            str(obj["ticker"]).upper().strip() for obj in data if obj.get("ticker")
        ]
        tickers = [
            t for t in tickers if t and all(c not in t for c in ("^", "/", "=", "."))
        ]
        return sorted(set(tickers))
    except Exception:
        return []


def download_symbol_files() -> pd.DataFrame:
    """
    Robust ticker collector:
      1) Try NasdaqTrader (HTTP symdir versions, not FTP);
      2) If empty — fall back to SEC JSON dataset.
    """
    syms = _download_from_nasdaqtrader()
    if not syms:
        syms = _download_from_sec_exchange()
    if not syms:
        raise RuntimeError("Не удалось скачать списки тикеров (NasdaqTrader/SEC).")
    df = pd.DataFrame({"ticker": syms})
    return df


def today_ymd() -> str:
    return datetime.date.today().isoformat()


def load_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path)


def save_df(df: pd.DataFrame, path: str) -> None:
    df.to_csv(path, index=False)


def append_trade(
    date: str, action: str, ticker: str, shares: int, price: float, note: str = ""
) -> None:
    hdr = ["date", "action", "ticker", "shares", "price", "note"]
    exists = os.path.exists(TRADES_PATH)
    with open(TRADES_PATH, "a", newline="") as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(hdr)
        w.writerow([date, action, ticker, shares, price, note])


def yf_fast_info(ticker: str) -> Tuple[Optional[float], Optional[int]]:
    try:
        t = yf.Ticker(ticker)
        fi = t.fast_info
        price = float(fi["last_price"])
        shares = fi.get("shares", None)
        if shares is None:
            info = t.get_info()
            shares = info.get("sharesOutstanding", None)
        return price, (int(shares) if shares is not None else None)
    except Exception:
        return None, None


def compute_metrics(tickers: List[str]) -> pd.DataFrame:
    data = []
    for tk in tickers:
        try:
            hist = yf.download(
                tk,
                period=f"{HISTORY_DAYS}d",
                interval="1d",
                progress=False,
                auto_adjust=True,
            )
            if hist is None or hist.empty or "Close" not in hist:
                continue
            close = hist["Close"].dropna()
            vol = hist.get("Volume", pd.Series(index=close.index, dtype=float)).fillna(
                0
            )
            if len(close) < 60:
                continue
            r20 = (
                (close.iloc[-1] / close.iloc[-21] - 1.0) if len(close) >= 21 else np.nan
            )
            r60 = (
                (close.iloc[-1] / close.iloc[-61] - 1.0) if len(close) >= 61 else np.nan
            )
            avg_dollar_vol = float((close * vol).tail(20).mean())
            vol_daily = float(np.log(close / close.shift(1)).std() * np.sqrt(252))
            last = float(close.iloc[-1])
            price, shares = yf_fast_info(tk)
            mcap = None
            if price is not None and shares is not None:
                mcap = float(price * shares)
            data.append(
                {
                    "ticker": tk,
                    "price": last,
                    "r20": r20,
                    "r60": r60,
                    "avg_dollar_vol": avg_dollar_vol,
                    "vol_annual": vol_daily,
                    "mcap": mcap,
                }
            )
        except Exception:
            continue
    df = pd.DataFrame(data)
    if df.empty:
        return df
    df = df[df["mcap"].notna()]
    df = df[df["mcap"] <= MICROCAP_MAX_USD]
    df = df[df["avg_dollar_vol"] >= MIN_DOLLAR_VOL]
    df = df.dropna(subset=["r20", "r60"]).reset_index(drop=True)
    return df


def build_candidate_universe() -> pd.DataFrame:
    universe = download_symbol_files()
    rng = np.random.default_rng(seed=int(time.time()) % 2**32)
    sample = universe.sample(
        min(UNIVERSE_SAMPLE, len(universe)), random_state=int(rng.integers(0, 10**9))
    )
    df = compute_metrics(sample["ticker"].tolist())
    if df.empty:
        return df
    df["score"] = 0.5 * df["r20"] + 0.5 * df["r60"]
    df = (
        df.sort_values(["score", "avg_dollar_vol"], ascending=[False, False])
        .head(120)
        .reset_index(drop=True)
    )
    return df


def ask_openai_portfolio(
    candidates_df: pd.DataFrame, current_holdings: Dict[str, Dict]
) -> Dict:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY не установлен в окружении.")

    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        use_new = True
    except Exception:
        import openai

        openai.api_key = api_key
        client = None
        use_new = False

    cd = candidates_df[
        ["ticker", "price", "mcap", "avg_dollar_vol", "r20", "r60", "vol_annual"]
    ].copy()
    cd = cd.round(
        {
            "price": 4,
            "mcap": 0,
            "avg_dollar_vol": 0,
            "r20": 4,
            "r60": 4,
            "vol_annual": 4,
        }
    )
    candidates = cd.to_dict(orient="records")

    sys_msg = (
        "You are a disciplined micro-cap portfolio strategist.\n"
        "Goal: Build a 3–6 stock portfolio of U.S.-listed micro-caps (market cap ≤ $300M),"
        " using whole-share positions only, for a $100 account. You must output strict JSON.\n"
        "Rules: Total weight must equal 100%. For each pick provide a stop_loss price (absolute USD).\n"
        "Avoid illiquid names (prefer higher avg dollar volume). Use momentum (r20,r60)."
        'JSON schema: {"portfolio":[{"ticker":"...","weight_pct":float,"stop_loss":float}],'
        ' "notes":"short rationale"}'
    )
    user_msg = (
        "Candidates:\n"
        f"{json.dumps(candidates)[:12000]}\n\n"
        f"Current holdings: {json.dumps(current_holdings)}\n"
        "Pick 3–6 tickers. Sum of weight_pct = 100. stop_loss is absolute price below current.\n"
        "Return ONLY the JSON object, no commentary."
    )

    if use_new:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.2,
        )
        out = resp.choices[0].message.content
    else:
        resp = openai.ChatCompletion.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.2,
        )
        out = resp.choices[0].message["content"]

    s = out.find("{")
    t = out.rfind("}")
    if s != -1 and t != -1 and t > s:
        return json.loads(out[s : t + 1])
    return json.loads(out)


def read_portfolio() -> pd.DataFrame:
    if not os.path.exists(PORTF_PATH):
        return pd.DataFrame(
            columns=["date", "ticker", "shares", "entry_price", "stop_loss"]
        )
    return pd.read_csv(PORTF_PATH)


def mark_to_market(portf: pd.DataFrame) -> Tuple[pd.DataFrame, float]:
    if portf.empty:
        return portf, 0.0
    prices = {}
    for tk in portf["ticker"]:
        try:
            p = (
                yf.download(
                    tk, period="5d", interval="1d", progress=False, auto_adjust=True
                )["Close"]
                .dropna()
                .iloc[-1]
            )
            prices[tk] = float(p)
        except Exception:
            prices[tk] = float(portf.loc[portf["ticker"] == tk, "entry_price"].iloc[0])
    portf["last_price"] = portf["ticker"].map(prices)
    portf["position_value"] = portf["shares"] * portf["last_price"]
    equity = float(portf["position_value"].sum())
    return portf, equity


def build_orders_from_target(
    target: Dict, capital: float, prices: Dict[str, float]
) -> List[Dict]:
    picks = target.get("portfolio", [])
    if not picks:
        return []
    wsum = sum([float(p["weight_pct"]) for p in picks])
    if wsum <= 0:
        return []
    for p in picks:
        p["weight_pct"] = 100.0 * float(p["weight_pct"]) / wsum

    orders = []
    for p in picks:
        tk = p["ticker"].upper()
        w = float(p["weight_pct"]) / 100.0
        stop = float(p["stop_loss"])
        px = prices.get(tk, None)
        if px is None or px <= 0:
            try:
                px = float(
                    yf.download(
                        tk, period="5d", interval="1d", progress=False, auto_adjust=True
                    )["Close"]
                    .dropna()
                    .iloc[-1]
                )
            except Exception:
                continue
        shares = int(math.floor((capital * w) / px))
        if shares <= 0:
            continue
        orders.append(
            {
                "action": "BUY",
                "ticker": tk,
                "shares": shares,
                "price": px,
                "stop_loss": stop,
            }
        )
    return orders


def apply_orders(portf: pd.DataFrame, orders: List[Dict]) -> pd.DataFrame:
    if not orders:
        return portf
    today = today_ymd()
    for od in orders:
        tk, sh, px, sl = (
            od["ticker"],
            int(od["shares"]),
            float(od["price"]),
            float(od["stop_loss"]),
        )
        row = {
            "date": today,
            "ticker": tk,
            "shares": sh,
            "entry_price": px,
            "stop_loss": sl,
        }
        portf = pd.concat([portf, pd.DataFrame([row])], ignore_index=True)
        append_trade(today, "BUY", tk, sh, px, f"SL={sl}")
    return portf


def stoploss_check_and_sell(portf: pd.DataFrame) -> Tuple[pd.DataFrame, List[Dict]]:
    if portf.empty:
        return portf, []
    today = today_ymd()
    sell_orders = []
    for _, row in portf.iterrows():
        tk = row["ticker"]
        sl = float(row["stop_loss"])
        try:
            p = float(
                yf.download(
                    tk, period="5d", interval="1d", progress=False, auto_adjust=True
                )["Close"]
                .dropna()
                .iloc[-1]
            )
        except Exception:
            p = float(row["entry_price"])
        if p <= sl:
            sell_orders.append(
                {
                    "action": "SELL",
                    "ticker": tk,
                    "shares": int(row["shares"]),
                    "price": p,
                }
            )
    if sell_orders:
        keep = []
        for _, row in portf.iterrows():
            if any(od["ticker"] == row["ticker"] for od in sell_orders):
                px = [
                    od["price"] for od in sell_orders if od["ticker"] == row["ticker"]
                ][0]
                append_trade(
                    today, "SELL", row["ticker"], int(row["shares"]), float(px), "STOP"
                )
            else:
                keep.append(row)
        portf = pd.DataFrame(keep, columns=portf.columns)
    return portf, sell_orders


def write_orders_csv(orders: List[Dict]) -> None:
    hdr = ["date", "action", "ticker", "shares", "price", "extra"]
    with open(ORDERS_PATH, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(hdr)
        for od in orders:
            extra = (
                f"stop_loss={od.get('stop_loss','')}" if od["action"] == "BUY" else ""
            )
            w.writerow(
                [
                    today_ymd(),
                    od["action"],
                    od["ticker"],
                    od["shares"],
                    f"{od['price']:.4f}",
                    extra,
                ]
            )


def log_performance(portf: pd.DataFrame, cash: float) -> None:
    _, equity = mark_to_market(portf.copy())
    total = equity + cash
    row = {"date": today_ymd(), "equity": equity, "cash": cash, "total": total}
    df = load_csv(PERF_PATH)
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    save_df(df, PERF_PATH)


def main():
    print("=== AI Micro-Cap Experiment — robust universe fetch ===")

    portf = read_portfolio()
    cash = START_CAPITAL_USD
    if not portf.empty:
        mp, equity = mark_to_market(portf.copy())
        cash = max(
            0.0, START_CAPITAL_USD - float((mp["entry_price"] * mp["shares"]).sum())
        )

    portf, sells = stoploss_check_and_sell(portf)
    if sells:
        write_orders_csv(sells)
        print(f"[{today_ymd()}] STOP SELL orders:", sells)

    dow = datetime.date.today().weekday()
    do_rebalance = (dow == REBALANCE_WEEKDAY) or portf.empty

    if do_rebalance:
        print("Building candidate universe…")
        cands = build_candidate_universe()
        if cands.empty:
            print(
                "No candidates built — try again later. (Check network or increase UNIVERSE_SAMPLE)."
            )
            log_performance(portf, cash)
            save_df(portf, PORTF_PATH)
            return

        prices = {r["ticker"]: float(r["price"]) for _, r in cands.iterrows()}
        holdings = {}
        for _, r in portf.iterrows():
            holdings[r["ticker"]] = {
                "shares": int(r["shares"]),
                "entry_price": float(r["entry_price"]),
                "stop_loss": float(r["stop_loss"]),
            }

        print("Asking ChatGPT for target portfolio…")
        target = ask_openai_portfolio(cands, holdings)

        orders = build_orders_from_target(target, START_CAPITAL_USD, prices)
        if orders:
            write_orders_csv(orders)
            print("Proposed BUY orders written to", ORDERS_PATH)
            portf = apply_orders(portf, orders)
        else:
            print("No BUY orders proposed.")

    log_performance(portf, cash)
    save_df(portf, PORTF_PATH)
    print("Done. Check:", ORDERS_PATH, PORTF_PATH, PERF_PATH, TRADES_PATH)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted.")
