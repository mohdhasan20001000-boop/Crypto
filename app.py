import time
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import ccxt
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="PRO AI Crypto Analyzer", page_icon="📊", layout="wide")

st.markdown("""
<style>
.block-container {padding-top: 1.3rem; padding-bottom: 2rem;}
[data-testid="stMetric"] {background: rgba(120,120,120,.08); padding: 12px; border-radius: 12px;}
</style>
""", unsafe_allow_html=True)

st.title("📊 PRO AI CRYPTO ANALYZER")
st.caption("Multi-coin market scanner • Technical indicators • Educational ML estimates")

DEFAULT_COINS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT", "BNB/USDT",
                 "DOGE/USDT", "ADA/USDT", "AVAX/USDT", "LINK/USDT", "TRX/USDT"]

@st.cache_resource
def get_exchange():
    # Public market data only; no API key and no trading permissions.
    return ccxt.binance({"enableRateLimit": True, "timeout": 20000})

def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def build_features(df):
    x = df.copy()
    x["return_1"] = x["close"].pct_change()
    x["return_3"] = x["close"].pct_change(3)
    x["return_6"] = x["close"].pct_change(6)
    x["ema_9"] = x["close"].ewm(span=9, adjust=False).mean()
    x["ema_21"] = x["close"].ewm(span=21, adjust=False).mean()
    x["ema_50"] = x["close"].ewm(span=50, adjust=False).mean()
    x["ema_gap_9"] = x["close"] / x["ema_9"] - 1
    x["ema_gap_21"] = x["close"] / x["ema_21"] - 1
    x["ema_gap_50"] = x["close"] / x["ema_50"] - 1
    x["rsi"] = rsi(x["close"])
    fast = x["close"].ewm(span=12, adjust=False).mean()
    slow = x["close"].ewm(span=26, adjust=False).mean()
    x["macd"] = fast - slow
    x["macd_signal"] = x["macd"].ewm(span=9, adjust=False).mean()
    x["macd_hist"] = x["macd"] - x["macd_signal"]
    x["volatility_10"] = x["return_1"].rolling(10).std()
    x["volume_change"] = x["volume"].pct_change()
    x["target"] = (x["close"].shift(-1) > x["close"]).astype(int)
    return x

FEATURES = ["return_1", "return_3", "return_6", "ema_gap_9", "ema_gap_21",
            "ema_gap_50", "rsi", "macd", "macd_signal", "macd_hist",
            "volatility_10", "volume_change"]

@st.cache_data(ttl=60, show_spinner=False)
def load_ohlcv(symbol, timeframe, limit, cache_key):
    exchange = get_exchange()
    rows = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    if not rows:
        raise ValueError("No candle data returned")
    df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df

def analyze(symbol, timeframe, limit, refresh_key):
    raw = load_ohlcv(symbol, timeframe, limit, refresh_key)
    data = build_features(raw).replace([np.inf, -np.inf], np.nan)
    usable = data.dropna(subset=FEATURES + ["target"]).copy()
    latest = data.dropna(subset=FEATURES).tail(1)
    if len(usable) < 80 or latest.empty:
        raise ValueError("Not enough clean candle history. Try a larger history or another timeframe.")

    # Chronological holdout: older rows train; newest 20% checks directional accuracy.
    split = int(len(usable) * 0.8)
    train = usable.iloc[:split]
    test = usable.iloc[split:]
    if train["target"].nunique() < 2:
        raise ValueError("Not enough examples of both directions to train the model.")

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("forest", RandomForestClassifier(
            n_estimators=250, max_depth=7, min_samples_leaf=4,
            class_weight="balanced", random_state=42, n_jobs=-1
        ))
    ])
    model.fit(train[FEATURES], train["target"])
    holdout_accuracy = accuracy_score(test["target"], model.predict(test[FEATURES])) if len(test) else np.nan

    # Refit on all labelled history before estimating the next candle.
    final_model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("forest", RandomForestClassifier(
            n_estimators=250, max_depth=7, min_samples_leaf=4,
            class_weight="balanced", random_state=42, n_jobs=-1
        ))
    ])
    final_model.fit(usable[FEATURES], usable["target"])
    probabilities = final_model.predict_proba(latest[FEATURES])[0]
    classes = list(final_model.named_steps["forest"].classes_)
    up_prob = float(probabilities[classes.index(1)]) if 1 in classes else 0.0
    down_prob = 1.0 - up_prob

    if up_prob >= 0.55:
        signal = "UP"
    elif up_prob <= 0.45:
        signal = "DOWN"
    else:
        signal = "NEUTRAL"

    last = data.iloc[-1]
    return {
        "symbol": symbol, "price": float(last["close"]), "signal": signal,
        "up": up_prob * 100, "down": down_prob * 100,
        "rsi": float(last["rsi"]), "macd": float(last["macd"]),
        "ema9": float(last["ema_9"]), "ema21": float(last["ema_21"]),
        "accuracy": float(holdout_accuracy) * 100 if not np.isnan(holdout_accuracy) else np.nan,
        "candles": raw, "updated": raw["timestamp"].iloc[-1]
    }

with st.sidebar:
    st.header("⚙️ Scanner Settings")
    coins_text = st.text_area("Trading pairs (comma separated)", ", ".join(DEFAULT_COINS), height=150)
    coins = list(dict.fromkeys([c.strip().upper() for c in coins_text.split(",") if c.strip()]))
    timeframe = st.selectbox("Timeframe", ["5m", "15m", "1h", "4h", "1d"], index=2)
    history = st.select_slider("Historical candles", options=[150, 200, 300, 500], value=300)
    refresh_seconds = st.selectbox("Auto refresh", [0, 30, 60, 120], index=2,
                                   format_func=lambda v: "Off" if v == 0 else f"{v} seconds")
    run_scan = st.button("🔄 Scan Market Now", use_container_width=True)
    st.caption("Uses public Binance market data. No API key is required.")

if "refresh_key" not in st.session_state:
    st.session_state.refresh_key = 0
if run_scan:
    st.session_state.refresh_key += 1

if not coins:
    st.warning("Enter at least one trading pair, for example BTC/USDT.")
    st.stop()

st.info("Signals estimate the direction of the next candle for the selected timeframe. They are not guaranteed forecasts.")
results = []
progress = st.progress(0)
status = st.empty()

for i, coin in enumerate(coins):
    status.write(f"Analyzing {coin} ({i+1}/{len(coins)})...")
    try:
        results.append(analyze(coin, timeframe, history, st.session_state.refresh_key))
    except Exception as e:
        st.warning(f"{coin}: {e}")
    progress.progress((i + 1) / len(coins))

status.empty()
progress.empty()

if not results:
    st.error("No coin could be analyzed. Check your internet connection, pair names, or Binance availability.")
    st.stop()

table = pd.DataFrame([{
    "Coin": r["symbol"], "Price (USDT)": r["price"], "AI Signal": r["signal"],
    "UP estimate (%)": round(r["up"], 1), "DOWN estimate (%)": round(r["down"], 1),
    "RSI": round(r["rsi"], 1), "Holdout accuracy (%)": round(r["accuracy"], 1)
} for r in results])

up_count = sum(r["signal"] == "UP" for r in results)
down_count = sum(r["signal"] == "DOWN" for r in results)
neutral_count = sum(r["signal"] == "NEUTRAL" for r in results)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Coins analyzed", len(results))
c2.metric("UP signals", up_count)
c3.metric("DOWN signals", down_count)
c4.metric("NEUTRAL", neutral_count)

st.subheader("🌐 Multi-Crypto Market Scanner")
st.dataframe(
    table.style.format({"Price (USDT)": "{:,.6f}", "UP estimate (%)": "{:.1f}",
                        "DOWN estimate (%)": "{:.1f}", "RSI": "{:.1f}",
                        "Holdout accuracy (%)": "{:.1f}"}),
    use_container_width=True, hide_index=True
)

selected = st.selectbox("Choose a coin for detailed analysis", [r["symbol"] for r in results])
item = next(r for r in results if r["symbol"] == selected)

st.subheader(f"🔎 {selected} Detailed Analysis")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Last price", f'{item["price"]:,.6f} USDT')
m2.metric("AI estimate", item["signal"])
m3.metric("UP probability estimate", f'{item["up"]:.1f}%')
m4.metric("DOWN probability estimate", f'{item["down"]:.1f}%')

left, right = st.columns([2, 1])
with left:
    candles = item["candles"]
    fig = go.Figure(data=[go.Candlestick(
        x=candles["timestamp"], open=candles["open"], high=candles["high"],
        low=candles["low"], close=candles["close"], name=selected
    )])
    fig.update_layout(title=f"{selected} Candlestick Chart ({timeframe})",
                      xaxis_title="Time (UTC)", yaxis_title="Price (USDT)",
                      height=470, xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.markdown("#### Technical indicators")
    st.metric("RSI (14)", f'{item["rsi"]:.2f}')
    st.metric("MACD", f'{item["macd"]:.6f}')
    st.metric("EMA 9", f'{item["ema9"]:.6f}')
    st.metric("EMA 21", f'{item["ema21"]:.6f}')
    st.metric("Historical holdout accuracy", f'{item["accuracy"]:.1f}%')

st.caption(f"Latest candle timestamp: {item['updated']} • Data source: Binance public OHLCV")
st.caption("Model: Random Forest classifier trained on historical technical features. Probability estimates are model outputs, not calibrated real-world odds. Backtest/holdout accuracy does not guarantee future performance.")
st.caption("Educational tool only. It does not place trades, connect wallets, or provide financial advice.")

if refresh_seconds > 0:
    st.caption(f"Auto-refresh is set to {refresh_seconds} seconds. Streamlit Cloud/browser sleep or rate limits may delay refresh.")
    time.sleep(refresh_seconds)
    st.rerun()
