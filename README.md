# PRO AI Crypto Analyzer

## Features
- Multi-coin market scanner using Binance public OHLCV data
- Random Forest next-candle direction estimate (UP / DOWN / NEUTRAL)
- RSI, MACD, EMA indicators
- Candlestick chart and dashboard
- No API key and no trading permissions required

## Run locally
1. Install Python 3.10 or newer.
2. Open a terminal in this folder.
3. Run:
   ```bash
   pip install -r requirements.txt
   streamlit run app.py
   ```
4. Open the local URL printed by Streamlit.

## Deploy
Push `app.py` and `requirements.txt` to a GitHub repository, then create a Streamlit Community Cloud app using `app.py` as the main file.

## Notes
- Uses public Binance market data; availability depends on region and exchange access.
- The model estimates the direction of the next candle, not guaranteed future price movement.
- Holdout accuracy is a historical test metric and does not guarantee future performance.
- This project does not execute trades, connect wallets, or request API keys.
