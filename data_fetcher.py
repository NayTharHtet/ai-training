import yfinance as yf
import pandas as pd

def fetch_market_data(ticker: str) -> dict:
    try:
        tk = yf.Ticker(ticker)
        info = tk.fast_info
        current_price = round(float(info.last_price), 2)
        prev_close = round(float(info.previous_close), 2)
        day_change_pct = round((current_price - prev_close) / prev_close * 100, 2)

        df = yf.download(ticker, period="1mo", interval="1d",
                         progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.dropna()

        sma_5 = round(float(df["Close"].tail(5).mean()), 2) if len(df) >= 5 else current_price
        sma_20 = round(float(df["Close"].tail(20).mean()), 2) if len(df) >= 20 else current_price
        volatility = round(float(df["Close"].pct_change().std() * 100), 2) if len(df) > 1 else 0.0
        five_day_return = round(
            (float(df["Close"].iloc[-1]) - float(df["Close"].iloc[-6])) /
            float(df["Close"].iloc[-6]) * 100, 2
        ) if len(df) >= 6 else 0.0

        return {
            "ticker": ticker,
            "current_price": current_price,
            "prev_close": prev_close,
            "day_change_pct": day_change_pct,
            "sma_5": sma_5,
            "sma_20": sma_20,
            "volatility": volatility,
            "five_day_return": five_day_return,
        }
    except Exception as e:
        return {"error": str(e)}
