# ticker_validator.py
import yfinance as yf

def is_valid_ticker(ticker: str) -> bool:
    t = yf.Ticker(ticker)
    try:
        info = t.fast_info
        return bool(info and info.last_price not in (None, 0))
    except Exception:
        return False
