# ticker_db.py
import pandas as pd
from pathlib import Path
from functools import lru_cache

DATA_PATH = Path("data") / "ticker_db.csv"

@lru_cache(maxsize=1)
def load_ticker_db():
    if not DATA_PATH.exists():
        return {}
    df = pd.read_csv(DATA_PATH)
    db = {}
    for _, row in df.iterrows():
        sym = str(row["Symbol"]).strip().upper()
        name = str(row.get("Security Name", "")).strip()
        if sym:
            db[sym] = name
    return db

def validate_ticker(ticker: str):
    db = load_ticker_db()
    t = str(ticker).strip().upper()
    name = db.get(t)
    return {
        "valid": bool(name),
        "ticker": t,
        "company_name": name or "",
    }
