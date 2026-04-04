# download_tickers.py
import pandas as pd
from pathlib import Path

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

def load_nasdaq_file(url: str) -> pd.DataFrame:
    df = pd.read_csv(url, sep="|")
    cols = {c.lower(): c for c in df.columns}

    # Prefer common names
    sym_col = None
    name_col = None

    # 1) Try exact matches
    for key, col in cols.items():
        if key == "symbol":
            sym_col = col
        if key == "security name":
            name_col = col

    # 2) Fallbacks – handle "ACT Symbol"
    if sym_col is None:
        for key, col in cols.items():
            if "act symbol" in key:
                sym_col = col
                break

    if name_col is None:
        for key, col in cols.items():
            if "security name" in key or "company name" in key:
                name_col = col
                break

    if sym_col is None:
        raise KeyError(f"No symbol column found in {url}. Columns: {list(df.columns)}")
    if name_col is None:
        name_col = sym_col  # fallback

    out = df[[sym_col, name_col]].rename(
        columns={sym_col: "Symbol", name_col: "Security Name"}
    )
    return out

def download_nasdaq_lists():
    urls = {
        "nasdaq": "ftp://ftp.nasdaqtrader.com/symboldirectory/nasdaqlisted.txt",
        "other":  "ftp://ftp.nasdaqtrader.com/symboldirectory/otherlisted.txt",
    }

    frames = []
    for name, url in urls.items():
        df = load_nasdaq_file(url)
        df["exchange"] = name
        frames.append(df)

    all_df = pd.concat(frames, ignore_index=True)
    all_df = all_df.dropna(subset=["Symbol"])
    all_df.to_csv(DATA_DIR / "ticker_db.csv", index=False)
    print(f"Saved {len(all_df)} tickers to data/ticker_db.csv")

if __name__ == "__main__":
    download_nasdaq_lists()
