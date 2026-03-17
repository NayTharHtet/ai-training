from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd
import yfinance as yf
from groq import Groq

DEFAULT_MODEL_PATH = "model_multi.pkl"
DEFAULT_OUTPUT_PATH = "output.json"
VALID_TICKERS  = ["AAPL", "NVDA", "TSLA"]
VALID_HORIZONS = [1, 7, 30]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="XGBoost predictor with LLM notes")
    parser.add_argument("--tickers",    type=str, default="ALL")
    parser.add_argument("--horizon",    type=int, default=7, choices=VALID_HORIZONS)
    parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output",     type=str, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--period",     type=str, default="6mo")
    return parser.parse_args()


def normalize_requested_tickers(tickers_raw: str) -> List[str]:
    raw = tickers_raw.strip().upper()
    if raw == "ALL":
        return VALID_TICKERS.copy()
    tickers = [t.strip().upper() for t in raw.split(",") if t.strip()]
    invalid = [t for t in tickers if t not in VALID_TICKERS]
    if invalid:
        raise ValueError(f"Invalid tickers: {invalid}. Allowed: {VALID_TICKERS}")
    return tickers


def load_artifact(model_path: str) -> Dict[str, Any]:
    path = Path(model_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    artifact = joblib.load(path)
    required_keys = {"models", "feature_columns", "base_features", "targets"}
    missing = required_keys - set(artifact.keys())
    if missing:
        raise ValueError(f"Model artifact missing keys: {sorted(missing)}")
    return artifact


def fetch_live_data(ticker: str, period: str) -> pd.DataFrame:
    df = yf.download(
        ticker, period=period, interval="1d",
        auto_adjust=False, progress=False, threads=False,
    )
    if df is None or df.empty:
        raise ValueError(f"No Yahoo Finance data returned for {ticker}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
    df = df.reset_index()
    expected_cols = ["Date", "Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns for {ticker}: {missing}")
    df["Ticker"] = ticker
    return df


def build_live_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], utc=True, errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["ret_1"]    = df["Close"].pct_change(1)
    df["ret_5"]    = df["Close"].pct_change(5)
    df["ret_20"]   = df["Close"].pct_change(20)
    df["vol_5"]    = df["ret_1"].rolling(5).std()
    df["vol_20"]   = df["ret_1"].rolling(20).std()
    df["ma_5"]     = df["Close"].rolling(5).mean()
    df["ma_20"]    = df["Close"].rolling(20).mean()
    df["ma_ratio"] = df["ma_5"] / df["ma_20"]
    df = df.replace([np.inf, -np.inf], np.nan)
    return df


def build_model_input_row(
    live_df: pd.DataFrame,
    ticker: str,
    feature_columns: List[str],
    base_features: List[str],
) -> tuple[pd.DataFrame, pd.Series]:
    working = live_df.copy()
    missing_base = [c for c in base_features if c not in working.columns]
    if missing_base:
        raise ValueError(f"Missing base feature columns for {ticker}: {missing_base}")
    working = working.dropna(subset=base_features).reset_index(drop=True)
    if working.empty:
        raise ValueError(f"Not enough recent data to compute features for {ticker}")
    latest_row = working.iloc[[-1]].copy()
    as_of_row  = latest_row.iloc[0]
    x = latest_row[base_features + ["Ticker"]].copy()
    x = pd.get_dummies(x, columns=["Ticker"], prefix="Ticker")
    for col in feature_columns:
        if col not in x.columns:
            x[col] = 0
    extra_cols = [c for c in x.columns if c not in feature_columns]
    if extra_cols:
        x = x.drop(columns=extra_cols)
    x = x[feature_columns]
    return x, as_of_row


def safe_float(val) -> float:
    try:
        if hasattr(val, "item"):  return float(val.item())
        if hasattr(val, "iloc"):  return float(val.iloc[0])
        return float(val)
    except Exception:
        return 0.0


# ── LLM NOTE via Groq (free) ─────────────────────────────────────
def generate_llm_note(
    client: Groq | None,
    *,
    ticker: str,
    horizon: int,
    direction: str,
    prob_up: float,
    prob_down: float,
    as_of_date: str,
) -> str:
    if client is None:
        return "LLM note unavailable: GROQ_API_KEY not set."

    prompt = (
        f"You are helping explain a stock prediction to a university project user.\n"
        f"Ticker: {ticker} | Horizon: {horizon} day(s) | Model: XGBoost\n"
        f"Predicted direction: {direction}\n"
        f"Probability UP: {prob_up:.2%} | Probability DOWN: {prob_down:.2%}\n"
        f"Data as of: {as_of_date}\n\n"
        f"Write 2 short sentences only. Keep it neutral. "
        f"Do not claim certainty. Mention this is model-based and not financial advice."
    )

    try:
        stream = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You write short, neutral financial notes for a dashboard."},
                {"role": "user",   "content": prompt},
            ],
            temperature=0.3,
            max_completion_tokens=120,
            stream=True,
        )
        full_note = ""
        print(f"[Groq] Note for {ticker}: ", end="", flush=True)
        for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            full_note += delta
            print(delta, end="", flush=True)
        print()
        return full_note.strip()
    except Exception as e:
        return f"Groq note unavailable: {e}"
# ─────────────────────────────────────────────────────────────────


def append_run(output_path: Path, run_entry: Dict[str, Any]) -> None:
    if output_path.exists():
        try:
            data = json.loads(output_path.read_text(encoding="utf-8"))
        except Exception:
            data = {"runs": []}
    else:
        data = {"runs": []}
    if "runs" not in data or not isinstance(data["runs"], list):
        data = {"runs": []}
    data["runs"].append(run_entry)
    output_path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def run_pipeline(
    tickers_raw: str,
    horizon: int,
    model_path: str,
    output_path: str,
    period: str,
) -> int:
    artifact       = load_artifact(model_path)
    models_by_name = artifact["models"]
    feature_columns: List[str] = artifact["feature_columns"]
    base_features:   List[str] = artifact["base_features"]

    # Always XGBoost — only real ML model
    if "xgboost" not in models_by_name:
        raise ValueError("No trained XGBoost model found in artifact.")
    xgb_models = models_by_name["xgboost"]
    if horizon not in xgb_models:
        raise ValueError(f"No XGBoost model found for horizon {horizon}. Available: {list(xgb_models.keys())}")
    selected_model = xgb_models[horizon]

    requested_tickers = normalize_requested_tickers(tickers_raw)

    groq_client = None
    groq_api_key = os.getenv("GROQ_API_KEY")
    if groq_api_key:
        groq_client = Groq(api_key=groq_api_key)

    results: List[Dict[str, Any]] = []
    errors:  List[Dict[str, str]] = []

    for ticker in requested_tickers:
        try:
            raw_df  = fetch_live_data(ticker, period)
            feat_df = build_live_features(raw_df)
            x_latest, latest_row = build_model_input_row(
                feat_df,
                ticker=ticker,
                feature_columns=feature_columns,
                base_features=base_features,
            )

            # XGBoost prediction
            pred   = int(selected_model.predict(x_latest)[0])
            prob_up = 0.5
            if hasattr(selected_model, "predict_proba"):
                proba = selected_model.predict_proba(x_latest)[0]
                if len(proba) >= 2:
                    prob_up = float(proba[1])

            direction  = "UP" if pred == 1 else "DOWN"
            prob_down  = round(1.0 - prob_up, 4)
            prob_up    = round(prob_up, 4)
            as_of_date = pd.to_datetime(latest_row["Date"]).strftime("%Y-%m-%d")

            # Groq generates the explanatory note
            llm_note = generate_llm_note(
                groq_client,
                ticker=ticker,
                horizon=horizon,
                direction=direction,
                prob_up=prob_up,
                prob_down=prob_down,
                as_of_date=as_of_date,
            )

            results.append({
                "ticker":        ticker,
                "horizon_days":  horizon,
                "model_name":    "xgboost",
                "as_of_date":    as_of_date,
                "direction":     direction,
                "prob_up":       prob_up,
                "prob_down":     prob_down,
                "expected_return": None,
                "note":          llm_note,
                "llm_note":      llm_note,
                "source":        "Yahoo Finance live data",
            })

        except Exception as e:
            errors.append({"ticker": ticker, "error": str(e)})

    run_entry = {
        "run_at": utc_now_iso(),
        "params": {
            "tickers":      requested_tickers,
            "horizon_days": horizon,
            "model_name":   "xgboost",
            "model_path":   str(Path(model_path).expanduser().resolve()),
            "period":       period,
        },
        "results": results,
        "errors":  errors,
    }

    output_path_p = Path(output_path).expanduser().resolve()
    append_run(output_path_p, run_entry)

    print(f"\n[OK] Wrote {len(results)} prediction(s) to: {output_path_p}")
    if errors:
        print(f"[WARN] {len(errors)} ticker(s) failed.")
    if results:
        print("\nSummary:")
        for item in results:
            print(
                f"  {item['ticker']} | {item['horizon_days']}d | {item['direction']}"
                f" | prob_up={item['prob_up']} | prob_down={item['prob_down']}"
                f" | as_of={item['as_of_date']}"
            )
    return 0


def main() -> int:
    args = parse_args()
    return run_pipeline(
        tickers_raw=args.tickers,
        horizon=args.horizon,
        model_path=args.model_path,
        output_path=args.output,
        period=args.period,
    )


if __name__ == "__main__":
    import sys
    raise SystemExit(main())
