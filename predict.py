#!/usr/bin/env python3
"""
predict.py

Single-file prediction script for an engineered stock CSV.

- Choose tickers: AAPL, NVDA, TSLA (or ALL)
- Choose horizon: 1, 7, 30 (based on available target columns in the CSV)
- Produces/updates output.json on every run by appending a new "run" entry.

Robust behaviors (so your demo doesn't "die"):
- Case-insensitive handling for 'Ticker' and 'Date' column names.
- Auto feature selection if expected feature columns don't exist.
- If a ticker has only ONE class in the training labels (all UP or all DOWN),
  falls back to a baseline prediction instead of crashing.
- Prints per-ticker errors to console (and still records them in JSON).
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

import sys

ISO_FMT = "%Y-%m-%dT%H:%M:%SZ"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime(ISO_FMT)


def _find_col(df: pd.DataFrame, wanted: str) -> Optional[str]:
    """Find a column name in df matching wanted (case-insensitive)."""
    wanted_l = wanted.lower()
    for c in df.columns:
        if str(c).lower() == wanted_l:
            return str(c)
    return None


def infer_available_horizons(df: pd.DataFrame) -> List[int]:
    """Find horizons based on columns like target_up_1d, target_up_7d, target_up_30d."""
    horizons = set()
    for c in df.columns:
        m = re.fullmatch(r"target_up_(\d+)d", str(c))
        if m:
            horizons.add(int(m.group(1)))
    return sorted(horizons)


def build_feature_list(df: pd.DataFrame, ticker_col: str, date_col: str) -> List[str]:
    """Prefer a known-good feature set if present; otherwise auto-pick numeric columns."""
    preferred = [
        "Open", "High", "Low", "Close", "Volume",
        "ret_1", "ret_5", "ret_20",
        "vol_5", "vol_20",
        "ma_5", "ma_20", "ma_ratio",
    ]
    present = [c for c in preferred if c in df.columns]
    if present:
        return present

    numeric_cols = df.select_dtypes(include=["number", "bool"]).columns.tolist()
    auto = []
    for c in numeric_cols:
        if c in (ticker_col, date_col):
            continue
        if re.match(r"^target_", str(c)):
            continue
        auto.append(str(c))
    return auto[:25]


@dataclass
class PredictionResult:
    ticker: str
    horizon_days: int
    as_of_date: str
    direction: str
    prob_up: Optional[float]
    prob_down: Optional[float]
    expected_return: Optional[float]
    model: str
    features_used: List[str]
    note: Optional[str] = None


def _baseline_from_return(expected_return: Optional[float]) -> Tuple[str, float, float]:
    if expected_return is None:
        return "UP", 0.5, 0.5
    if expected_return > 0:
        return "UP", 0.6, 0.4
    if expected_return < 0:
        return "DOWN", 0.4, 0.6
    return "UP", 0.5, 0.5


def train_and_predict_for_ticker(
    df: pd.DataFrame,
    ticker: str,
    horizon_days: int,
    features: List[str],
    ticker_col: str,
    date_col: str,
    min_train_rows: int = 250,
    random_state: int = 42,
) -> PredictionResult:
    up_col = f"target_up_{horizon_days}d"
    ret_col = f"target_ret_{horizon_days}d"

    if up_col not in df.columns:
        raise ValueError(f"Missing target column '{up_col}' in CSV.")

    df_t = df[df[ticker_col].astype(str).str.upper() == ticker].copy()
    if df_t.empty:
        raise ValueError(f"No rows found for ticker '{ticker}'.")

    df_t[date_col] = pd.to_datetime(df_t[date_col], errors="coerce", utc=True)
    df_t = df_t.sort_values(date_col).reset_index(drop=True)

    as_of_row = df_t.iloc[-1].copy()
    as_of_date = as_of_row[date_col]
    as_of_date_str = as_of_date.strftime("%Y-%m-%d") if pd.notna(as_of_date) else "unknown"

    train_df = df_t.dropna(subset=[up_col]).copy()
    if len(train_df) >= 2:
        train_df = train_df.iloc[:-1].copy()

    need_cols = [c for c in features if c in train_df.columns]
    if not need_cols:
        raise ValueError("No usable feature columns found for this CSV/ticker.")

    X = train_df[need_cols].replace([np.inf, -np.inf], np.nan).dropna()
    y = train_df.loc[X.index, up_col].astype(int)

    y_ret = None
    if ret_col in train_df.columns:
        y_ret = train_df.loc[X.index, ret_col]

    uniq = y.nunique(dropna=True)
    if uniq < 2 or len(y) < 20:
        expected_return = None
        if y_ret is not None and y_ret.notna().sum() > 0:
            expected_return = float(y_ret.dropna().mean())
            direction, prob_up, prob_down = _baseline_from_return(expected_return)
            note = "baseline:insufficient_label_var_used_mean_return"
        else:
            maj = int(y.mode().iloc[0]) if len(y) else 1
            direction = "UP" if maj == 1 else "DOWN"
            prob_up = 1.0 if direction == "UP" else 0.0
            prob_down = 1.0 if direction == "DOWN" else 0.0
            note = "baseline:single_class_or_too_few_rows"

        return PredictionResult(
            ticker=ticker,
            horizon_days=horizon_days,
            as_of_date=as_of_date_str,
            direction=direction,
            prob_up=prob_up,
            prob_down=prob_down,
            expected_return=expected_return,
            model="Baseline (fallback)",
            features_used=need_cols,
            note=note,
        )

    clf = DecisionTreeClassifier(random_state=random_state, max_depth=5, min_samples_leaf=25)
    clf.fit(X.values, y.values)

    reg = None
    if y_ret is not None and y_ret.notna().sum() > 50:
        reg = DecisionTreeRegressor(random_state=random_state, max_depth=5, min_samples_leaf=25)
        reg.fit(X.values, y_ret.values)

    X_pred = pd.DataFrame([as_of_row[need_cols].to_dict()])[need_cols]
    X_pred = X_pred.replace([np.inf, -np.inf], np.nan).ffill(axis=1).fillna(0.0)

    pred_class = int(clf.predict(X_pred.values)[0])
    direction = "UP" if pred_class == 1 else "DOWN"

    prob_up = prob_down = None
    if hasattr(clf, "predict_proba"):
        probs = clf.predict_proba(X_pred.values)[0]
        class_to_prob = {int(c): float(p) for c, p in zip(clf.classes_, probs)}
        prob_up = class_to_prob.get(1)
        prob_down = class_to_prob.get(0)

    expected_return = float(reg.predict(X_pred.values)[0]) if reg is not None else None

    note = None
    if len(train_df) < min_train_rows:
        note = f"warn:train_rows_low({len(train_df)})"

    return PredictionResult(
        ticker=ticker,
        horizon_days=horizon_days,
        as_of_date=as_of_date_str,
        direction=direction,
        prob_up=prob_up,
        prob_down=prob_down,
        expected_return=expected_return,
        model="DecisionTree (depth=5, min_leaf=25)",
        features_used=need_cols,
        note=note,
    )


def load_existing_output(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"schema_version": 1, "updated_at": None, "runs": []}
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return {"schema_version": 1, "updated_at": None, "runs": []}
        data.setdefault("schema_version", 1)
        data.setdefault("runs", [])
        return data
    except Exception:
        backup = path.with_suffix(".corrupt.backup.json")
        try:
            path.replace(backup)
        except Exception:
            pass
        return {"schema_version": 1, "updated_at": None, "runs": []}


def append_run(output_path: Path, run_entry: Dict[str, Any]) -> None:
    data = load_existing_output(output_path)
    data["updated_at"] = utc_now_iso()
    if not isinstance(data.get("runs"), list):
        data["runs"] = []
    data["runs"].append(run_entry)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a simple tree model per ticker and write predictions to output.json")
    p.add_argument("--csv", default="engineered_master_1_7_30.csv", help="Path to engineered CSV")
    p.add_argument("--tickers", default="ALL", help="Comma-separated tickers or ALL")
    p.add_argument("--horizon", type=int, default=1, help="Prediction horizon (days)")
    p.add_argument("--output", default="output.json", help="Output JSON path")
    p.add_argument("--min-train-rows", type=int, default=250, help="Minimum train rows before warning")
    p.add_argument("--random-state", type=int, default=42, help="Random seed")
    return p.parse_args()

def interactive_prompt(default_csv: str, default_output: str):
    print("\n=== Stock Predictor ===")
    print("Tickers: AAPL, NVDA, TSLA, ALL")
    tickers_in = input("Choose tickers (e.g. ALL or AAPL,NVDA) [ALL]: ").strip() or "ALL"

    horizon_in = input("Choose horizon days [1/7/30] (default 7): ").strip() or "7"
    try:
        horizon = int(horizon_in)
    except ValueError:
        horizon = 7

    csv_in = input(f"CSV path [{default_csv}]: ").strip() or default_csv
    out_in = input(f"Output JSON [{default_output}]: ").strip() or default_output

    # Return values in same format your argparse uses
    return csv_in, tickers_in, horizon, out_in

def run_pipeline(
    csv_path: str,
    tickers: str,
    horizon: int,
    output_path: str,
    min_train_rows: int = 200,
    random_state: int = 42,
) -> int:
    """
    Runs the prediction pipeline and writes/updates output JSON.
    Returns process exit code (0 = OK).
    """
    csv_path_p = Path(csv_path).expanduser().resolve()
    if not csv_path_p.exists():
        print(f"[ERROR] CSV not found: {csv_path_p}")
        return 2

    df = pd.read_csv(csv_path_p)

    ticker_col = _find_col(df, "Ticker")
    date_col = _find_col(df, "Date")
    if not ticker_col or not date_col:
        print("[ERROR] CSV must contain 'Ticker' and 'Date' columns (any casing).")
        print(f"        Columns found: {list(df.columns)}")
        return 2

    available_horizons = infer_available_horizons(df)
    if horizon not in available_horizons:
        print(f"[ERROR] Horizon {horizon} not available. Available horizons: {available_horizons}")
        return 2

    all_tickers = sorted(df[ticker_col].dropna().astype(str).str.upper().unique().tolist())

    tickers_str = str(tickers).strip().upper()
    if tickers_str == "ALL":
        preferred = [t for t in ["AAPL", "NVDA", "TSLA"] if t in all_tickers]
        tickers_list = preferred if preferred else all_tickers
    else:
        tickers_list = [t.strip().upper() for t in str(tickers).split(",") if t.strip()]
        bad = [t for t in tickers_list if t not in all_tickers]
        if bad:
            print(f"[ERROR] Unknown tickers {bad}. Available: {all_tickers}")
            return 2

    features = build_feature_list(df, ticker_col=ticker_col, date_col=date_col)
    if not features:
        print("[ERROR] No usable feature columns found in CSV.")
        return 2

    results: List[Dict[str, Any]] = []
    errors: List[Dict[str, str]] = []

    for t in tickers_list:
        try:
            pr = train_and_predict_for_ticker(
                df=df,
                ticker=t,
                horizon_days=horizon,
                features=features,
                ticker_col=ticker_col,
                date_col=date_col,
                min_train_rows=min_train_rows,
                random_state=random_state,
            )
            results.append(asdict(pr))
        except Exception as e:
            msg = str(e)
            print(f"[ERROR] {t} failed: {msg}")
            errors.append({"ticker": t, "error": msg})

    run_entry = {
        "run_at": utc_now_iso(),
        "params": {
            "csv": str(csv_path_p),
            "tickers": tickers_list,
            "horizon_days": horizon,
            "features_used": features,
            "model": "DecisionTreeClassifier + optional DecisionTreeRegressor (with fallback baseline)",
            "random_state": random_state,
            "min_train_rows": min_train_rows,
        },
        "results": results,
        "errors": errors,
    }

    output_path_p = Path(output_path).expanduser().resolve()
    append_run(output_path_p, run_entry)

    print(f"[OK] Wrote {len(results)} prediction(s) to: {output_path_p}")
    if errors:
        print(f"[WARN] {len(errors)} ticker(s) failed. See 'errors' in the JSON.")
    return 0

def main() -> int:
    args = parse_args()
    return run_pipeline(
    csv_path=args.csv,
    tickers=args.tickers,
    horizon=args.horizon,
    output_path=args.output,
    min_train_rows=args.min_train_rows,
    random_state=args.random_state,
)

   


if __name__ == "__main__":
    if len(sys.argv) == 1:
        csv_path, tickers, horizon, out_path = interactive_prompt(
            default_csv="engineered_master_1_7_30.csv",
            default_output="output.json"
        )
        raise SystemExit(run_pipeline(csv_path, tickers, horizon, out_path))
    else:
        raise SystemExit(main())

