from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier


CSV_PATH = "engineered_master_1_7_30.csv"
MODEL_PATH = "model_multi.pkl"

BASE_FEATURES = [
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "ret_1",
    "ret_5",
    "ret_20",
    "vol_5",
    "vol_20",
    "ma_5",
    "ma_20",
    "ma_ratio",
]

TARGETS = {
    1: "target_up_1d",
    7: "target_up_7d",
    30: "target_up_30d",
}


def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    if "Date" not in df.columns or "Ticker" not in df.columns:
        raise ValueError("CSV must contain 'Date' and 'Ticker' columns.")

    df["Date"] = pd.to_datetime(df["Date"], utc=True, errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values(["Date", "Ticker"]).reset_index(drop=True)

    return df


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, List[str]]:
    missing = [col for col in BASE_FEATURES if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")

    x = df[BASE_FEATURES + ["Ticker"]].copy()

    # One-hot encode ticker so models can learn stock-specific patterns
    x = pd.get_dummies(x, columns=["Ticker"], prefix="Ticker")

    feature_columns = x.columns.tolist()
    return x, feature_columns


def time_split(x: pd.DataFrame, y: pd.Series, split_ratio: float = 0.8):
    split_index = int(len(x) * split_ratio)

    x_train = x.iloc[:split_index].copy()
    x_test = x.iloc[split_index:].copy()
    y_train = y.iloc[:split_index].copy()
    y_test = y.iloc[split_index:].copy()

    return x_train, x_test, y_train, y_test


def build_model(model_name: str) -> Any:
    if model_name == "logistic_regression":
        return LogisticRegression(max_iter=1000, random_state=42)

    if model_name == "decision_tree":
        return DecisionTreeClassifier(
            max_depth=5,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
        )

    if model_name == "xgboost":
        return XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42,
        )

    raise ValueError(f"Unknown model name: {model_name}")


def main() -> int:
    csv_path = Path(CSV_PATH)
    if not csv_path.exists():
        print(f"[ERROR] CSV not found: {csv_path.resolve()}")
        return 1

    df = load_data(str(csv_path))
    x_all, feature_columns = prepare_features(df)

    model_names = [
        "logistic_regression",
        "decision_tree",
        "xgboost",
    ]

    trained_models: Dict[str, Dict[int, Any]] = {
        model_name: {} for model_name in model_names
    }

    metrics: Dict[str, Dict[int, Dict[str, float]]] = {
        model_name: {} for model_name in model_names
    }

    for horizon, target_col in TARGETS.items():
        if target_col not in df.columns:
            print(f"[WARN] Skipping {horizon}d models because '{target_col}' is missing.")
            continue

        working = x_all.copy()
        working[target_col] = df[target_col]

        # Remove rows where target is missing
        working = working.dropna(subset=[target_col]).reset_index(drop=True)

        x = working.drop(columns=[target_col])
        y = working[target_col].astype(int)

        if len(x) < 50:
            print(f"[WARN] Not enough rows to train {horizon}d models.")
            continue

        x_train, x_test, y_train, y_test = time_split(x, y, split_ratio=0.8)

        print(f"\n==============================")
        print(f"=== HORIZON: {horizon} DAY ===")
        print(f"==============================")

        for model_name in model_names:
            model = build_model(model_name)
            model.fit(x_train, y_train)

            preds = model.predict(x_test)
            acc = accuracy_score(y_test, preds)

            trained_models[model_name][horizon] = model
            metrics[model_name][horizon] = {
                "accuracy": float(acc),
                "train_rows": int(len(x_train)),
                "test_rows": int(len(x_test)),
            }

            print(f"\n--- {model_name.upper()} ---")
            print(f"Accuracy: {acc:.4f}")
            print(classification_report(y_test, preds, digits=4))

    # Check if at least one model got trained
    anything_trained = any(trained_models[name] for name in trained_models)
    if not anything_trained:
        print("[ERROR] No models were trained.")
        return 1

    artifact = {
        "models": trained_models,
        "feature_columns": feature_columns,
        "base_features": BASE_FEATURES,
        "targets": TARGETS,
        "metrics": metrics,
    }

    joblib.dump(artifact, MODEL_PATH)
    print(f"\n[OK] Saved combined model artifact to: {Path(MODEL_PATH).resolve()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())