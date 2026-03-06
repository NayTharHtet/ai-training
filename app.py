from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timedelta

from pathlib import Path
from typing import Any, Dict, Optional

from flask import Flask, jsonify, make_response, render_template, request, send_from_directory

BASE_DIR = Path(__file__).resolve().parent
PREDICT_PY = BASE_DIR / "predict.py"
OUTPUT_JSON = BASE_DIR / "output.json"

ALLOWED_TICKERS = {"AAPL", "NVDA", "TSLA", "ALL"}
ALLOWED_HORIZONS = {1, 7, 30}

app = Flask(__name__)  # uses ./templates and ./static by default [web:709][web:710]

@app.get("/__debug_paths")
def __debug_paths():
    return jsonify({
        "cwd": str(Path.cwd()),
        "app_root_path": app.root_path,
        "template_folder": app.template_folder,
        "static_folder": app.static_folder,
        "index_exists": (Path(app.root_path) / (app.template_folder or "templates") / "index.html").exists(),
        "index_path": str(Path(app.root_path) / (app.template_folder or "templates") / "index.html"),
    })

@app.get("/__debug_index_len")
def __debug_index_len():
    from pathlib import Path
    p = Path(app.root_path) / (app.template_folder or "templates") / "index.html"
    if not p.exists():
        return jsonify({"exists": False, "path": str(p)})
    txt = p.read_text(encoding="utf-8", errors="replace")
    return jsonify({"exists": True, "path": str(p), "length": len(txt), "first_120": txt[:120]})




def _no_cache(resp):
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    resp.headers["Pragma"] = "no-cache"
    resp.headers["Expires"] = "0"
    return resp


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"schema_version": 1, "updated_at": None, "runs": []}
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {"schema_version": 1, "updated_at": None, "runs": []}


def _latest_run(data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    runs = data.get("runs")
    if not isinstance(runs, list) or not runs:
        return None
    last = runs[-1]
    return last if isinstance(last, dict) else None


def _pick_result(run: Dict[str, Any], ticker: str) -> Optional[Dict[str, Any]]:
    results = run.get("results")
    if not isinstance(results, list):
        return None
    t = ticker.strip().upper()
    for r in results:
        if isinstance(r, dict) and str(r.get("ticker", "")).upper() == t:
            return r
    return None


def _pick_error(run: Dict[str, Any], ticker: str) -> Optional[Dict[str, Any]]:
    errors = run.get("errors")
    if not isinstance(errors, list):
        return None
    t = ticker.strip().upper()
    for e in errors:
        if isinstance(e, dict) and str(e.get("ticker", "")).upper() == t:
            return e
    return None


def _validate_inputs(tickers: str, horizon: int) -> Optional[str]:
    tickers = (tickers or "ALL").strip().upper()
    if tickers == "ALL":
        ok_tickers = True
    else:
        parts = [p.strip().upper() for p in tickers.split(",") if p.strip()]
        ok_tickers = len(parts) > 0 and all(p in ALLOWED_TICKERS and p != "ALL" for p in parts)

    if not ok_tickers:
        return "Invalid tickers. Use ALL or comma list of: AAPL,NVDA,TSLA"

    if horizon not in ALLOWED_HORIZONS:
        return "Invalid horizon. Use 1, 7, or 30."

    return None


def _run_predict_py(tickers: str, horizon: int) -> subprocess.CompletedProcess:
    cmd = [
        sys.executable, str(PREDICT_PY),
        "--tickers", tickers,
        "--horizon", str(horizon),
        "--output", str(OUTPUT_JSON),
    ]
    return subprocess.run(cmd, capture_output=True, text=True)


@app.get("/")
def index():
    return render_template("index.html")



@app.get("/output.json")
def output_json():
    data = _read_json(OUTPUT_JSON)
    resp = make_response(jsonify(data))
    return _no_cache(resp)


@app.post("/run_predict")
def run_predict():
    body = request.get_json(silent=True) or {}
    tickers = str(body.get("tickers", "ALL")).strip().upper()
    horizon = int(body.get("horizon", 7))
    ticker_for_ui = str(body.get("ticker", "")).strip().upper()


    err = _validate_inputs(tickers, horizon)
    if err:
        return _no_cache(make_response(jsonify({"ok": False, "error": err}), 400))

    if not PREDICT_PY.exists():
        return _no_cache(make_response(jsonify({"ok": False, "error": "predict.py not found"}), 500))

    proc = _run_predict_py(tickers, horizon)
    if proc.returncode != 0:
        return _no_cache(make_response(jsonify({
            "ok": False,
            "error": "predict.py failed",
            "stdout": proc.stdout,
            "stderr": proc.stderr
        }), 500))

    data = _read_json(OUTPUT_JSON)
    last = _latest_run(data)

    picked_result = None
    picked_error = None
    if last and ticker_for_ui:
        picked_result = _pick_result(last, ticker_for_ui)
        picked_error = _pick_error(last, ticker_for_ui)
        
    target_date = None
    if picked_result and picked_result.get("as_of_date"):
        try:
            d0 = datetime.strptime(picked_result["as_of_date"], "%Y-%m-%d").date()
            target_date = (d0 + timedelta(days=int(horizon))).isoformat()
        except Exception:
            target_date = None



    return _no_cache(make_response(jsonify({
    "ok": True,
    "data": data,
    "latest_run": last,
    "picked": {
        "ticker": ticker_for_ui or None,
        "result": picked_result,
        "error": picked_error,
        "target_date": target_date
    }
}), 200))



if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
