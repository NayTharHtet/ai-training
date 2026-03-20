from __future__ import annotations

import json
import re
import subprocess
import sys
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
<<<<<<< HEAD
from typing import Any, Dict

import yfinance as yf
from dotenv import load_dotenv
from flask import Flask, jsonify, make_response, render_template, request

load_dotenv()

=======
from typing import Any, Dict, Optional

import yfinance as yf
from flask import Flask, jsonify, make_response, render_template, request

>>>>>>> cd18f4651e0eb348828f24c9d2f2f0d980b65797
BASE_DIR     = Path(__file__).resolve().parent
PREDICT_PY   = BASE_DIR / "predict.py"
OUTPUT_JSON  = BASE_DIR / "output.json"

XGB_TICKERS      = {"AAPL", "NVDA", "TSLA"}
ALLOWED_HORIZONS = {1, 7, 30}

app = Flask(__name__)

<<<<<<< HEAD

=======
>>>>>>> cd18f4651e0eb348828f24c9d2f2f0d980b65797
def _no_cache(resp):
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    resp.headers["Pragma"]        = "no-cache"
    resp.headers["Expires"]       = "0"
    return resp

<<<<<<< HEAD

=======
>>>>>>> cd18f4651e0eb348828f24c9d2f2f0d980b65797
def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"schema_version": 1, "updated_at": None, "runs": []}
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {"schema_version": 1, "updated_at": None, "runs": []}

<<<<<<< HEAD

=======
>>>>>>> cd18f4651e0eb348828f24c9d2f2f0d980b65797
def _latest_run(data):
    runs = data.get("runs")
    if not isinstance(runs, list) or not runs:
        return None
    last = runs[-1]
    return last if isinstance(last, dict) else None

<<<<<<< HEAD

=======
>>>>>>> cd18f4651e0eb348828f24c9d2f2f0d980b65797
def _pick_result(run, ticker):
    results = run.get("results")
    if not isinstance(results, list):
        return None
    t = ticker.strip().upper()
    for r in results:
        if isinstance(r, dict) and str(r.get("ticker", "")).upper() == t:
            return r
    return None

<<<<<<< HEAD

=======
>>>>>>> cd18f4651e0eb348828f24c9d2f2f0d980b65797
def _run_predict_py(tickers: str, horizon: int):
    cmd = [sys.executable, str(PREDICT_PY),
           "--tickers", tickers, "--horizon", str(horizon), "--output", str(OUTPUT_JSON)]
    return subprocess.run(cmd, capture_output=True, text=True)

<<<<<<< HEAD

def _openrouter_call(model: str, prompt: str, api_key: str) -> str:
    import requests as req
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:7860",
        "X-Title": "Market Oracle"
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 200,
    }
    r = req.post("https://openrouter.ai/api/v1/chat/completions",
                 headers=headers, json=payload, timeout=20)
    if r.status_code != 200:
        raise ValueError(f"HTTP {r.status_code}: {r.text[:300]}")
    resp = r.json()
    if "choices" not in resp:
        raise ValueError(f"Unexpected response: {str(resp)[:300]}")
    return resp["choices"][0]["message"]["content"]


=======
>>>>>>> cd18f4651e0eb348828f24c9d2f2f0d980b65797
@app.get("/")
def index():
    return render_template("index.html")

<<<<<<< HEAD

=======
>>>>>>> cd18f4651e0eb348828f24c9d2f2f0d980b65797
@app.get("/output.json")
def output_json():
    data = _read_json(OUTPUT_JSON)
    return _no_cache(make_response(jsonify(data)))

<<<<<<< HEAD

=======
>>>>>>> cd18f4651e0eb348828f24c9d2f2f0d980b65797
@app.get("/api/chart")
def api_chart():
    ticker = request.args.get("ticker", "AAPL").strip().upper()
    days   = int(request.args.get("days", 365))

<<<<<<< HEAD
    if days > 1825:   period = "max"
    elif days > 730:  period = "5y"
    elif days > 365:  period = "2y"
    else:             period = "1y"
=======
    if days > 1825:
        period = "max"
    elif days > 730:
        period = "5y"
    elif days > 365:
        period = "2y"
    else:
        period = "1y"
>>>>>>> cd18f4651e0eb348828f24c9d2f2f0d980b65797

    try:
        df = yf.download(ticker, period=period, interval="1d",
                         progress=False, auto_adjust=True)
<<<<<<< HEAD
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
=======

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

>>>>>>> cd18f4651e0eb348828f24c9d2f2f0d980b65797
        df = df.dropna()
        if df.empty:
            raise ValueError(f"No data returned for {ticker}")

        candles = []
        for ts, row in df.iterrows():
            candles.append({
                "time":  int(ts.timestamp()),
                "open":  round(float(row["Open"]),  4),
                "high":  round(float(row["High"]),  4),
                "low":   round(float(row["Low"]),   4),
                "close": round(float(row["Close"]), 4),
            })
<<<<<<< HEAD
=======

>>>>>>> cd18f4651e0eb348828f24c9d2f2f0d980b65797
        last_close = candles[-1]["close"] if candles else None
        return _no_cache(make_response(jsonify({
            "ok": True, "ticker": ticker,
            "last_close": last_close, "candles": candles
        })))
<<<<<<< HEAD
=======

>>>>>>> cd18f4651e0eb348828f24c9d2f2f0d980b65797
    except Exception as e:
        return _no_cache(make_response(jsonify({
            "ok": False, "error": str(e), "candles": []
        }), 500))

<<<<<<< HEAD

@app.post("/run_predict")
def run_predict():
    body      = request.get_json(silent=True) or {}
    ticker_ui = str(body.get("ticker", "")).strip().upper()
=======
@app.post("/run_predict")
def run_predict():
    body      = request.get_json(silent=True) or {}
    ticker_ui = str(body.get("ticker",  "")).strip().upper()
>>>>>>> cd18f4651e0eb348828f24c9d2f2f0d980b65797
    horizon   = int(body.get("horizon", 7))

    if horizon not in ALLOWED_HORIZONS:
        return _no_cache(make_response(jsonify({"ok": False, "error": "Invalid horizon."}), 400))

    xgb_result  = None
    xgb_error   = None
    target_date = None

    if ticker_ui in XGB_TICKERS:
        if not PREDICT_PY.exists():
            xgb_error = "predict.py not found"
        else:
            proc = _run_predict_py(ticker_ui, horizon)
            if proc.returncode != 0:
                xgb_error = proc.stderr[:400] or "predict.py failed"
            else:
                data = _read_json(OUTPUT_JSON)
                last = _latest_run(data)
                if last:
                    xgb_result = _pick_result(last, ticker_ui)
                    if xgb_result:
                        try:
                            d0 = datetime.strptime(xgb_result["as_of_date"], "%Y-%m-%d").date()
                            target_date = (d0 + timedelta(days=int(xgb_result["horizon_days"]))).isoformat()
                        except Exception:
                            target_date = None
    else:
        xgb_error = f"XGBoost only supports AAPL, NVDA, TSLA. Showing LLM insights only for {ticker_ui}."

<<<<<<< HEAD
=======


>>>>>>> cd18f4651e0eb348828f24c9d2f2f0d980b65797
    llm = _call_llms(ticker_ui, horizon)

    return _no_cache(make_response(jsonify({
        "ok": True,
        "picked": {
            "ticker":      ticker_ui,
            "result":      xgb_result,
            "error":       xgb_error,
            "target_date": target_date,
        },
        "llm": llm,
    }), 200))


def _call_llms(ticker: str, horizon: int) -> dict:
    import os
    llm_out = {}
<<<<<<< HEAD
    or_key  = os.environ.get("OPENROUTER_API_KEY", "")

    prompt = (
        f"You are a financial risk analyst. Analyze {ticker} stock for "
        f"the next {horizon} day(s). Return JSON only with keys: "
        f"risk_level (LOW/MEDIUM/HIGH), price_estimate (number), "
        f"trend (UP/DOWN/SIDEWAYS), insight (max 1 sentence)."
    )

    # ── Qwen 3.5 Flash (free) ──
    try:
        if not or_key:
            raise ValueError("OPENROUTER_API_KEY not set")
        txt = _openrouter_call("qwen/qwen3.5-flash-02-23", prompt, or_key)
        m   = re.search(r'\{[^{}]*\}', txt, re.DOTALL)
        llm_out["chat"] = json.loads(m.group()) if m else {
            "insight": txt[:200], "trend": "–", "risk_level": "MEDIUM", "price_estimate": None
        }
    except Exception as e:
        llm_out["chat"] = {"insight": f"Qwen error: {e}", "trend": "–",
                            "risk_level": "MEDIUM", "price_estimate": None}


    # ── Google Gemma via OpenRouter (free) ──
    try:
        if not or_key:
            raise ValueError("OPENROUTER_API_KEY not set")
        txt = _openrouter_call("google/gemma-3-4b-it:free", prompt, or_key)
        m   = re.search(r'\{[^{}]*\}', txt, re.DOTALL)
=======

    # ── Gemini ──
    try:
        from google import genai
        client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY", ""))
        prompt = (
            f"You are a financial risk analyst. Analyze {ticker} stock for "
            f"the next {horizon} day(s). Return JSON only with keys: "
            f"risk_level (LOW/MEDIUM/HIGH), price_estimate (number), "
            f"trend (UP/DOWN/SIDEWAYS), insight (max 1 sentence)."
        )
        r   = client.models.generate_content(model="gemini-2.0-flash", contents=prompt)
        txt = r.text
        m   = re.search(r'\{.*?\}', txt, re.DOTALL)
        llm_out["gemini"] = json.loads(m.group()) if m else {
            "insight": txt[:200], "trend": "–", "risk_level": "MEDIUM", "price_estimate": None
        }
    except Exception as e:
        llm_out["gemini"] = {"insight": f"Gemini error: {e}", "trend": "–",
                              "risk_level": "MEDIUM", "price_estimate": None}

    # ── DeepSeek ──
    try:
        import requests as req
        ds_key  = os.environ.get("DEEPSEEK_API_KEY", "")
        headers = {"Authorization": f"Bearer {ds_key}", "Content-Type": "application/json"}
        payload = {
            "model": "deepseek-chat",
            "messages": [{"role": "user", "content": (
                f"You are a financial risk analyst. Analyze {ticker} stock for "
                f"the next {horizon} day(s). Return JSON only with keys: "
                f"risk_level (LOW/MEDIUM/HIGH), price_estimate (number), "
                f"trend (UP/DOWN/SIDEWAYS), insight (max 1 sentence)."
            )}],
            "max_tokens": 200,
        }
        r   = req.post("https://api.deepseek.com/v1/chat/completions",
                       headers=headers, json=payload, timeout=15)
        txt = r.json()["choices"][0]["message"]["content"]
        m   = re.search(r'\{.*?\}', txt, re.DOTALL)
>>>>>>> cd18f4651e0eb348828f24c9d2f2f0d980b65797
        llm_out["deepseek"] = json.loads(m.group()) if m else {
            "insight": txt[:200], "trend": "–", "risk_level": "MEDIUM", "price_estimate": None
        }
    except Exception as e:
<<<<<<< HEAD
        llm_out["deepseek"] = {"insight": f"Gemma error: {e}", "trend": "–",
=======
        llm_out["deepseek"] = {"insight": f"DeepSeek error: {e}", "trend": "–",
>>>>>>> cd18f4651e0eb348828f24c9d2f2f0d980b65797
                                "risk_level": "MEDIUM", "price_estimate": None}

    # ── Groq (free) ──
    try:
        from groq import Groq as GroqClient
        groq_client = GroqClient(api_key=os.environ.get("GROQ_API_KEY", ""))
<<<<<<< HEAD
=======
        prompt = (
            f"You are a financial risk analyst. Analyze {ticker} stock for "
            f"the next {horizon} day(s). Return JSON only with keys: "
            f"risk_level (LOW/MEDIUM/HIGH), price_estimate (number), "
            f"trend (UP/DOWN/SIDEWAYS), insight (max 1 sentence)."
        )
>>>>>>> cd18f4651e0eb348828f24c9d2f2f0d980b65797
        chat = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile",
            max_tokens=200,
        )
        txt = chat.choices[0].message.content
<<<<<<< HEAD
        m   = re.search(r'\{[^{}]*\}', txt, re.DOTALL)
=======
        m   = re.search(r'\{.*?\}', txt, re.DOTALL)
>>>>>>> cd18f4651e0eb348828f24c9d2f2f0d980b65797
        llm_out["groq"] = json.loads(m.group()) if m else {
            "insight": txt[:200], "trend": "–", "risk_level": "MEDIUM", "price_estimate": None
        }
    except Exception as e:
        llm_out["groq"] = {"insight": f"Groq error: {e}", "trend": "–",
                            "risk_level": "MEDIUM", "price_estimate": None}

    return llm_out

<<<<<<< HEAD
    # ── Qwen (free) with fallback chain ──
    QWEN_MODELS = [
        "qwen/qwen3.5-flash-02-23",
        "qwen/qwen2.5-72b-instruct:free",
        "meta-llama/llama-3.3-70b-instruct:free",
        "mistralai/mistral-small-3.1-24b-instruct:free",
    ]
    llm_out["chat"] = None
    for model_id in QWEN_MODELS:
        try:
            if not or_key:
                raise ValueError("OPENROUTER_API_KEY not set")
            txt = _openrouter_call(model_id, prompt, or_key)
            m   = re.search(r'\{[^{}]*\}', txt, re.DOTALL)
            llm_out["chat"] = json.loads(m.group()) if m else {
                "insight": txt[:200], "trend": "–", "risk_level": "MEDIUM", "price_estimate": None
            }
            break  # ✅ success — stop trying
        except Exception:
            continue  # failed — try next model
    if not llm_out["chat"]:
        llm_out["chat"] = {"insight": "All Qwen models unavailable.", "trend": "–",
                            "risk_level": "MEDIUM", "price_estimate": None}



=======
>>>>>>> cd18f4651e0eb348828f24c9d2f2f0d980b65797
@app.get("/api/news")
def api_news():
    import feedparser
    ticker = request.args.get("ticker", "AAPL").strip().upper()
    url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"
    try:
        feed = feedparser.parse(url)
        articles = []
        for entry in feed.entries[:6]:
            articles.append({
                "title":     entry.get("title", ""),
                "link":      entry.get("link",  ""),
                "published": entry.get("published", ""),
                "summary":   entry.get("summary", "")[:200],
            })
        if not articles:
            raise ValueError(f"No articles found for {ticker}")
        return _no_cache(make_response(jsonify({"ok": True, "ticker": ticker, "articles": articles})))
    except Exception as e:
        return _no_cache(make_response(jsonify({"ok": False, "error": str(e), "articles": []})))


<<<<<<< HEAD
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860, debug=False)
=======

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860, debug=False)
>>>>>>> cd18f4651e0eb348828f24c9d2f2f0d980b65797
