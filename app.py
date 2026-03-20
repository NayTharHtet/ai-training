from __future__ import annotations

import json
import re
import subprocess
import sys
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict

import yfinance as yf
from dotenv import load_dotenv
from flask import Flask, jsonify, make_response, render_template, request

load_dotenv()

BASE_DIR    = Path(__file__).resolve().parent
OUTPUT_JSON = BASE_DIR / "output.json"

app = Flask(__name__)


def _no_cache(resp):
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    resp.headers["Pragma"]        = "no-cache"
    resp.headers["Expires"]       = "0"
    return resp


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


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/chart")
def api_chart():
    ticker = request.args.get("ticker", "AAPL").strip().upper()
    days   = int(request.args.get("days", 365))

    if days > 1825:   period = "max"
    elif days > 730:  period = "5y"
    elif days > 365:  period = "2y"
    else:             period = "1y"

    try:
        df = yf.download(ticker, period=period, interval="1d",
                         progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
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
        last_close = candles[-1]["close"] if candles else None
        return _no_cache(make_response(jsonify({
            "ok": True, "ticker": ticker,
            "last_close": last_close, "candles": candles
        })))
    except Exception as e:
        return _no_cache(make_response(jsonify({
            "ok": False, "error": str(e), "candles": []
        }), 500))


@app.post("/run_predict")
def run_predict():
    body      = request.get_json(silent=True) or {}
    ticker_ui = str(body.get("ticker", "")).strip().upper()

    llm = _call_llms(ticker_ui, 1)

    return _no_cache(make_response(jsonify({
        "ok": True,
        "llm": llm,
    }), 200))


def _call_llms(ticker: str, horizon: int) -> dict:
    import os
    llm_out = {}
    or_key  = os.environ.get("OPENROUTER_API_KEY", "")

    prompt = (
        f"You are a financial risk analyst. Analyze {ticker} stock for "
        f"the next {horizon} day(s). Return JSON only with keys: "
        f"risk_level (LOW/MEDIUM/HIGH), price_estimate (number), "
        f"trend (UP/DOWN/SIDEWAYS), insight (max 1 sentence)."
    )

    # ── Qwen with fallback chain ──
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
            break
        except Exception:
            continue
    if not llm_out["chat"]:
        llm_out["chat"] = {"insight": "All Qwen models unavailable.", "trend": "–",
                            "risk_level": "MEDIUM", "price_estimate": None}

    # ── Gemma with fallback chain ──
    GEMMA_MODELS = [
        "google/gemma-3-4b-it:free",
        "google/gemma-3-12b-it:free",
        "google/gemma-2-9b-it:free",
        "microsoft/phi-3-mini-128k-instruct:free",
    ]
    llm_out["deepseek"] = None
    for model_id in GEMMA_MODELS:
        try:
            if not or_key:
                raise ValueError("OPENROUTER_API_KEY not set")
            txt = _openrouter_call(model_id, prompt, or_key)
            m   = re.search(r'\{[^{}]*\}', txt, re.DOTALL)
            llm_out["deepseek"] = json.loads(m.group()) if m else {
                "insight": txt[:200], "trend": "–", "risk_level": "MEDIUM", "price_estimate": None
            }
            break
        except Exception:
            continue
    if not llm_out["deepseek"]:
        llm_out["deepseek"] = {"insight": "All Gemma models unavailable.", "trend": "–",
                                "risk_level": "MEDIUM", "price_estimate": None}

    # ── Groq with fallback chain ──
    GROQ_MODELS = [
        "llama-3.3-70b-versatile",
        "llama-3.1-70b-versatile",
        "mixtral-8x7b-32768",
        "gemma2-9b-it",
    ]
    llm_out["groq"] = None
    for model_id in GROQ_MODELS:
        try:
            from groq import Groq as GroqClient
            groq_client = GroqClient(api_key=os.environ.get("GROQ_API_KEY", ""))
            chat = groq_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=model_id,
                max_tokens=200,
            )
            txt = chat.choices[0].message.content
            m   = re.search(r'\{[^{}]*\}', txt, re.DOTALL)
            llm_out["groq"] = json.loads(m.group()) if m else {
                "insight": txt[:200], "trend": "–", "risk_level": "MEDIUM", "price_estimate": None
            }
            break
        except Exception:
            continue
    if not llm_out["groq"]:
        llm_out["groq"] = {"insight": "All Groq models unavailable.", "trend": "–",
                            "risk_level": "MEDIUM", "price_estimate": None}

    return llm_out


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


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860, debug=False)
