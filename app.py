from __future__ import annotations

import os
import json
import csv
import re
import io
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import yfinance as yf
from dotenv import load_dotenv
from flask import Flask, jsonify, make_response, render_template, request

load_dotenv(override=True)



BASE_DIR = Path(__file__).resolve().parent
OUTPUT_JSON = BASE_DIR / "output.json"

app = Flask(__name__)

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

        sma_5  = round(float(df["Close"].tail(5).mean()),  2) if len(df) >= 5  else current_price
        sma_20 = round(float(df["Close"].tail(20).mean()), 2) if len(df) >= 20 else current_price
        volatility = round(float(df["Close"].pct_change().std() * 100), 2) if len(df) > 1 else 0.0
        five_day_return = round(
            (float(df["Close"].iloc[-1]) - float(df["Close"].iloc[-6])) /
            float(df["Close"].iloc[-6]) * 100, 2
        ) if len(df) >= 6 else 0.0

        print(f"DEBUG fetch_market_data >>> {ticker} current_price={current_price}")
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
        print(f"DEBUG fetch_market_data ERROR >>> {e}")
        return {"error": str(e)}

# ── FIX 1: Fetch the actual HISTORICAL closing price for a specific date ──
def fetch_historical_close(ticker: str, date_str: str) -> float | None:
    """
    Returns the closing price for `ticker` on `date_str` (YYYY-MM-DD) using
    historical data instead of the live price.  This is what the backtester
    should use so the actual_price column reflects the correct next-day close.
    """
    try:
        target_dt = datetime.strptime(date_str, "%Y-%m-%d")
        # Fetch a small window around the target date (±3 days handles weekends/holidays)
        start = (target_dt - timedelta(days=3)).strftime("%Y-%m-%d")
        end   = (target_dt + timedelta(days=4)).strftime("%Y-%m-%d")
        df = yf.download(ticker, start=start, end=end, interval="1d",
                         progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.dropna()
        if df.empty:
            return None
        df.index = pd.to_datetime(df.index)
        available = df[df.index <= pd.Timestamp(target_dt)]
        if available.empty:
            available = df
        return round(float(available["Close"].iloc[-1]), 2)
    except Exception as e:
        print(f"DEBUG fetch_historical_close ERROR >>> {ticker} {date_str}: {e}")
        return None

def build_prompt(ticker: str, data: dict) -> str:
    sma5  = data.get("sma_5",  data.get("current_price", 0))
    sma20 = data.get("sma_20", data.get("current_price", 0))
    five_day = data.get("five_day_return", 0)
    day_chg  = data.get("day_change_pct", 0)

    # Determine clear directional signal
    bullish_signals = sum([
        sma5 > sma20,
        five_day > 0,
        day_chg > 0,
    ])
    bearish_signals = sum([
        sma5 < sma20,
        five_day < 0,
        day_chg < 0,
    ])

    if bullish_signals >= 2:
        direction_instruction = "The data clearly shows BULLISH momentum. pct_change MUST be positive (between +0.5 and +3.0). trend MUST be UP."
    elif bearish_signals >= 2:
        direction_instruction = "The data clearly shows BEARISH momentum. pct_change MUST be negative (between -0.5 and -3.0). trend MUST be DOWN."
    else:
        direction_instruction = "Data is mixed. pct_change should be small (between -0.5 and +0.5). trend should be SIDEWAYS."

    return (
        f"You are a financial analyst predicting {ticker} for the next 1 trading day.\n"
        f"LIVE MARKET DATA:\n"
        f"- Current price: ${data['current_price']}\n"
        f"- Day change: {day_chg}%\n"
        f"- 5-day return: {five_day}%\n"
        f"- SMA5: ${sma5} | SMA20: ${sma20}\n"
        f"- Daily volatility: {data.get('volatility', 0)}%\n\n"
        f"INSTRUCTION: {direction_instruction}\n\n"
        f"Return ONLY valid JSON with these keys:\n"
        f"risk_level (LOW/MEDIUM/HIGH), "
        f"pct_change (float), "
        f"trend (UP/DOWN/SIDEWAYS), "
        f"insight (1 sentence mentioning the price).\n"
        f"No extra text. Just JSON."
    )

def is_valid_ticker(ticker: str) -> bool:
    try:
        price = yf.Ticker(ticker).fast_info.last_price
        return price is not None and float(price) > 0
    except Exception:
        return False

@app.get("/")
def index():
    return render_template("index.html")

@app.get("/history")
def history():
    update_predictions_log()

    file_path = BASE_DIR / "predictions_log.csv"

    if file_path.exists():
        df = pd.read_csv(file_path)
        records = df.fillna("").to_dict(orient="records")

        total_predictions = len(df)
        checked_df = df[df["actual_price"].notna()] if "actual_price" in df.columns else pd.DataFrame()
        checked_results = len(checked_df)

        if "is_correct" in df.columns:
            correct_count = len(df[df["is_correct"] == True]) + len(df[df["is_correct"] == "True"])
        else:
            correct_count = 0

        if "error_percent" in df.columns and checked_results > 0:
            error_series = pd.to_numeric(df["error_percent"], errors="coerce").dropna()
            avg_error_percent = round(error_series.mean(), 2) if not error_series.empty else 0
        else:
            avg_error_percent = 0
    else:
        records = []
        total_predictions = 0
        checked_results = 0
        correct_count = 0
        avg_error_percent = 0

    return render_template(
        "history.html",
        records=records,
        total_predictions=total_predictions,
        checked_results=checked_results,
        correct_count=correct_count,
        avg_error_percent=avg_error_percent
    )

def cleanup_old_predictions(days_to_keep: int = 3):
    """Delete rows where timestamp is older than `days_to_keep` days,
    UNLESS the row has been manually saved (saved == True)."""
    file_path = BASE_DIR / "predictions_log.csv"
    if not file_path.exists():
        return

    df = pd.read_csv(file_path)
    if df.empty:
        return

    cutoff = datetime.now() - timedelta(days=days_to_keep)

    # Add 'saved' column if it doesn't exist
    if "saved" not in df.columns:
        df["saved"] = False

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    # Keep rows that are: newer than cutoff OR manually saved
    df = df[(df["timestamp"] >= cutoff) | (df["saved"] == True)]

    df.to_csv(file_path, index=False)

@app.get("/output.json")
def output_json():
    data = _read_json(OUTPUT_JSON)
    return _no_cache(make_response(jsonify(data)))

@app.get("/api/chart")
def api_chart():
    ticker = request.args.get("ticker", "AAPL").strip().upper()
    days = int(request.args.get("days", 365))

    if days > 1825: period = "max"
    elif days > 730: period = "5y"
    elif days > 365: period = "2y"
    else:            period = "1y"

    try:
        import time
        for attempt in range(3):
            df = yf.download(ticker, period=period, interval="1d",
                             progress=False, auto_adjust=True, timeout=10)
            if not df.empty:
                break
            time.sleep(5 * (attempt + 1))
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

def save_prediction_log(ticker, current_price, llm, summary):
    file_path = BASE_DIR / "predictions_log.csv"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ── FIX 3: Skip weekends when calculating the target date ──
    target_dt = datetime.now() + timedelta(days=1)
    while target_dt.weekday() >= 5:   # 5=Sat, 6=Sun
        target_dt += timedelta(days=1)
    target_date = target_dt.strftime("%Y-%m-%d")

    chat_price          = llm.get("chat",     {}).get("price_estimate", "")
    deepseek_price      = llm.get("deepseek", {}).get("price_estimate", "")
    gemini_price        = llm.get("gemini",   {}).get("price_estimate", "")
    groq_price          = llm.get("groq",     {}).get("price_estimate", "")
    consensus_price     = summary.get("predicted_price", "")
    consensus_direction = summary.get("decision", "")

    file_exists = file_path.exists()

    with open(file_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "timestamp", "ticker", "current_price", "chat_price", "deepseek_price",
                "gemini_price", "groq_price", "consensus_price", "consensus_direction",
                "target_date", "actual_price", "actual_direction", "is_correct",
                "error_amount", "error_percent"
            ])
        writer.writerow([
            timestamp, ticker, current_price,
            chat_price, deepseek_price, gemini_price, groq_price,
            consensus_price, consensus_direction,
            target_date, "", "", "", "", ""
        ])

def update_predictions_log():
    file_path = BASE_DIR / "predictions_log.csv"

    if not file_path.exists():
        return

    df = pd.read_csv(file_path)
    if df.empty:
        return

    today_str = datetime.now().strftime("%Y-%m-%d")

    for i, row in df.iterrows():
        target_date  = str(row.get("target_date",  "")).strip()
        actual_price = str(row.get("actual_price", "")).strip()

        # Skip already-updated rows
        if actual_price not in ("", "nan", "None"):
            continue

        # Only update when the target date has arrived
        if not target_date or target_date > today_str:
            continue

        ticker              = str(row.get("ticker",              "")).strip().upper()
        current_price       = pd.to_numeric(row.get("current_price",   ""), errors="coerce")
        predicted_price     = pd.to_numeric(row.get("consensus_price", ""), errors="coerce")
        predicted_direction = str(row.get("consensus_direction", "")).strip().upper()

        try:
            # ── FIX 1 APPLIED: use historical close for the target date ──
            real_price = fetch_historical_close(ticker, target_date)

            if real_price is None:
                print(f"DEBUG update_predictions_log: no historical close for {ticker} on {target_date}")
                continue

            df.at[i, "actual_price"] = real_price

            # ── FIX 4: HOLD band — treat <0.5% move as HOLD, not BUY/SELL ──
            if pd.isna(current_price):
                actual_direction = ""
            else:
                pct_move = (real_price - float(current_price)) / float(current_price) * 100
                if abs(pct_move) < 0.5:
                    actual_direction = "HOLD"
                elif pct_move > 0:
                    actual_direction = "BUY"
                else:
                    actual_direction = "SELL"

            df.at[i, "actual_direction"] = actual_direction

            is_correct = str(predicted_direction) == str(actual_direction)
            df.at[i, "is_correct"] = is_correct

            if not pd.isna(predicted_price):
                error_amount  = abs(float(predicted_price) - real_price)
                error_percent = (error_amount / real_price) * 100 if real_price != 0 else 0
                df.at[i, "error_amount"]  = round(error_amount, 2)
                df.at[i, "error_percent"] = round(error_percent, 2)

        except Exception as e:
            print(f"Failed to update row {i} for {ticker}: {e}")

    df.to_csv(file_path, index=False)

@app.route("/export_csv")
def export_csv():
    """Export predictions_log.csv as a downloadable file."""
    log_path = Path("predictions_log.csv")
    
    if not log_path.exists():
        return jsonify({"error": "No predictions log found"}), 404
    
    try:
        with open(log_path, "r", newline="", encoding="utf-8") as f:
            csv_content = f.read()
        
        output = io.BytesIO()
        output.write(csv_content.encode("utf-8"))
        output.seek(0)
        
        response = make_response(output.getvalue())
        response.headers["Content-Disposition"] = "attachment; filename=predictions_log.csv"
        response.headers["Content-Type"] = "text/csv"
        return response
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.post("/run_predict")
def run_predict():
    body      = request.get_json(silent=True) or {}
    ticker_ui = str(body.get("ticker", "AAPL")).strip().upper()

    if not is_valid_ticker(ticker_ui):
        return _no_cache(make_response(jsonify({
            "ok": False,
            "error": f"Ticker '{ticker_ui}' not found. Please check the symbol."
        }), 400))

    today       = datetime.today()
    target_dt   = today + timedelta(days=1)
    while target_dt.weekday() >= 5:
        target_dt += timedelta(days=1)
    target_date = target_dt.strftime("%Y-%m-%d")
    as_of_date  = today.strftime("%Y-%m-%d")

    market_data = fetch_market_data(ticker_ui)
    llm         = _call_llms(ticker_ui, market_data)
    summary     = ensemble_summary(llm, current_price=market_data.get("current_price", 0))
    save_prediction_log(ticker_ui, market_data.get("current_price", 0), llm, summary)

    return _no_cache(make_response(jsonify({
        "ok": True,
        "picked": {
            "ticker":      ticker_ui,
            "result":      None,
            "error":       None,
            "target_date": target_date,
            "as_of_date":  as_of_date,
        },
        "llm":     llm,
        "summary": summary,
    }), 200))

def _call_llms(ticker: str, market_data: dict) -> dict:
    import os
    llm_out    = {}
    or_key     = os.environ.get("OPENROUTER_API_KEY", "")
    last_close = market_data.get("current_price") if "error" not in market_data else None

    prompt = build_prompt(ticker, market_data) if last_close else (
        f"Analyze {ticker} for next 1 trading day. "
        f"Return JSON only: risk_level (LOW/MEDIUM/HIGH), "
        f"pct_change (float -5 to 5), trend (UP/DOWN/SIDEWAYS), insight (1 sentence)."
    )

    def parse_result(txt, model_name):
        print(f"DEBUG {model_name} RAW >>> {txt[:300]}")
        try:
            m = re.search(r'\{[^{}]*\}', txt, re.DOTALL)
            if not m:
                return {"insight": txt[:200] or "No response", "trend": "SIDEWAYS",
                        "risk_level": "MEDIUM", "price_estimate": last_close}
            raw = json.loads(m.group())
            pct = float(raw.get("pct_change", 0))
            pct = max(-5.0, min(5.0, pct))
            raw["price_estimate"] = round(last_close * (1 + pct / 100), 2) if last_close else None
            raw.setdefault("trend",      "SIDEWAYS")
            raw.setdefault("risk_level", "MEDIUM")
            print(f"DEBUG {model_name} price_estimate >>> {raw['price_estimate']}")
            return raw
        except Exception as ex:
            print(f"DEBUG {model_name} parse error >>> {ex}")
            return {"insight": "Parse error", "trend": "SIDEWAYS",
                    "risk_level": "MEDIUM", "price_estimate": last_close}

    def openrouter_call(model: str, label: str):
        try:
            if not or_key:
                raise ValueError("OPENROUTER_API_KEY not set")
            import requests as req
            headers = {
                "Authorization": f"Bearer {or_key}",
                "Content-Type":  "application/json",
                "HTTP-Referer":  "http://localhost:7860",
                "X-Title":       "Market Oracle",
            }
            payload = {
                "model":      model,
                "messages":   [{"role": "user", "content": prompt}],
                "max_tokens": 200,
            }
            r = req.post("https://openrouter.ai/api/v1/chat/completions",
                         headers=headers, json=payload, timeout=20)
            if r.status_code != 200:
                raise ValueError(f"HTTP {r.status_code}: {r.text[:300]}")
            content = r.json()["choices"][0]["message"].get("content") or ""
            return parse_result(content, label)
        except Exception as e:
            print(f"DEBUG {label} failed >>> {e}")
            return {"insight": f"Model unavailable: {e}", "trend": "SIDEWAYS",
                    "risk_level": "MEDIUM", "price_estimate": last_close}

    llm_out["chat"]     = openrouter_call(model="openai/gpt-4o",                  label="CHATGPT")
    llm_out["deepseek"] = openrouter_call(model="deepseek/deepseek-chat-v3-0324", label="DEEPSEEK")
    llm_out["gemini"]   = openrouter_call(model="google/gemini-2.5-flash",         label="GEMINI")

    try:
        from groq import Groq as GroqClient
        groq_client = GroqClient(api_key=os.environ.get("GROQ_API_KEY", ""))
        chat = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile",
            max_tokens=200,
        )
        txt = chat.choices[0].message.content or ""
        llm_out["groq"] = parse_result(txt, "GROQ")
    except Exception as e:
        print(f"DEBUG GROQ failed >>> {e}")
        llm_out["groq"] = {"insight": f"Model unavailable: {e}", "trend": "SIDEWAYS",
                           "risk_level": "MEDIUM", "price_estimate": last_close}

    return llm_out

def ensemble_summary(llm_out: dict, current_price: float = 0.0) -> dict:
    models       = ["chat", "deepseek", "gemini", "groq"]
    risk_weights = {"LOW": 3, "MEDIUM": 2, "HIGH": 1}

    weighted_prices = []
    total_weight    = 0
    trends          = []
    risks           = []

    for m in models:
        data  = llm_out.get(m, {})
        price = data.get("price_estimate")
        risk  = data.get("risk_level", "MEDIUM").upper()
        trend = data.get("trend",      "SIDEWAYS").upper()
        risks.append(risk)
        trends.append(trend)
        if price is not None:
            w = risk_weights.get(risk, 2)
            weighted_prices.append(price * w)
            total_weight += w

    final_price = round(sum(weighted_prices) / total_weight, 2) if total_weight > 0 else None

    prices = [llm_out[m].get("price_estimate") for m in models if llm_out.get(m, {}).get("price_estimate")]
    if len(prices) >= 2:
        avg        = sum(prices) / len(prices)
        std_dev    = (sum((p - avg) ** 2 for p in prices) / len(prices)) ** 0.5
        confidence = max(0, round(100 - (std_dev / avg * 100), 1))
        confidence = min(confidence, 99)
    else:
        confidence = 50

    up        = sum(1 for t in trends if "UP"   in t)
    down      = sum(1 for t in trends if "DOWN" in t)
    high_risk = risks.count("HIGH")

    # ── FIX 5: Lowered thresholds + HOLD band for small predicted moves ──
    price_pct_diff = (
        abs(final_price - current_price) / current_price * 100
        if final_price and current_price else 999
    )

    if price_pct_diff < 0.5:
        decision, risk_label = "HOLD", "LOW"
    elif up >= 2 and high_risk <= 1:
        decision, risk_label = "BUY", "LOW"
    elif down >= 2 or high_risk >= 3:
        decision, risk_label = "SELL", "HIGH"
    elif up == 1 and down == 0:
        decision, risk_label = "HOLD", "MEDIUM"
    else:
        decision, risk_label = "CAUTION", "HIGH"

    return {
        "decision":       decision,
        "risk":           risk_label,
        "confidence":     confidence,
        "votes":          {"up": up, "down": down, "neutral": len(models) - up - down},
        "message": (
            f"{max(up, down)}/4 AI models agree. "
            + ("Upward potential — possible buying opportunity." if decision == "BUY"
               else "Downward risk — not recommended to buy."     if decision == "SELL"
               else "Minimal movement expected — hold position."  if decision == "HOLD"
               else "Mixed signals — safer to wait.")
        ),
        "current_price":  current_price,
        "predicted_price": final_price,
    }

@app.get("/api/news")
def api_news():
    import feedparser
    ticker = request.args.get("ticker", "AAPL").strip().upper()
    url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"
    try:
        feed     = feedparser.parse(url)
        articles = []
        for entry in feed.entries[:6]:
            articles.append({
                "title":     entry.get("title",     ""),
                "link":      entry.get("link",      ""),
                "published": entry.get("published", ""),
                "summary":   entry.get("summary",   "")[:200],
            })
        if not articles:
            raise ValueError(f"No articles found for {ticker}")
        return _no_cache(make_response(jsonify({"ok": True, "ticker": ticker, "articles": articles})))
    except Exception as e:
        return _no_cache(make_response(jsonify({"ok": False, "error": str(e), "articles": []})))

@app.get("/api/search")
def api_search():
    query = request.args.get("q", "").strip().upper()
    if not query:
        return jsonify([])
    ticker_db_path = BASE_DIR / "data" / "ticker_db.csv"
    df      = pd.read_csv(ticker_db_path)
    results = df[df["Symbol"].str.upper().str.startswith(query)].head(20)
    return jsonify([{"t": row["Symbol"], "n": row["Security Name"]}
                    for _, row in results.iterrows()])

@app.post("/api/feedback")
def api_feedback():
    body      = request.get_json(silent=True) or {}
    name      = body.get("name",    "Anonymous").strip()
    email     = body.get("email",   "").strip()
    message   = body.get("message", "").strip()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if not message:
        return jsonify({"ok": False, "error": "Message is required"}), 400

    feedback_path = BASE_DIR / "feedback.csv"
    file_exists   = feedback_path.exists()
    with open(feedback_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "name", "email", "message"])
        writer.writerow([timestamp, name, email, message])

    return jsonify({"ok": True, "message": "Feedback received!"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860, debug=False)
