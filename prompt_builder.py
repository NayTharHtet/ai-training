def build_prompt(ticker: str, data: dict, horizon: int) -> str:
    return (
        f"You are a financial risk analyst. Here is real market data for {ticker}:\n"
        f"- Current price: ${data['current_price']}\n"
        f"- Previous close: ${data['prev_close']}\n"
        f"- Day change: {data['day_change_pct']}%\n"
        f"- 5-day SMA: ${data['sma_5']}\n"
        f"- 20-day SMA: ${data['sma_20']}\n"
        f"- 5-day return: {data['five_day_return']}%\n"
        f"- Volatility: {data['volatility']}%\n\n"
        f"Analyze the next {horizon} day(s). "
        f"Return JSON only with keys: "
        f"risk_level (LOW/MEDIUM/HIGH), "
        f"pct_change (float between -5 and 5), "
        f"trend (UP/DOWN/SIDEWAYS), "
        f"insight (max 1 sentence). "
        f"No text outside JSON."
    )
