import asyncio
from datetime import datetime, timedelta
from data_fetcher import fetch_all_intraday, NIFTY_20
from technical_analyzer import compute_indicators
from news_analyzer import analyze_all_news
from recommender import recommend_stocks
import yfinance as yf
import pandas as pd
import numpy as np

def safe_scalar(val):
    if isinstance(val, (pd.Series, np.ndarray)):
        val = val.item() if val.size == 1 else None
    if val is None or pd.isna(val):
        return '-'
    return f"{val:.2f}" if isinstance(val, float) or isinstance(val, int) else str(val)

def get_today_close(symbol):
    df = yf.download(symbol, period='2d', interval='1d', progress=False)
    if df.empty:
        return None
    return df['Close'].iloc[-1]

def get_next_day_min_max(symbol):
    today = datetime.now().date()
    next_day = today + timedelta(days=1)
    df = yf.download(symbol, start=str(next_day), end=str(next_day + timedelta(days=1)), interval='5m', progress=False)
    if df is None or df.empty or 'Close' not in df:
        return None, None
    min_price = df['Close'].min()
    max_price = df['Close'].max()
    return min_price, max_price

def predict_close(rule_confidence, close):
    if close is None or close == '-':
        return '-'
    try:
        close = float(close)
    except Exception:
        return '-'
    if rule_confidence >= 2:
        return f"{close * 1.01:.2f}"  # +1%
    elif rule_confidence <= 0:
        return f"{close * 0.99:.2f}"  # -1%
    else:
        return f"{close:.2f}"

def print_tomorrow_recommendations():
    print("Fetching intraday data...")
    intraday_data = fetch_all_intraday(NIFTY_20)
    print("Computing technical indicators...")
    technical_scores = {symbol: compute_indicators(df) for symbol, df in intraday_data.items()}
    print("Analyzing news and sentiment...")
    sentiment_scores = asyncio.run(analyze_all_news(NIFTY_20))
    print("\nTomorrow's Top 5 Intraday Stock Picks:")
    print(f"{'#':<2} {'Symbol':<15} {'Conf':<6} {'Close':<10} {'Pred Close':<12} {'Next Min':<10} {'Next Max':<10} Reason")
    recommendations = recommend_stocks(technical_scores, sentiment_scores)
    for idx, rec in enumerate(recommendations, 1):
        symbol = rec['symbol']
        close = safe_scalar(get_today_close(symbol))
        min_price, max_price = get_next_day_min_max(symbol)
        min_price = safe_scalar(min_price)
        max_price = safe_scalar(max_price)
        pred_close = predict_close(rec['confidence_score'], close)
        print(f"{idx:<2} {symbol:<15} {rec['confidence_score']:<6} {close:<10} {pred_close:<12} {min_price:<10} {max_price:<10} {rec['reason']}")

if __name__ == "__main__":
    print_tomorrow_recommendations() 