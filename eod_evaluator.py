import yfinance as yf
import json
from datetime import datetime
import os

def fetch_eod_price(symbol):
    """
    Fetch the latest EOD close price for a stock symbol.
    """
    df = yf.download(symbol, period='2d', interval='1d', progress=False)
    if df.empty:
        return None
    return df['Close'].iloc[-1]

def evaluate_recommendations(recommendations, intraday_data):
    """
    For each recommended stock, check if the EOD price moved in the predicted direction.
    Returns a list of dicts with accuracy info.
    """
    results = []
    for rec in recommendations:
        symbol = rec['symbol']
        confidence = rec['confidence_score']
        reason = rec['reason']
        df = intraday_data.get(symbol)
        if df is None or df.empty:
            results.append({'symbol': symbol, 'result': 'no_data'})
            continue
        open_price = df['Open'].iloc[0]
        eod_price = fetch_eod_price(symbol)
        if eod_price is None:
            results.append({'symbol': symbol, 'result': 'no_eod'})
            continue
        # If confidence > 1, expect bullish; < 0, expect bearish
        expected = 'up' if confidence > 1 else 'down' if confidence < 0 else 'neutral'
        actual = 'up' if eod_price > open_price else 'down' if eod_price < open_price else 'neutral'
        accuracy = 'accurate' if expected == actual and expected != 'neutral' else 'miss'
        results.append({
            'symbol': symbol,
            'open_price': open_price,
            'eod_price': eod_price,
            'expected': expected,
            'actual': actual,
            'accuracy': accuracy,
            'reason': reason
        })
    return results

def log_results(timestamp, recommendations, eod_results, log_path='logs'):
    os.makedirs(log_path, exist_ok=True)
    log_data = {
        'timestamp': timestamp,
        'suggested_stocks': recommendations,
        'eod_performance': eod_results
    }
    fname = os.path.join(log_path, f'stock_log_{timestamp}.json')
    with open(fname, 'w') as f:
        json.dump(log_data, f, indent=2, default=str)
    return fname 