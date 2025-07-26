import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

# Top 20 Nifty 50 stocks (sample, can be updated)
NIFTY_20 = [
    'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ICICIBANK.NS',
    'HINDUNILVR.NS', 'ITC.NS', 'LT.NS', 'SBIN.NS', 'BHARTIARTL.NS',
    'BAJFINANCE.NS', 'KOTAKBANK.NS', 'ASIANPAINT.NS', 'HCLTECH.NS',
    'MARUTI.NS', 'AXISBANK.NS', 'SUNPHARMA.NS', 'TITAN.NS',
    'ULTRACEMCO.NS', 'WIPRO.NS'
]

def fetch_intraday_data(symbol, period_days=60, interval='5m'):
    """
    Fetch 5-min interval data for the past `period_days` for a given stock symbol.
    Returns a pandas DataFrame.
    """
    period = f"{period_days}d"
    df = yf.download(symbol, period=period, interval=interval, progress=False)
    df = df.reset_index()
    return df

def fetch_all_intraday(symbols=NIFTY_20, period_days=60, interval='5m'):
    """
    Fetch intraday data for all symbols. Returns a dict of DataFrames.
    """
    data = {}
    for symbol in symbols:
        try:
            df = fetch_intraday_data(symbol, period_days, interval)
            data[symbol] = df
        except Exception as e:
            print(f"Error fetching {symbol}: {e}")
    return data 