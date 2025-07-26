import pandas as pd
import ta

def compute_indicators(df):
    """
    Compute RSI, MACD, EMA indicators and return signals.
    Returns a dict with indicator values and bullish/bearish signals.
    """
    result = {}
    if df is None or df.empty:
        return result
    close = df['Close'].squeeze()  # Ensure 1D Series
    # RSI
    rsi = ta.momentum.RSIIndicator(close).rsi()
    result['rsi'] = rsi.iloc[-1]
    result['rsi_signal'] = 'bullish' if rsi.iloc[-1] < 30 else 'bearish' if rsi.iloc[-1] > 70 else 'neutral'
    # MACD
    macd = ta.trend.MACD(close)
    macd_diff = macd.macd_diff().iloc[-1]
    result['macd'] = macd.macd().iloc[-1]
    result['macd_signal'] = 'bullish' if macd_diff > 0 else 'bearish' if macd_diff < 0 else 'neutral'
    # EMA
    ema_20 = ta.trend.EMAIndicator(close, window=20).ema_indicator().iloc[-1]
    ema_50 = ta.trend.EMAIndicator(close, window=50).ema_indicator().iloc[-1]
    result['ema_20'] = ema_20
    result['ema_50'] = ema_50
    result['ema_signal'] = 'bullish' if ema_20 > ema_50 else 'bearish' if ema_20 < ema_50 else 'neutral'
    return result 