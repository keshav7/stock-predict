def recommend_stocks(technical_scores, sentiment_scores, top_n=5):
    """
    Combine technical and sentiment scores to recommend top stocks.
    Returns a list of dicts: symbol, confidence_score, reason.
    """
    recommendations = []
    for symbol in technical_scores:
        tech = technical_scores[symbol]
        sent = sentiment_scores.get(symbol, {'sentiment_score': 0.0})
        # Simple scoring: bullish signals + sentiment
        score = 0
        reasons = []
        if tech.get('rsi_signal') == 'bullish':
            score += 1
            reasons.append('RSI bullish')
        if tech.get('macd_signal') == 'bullish':
            score += 1
            reasons.append('MACD bullish')
        if tech.get('ema_signal') == 'bullish':
            score += 1
            reasons.append('EMA bullish')
        score += sent['sentiment_score']  # sentiment is -1 to 1
        if sent['sentiment_score'] > 0.2:
            reasons.append('Positive news sentiment')
        elif sent['sentiment_score'] < -0.2:
            reasons.append('Negative news sentiment')
        recommendations.append({
            'symbol': symbol,
            'confidence_score': round(score, 2),
            'reason': ' + '.join(reasons) or 'Neutral'
        })
    # Sort by confidence_score descending
    recommendations.sort(key=lambda x: x['confidence_score'], reverse=True)
    return recommendations[:top_n] 