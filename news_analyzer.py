import requests
import openai
import asyncio
import os
from datetime import datetime, timedelta

NEWS_API_KEY = os.getenv('NEWS_API_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
openai.api_key = OPENAI_API_KEY

NEWS_ENDPOINT = 'https://newsapi.org/v2/everything'

async def fetch_news(symbol, session, days=2):
    """
    Fetch recent news articles for a stock symbol using NewsAPI.
    """
    if not NEWS_API_KEY:
        raise RuntimeError("NEWS_API_KEY environment variable not set.")
    params = {
        'q': symbol.replace('.NS', ''),
        'from': (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d'),
        'sortBy': 'publishedAt',
        'apiKey': str(NEWS_API_KEY),
        'language': 'en',
        'pageSize': 5
    }
    # Remove any None values
    params = {k: v for k, v in params.items() if v is not None}
    async with session.get(NEWS_ENDPOINT, params=params) as resp:
        data = await resp.json()
        return data.get('articles', [])

async def summarize_article(article_text):
    """
    Summarize article using OpenAI API (GPT-3.5-turbo).
    """
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": f"Summarize this news article: {article_text}"}],
            max_tokens=60
        )
        return response['choices'][0]['message']['content']
    except Exception as e:
        return ""

async def analyze_sentiment(text):
    """
    Use OpenAI API to analyze sentiment. Returns score (-1 to 1).
    """
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": f"What is the sentiment of this news? Respond with a number between -1 (very negative) and 1 (very positive): {text}"}],
            max_tokens=10
        )
        score = float(response['choices'][0]['message']['content'].strip())
        return score
    except Exception as e:
        return 0.0

async def analyze_news_for_stock(symbol, session):
    articles = await fetch_news(symbol, session)
    summaries = []
    sentiment_scores = []
    for article in articles:
        summary = await summarize_article(article.get('content', '') or article.get('description', ''))
        summaries.append(summary)
        sentiment = await analyze_sentiment(summary)
        sentiment_scores.append(sentiment)
    avg_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0.0
    return {
        'symbol': symbol,
        'summaries': summaries,
        'sentiment_score': avg_sentiment
    }

async def analyze_all_news(symbols):
    import aiohttp
    results = []
    async with aiohttp.ClientSession() as session:
        tasks = [analyze_news_for_stock(symbol, session) for symbol in symbols]
        results = await asyncio.gather(*tasks)
    return {r['symbol']: r for r in results} 