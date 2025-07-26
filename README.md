# Stock Analysis Agent

A Python-based agent for daily intraday stock analysis and recommendation for top Indian stocks (Nifty 50).

## Features
- Fetches 5-min interval intraday data for top 20 Nifty 50 stocks (60 days)
- Computes technical indicators (RSI, MACD, EMA)
- Fetches and summarizes news, performs sentiment analysis (OpenAI API)
- Combines technical and sentiment scores for recommendations
- Evaluates EOD accuracy and logs results
- Outputs daily summary to terminal and JSON log

## Setup
1. **Clone the repo**
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Set API keys** (required for news and LLM):
   - `NEWS_API_KEY` (from https://newsapi.org/)
   - `OPENAI_API_KEY` (from https://platform.openai.com/)
   
   You can set them in your shell or a `.env` file:
   ```bash
   export NEWS_API_KEY=your_newsapi_key
   export OPENAI_API_KEY=your_openai_key
   ```

## Usage
```bash
python main.py
```

## Module Structure
- `data_fetcher.py` – Fetches 5-min interval data for 60 days
- `technical_analyzer.py` – Computes RSI, MACD, EMA and signals
- `news_analyzer.py` – Fetches news, summarizes, and analyzes sentiment (async)
- `recommender.py` – Combines scores and recommends top stocks
- `eod_evaluator.py` – Evaluates EOD accuracy and logs results
- `main.py` – Orchestrates the workflow

## Notes
- News and LLM APIs may incur costs or rate limits
- All logs are saved in the `logs/` directory
- You can update the stock list in `data_fetcher.py` (`NIFTY_20`) 