# Enhanced Module Documentation

The crypto_bot_fusion_v1 project includes several enhanced modules that use real data APIs when available and fall back to simulated data when necessary:

## enhanced_news_fetcher.py
- Uses the CryptoCompare News API to fetch real cryptocurrency news
- Analyzes sentiment based on actual news content
- Falls back to simulated news if the API is unavailable or fails

## enhanced_social_sentiment.py
- Connects to multiple social sentiment APIs (The Tie, StockGeist, Santiment)
- Provides real-time social media sentiment analysis for cryptocurrencies
- Includes fallback mechanisms to generate simulated data when APIs are unavailable

## predictive_technical_analysis.py
- Performs advanced technical analysis beyond basic indicators
- Identifies chart patterns and trend strength
- Detects support and resistance levels
- Provides conviction scores for trading decisions

## enhanced_gpt_fusion_analyzer.py
- Combines all data sources using GPT for intelligent analysis
- Integrates predictive technical analysis with market sentiment
- Provides detailed rationale for trading decisions
- Includes confidence and conviction metrics

## Setting Up API Keys

To use the enhanced modules with real data, add your API keys to the .env file: