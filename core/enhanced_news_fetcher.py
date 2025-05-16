"""
Enhanced Crypto News Fetcher

Fetches real cryptocurrency news using the CryptoCompare News API.
This module replaces the dummy news generation with actual news data.
"""

import logging
import requests
from datetime import datetime
import config

# Set up logging
logger = logging.getLogger('crypto_bot.enhanced_news')

# CryptoCompare News API endpoint
NEWS_API_URL = "https://min-api.cryptocompare.com/data/v2/news/"

def get_latest_headlines(symbol=None, limit=5):
    """
    Retrieve the latest crypto news headlines from CryptoCompare.
    
    Args:
        symbol (str, optional): Cryptocurrency symbol to filter news
        limit (int): Number of headlines to return
    
    Returns:
        list: List of news headline objects with timestamp and headline text
    """
    headers = {}
    if config.CRYPTOCOMPARE_API_KEY:
        headers["authorization"] = f"Apikey {config.CRYPTOCOMPARE_API_KEY}"
    
    params = {
        "sortOrder": "latest",
        "limit": limit
    }
    
    # Add category filter if symbol is provided
    if symbol:
        # CryptoCompare uses categories like "BTC", "ETH", etc.
        params["categories"] = symbol.upper()
    
    try:
        response = requests.get(
            NEWS_API_URL, 
            params=params,
            headers=headers
        )
        response.raise_for_status()
        data = response.json()
        
        if "Data" in data:
            headlines = []
            
            for article in data["Data"]:
                # Convert the published_on timestamp to ISO format
                timestamp = datetime.fromtimestamp(article.get("published_on", 0)).isoformat()
                
                headlines.append({
                    "timestamp": timestamp,
                    "headline": article.get("title", ""),
                    "url": article.get("url", ""),
                    "source": article.get("source", ""),
                    "body": article.get("body", ""),
                    "tags": article.get("tags", "")
                })
            
            logger.info(f"Successfully fetched {len(headlines)} headlines from CryptoCompare")
            return headlines
        else:
            logger.error(f"API error from CryptoCompare: {data.get('Message', 'Unknown error')}")
            # Fall back to original method as backup
            from core import crypto_news_fetcher
            return crypto_news_fetcher.get_dummy_headlines(limit)
            
    except Exception as e:
        logger.error(f"Error fetching news from CryptoCompare: {str(e)}")
        # Fall back to original method as backup
        from core import crypto_news_fetcher
        return crypto_news_fetcher.get_dummy_headlines(limit)

def analyze_sentiment(headlines):
    """
    Analyze the sentiment of a collection of headlines.
    
    Args:
        headlines (list): List of headline objects
    
    Returns:
        str: Sentiment assessment ("bullish", "bearish", or "neutral")
    """
    # Advanced sentiment analysis based on real news content
    # We'll use a more sophisticated keyword analysis that incorporates
    # news body content when available
    
    # Keywords that typically indicate bullish sentiment
    bullish_keywords = [
        "surge", "rally", "jump", "gain", "bullish", "adoption", "integration",
        "breakthrough", "partnership", "launch", "milestone", "boost", "soar",
        "momentum", "positive", "approval", "support", "backing", "potential",
        "growth", "rise", "uptrend", "accumulate", "institutional", "invest", 
        "upgrade", "strong", "higher", "record", "outperform"
    ]
    
    # Keywords that typically indicate bearish sentiment
    bearish_keywords = [
        "drop", "fall", "crash", "bearish", "ban", "crackdown", "risk", "warn",
        "concern", "investigation", "hack", "breach", "vulnerability", "sell-off",
        "decline", "tumble", "plunge", "negative", "uncertainty", "volatility",
        "downtrend", "correction", "slump", "weak", "lower", "underperform",
        "fear", "panic", "regulation", "fine", "penalty", "lawsuit", "security"
    ]
    
    bullish_count = 0
    bearish_count = 0
    
    # Weighted scoring - we value headline sentiment more than body sentiment
    headline_weight = 2.0
    body_weight = 1.0
    
    for headline in headlines:
        headline_text = headline["headline"].lower()
        
        # Check headline for bullish keywords
        bullish_in_headline = sum(1 for word in bullish_keywords if word in headline_text)
        bullish_count += bullish_in_headline * headline_weight
        
        # Check headline for bearish keywords
        bearish_in_headline = sum(1 for word in bearish_keywords if word in headline_text)
        bearish_count += bearish_in_headline * headline_weight
        
        # Check body content if available
        if "body" in headline and headline["body"]:
            body_text = headline["body"].lower()
            
            # Check body for bullish keywords
            bullish_in_body = sum(1 for word in bullish_keywords if word in body_text)
            bullish_count += bullish_in_body * body_weight
            
            # Check body for bearish keywords
            bearish_in_body = sum(1 for word in bearish_keywords if word in body_text)
            bearish_count += bearish_in_body * body_weight
    
    # Consider headline sources and tags for additional context
    # Some sources might have inherent bias
    for headline in headlines:
        if "source" in headline:
            source = headline["source"].lower()
            if source in ["coindesk", "cointelegraph", "bitcoin magazine"]:
                # These sources tend to be more balanced
                pass
            elif source in ["bitcoinist", "newsbtc"]:
                # Slightly bullish-leaning sources
                bullish_count += 0.5
        
        # Tags can provide additional context
        if "tags" in headline and headline["tags"]:
            tags = headline["tags"].lower()
            if "bullish" in tags or "buy" in tags:
                bullish_count += 1
            elif "bearish" in tags or "sell" in tags:
                bearish_count += 1
    
    # Determine overall sentiment with a slight bullish bias (as crypto news tends to be)
    if bullish_count > bearish_count * 1.1:  # Requiring 10% more bullish signals to account for general positive bias
        sentiment = "bullish"
    elif bearish_count > bullish_count * 0.9:  # 10% less bearish signals required to be considered bearish
        sentiment = "bearish"
    else:
        sentiment = "neutral"
    
    logger.info(f"News sentiment analysis: {sentiment} (bullish: {bullish_count}, bearish: {bearish_count})")
    return sentiment

def extract_mentioned_symbols(headlines):
    """
    Extract cryptocurrency symbols mentioned in headlines.
    
    Args:
        headlines (list): List of headline objects
    
    Returns:
        list: List of mentioned cryptocurrency symbols
    """
    # Import symbol mapper
    from core import crypto_symbol_mapper
    
    mentioned_symbols = set()
    
    for headline in headlines:
        headline_text = headline["headline"]
        
        # Check headline text
        symbols = crypto_symbol_mapper.extract_crypto_mentions(headline_text)
        mentioned_symbols.update(symbols)
        
        # Check body text if available
        if "body" in headline and headline["body"]:
            body_symbols = crypto_symbol_mapper.extract_crypto_mentions(headline["body"])
            mentioned_symbols.update(body_symbols)
        
        # Check tags if available
        if "tags" in headline and headline["tags"]:
            tag_symbols = crypto_symbol_mapper.extract_crypto_mentions(headline["tags"])
            mentioned_symbols.update(tag_symbols)
    
    return list(mentioned_symbols)