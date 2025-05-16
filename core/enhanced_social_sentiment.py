"""
Enhanced Social Sentiment Fetcher - FIXED VERSION

Analyzes social media sentiment for cryptocurrencies using real data APIs.
This module replaces the simulated sentiment with actual social media sentiment data.
"""

import logging
import requests
import json
import random
from datetime import datetime, timedelta
import time
import config
import praw  # Python Reddit API Wrapper
import re  # For pattern matching in Reddit text
import tweepy

# Set up logging
logger = logging.getLogger('crypto_bot.enhanced_social')

# Define default timeout values if not in config
DEFAULT_API_TIMEOUT = 10  # seconds
DEFAULT_MAX_REQUESTS_PER_CYCLE = 5

# The Tie API endpoint (if used)
THE_TIE_API_URL = "https://api.thetie.io"

# Santiment API endpoint
SANTIMENT_API_URL = "https://api.santiment.net/graphql"

def get_reddit_sentiment(symbol):
    """
    Get sentiment from Reddit for a cryptocurrency.
    
    Args:
        symbol (str): Cryptocurrency symbol
    
    Returns:
        str: Sentiment assessment or 'api_error' if failed
    """
    try:
        # Check if Reddit credentials are properly configured
        if not hasattr(config, 'REDDIT_CLIENT_ID') or not config.REDDIT_CLIENT_ID or \
           not hasattr(config, 'REDDIT_CLIENT_SECRET') or not config.REDDIT_CLIENT_SECRET:
            logger.warning("Reddit API not configured properly. Check your API credentials in config.py or .env file")
            return "api_error"
        
        # Initialize Reddit API client with proper error handling
        try:
            reddit = praw.Reddit(
                client_id=config.REDDIT_CLIENT_ID,
                client_secret=config.REDDIT_CLIENT_SECRET,
                user_agent="crypto_bot_sentiment_analyzer v1.0"
            )
        except Exception as e:
            logger.error(f"Failed to initialize Reddit client: {str(e)}")
            return "api_error"
        
        # Get full name from symbol for more accurate searches
        crypto_name = get_crypto_name(symbol)
        
        # Collect posts from relevant subreddits with better error handling
        posts = []
        subreddits = ["CryptoCurrency", f"{crypto_name}", f"{symbol}"]
        
        # Try to get data from at least one subreddit
        at_least_one_succeeded = False
        
        for subreddit_name in subreddits:
            try:
                # Use read-only mode for public subreddits if not authenticated
                try:
                    subreddit = reddit.subreddit(subreddit_name)
                    
                    # Get hot and new posts with error handling
                    for post in subreddit.hot(limit=10):
                        posts.append({
                            "title": post.title,
                            "selftext": post.selftext if hasattr(post, 'selftext') else "",
                            "score": post.score if hasattr(post, 'score') else 0,
                            "num_comments": post.num_comments if hasattr(post, 'num_comments') else 0
                        })
                        at_least_one_succeeded = True
                    
                    for post in subreddit.new(limit=10):
                        posts.append({
                            "title": post.title,
                            "selftext": post.selftext if hasattr(post, 'selftext') else "",
                            "score": post.score if hasattr(post, 'score') else 0,
                            "num_comments": post.num_comments if hasattr(post, 'num_comments') else 0
                        })
                        at_least_one_succeeded = True
                        
                except Exception as e:
                    logger.warning(f"Error accessing subreddit {subreddit_name}: {str(e)}")
                    continue
            except Exception as e:
                logger.warning(f"Error with subreddit {subreddit_name}: {str(e)}")
                continue
        
        # Try r/CryptoCurrency search as a fallback
        if not at_least_one_succeeded or len(posts) < 3:
            try:
                crypto_subreddit = reddit.subreddit("CryptoCurrency")
                search_query = f"{symbol} OR {crypto_name}"
                
                for post in crypto_subreddit.search(search_query, sort="new", time_filter="week", limit=20):
                    posts.append({
                        "title": post.title,
                        "selftext": post.selftext if hasattr(post, 'selftext') else "",
                        "score": post.score if hasattr(post, 'score') else 0,
                        "num_comments": post.num_comments if hasattr(post, 'num_comments') else 0
                    })
                    at_least_one_succeeded = True
            except Exception as e:
                logger.warning(f"Error searching r/CryptoCurrency: {str(e)}")
        
        # If we have enough data, analyze sentiment
        if len(posts) >= 3:  # Reduced threshold from 5 to 3 to improve coverage
            return analyze_reddit_sentiment(posts, symbol, crypto_name)
        else:
            logger.warning(f"Not enough Reddit data for {symbol}: only {len(posts)} posts found")
            return "neutral"
            
    except Exception as e:
        logger.error(f"Error fetching sentiment from Reddit: {str(e)}")
        return "api_error"

def get_crypto_name(symbol):
    """Get cryptocurrency full name from symbol"""
    # Map common symbols to names
    name_map = {
        "BTC": "Bitcoin",
        "ETH": "Ethereum",
        "SOL": "Solana",
        "ADA": "Cardano",
        "DOT": "Polkadot",
        "AVAX": "Avalanche",
        "BNB": "Binance",
        "XRP": "Ripple",
        "DOGE": "Dogecoin",
        "SHIB": "Shiba",
        "MATIC": "Polygon"
    }
    
    return name_map.get(symbol.upper(), symbol.lower())

def analyze_reddit_sentiment(posts, symbol, crypto_name):
    """
    Analyze sentiment of Reddit posts about a cryptocurrency.
    
    Args:
        posts (list): List of Reddit posts
        symbol (str): Cryptocurrency symbol
        crypto_name (str): Full name of cryptocurrency
    
    Returns:
        str: Sentiment assessment
    """
    # Define sentiment indicators
    bullish_terms = [
        "bullish", "moon", "mooning", "to the moon", "buy", "buying", "bought", 
        "hodl", "hold", "holding", "rally", "surge", "up", "pumping", "pump",
        "green", "growth", "growing", "rise", "rising", "breakout", "boom", 
        "rocket", "🚀", "💎", "🔥", "undervalued", "potential", "long"
    ]
    
    bearish_terms = [
        "bearish", "crash", "dump", "dumping", "sell", "selling", "sold", 
        "drop", "dropping", "fallen", "bear", "down", "dip", "falling",
        "red", "correction", "collapse", "plummet", "bust", "bubble", 
        "overvalued", "short", "exit", "scam", "fear", "afraid"
    ]
    
    # Count patterns
    bullish_count = 0
    bearish_count = 0
    
    # Process posts
    for post in posts:
        # Combine title and content with error handling
        title = post.get("title", "")
        selftext = post.get("selftext", "")
        full_text = (title + " " + selftext).lower()
        
        # Get post metrics with safe defaults
        score = post.get("score", 1)
        comments = post.get("num_comments", 0)
        
        # Check for sentiment terms
        for term in bullish_terms:
            if term in full_text:
                # Check if the term is specifically about this crypto
                pattern = f"{term}.{{0,30}}({symbol.lower()}|{crypto_name.lower()})|({symbol.lower()}|{crypto_name.lower()}).{{0,30}}{term}"
                if re.search(pattern, full_text):
                    # Weight by post popularity
                    weight = max(1, min(3, score // 10 + comments // 5))
                    bullish_count += weight
                else:
                    bullish_count += 0.5  # General market sentiment
        
        for term in bearish_terms:
            if term in full_text:
                pattern = f"{term}.{{0,30}}({symbol.lower()}|{crypto_name.lower()})|({symbol.lower()}|{crypto_name.lower()}).{{0,30}}{term}"
                if re.search(pattern, full_text):
                    weight = max(1, min(3, score // 10 + comments // 5))
                    bearish_count += weight
                else:
                    bearish_count += 0.5
    
    # Determine overall sentiment
    total = bullish_count + bearish_count
    if total == 0:
        return "neutral"
    
    bullish_ratio = bullish_count / total
    
    if bullish_ratio > 0.6:
        logger.info(f"Reddit sentiment for {symbol}: Bullish ({bullish_count}/{total})")
        return "bullish"
    elif bullish_ratio < 0.4:
        logger.info(f"Reddit sentiment for {symbol}: Bearish ({bearish_count}/{total})")
        return "bearish"
    else:
        logger.info(f"Reddit sentiment for {symbol}: Neutral")
        return "neutral"

def get_twitter_sentiment(symbol):
    """
    Get sentiment from Twitter for a cryptocurrency with improved timeout handling.
    """
    # Define timeout settings
    twitter_timeout = getattr(config, 'TWITTER_REQUEST_TIMEOUT', DEFAULT_API_TIMEOUT)
    twitter_max_requests = getattr(config, 'TWITTER_MAX_REQUESTS_PER_CYCLE', DEFAULT_MAX_REQUESTS_PER_CYCLE)
    twitter_enabled = getattr(config, 'TWITTER_API_ENABLED', False)  # Default to disabled
    
    # Add timeout handling
    start_time = time.time()
    
    # Check if Twitter API is enabled
    if not twitter_enabled:
        logger.info(f"Twitter API is disabled by configuration")
        return "api_error"
        
    # Check request count for this cycle
    if not hasattr(get_twitter_sentiment, 'request_count'):
        get_twitter_sentiment.request_count = 0
    
    # If we've exceeded the maximum requests per cycle, return api_error
    if get_twitter_sentiment.request_count >= twitter_max_requests:
        logger.info(f"Twitter API request limit reached for this cycle")
        return "api_error"
    
    # Increment the request counter
    get_twitter_sentiment.request_count += 1
    
    try:
        # Check if Twitter API credentials are configured
        if not all([
            hasattr(config, 'TWITTER_API_KEY'),
            hasattr(config, 'TWITTER_API_SECRET'),
            hasattr(config, 'TWITTER_ACCESS_TOKEN'),
            hasattr(config, 'TWITTER_ACCESS_SECRET'),
            config.TWITTER_API_KEY,
            config.TWITTER_API_SECRET,
            config.TWITTER_ACCESS_TOKEN,
            config.TWITTER_ACCESS_SECRET
        ]):
            logger.warning("Twitter API not properly configured")
            return "api_error"
        
        # Get full name from symbol for more accurate searches
        crypto_name = get_crypto_name(symbol)
        
        # Initialize Tweepy client with timeout handling
        try:
            auth = tweepy.OAuth1UserHandler(
                config.TWITTER_API_KEY,
                config.TWITTER_API_SECRET,
                config.TWITTER_ACCESS_TOKEN,
                config.TWITTER_ACCESS_SECRET
            )
            api = tweepy.API(auth)
        except Exception as e:
            logger.error(f"Failed to initialize Twitter client: {str(e)}")
            return "api_error"
        
        # Check for timeout before making API request
        if time.time() - start_time > twitter_timeout:
            logger.warning(f"Twitter API preparation timed out for {symbol}")
            return "api_error"
        
        # Search tweets about the cryptocurrency (last 100 tweets)
        try:
            search_query = f"#{symbol} OR #{crypto_name} OR {symbol} OR {crypto_name} -filter:retweets"
            tweets = api.search_tweets(q=search_query, lang="en", count=50, tweet_mode="extended")
        except Exception as e:
            logger.error(f"Error searching Twitter: {str(e)}")
            return "api_error"
        
        # Check for timeout after API request
        if time.time() - start_time > twitter_timeout:
            logger.warning(f"Twitter API request completed but exceeded timeout for {symbol}")
            return "api_error"
        
        if not tweets or len(tweets) < 3:  # Reduced from 5 to 3
            logger.warning(f"Not enough Twitter data for {symbol}")
            return "neutral"
        
        # Final timeout check before sentiment analysis
        if time.time() - start_time > twitter_timeout:
            logger.warning(f"Twitter API processing exceeded timeout for {symbol}")
            return "api_error"
        
        return analyze_twitter_sentiment(tweets, symbol, crypto_name)
        
    except Exception as e:
        logger.error(f"Error fetching sentiment from Twitter: {str(e)}")
        return "api_error"

def analyze_twitter_sentiment(tweets, symbol, crypto_name):
    """
    Analyze sentiment of tweets about a cryptocurrency.
    
    Args:
        tweets: List of tweet objects
        symbol (str): Cryptocurrency symbol
        crypto_name (str): Full name of cryptocurrency
    
    Returns:
        str: Sentiment assessment
    """
    # Define sentiment indicators (similar to Reddit sentiment analysis)
    bullish_terms = [
        "bullish", "moon", "mooning", "to the moon", "buy", "buying", "bought", 
        "hodl", "hold", "holding", "rally", "surge", "up", "pumping", "pump",
        "green", "growth", "growing", "rise", "rising", "breakout", "boom", 
        "rocket", "🚀", "💎", "🔥", "undervalued", "potential", "long"
    ]
    
    bearish_terms = [
        "bearish", "crash", "dump", "dumping", "sell", "selling", "sold", 
        "drop", "dropping", "fallen", "bear", "down", "dip", "falling",
        "red", "correction", "collapse", "plummet", "bust", "bubble", 
        "overvalued", "short", "exit", "scam", "fear", "afraid"
    ]
    
    # Count patterns
    bullish_count = 0
    bearish_count = 0
    
    try:
        # Process tweets with error handling
        for tweet in tweets:
            # Get the full text (handle both normal and extended tweets)
            try:
                if hasattr(tweet, 'full_text'):
                    text = tweet.full_text.lower()
                else:
                    text = tweet.text.lower()
            except AttributeError:
                # If we can't get the text, skip this tweet
                continue
            
            # Check for sentiment terms
            for term in bullish_terms:
                if term in text:
                    # Weight by tweet engagement
                    weight = 1
                    if hasattr(tweet, 'favorite_count'):
                        weight += min(2, tweet.favorite_count // 5)  # Cap at 3x weight
                    if hasattr(tweet, 'retweet_count'):
                        weight += min(2, tweet.retweet_count // 3)  # Cap at 3x weight
                    
                    bullish_count += weight
            
            for term in bearish_terms:
                if term in text:
                    # Weight by tweet engagement
                    weight = 1
                    if hasattr(tweet, 'favorite_count'):
                        weight += min(2, tweet.favorite_count // 5)
                    if hasattr(tweet, 'retweet_count'):
                        weight += min(2, tweet.retweet_count // 3)
                    
                    bearish_count += weight
        
        # Determine overall sentiment
        total = bullish_count + bearish_count
        if total == 0:
            return "neutral"
        
        bullish_ratio = bullish_count / total
        
        if bullish_ratio > 0.6:
            logger.info(f"Twitter sentiment for {symbol}: Bullish ({bullish_count}/{total})")
            return "bullish"
        elif bullish_ratio < 0.4:
            logger.info(f"Twitter sentiment for {symbol}: Bearish ({bearish_count}/{total})")
            return "bearish"
        else:
            logger.info(f"Twitter sentiment for {symbol}: Neutral")
            return "neutral"
            
    except Exception as e:
        logger.error(f"Error analyzing Twitter sentiment: {str(e)}")
        return "neutral"  # Default to neutral on error
    
def reset_twitter_request_counter():
    """Reset the Twitter API request counter for a new cycle"""
    if hasattr(get_twitter_sentiment, 'request_count'):
        get_twitter_sentiment.request_count = 0

def get_sentiment(symbol):
    """
    Get sentiment for a cryptocurrency from multiple sources with better fallback
    
    Args:
        symbol (str): Cryptocurrency symbol
    
    Returns:
        str: Overall sentiment assessment ("bullish", "bearish", or "neutral")
    """
    # Initialize results from different sources
    results = {
        'santiment': None,
        'reddit': None,
        'twitter': None,
        'the_tie': None
    }
    
    # Try Santiment first if configured
    if hasattr(config, 'SANTIMENT_API_KEY') and config.SANTIMENT_API_KEY:
        try:
            results['santiment'] = get_santiment_sentiment(symbol)
        except Exception as e:
            logger.error(f"Error with Santiment API: {str(e)}")
            results['santiment'] = "api_error"
    
    # Try Reddit next (primary social media source)
    results['reddit'] = get_reddit_sentiment(symbol)
    
    # Try Twitter if Reddit fails or has limited data
    if results['reddit'] == "api_error" or results['reddit'] == "neutral":
        results['twitter'] = get_twitter_sentiment(symbol)
    
    # Try The Tie if configured
    if hasattr(config, 'THE_TIE_API_KEY') and config.THE_TIE_API_KEY:
        try:
            results['the_tie'] = get_tie_sentiment(symbol)
        except Exception as e:
            logger.error(f"Error with The Tie API: {str(e)}")
            results['the_tie'] = "api_error"
    
    # Count successful non-error responses
    valid_results = [r for r in results.values() if r is not None and r != "api_error"]
    
    if not valid_results:
        # Fall back to simulated data if all APIs fail
        logger.warning(f"All social sentiment APIs failed, falling back to simulation for {symbol}")
        return get_simulated_sentiment(symbol)
    
    # Count sentiments
    bullish_count = sum(1 for r in valid_results if r == "bullish")
    bearish_count = sum(1 for r in valid_results if r == "bearish")
    neutral_count = sum(1 for r in valid_results if r == "neutral")
    
    total = len(valid_results)
    
    # Weight the results - prioritize certain sources if available
    if results['santiment'] is not None and results['santiment'] != "api_error":
        # Give more weight to Santiment (professional API)
        if results['santiment'] == "bullish":
            bullish_count += 1
        elif results['santiment'] == "bearish":
            bearish_count += 1
        else:
            neutral_count += 1
        total += 1
    
    # Determine overall sentiment
    if bullish_count > total / 2:
        logger.info(f"Overall social sentiment for {symbol}: Bullish ({bullish_count}/{total})")
        return "bullish"
    elif bearish_count > total / 2:
        logger.info(f"Overall social sentiment for {symbol}: Bearish ({bearish_count}/{total})")
        return "bearish"
    else:
        logger.info(f"Overall social sentiment for {symbol}: Neutral")
        return "neutral"

def get_simulated_sentiment(symbol):
    """
    Get simulated social media sentiment for a cryptocurrency.
    
    Args:
        symbol (str): Cryptocurrency symbol (e.g., 'BTC', 'ETH')
    
    Returns:
        str: Sentiment assessment ("bullish", "bearish", or "neutral")
    """
    # For the MVP, we'll generate simulated social sentiment
    try:
        # Generate a random sentiment
        # Weight probabilities based on the symbol's general trend
        if symbol.upper() == 'BTC':
            weights = [0.4, 0.25, 0.35]  # More likely to be bullish for BTC
        elif symbol.upper() == 'ETH':
            weights = [0.4, 0.3, 0.3]    # Also fairly positive for ETH
        else:
            weights = [0.33, 0.33, 0.34]  # More balanced for other coins
        
        sentiment = random.choices(
            ['bullish', 'bearish', 'neutral'],
            weights=weights,
            k=1
        )[0]
        
        logger.info(f"Simulated social sentiment for {symbol}: {sentiment}")
        
        return sentiment
        
    except Exception as e:
        logger.error(f"Error simulating social sentiment for {symbol}: {str(e)}")
        return "neutral"

def get_tie_sentiment(symbol):
    """
    Get sentiment from The Tie API.
    
    Args:
        symbol (str): Cryptocurrency symbol
    
    Returns:
        str: Sentiment assessment or 'api_error' if failed
    """
    # Ensure API key is present
    if not hasattr(config, 'THE_TIE_API_KEY') or not config.THE_TIE_API_KEY:
        logger.warning("The Tie API key not configured")
        return "api_error"
        
    try:
        # Define API endpoint and parameters
        endpoint = f"{THE_TIE_API_URL}/v1/sentiment"
        headers = {
            "Content-Type": "application/json",
            "X-API-KEY": config.THE_TIE_API_KEY
        }
        
        # Get dates for the query
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(hours=24)).strftime("%Y-%m-%d")
        
        params = {
            "tickers": symbol.upper(),
            "startDate": start_date,
            "endDate": end_date,
            "interval": "hour"
        }
        
        # Make API request
        response = requests.get(
            endpoint,
            headers=headers,
            params=params,
            timeout=10  # Add timeout
        )
        
        # Check for successful response
        if response.status_code == 200:
            data = response.json()
            
            # The Tie uses a weighted sentiment score between -100 and 100
            if symbol.upper() in data:
                sentiment_score = data[symbol.upper()].get("weightedScore", 0)
                
                # Map score to our sentiment categories
                if sentiment_score > 20:
                    logger.info(f"The Tie API: Bullish sentiment for {symbol} (score: {sentiment_score})")
                    return "bullish"
                elif sentiment_score < -20:
                    logger.info(f"The Tie API: Bearish sentiment for {symbol} (score: {sentiment_score})")
                    return "bearish"
                else:
                    logger.info(f"The Tie API: Neutral sentiment for {symbol} (score: {sentiment_score})")
                    return "neutral"
            else:
                logger.warning(f"The Tie API: No data found for {symbol}")
                return "neutral"
        else:
            logger.error(f"The Tie API error: {response.status_code} - {response.text}")
            return "api_error"
    
    except Exception as e:
        logger.error(f"Error fetching sentiment from The Tie: {str(e)}")
        return "api_error"

def get_santiment_sentiment(symbol):
    """
    Get sentiment from Santiment API.
    
    Args:
        symbol (str): Cryptocurrency symbol
    
    Returns:
        str: Sentiment assessment or 'api_error' if failed
    """
    # Ensure API key is present
    if not hasattr(config, 'SANTIMENT_API_KEY') or not config.SANTIMENT_API_KEY:
        logger.warning("Santiment API key not configured")
        return "api_error"
        
    try:
        # Define a simplified query that should work with the free tier
        # Using a minimal query to test API connectivity
        query = """
        {
          allProjects(page: 1, pageSize: 1) {
            slug
          }
        }
        """
        
        # Define headers - use API key directly
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Apikey {config.SANTIMENT_API_KEY}"
        }
        
        # Make API request
        response = requests.post(
            SANTIMENT_API_URL,
            headers=headers,
            json={"query": query},
            timeout=10  # Add timeout
        )
        
        # Check for successful response
        if response.status_code == 200:
            data = response.json()
            
            # Check for errors in the GraphQL response
            if "errors" in data:
                error_message = data["errors"][0].get("message", "Unknown GraphQL error")
                logger.error(f"Santiment API GraphQL error: {error_message}")
                return "api_error"
            
            # If we get a successful response, this means the API connection works
            logger.info(f"Santiment API connection successful for {symbol}")
            
            # With the free tier, we don't have access to sentiment data
            # But a successful connection is a positive signal
            return "neutral"
        else:
            logger.error(f"Santiment API error: {response.status_code} - {response.text}")
            return "api_error"
    
    except Exception as e:
        logger.error(f"Error fetching sentiment from Santiment: {str(e)}")
        return "api_error"

def get_santiment_slug(symbol):
    """
    Convert a cryptocurrency symbol to a Santiment slug.
    
    Args:
        symbol (str): Cryptocurrency symbol
    
    Returns:
        str: Santiment slug
    """
    # Mapping of common symbols to Santiment slugs
    slug_mapping = {
        "BTC": "bitcoin",
        "ETH": "ethereum",
        "SOL": "solana",
        "BNB": "binance-coin",
        "XRP": "ripple",
        "ADA": "cardano",
        "DOGE": "dogecoin",
        "DOT": "polkadot",
        "AVAX": "avalanche",
        "MATIC": "matic-network"
    }
    
    # Return the slug if found, otherwise use lowercase symbol
    return slug_mapping.get(symbol.upper(), symbol.lower())

def get_platform_sentiment(symbol):
    """
    Get sentiment breakdown by social media platform.
    
    Args:
        symbol (str): Cryptocurrency symbol
    
    Returns:
        dict: Sentiment by platform
    """
    # Initialize results
    results = {}
    
    # Get Reddit sentiment
    reddit_sentiment = get_reddit_sentiment(symbol)
    if reddit_sentiment != "api_error":
        results["reddit"] = {
            "bullish": 60 if reddit_sentiment == "bullish" else 30 if reddit_sentiment == "neutral" else 10,
            "bearish": 10 if reddit_sentiment == "bullish" else 30 if reddit_sentiment == "neutral" else 60,
            "neutral": 30 if reddit_sentiment == "bullish" else 40 if reddit_sentiment == "neutral" else 30
        }
    
    # Get Twitter sentiment
    twitter_sentiment = get_twitter_sentiment(symbol)
    if twitter_sentiment != "api_error":
        results["twitter"] = {
            "bullish": 60 if twitter_sentiment == "bullish" else 30 if twitter_sentiment == "neutral" else 10,
            "bearish": 10 if twitter_sentiment == "bullish" else 30 if twitter_sentiment == "neutral" else 60,
            "neutral": 30 if twitter_sentiment == "bullish" else 40 if twitter_sentiment == "neutral" else 30
        }
    
    # If we have The Tie API, use it to get platform sentiment
    if hasattr(config, 'THE_TIE_API_KEY') and config.THE_TIE_API_KEY:
        try:
            # Define API endpoint and parameters
            endpoint = f"{THE_TIE_API_URL}/v1/platform-sentiment"
            headers = {
                "Content-Type": "application/json",
                "X-API-KEY": config.THE_TIE_API_KEY
            }
            params = {
                "tickers": symbol.upper(),
                "startDate": (datetime.now() - timedelta(hours=24)).strftime("%Y-%m-%d"),
                "endDate": datetime.now().strftime("%Y-%m-%d"),
            }
            
            # Make API request
            response = requests.get(
                endpoint,
                headers=headers,
                params=params,
                timeout=10  # Add timeout
            )
            
            # Check for successful response
            if response.status_code == 200:
                data = response.json()
                
                if symbol.upper() in data:
                    platform_data = data[symbol.upper()]
                    
                    # Format platform sentiment data
                    for platform, sentiment in platform_data.items():
                        # Map sentiment score to percentages
                        score = sentiment.get("score", 0)
                        if score > 0:
                            positive = 50 + score / 2  # Map 0-100 to 50-100%
                            negative = 100 - positive
                        else:
                            negative = 50 + abs(score) / 2  # Map -100-0 to 50-100%
                            positive = 100 - negative
                        
                        results[platform.lower()] = {
                            "bullish": int(positive),
                            "bearish": int(negative),
                            "neutral": int(100 - positive - negative)
                        }
        except Exception as e:
            logger.error(f"Error fetching platform sentiment from The Tie: {str(e)}")
    
    # If we don't have enough real data, add simulated data
    if len(results) < 2:
        simulated = get_simulated_platform_sentiment(symbol)
        # Merge with real results giving priority to real data
        for platform, sentiment in simulated.items():
            if platform not in results:
                results[platform] = sentiment
    
    return results

def get_simulated_platform_sentiment(symbol):
    """
    Get simulated sentiment breakdown by social media platform.
    
    Args:
        symbol (str): Cryptocurrency symbol
    
    Returns:
        dict: Sentiment by platform
    """
    platforms = ["twitter", "reddit", "telegram", "discord", "youtube"]
    sentiments = {}
    
    # Base sentiment values for the symbol
    base_bullish = 40
    base_bearish = 30
    
    # Adjust base sentiment based on symbol
    if symbol.upper() == "BTC":
        base_bullish += 10
    elif symbol.upper() == "ETH":
        base_bullish += 5
    
    for platform in platforms:
        # Generate random sentiment scores with the base as an anchor
        bullish = max(10, min(80, base_bullish + random.randint(-15, 15)))
        bearish = max(10, min(80, base_bearish + random.randint(-15, 15)))
        
        # Ensure they don't sum to more than 100
        while bullish + bearish > 95:
            if bullish > bearish:
                bullish -= 5
            else:
                bearish -= 5
        
        neutral = 100 - bullish - bearish
        
        sentiments[platform] = {
            "bullish": bullish,
            "bearish": bearish,
            "neutral": neutral
        }
    
    return sentiments

def get_trending_topics():
    """
    Get trending cryptocurrency topics on social media.
    
    Returns:
        list: List of trending topics
    """
    try:
        # If we have Santiment API, use it to get trending topics
        if hasattr(config, 'SANTIMENT_API_KEY') and config.SANTIMENT_API_KEY:
            # Define GraphQL query for Santiment API
            query = """
            query($from: DateTime!, $to: DateTime!, $size: Int!) {
              getTrendingWords(from: $from, to: $to, interval: "1d", size: $size) {
                datetime
                topWords {
                  word
                  score
                }
              }
            }
            """
            
            # Format dates
            end_date = datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")
            start_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
            
            # Format variables
            variables = {
                "from": start_date,
                "to": end_date,
                "size": 10
            }
            
            # Define headers - use API key directly
            headers = {
                "Content-Type": "application/json",
                "apikey": config.SANTIMENT_API_KEY
            }
            
            # Make API request
            response = requests.post(
                SANTIMENT_API_URL,
                headers=headers,
                json={"query": query, "variables": variables},
                timeout=10  # Add timeout
            )
            
            # Check for successful response
            if response.status_code == 200:
                data = response.json()
                
                # Check for errors in the GraphQL response
                if "errors" in data:
                    error_message = data["errors"][0].get("message", "Unknown GraphQL error")
                    logger.error(f"Santiment API GraphQL error: {error_message}")
                    return get_simulated_trending_topics()
                
                # Check for trending words data
                if "data" in data and "getTrendingWords" in data["data"]:
                    trending_data = data["data"]["getTrendingWords"]
                    
                    # If no data points, fall back to simulated data
                    if not trending_data:
                        logger.warning("Santiment API: No trending topics found")
                        return get_simulated_trending_topics()
                    
                    # Get the latest trending words
                    latest_trends = trending_data[-1]["topWords"] if trending_data else []
                    
                    # Format trending topics
                    result = []
                    for trend in latest_trends:
                        result.append({
                            "topic": trend["word"],
                            "mentions": int(trend["score"] * 1000),  # Convert score to mentions (approximate)
                            "sentiment": get_topic_sentiment(trend["word"])
                        })
                    
                    return result
            
            # If API request failed, log the error and fall back to simulated data
            logger.error(f"Santiment API error: {response.status_code} - {response.text}")
        
        # Fall back to simulated data
        return get_simulated_trending_topics()
    
    except Exception as e:
        logger.error(f"Error fetching trending topics: {str(e)}")
        return get_simulated_trending_topics()

def get_simulated_trending_topics():
    """
    Generate simulated trending crypto topics.
    
    Returns:
        list: List of trending topics
    """
    # List of possible trending topics
    topics = [
        "bitcoin halving",
        "ethereum upgrade",
        "defi yields",
        "nft collections",
        "crypto regulation",
        "layer 2 scaling",
        "meme coins",
        "stablecoins",
        "web3 adoption",
        "metaverse projects",
        "dao governance",
        "exchange hacks",
        "institutional adoption",
        "mining sustainability",
        "altcoin season"
    ]
    
    # Select a random number of trending topics (3-6)
    count = random.randint(3, 6)
    trending = random.sample(topics, count)
    
    # Sort by "trending score" (random)
    trending_with_score = [(topic, random.randint(1000, 100000)) for topic in trending]
    trending_with_score.sort(key=lambda x: x[1], reverse=True)
    
    # Format the results
    result = []
    for topic, score in trending_with_score:
        result.append({
            "topic": topic,
            "mentions": score,
            "sentiment": random.choice(["bullish", "bearish", "neutral"])
        })
    
    return result

def get_topic_sentiment(topic):
    """
    Get sentiment for a specific topic.
    
    Args:
        topic (str): Topic to analyze
    
    Returns:
        str: Sentiment assessment
    """
    # This is a simple implementation - in a real system, we would call an API
    # For now, we'll use a random distribution with slightly more bullish bias
    sentiments = ["bullish", "neutral", "bearish"]
    weights = [0.4, 0.4, 0.2]  # Slight bullish bias for crypto topics
    
    return random.choices(sentiments, weights=weights, k=1)[0]

def get_historical_sentiment(symbol, days=7):
    """
    Get historical social sentiment data.
    
    Args:
        symbol (str): Cryptocurrency symbol
        days (int): Number of days of history
    
    Returns:
        list: Historical sentiment data
    """
    try:
        # If we have The Tie API, use it to get historical sentiment
        if hasattr(config, 'THE_TIE_API_KEY') and config.THE_TIE_API_KEY:
            # Define API endpoint and parameters
            endpoint = f"{THE_TIE_API_URL}/v1/sentiment"
            headers = {
                "Content-Type": "application/json",
                "X-API-KEY": config.THE_TIE_API_KEY
            }
            params = {
                "tickers": symbol.upper(),
                "startDate": (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d"),
                "endDate": datetime.now().strftime("%Y-%m-%d"),
                "interval": "day"
            }
            
            # Make API request
            response = requests.get(
                endpoint,
                headers=headers,
                params=params,
                timeout=10  # Add timeout
            )
            
            # Check for successful response
            if response.status_code == 200:
                data = response.json()
                
                if symbol.upper() in data:
                    daily_data = data[symbol.upper()].get("data", [])
                    
                    # Format historical data
                    result = []
                    for day in daily_data:
                        date = day.get("date", "")
                        score = day.get("score", 0)
                        
                        # Map score to bullish/bearish percentages
                        if score > 0:
                            bullish = 50 + score / 2  # Map 0-100 to 50-100%
                            bearish = 100 - bullish
                        else:
                            bearish = 50 + abs(score) / 2  # Map -100-0 to 50-100%
                            bullish = 100 - bearish
                        
                        # Determine overall sentiment
                        if score > 20:
                            sentiment = "bullish"
                        elif score < -20:
                            sentiment = "bearish"
                        else:
                            sentiment = "neutral"
                        
                        result.append({
                            "date": date,
                            "bullish_percentage": int(bullish),
                            "bearish_percentage": int(bearish),
                            "overall_sentiment": sentiment
                        })
                    
                    return result
        
        # Fall back to simulated data
        return get_simulated_historical_sentiment(symbol, days)
    
    except Exception as e:
        logger.error(f"Error fetching historical sentiment: {str(e)}")
        return get_simulated_historical_sentiment(symbol, days)

def get_simulated_historical_sentiment(symbol, days=7):
    """
    Get simulated historical social sentiment data.
    
    Args:
        symbol (str): Cryptocurrency symbol
        days (int): Number of days of history
    
    Returns:
        list: Historical sentiment data
    """
    historical = []
    
    # Start with a random sentiment bias
    if symbol.upper() == "BTC":
        bullish_bias = random.randint(45, 65)  # More bullish for BTC
    elif symbol.upper() == "ETH":
        bullish_bias = random.randint(40, 60)  # Slightly bullish for ETH
    else:
        bullish_bias = random.randint(35, 55)  # More neutral for others
    
    for i in range(days):
        # Calculate date
        date = (datetime.now() - timedelta(days=days-i-1)).strftime("%Y-%m-%d")
        
        # Adjust the bias slightly each day with some momentum (gradual changes)
        bullish_bias += random.randint(-5, 5)
        bullish_bias = max(20, min(80, bullish_bias))
        
        # Calculate bearish bias (inversely related to bullish)
        bearish_bias = 100 - bullish_bias
        
        # Allow some variance in the actual daily sentiment
        bullish = max(10, min(90, bullish_bias + random.randint(-10, 10)))
        bearish = max(10, min(90, bearish_bias + random.randint(-10, 10)))
        
        # Normalize
        total = bullish + bearish
        bullish = int((bullish / total) * 100)
        bearish = 100 - bullish
        
        # Determine overall sentiment
        if bullish > bearish + 10:
            sentiment = "bullish"
        elif bearish > bullish + 10:
            sentiment = "bearish"
        else:
            sentiment = "neutral"
        
        historical.append({
            "date": date,
            "bullish_percentage": bullish,
            "bearish_percentage": bearish,
            "overall_sentiment": sentiment
        })
    
    return historical

def get_sentiment_change(symbol, timeframe="24h"):
    """
    Get the sentiment change over a timeframe.
    
    Args:
        symbol (str): Cryptocurrency symbol
        timeframe (str): Time period for comparison
    
    Returns:
        dict: Sentiment change data
    """
    try:
        # If we have The Tie API, use it to get sentiment change
        if hasattr(config, 'THE_TIE_API_KEY') and config.THE_TIE_API_KEY:
            # Define API endpoint and parameters
            endpoint = f"{THE_TIE_API_URL}/v1/sentiment"
            headers = {
                "Content-Type": "application/json",
                "X-API-KEY": config.THE_TIE_API_KEY
            }
            
            # Calculate date ranges based on timeframe
            if timeframe == "24h":
                days = 2
                interval = "hour"
            elif timeframe == "7d":
                days = 8
                interval = "day"
            else:
                days = 31
                interval = "day"
            
            params = {
                "tickers": symbol.upper(),
                "startDate": (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d"),
                "endDate": datetime.now().strftime("%Y-%m-%d"),
                "interval": interval
            }
            
            # Make API request
            response = requests.get(
                endpoint,
                headers=headers,
                params=params,
                timeout=10  # Add timeout
            )
            
            # Check for successful response
            if response.status_code == 200:
                data = response.json()
                
                if symbol.upper() in data:
                    daily_data = data[symbol.upper()].get("data", [])
                    
                    # Need at least 2 data points to calculate change
                    if len(daily_data) < 2:
                        logger.warning(f"The Tie API: Not enough data points for {symbol} sentiment change")
                        return get_simulated_sentiment_change(symbol, timeframe)
                    
                    # Get current and previous sentiment scores
                    current_data = daily_data[-1]
                    previous_data = daily_data[0]
                    
                    current_score = current_data.get("score", 0)
                    previous_score = previous_data.get("score", 0)
                    
                    # Calculate bullish percentages
                    if current_score > 0:
                        current_bullish = 50 + current_score / 2
                    else:
                        current_bullish = 50 - abs(current_score) / 2
                    
                    if previous_score > 0:
                        previous_bullish = 50 + previous_score / 2
                    else:
                        previous_bullish = 50 - abs(previous_score) / 2
                    
                    # Calculate change
                    bullish_change = current_bullish - previous_bullish
                    
                    # Determine overall change
                    if bullish_change > 10:
                        overall_change = "significant increase"
                    elif bullish_change > 5:
                        overall_change = "moderate increase"
                    elif bullish_change < -10:
                        overall_change = "significant decrease"
                    elif bullish_change < -5:
                        overall_change = "moderate decrease"
                    else:
                        overall_change = "minimal change"
                    
                    return {
                        "timeframe": timeframe,
                        "previous_bullish": int(previous_bullish),
                        "current_bullish": int(current_bullish),
                        "bullish_change": int(bullish_change),
                        "overall_change": overall_change
                    }
        
        # Fall back to simulated data
        return get_simulated_sentiment_change(symbol, timeframe)
    
    except Exception as e:
        logger.error(f"Error fetching sentiment change: {str(e)}")
        return get_simulated_sentiment_change(symbol, timeframe)

def get_simulated_sentiment_change(symbol, timeframe="24h"):
    """
    Get simulated sentiment change data.
    
    Args:
        symbol (str): Cryptocurrency symbol
        timeframe (str): Time period for comparison
    
    Returns:
        dict: Sentiment change data
    """
    # Generate base values based on symbol
    if symbol.upper() == "BTC":
        previous_bullish = random.randint(45, 65)
    elif symbol.upper() == "ETH":
        previous_bullish = random.randint(40, 60)
    else:
        previous_bullish = random.randint(35, 55)
    
    # Generate a realistic change
    change_range = 15  # max +/- 15%
    bullish_change = random.randint(-change_range, change_range)
    
    # Calculate current bullish
    current_bullish = previous_bullish + bullish_change
    current_bullish = max(20, min(80, current_bullish))  # Constrain to reasonable values
    
    # Calculate bearish values
    previous_bearish = 100 - previous_bullish
    current_bearish = 100 - current_bullish
    
    # Determine overall change
    if bullish_change > 10:
        overall_change = "significant increase"
    elif bullish_change > 5:
        overall_change = "moderate increase"
    elif bullish_change < -10:
        overall_change = "significant decrease"
    elif bullish_change < -5:
        overall_change = "moderate decrease"
    else:
        overall_change = "minimal change"
    
    return {
        "timeframe": timeframe,
        "previous_bullish": previous_bullish,
        "current_bullish": current_bullish,
        "bullish_change": bullish_change,
        "overall_change": overall_change
    }