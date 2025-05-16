# Create a new file: data_aggregator.py
"""
Data Aggregator

Centralizes all data collection for the trading bot.
Provides a single clean interface for gathering all necessary information.
"""

import logging
from datetime import datetime

# Set up logging
logger = logging.getLogger('crypto_bot.data_aggregator')

class DataAggregator:
    """Centralized data collection for the trading bot."""
    
    def __init__(self):
        """Initialize the data aggregator."""
        # Import components only when needed (lazy loading)
        from core import enhanced_news_fetcher, enhanced_social_sentiment
        from core import whale_data_provider, fear_greed_fetcher
        from core import crypto_compare_api, technical_indicator_fetcher
        
        self.news_fetcher = enhanced_news_fetcher
        self.social_sentiment = enhanced_social_sentiment
        self.whale_data = whale_data_provider
        self.fear_greed = fear_greed_fetcher
        self.market_data = crypto_compare_api
        self.technical = technical_indicator_fetcher
    
    def gather_all_data(self, symbol):
        """
        Gather all data for a specific cryptocurrency.
        
        Args:
            symbol (str): Cryptocurrency symbol (e.g., 'BTC', 'ETH')
            
        Returns:
            dict: Complete context with all data points
        """
        logger.info(f"Gathering comprehensive data for {symbol}")
        
        # Create base context
        context = {
            "timestamp": datetime.now().isoformat(),
            "symbol": symbol,
        }
        
        # Add news data with error handling
        try:
            news_headlines = self.news_fetcher.get_latest_headlines(limit=5)
            news_sentiment = self.news_fetcher.analyze_sentiment(news_headlines)
            context["news_headlines"] = news_headlines
            context["news_sentiment"] = news_sentiment
        except Exception as e:
            logger.error(f"Error fetching news data: {str(e)}")
            context["news_headlines"] = []
            context["news_sentiment"] = "neutral"
        
        # Add each data source with proper error handling
        # [Add similar blocks for each data source]
        
        # Get global market context
        try:
            global_data = self.market_data.get_global_market_data()
            if global_data:
                context["global_market"] = {
                    'market_cap_usd': global_data.get('market_cap_usd', 0),
                    'btc_dominance': global_data.get('btc_dominance', 0),
                    'eth_dominance': global_data.get('eth_dominance', 0),
                    'updated_at': global_data.get('updated_at', 0)
                }
        except Exception as e:
            logger.error(f"Error fetching global market data: {str(e)}")
        
        return context