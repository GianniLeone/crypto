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
        # Import components lazily to avoid circular imports
        # All modules will be imported only when needed
        logger.info("Initializing data aggregator")
        self._modules_loaded = False
        self._modules = {}
    
    def _load_modules(self):
        """Load all required modules for data collection."""
        if self._modules_loaded:
            return
            
        try:
            # Market data
            from core import crypto_compare_api
            self._modules['market_data'] = crypto_compare_api
            
            # Technical indicators
            from core import technical_indicator_fetcher
            self._modules['technical'] = technical_indicator_fetcher
            
            # Enhanced market analysis
            from core import predictive_technical_analysis
            self._modules['predictive'] = predictive_technical_analysis
            
            # News and sentiment
            from core import enhanced_news_fetcher
            self._modules['news'] = enhanced_news_fetcher
            
            # Social sentiment
            from core import enhanced_social_sentiment
            self._modules['social'] = enhanced_social_sentiment
            
            # Whale data
            from core import unified_whale_data
            self._modules['whale'] = unified_whale_data
            
            # Fear & Greed index
            from core import fear_greed_fetcher
            self._modules['fear_greed'] = fear_greed_fetcher
            
            self._modules_loaded = True
            logger.info("Successfully loaded all data modules")
            
        except Exception as e:
            logger.error(f"Error loading modules: {str(e)}")
            raise
    
    def gather_all_data(self, symbol):
        """
        Gather all data for a specific cryptocurrency.
        
        Args:
            symbol (str): Cryptocurrency symbol (e.g., 'BTC', 'ETH')
            
        Returns:
            dict: Complete context with all data points
        """
        logger.info(f"Gathering comprehensive data for {symbol}")
        
        # Ensure all modules are loaded
        self._load_modules()
        
        # Create base context
        context = {
            "timestamp": datetime.now().isoformat(),
            "symbol": symbol,
        }
        
        # 1. Add market data with error handling
        try:
            market_data = self._modules['market_data'].get_market_data(symbol)
            if market_data:
                context["crypto_data"] = market_data
        except Exception as e:
            logger.error(f"Error fetching market data: {str(e)}")
        
        # 2. Add news data with error handling
        try:
            news_headlines = self._modules['news'].get_latest_headlines(symbol=symbol, limit=5)
            news_sentiment = self._modules['news'].analyze_sentiment(news_headlines)
            context["news_headlines"] = news_headlines
            context["news_sentiment"] = news_sentiment
        except Exception as e:
            logger.error(f"Error fetching news data: {str(e)}")
            context["news_headlines"] = []
            context["news_sentiment"] = "neutral"
        
        # 3. Add technical indicators with error handling
        try:
            context["rsi"] = self._modules['technical'].get_rsi(symbol)
            context["macd_crossover"] = self._modules['technical'].get_macd_signal(symbol)
            context["volume_spike"] = self._modules['technical'].check_volume_spike(symbol)
            context["price_action"] = self._modules['technical'].get_price_trend(symbol)
        except Exception as e:
            logger.error(f"Error fetching technical indicators: {str(e)}")
        
        # 4. Add whale transaction data with error handling
        try:
            context["whale_transactions"] = self._modules['whale'].get_recent_activity(symbol)
            # Get detailed whale transactions
            recent_whale_txs = self._modules['whale'].get_whale_transactions(symbol, count=3)
            if recent_whale_txs:
                context["recent_whale_transactions"] = recent_whale_txs
        except Exception as e:
            logger.error(f"Error fetching whale data: {str(e)}")
            context["whale_transactions"] = "neutral"
        
        # 5. Add social sentiment with error handling
        try:
            context["social_media_sentiment"] = self._modules['social'].get_sentiment(symbol)
        except Exception as e:
            logger.error(f"Error fetching social sentiment: {str(e)}")
            context["social_media_sentiment"] = "neutral"
        
        # 6. Add Fear & Greed Index with error handling
        try:
            context["fear_greed_index"] = self._modules['fear_greed'].get_current_index()
        except Exception as e:
            logger.error(f"Error fetching Fear & Greed Index: {str(e)}")
            context["fear_greed_index"] = "Neutral"
        
        # 7. Add global market context with error handling
        try:
            global_data = self._modules['market_data'].get_global_market_data()
            if global_data:
                context["global_market"] = {
                    'market_cap_usd': global_data.get('market_cap_usd', 0),
                    'btc_dominance': global_data.get('btc_dominance', 0),
                    'eth_dominance': global_data.get('eth_dominance', 0),
                    'updated_at': global_data.get('updated_at', 0)
                }
        except Exception as e:
            logger.error(f"Error fetching global market data: {str(e)}")
        
        # 8. Add predictive technical analysis with error handling
        try:
            # Only run predictive analysis if requested in config
            import config
            if getattr(config, 'USE_PREDICTIVE_ANALYSIS', True):
                predictive_data = self._modules['predictive'].get_analysis(symbol)
                if predictive_data and predictive_data.get('status') == 'success':
                    context["predictive_analysis"] = predictive_data
        except Exception as e:
            logger.error(f"Error fetching predictive analysis: {str(e)}")
        
        # 9. Add time context
        context["time_of_day"] = self._get_time_of_day()
        
        logger.info(f"Successfully gathered comprehensive data for {symbol}")
        return context
    
    def _get_time_of_day(self):
        """Helper function to get time of day"""
        current_hour = datetime.now().hour
        if 0 <= current_hour < 8:
            return "night"
        elif 8 <= current_hour < 12:
            return "morning"
        elif 12 <= current_hour < 18:
            return "afternoon"
        else:
            return "evening"
    
    def get_symbol_price(self, symbol):
        """
        Get the current price for a symbol.
        
        Args:
            symbol (str): Cryptocurrency symbol
            
        Returns:
            float: Current price
        """
        self._load_modules()
        return self._modules['market_data'].get_price(symbol)
    
    def get_global_market_data(self):
        """
        Get global market data.
        
        Returns:
            dict: Global market data
        """
        self._load_modules()
        return self._modules['market_data'].get_global_market_data()
    
    def get_trending_coins(self):
        """
        Get trending coins.
        
        Returns:
            list: Trending coin data
        """
        self._load_modules()
        return self._modules['market_data'].get_trending_coins()