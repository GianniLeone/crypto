"""
CryptoCompare API Integration Module

Retrieves cryptocurrency market data from the CryptoCompare API.
Provides price, volume, market cap, and other relevant metrics.
Replacement for CoinGecko with more reliable service.
"""

import requests
import logging
import time
import json
from datetime import datetime
import config

# Set up logging
logger = logging.getLogger('crypto_bot.cryptocompare')

# CryptoCompare API endpoints
PRICE_URL = "https://min-api.cryptocompare.com/data/price"
MULTI_PRICE_URL = "https://min-api.cryptocompare.com/data/pricemultifull"
HISTORICAL_DAILY_URL = "https://min-api.cryptocompare.com/data/v2/histoday"
HISTORICAL_HOURLY_URL = "https://min-api.cryptocompare.com/data/v2/histohour"
COIN_LIST_URL = "https://min-api.cryptocompare.com/data/all/coinlist"
GLOBAL_STATS_URL = "https://min-api.cryptocompare.com/data/top/totalvolfull"
TOP_PAIRS_URL = "https://min-api.cryptocompare.com/data/top/pairs"

# Cache to store API responses (reduce redundant calls)
response_cache = {}
cache_expiry = {}

def make_api_request(url, params, cache_duration=300):
    """
    Make a request to the CryptoCompare API with caching.
    
    Args:
        url (str): API endpoint URL
        params (dict): Query parameters
        cache_duration (int): Cache duration in seconds
    
    Returns:
        dict: API response or None if failed
    """
    # Generate cache key
    cache_key = f"{url}:{str(params)}"
    
    # Check cache first
    current_time = time.time()
    if cache_key in response_cache and current_time < cache_expiry.get(cache_key, 0):
        logger.debug(f"Using cached response for {url}")
        return response_cache[cache_key]
    
    # Add API key to headers if configured
    headers = {}
    if hasattr(config, 'CRYPTOCOMPARE_API_KEY') and config.CRYPTOCOMPARE_API_KEY:
        headers["authorization"] = f"Apikey {config.CRYPTOCOMPARE_API_KEY}"
    
    try:
        # Make the request
        response = requests.get(url, params=params, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        # Cache the response
        response_cache[cache_key] = data
        cache_expiry[cache_key] = current_time + cache_duration
        
        return data
    
    except requests.exceptions.RequestException as e:
        logger.error(f"Error making request to CryptoCompare API: {str(e)}")
        return None

def get_price(symbol, currency='USD'):
    """
    Get the current price of a cryptocurrency.
    
    Args:
        symbol (str): Cryptocurrency symbol (e.g., 'BTC', 'ETH')
        currency (str): Currency to get price in (default: 'USD')
    
    Returns:
        float: Current price or None if failed
    """
    try:
        # Fetch price data
        data = make_api_request(PRICE_URL, {"fsym": symbol, "tsyms": currency})
        
        if data and currency in data:
            price = data[currency]
            logger.info(f"CryptoCompare price for {symbol}: {price} {currency}")
            return price
        else:
            logger.warning(f"Could not get price for {symbol}")
            return None
    
    except Exception as e:
        logger.error(f"Error getting price for {symbol}: {str(e)}")
        return None

def get_market_data(symbol):
    """
    Get comprehensive market data for a cryptocurrency.
    
    Args:
        symbol (str): Cryptocurrency symbol (e.g., 'BTC', 'ETH')
    
    Returns:
        dict: Market data or None if failed
    """
    try:
        # Fetch full price data
        data = make_api_request(MULTI_PRICE_URL, {
            "fsyms": symbol,
            "tsyms": "USD"
        })
        
        if data and "RAW" in data and symbol in data["RAW"] and "USD" in data["RAW"][symbol]:
            market_data = data["RAW"][symbol]["USD"]
            
            # Also get historical data for price changes
            hist_data = get_historical_data(symbol, days=30)  # Last 30 days
            
            # Calculate price changes
            price_change_24h = 0
            price_change_7d = 0
            price_change_30d = 0
            
            if hist_data and len(hist_data) > 0:
                current_price = market_data["PRICE"]
                
                if len(hist_data) >= 1:  # At least 1 day of data
                    day1_price = hist_data[0]["price"]
                    price_change_24h = ((current_price - day1_price) / day1_price) * 100
                
                if len(hist_data) >= 7:  # At least 7 days of data
                    day7_price = hist_data[6]["price"]
                    price_change_7d = ((current_price - day7_price) / day7_price) * 100
                
                if len(hist_data) >= 30:  # Full 30 days of data
                    day30_price = hist_data[29]["price"]
                    price_change_30d = ((current_price - day30_price) / day30_price) * 100
            
            # Extract relevant metrics
            result = {
                "symbol": symbol,
                "name": data.get("DISPLAY", {}).get(symbol, {}).get("USD", {}).get("FROMSYMBOL", symbol),
                "current_price": market_data["PRICE"],
                "market_cap": market_data.get("MKTCAP", 0),
                "total_volume": market_data.get("TOTALVOLUME24H", 0),
                "high_24h": market_data.get("HIGH24HOUR", 0),
                "low_24h": market_data.get("LOW24HOUR", 0),
                "price_change_24h": price_change_24h,
                "price_change_7d": price_change_7d,
                "price_change_30d": price_change_30d,
                "market_cap_rank": 0,  # Not directly available
                "supply": {
                    "total": market_data.get("SUPPLY", 0),
                    "circulating": market_data.get("CIRCULATINGSUPPLY", 0),
                    "max": market_data.get("MAXSUPPLY", 0)
                }
            }
            
            logger.info(f"Retrieved market data for {symbol} from CryptoCompare")
            return result
        
        logger.warning(f"Failed to get market data for {symbol} from CryptoCompare")
        return None
    
    except Exception as e:
        logger.error(f"Error getting market data for {symbol}: {str(e)}")
        return None

def get_historical_data(symbol, days=30, interval='daily'):
    """
    Get historical price data for a cryptocurrency.
    
    Args:
        symbol (str): Cryptocurrency symbol (e.g., 'BTC', 'ETH')
        days (int): Number of days of history
        interval (str): Data interval ('daily' or 'hourly')
    
    Returns:
        list: Historical price data or None if failed
    """
    try:
        # Map interval to API parameter
        if interval == 'hourly':
            api_url = HISTORICAL_HOURLY_URL
        else:  # 'daily'
            api_url = HISTORICAL_DAILY_URL
        
        # Fetch historical data
        data = make_api_request(api_url, {
            "fsym": symbol,
            "tsym": "USD",
            "limit": days
        })
        
        if data and "Data" in data and "Data" in data["Data"]:
            ohlcv_data = data["Data"]["Data"]
            
            # Format the data
            result = []
            for candle in ohlcv_data:
                timestamp = candle["time"]
                
                result.append({
                    "timestamp": timestamp,
                    "datetime": datetime.fromtimestamp(timestamp).isoformat(),
                    "price": candle["close"],
                    "volume": candle["volumefrom"],
                    "market_cap": 0  # Not directly available
                })
            
            logger.info(f"Retrieved {len(result)} historical data points for {symbol} from CryptoCompare")
            return result
        
        logger.warning(f"Failed to get historical data for {symbol} from CryptoCompare")
        return None
    
    except Exception as e:
        logger.error(f"Error getting historical data for {symbol}: {str(e)}")
        return None

def get_ohlc_data(symbol, days=30):
    """
    Get OHLC (Open, High, Low, Close) data for a cryptocurrency.
    
    Args:
        symbol (str): Cryptocurrency symbol (e.g., 'BTC', 'ETH')
        days (int): Number of days of history
    
    Returns:
        list: OHLC data or None if failed
    """
    try:
        # Fetch OHLC data
        data = make_api_request(HISTORICAL_DAILY_URL, {
            "fsym": symbol,
            "tsym": "USD",
            "limit": days
        })
        
        if data and "Data" in data and "Data" in data["Data"]:
            ohlcv_data = data["Data"]["Data"]
            
            # Format data into a standard structure
            result = []
            for candle in ohlcv_data:
                timestamp = candle["time"]
                result.append({
                    "timestamp": timestamp,
                    "open": candle["open"],
                    "high": candle["high"],
                    "low": candle["low"],
                    "close": candle["close"]
                })
            
            logger.info(f"Retrieved {len(result)} OHLC data points for {symbol} from CryptoCompare")
            return result
        
        logger.warning(f"Failed to get OHLC data for {symbol} from CryptoCompare")
        return None
    
    except Exception as e:
        logger.error(f"Error getting OHLC data for {symbol}: {str(e)}")
        return None

def get_global_market_data():
    """
    Get global cryptocurrency market data.
    
    Returns:
        dict: Global market data or None if failed
    """
    try:
        # Fetch global market data (top coins by volume)
        data = make_api_request(GLOBAL_STATS_URL, {
            "limit": 20,
            "tsym": "USD"
        })
        
        if data and "Data" in data:
            coins_data = data["Data"]
            
            # Calculate total market cap and volume
            total_market_cap = 0
            total_volume = 0
            
            # Get BTC and ETH dominance
            btc_market_cap = 0
            eth_market_cap = 0
            
            for coin in coins_data:
                coin_info = coin.get("RAW", {}).get("USD", {})
                market_cap = coin_info.get("MKTCAP", 0)
                total_market_cap += market_cap
                total_volume += coin_info.get("TOTALVOLUME24H", 0)
                
                if coin.get("CoinInfo", {}).get("Name") == "BTC":
                    btc_market_cap = market_cap
                elif coin.get("CoinInfo", {}).get("Name") == "ETH":
                    eth_market_cap = market_cap
            
            # Calculate dominance
            btc_dominance = (btc_market_cap / total_market_cap * 100) if total_market_cap > 0 else 0
            eth_dominance = (eth_market_cap / total_market_cap * 100) if total_market_cap > 0 else 0
            
            # Format the result
            result = {
                "market_cap_usd": total_market_cap,
                "volume_24h_usd": total_volume,
                "market_cap_change_24h": 0,  # Not directly available
                "active_cryptocurrencies": len(coins_data),
                "markets": 0,  # Not directly available
                "btc_dominance": btc_dominance,
                "eth_dominance": eth_dominance,
                "updated_at": int(time.time())
            }
            
            logger.info("Retrieved global market data from CryptoCompare")
            return result
        
        logger.warning("Failed to get global market data from CryptoCompare")
        return None
    
    except Exception as e:
        logger.error(f"Error getting global market data: {str(e)}")
        return None

def get_trending_coins():
    """
    Get trending coins over the last 24 hours.
    
    Returns:
        list: Trending coins or None if failed
    """
    try:
        # Use top volume coins as "trending"
        data = make_api_request(GLOBAL_STATS_URL, {
            "limit": 15,
            "tsym": "USD"
        })
        
        if data and "Data" in data:
            coins_data = data["Data"]
            
            # Format the results
            result = []
            for i, coin in enumerate(coins_data):
                coin_info = coin.get("CoinInfo", {})
                
                result.append({
                    "id": coin_info.get("Name", "").lower(),
                    "name": coin_info.get("FullName", ""),
                    "symbol": coin_info.get("Name", ""),
                    "market_cap_rank": i + 1,
                    "thumb": coin_info.get("ImageUrl", ""),
                    "score": 15 - i  # Simple score based on rank
                })
            
            logger.info(f"Retrieved {len(result)} trending coins from CryptoCompare")
            return result
        
        logger.warning("Failed to get trending coins from CryptoCompare")
        return None
    
    except Exception as e:
        logger.error(f"Error getting trending coins: {str(e)}")
        return None

def convert_to_ta_format(ohlc_data):
    """
    Convert CryptoCompare OHLC data to the format expected by technical_indicator_fetcher.
    
    Args:
        ohlc_data (list): OHLC data from CryptoCompare
    
    Returns:
        list: OHLC data in technical_indicator_fetcher format
    """
    if not ohlc_data:
        return None
    
    result = []
    for candle in ohlc_data:
        result.append({
            "timestamp": candle["timestamp"],
            "open": candle["open"],
            "high": candle["high"],
            "low": candle["low"],
            "close": candle["close"],
            "volume": 0  # Volume might not be available
        })
    
    return result

def get_defi_stats():
    """
    Get global DeFi statistics.
    
    Note: CryptoCompare doesn't have direct DeFi stats like CoinGecko,
    this is a simplified version.
    
    Returns:
        dict: DeFi statistics or None if failed
    """
    try:
        # Get top DeFi coins (simplified approach)
        defi_coins = ["AAVE", "UNI", "COMP", "MKR", "SNX", "YFI", "CAKE", "CRV"]
        
        # Fetch price data for these coins
        params = {
            "fsyms": ",".join(defi_coins),
            "tsyms": "USD"
        }
        
        data = make_api_request(MULTI_PRICE_URL, params)
        
        if data and "RAW" in data:
            raw_data = data["RAW"]
            
            # Calculate total DeFi market cap
            defi_market_cap = 0
            
            for coin in defi_coins:
                if coin in raw_data and "USD" in raw_data[coin]:
                    defi_market_cap += raw_data[coin]["USD"].get("MKTCAP", 0)
            
            # Get ETH market cap for comparison
            eth_data = make_api_request(MULTI_PRICE_URL, {"fsyms": "ETH", "tsyms": "USD"})
            eth_market_cap = 0
            
            if eth_data and "RAW" in eth_data and "ETH" in eth_data["RAW"] and "USD" in eth_data["RAW"]["ETH"]:
                eth_market_cap = eth_data["RAW"]["ETH"]["USD"].get("MKTCAP", 0)
            
            # Calculate ratios
            defi_to_eth_ratio = (defi_market_cap / eth_market_cap) if eth_market_cap > 0 else 0
            
            # Find top coin
            top_coin_name = None
            top_coin_market_cap = 0
            
            for coin in defi_coins:
                if coin in raw_data and "USD" in raw_data[coin]:
                    market_cap = raw_data[coin]["USD"].get("MKTCAP", 0)
                    if market_cap > top_coin_market_cap:
                        top_coin_market_cap = market_cap
                        top_coin_name = coin
            
            top_coin_defi_dominance = (top_coin_market_cap / defi_market_cap * 100) if defi_market_cap > 0 else 0
            
            result = {
                "defi_market_cap": defi_market_cap,
                "eth_market_cap": eth_market_cap,
                "defi_to_eth_ratio": defi_to_eth_ratio,
                "top_coin_name": top_coin_name,
                "top_coin_defi_dominance": top_coin_defi_dominance
            }
            
            logger.info("Retrieved DeFi statistics")
            return result
        
        logger.warning("Failed to get DeFi statistics")
        return None
    
    except Exception as e:
        logger.error(f"Error getting DeFi statistics: {str(e)}")
        return None

def clear_cache():
    """
    Clear the API response cache.
    """
    global response_cache, cache_expiry
    response_cache = {}
    cache_expiry = {}
    logger.info("CryptoCompare API cache cleared")