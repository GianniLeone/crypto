"""
Free Market Data Provider

Uses truly free cryptocurrency API sources with generous limits:
1. Binance API (primary) - No API key required, 1200 requests/minute
2. CryptoCompare API (secondary) - With your existing API key
3. Fallback mechanisms for when APIs fail

Features:
- Robust caching system to minimize API calls
- No API keys required for primary source
- Focused on exchanges with confirmed free public endpoints
"""

import requests
import logging
import time
import json
from datetime import datetime, timedelta
import os
from functools import lru_cache
import random

# Set up logging
logger = logging.getLogger('crypto_bot.market_data')

# API endpoints
# Binance
BINANCE_TICKER_URL = "https://api.binance.com/api/v3/ticker/24hr"
BINANCE_PRICE_URL = "https://api.binance.com/api/v3/ticker/price"
BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"
BINANCE_EXCHANGE_INFO = "https://api.binance.com/api/v3/exchangeInfo"

# CryptoCompare (as fallback)
CRYPTOCOMPARE_PRICE_URL = "https://min-api.cryptocompare.com/data/price"
CRYPTOCOMPARE_MULTI_PRICE_URL = "https://min-api.cryptocompare.com/data/pricemultifull"
CRYPTOCOMPARE_HISTO_URL = "https://min-api.cryptocompare.com/data/v2/histoday"
CRYPTOCOMPARE_GLOBAL_URL = "https://min-api.cryptocompare.com/data/top/totalvolfull"

# Cache directory
CACHE_DIR = "data/cache/market_data"
os.makedirs(CACHE_DIR, exist_ok=True)

# Cache settings
CACHE_EXPIRY = {
    'price': 60,           # 1 minute for prices
    'market_data': 300,    # 5 minutes for market data
    'historical': 3600,    # 1 hour for historical data
    'global': 900,         # 15 minutes for global data
    'trending': 1800       # 30 minutes for trending data
}

# Mapping symbol to trading pair
def get_trading_pair(symbol, quote='USDT'):
    """Convert symbol to Binance trading pair."""
    return f"{symbol.upper()}{quote}"

# Common currency mappings for major symbols
# This helps us handle coins that might not be on Binance
SYMBOL_MAPPINGS = {
    'BTC': 'BTCUSDT',
    'ETH': 'ETHUSDT',
    'SOL': 'SOLUSDT',
    'BNB': 'BNBUSDT',
    'XRP': 'XRPUSDT',
    'ADA': 'ADAUSDT',
    'DOGE': 'DOGEUSDT',
    'DOT': 'DOTUSDT',
    'AVAX': 'AVAXUSDT',
    'MATIC': 'MATICUSDT',
    'LTC': 'LTCUSDT',
    'LINK': 'LINKUSDT',
    'UNI': 'UNIUSDT',
    'SHIB': 'SHIBUSDT',
    'TRX': 'TRXUSDT',
}

# Keep a list of valid trading pairs on Binance
VALID_PAIRS = set()

# Function to initialize valid trading pairs from Binance
def init_valid_pairs():
    """Initialize the list of valid trading pairs from Binance."""
    global VALID_PAIRS
    if VALID_PAIRS:  # Already initialized
        return
    
    try:
        response = requests.get(BINANCE_EXCHANGE_INFO, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if 'symbols' in data:
                for symbol_data in data['symbols']:
                    if symbol_data.get('status') == 'TRADING':
                        VALID_PAIRS.add(symbol_data.get('symbol'))
                logger.info(f"Initialized {len(VALID_PAIRS)} valid trading pairs from Binance")
        else:
            logger.warning(f"Failed to get exchange info from Binance: {response.status_code}")
    except Exception as e:
        logger.error(f"Error initializing valid pairs: {str(e)}")
        # Initialize with common pairs as fallback
        for pair in SYMBOL_MAPPINGS.values():
            VALID_PAIRS.add(pair)

# Initialize valid pairs
init_valid_pairs()

def get_cache_path(key):
    """Get path for cache file by key."""
    safe_key = "".join(c if c.isalnum() else "_" for c in key)
    return os.path.join(CACHE_DIR, f"{safe_key}.json")

def save_to_cache(key, data, expiry_type='price'):
    """Save data to cache with specified expiry type."""
    try:
        expiry_seconds = CACHE_EXPIRY.get(expiry_type, 300)
        cache_data = {
            'data': data,
            'expiry': (datetime.now() + timedelta(seconds=expiry_seconds)).timestamp()
        }
        
        with open(get_cache_path(key), 'w') as f:
            json.dump(cache_data, f)
            
        return True
    except Exception as e:
        logger.warning(f"Failed to save to cache: {str(e)}")
        return False

def load_from_cache(key):
    """Load data from cache if not expired."""
    cache_path = get_cache_path(key)
    
    if not os.path.exists(cache_path):
        return None
    
    try:
        with open(cache_path, 'r') as f:
            cache_data = json.load(f)
        
        if datetime.now().timestamp() > cache_data['expiry']:
            return None
            
        return cache_data['data']
    except Exception as e:
        logger.warning(f"Failed to load from cache: {str(e)}")
        return None

def make_api_request(url, params=None, headers=None, timeout=10):
    """Make API request with error handling."""
    try:
        response = requests.get(url, params=params, headers=headers, timeout=timeout)
        
        # Check if we got a valid response
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 429:  # Rate limit
            logger.warning(f"Rate limit hit for {url}")
            return None
        else:
            logger.error(f"API request failed with status {response.status_code}: {url}")
            return None
            
    except requests.exceptions.RequestException as e:
        logger.error(f"Request error: {str(e)}")
        return None
    except ValueError as e:
        logger.error(f"JSON decode error: {str(e)}")
        return None

def get_price_binance(symbol, currency='USDT'):
    """Get price from Binance API."""
    # Check cache first
    cache_key = f"price_binance_{symbol}_{currency}"
    cached_price = load_from_cache(cache_key)
    if cached_price is not None:
        return cached_price
    
    # Format trading pair
    if symbol.upper() in SYMBOL_MAPPINGS:
        trading_pair = SYMBOL_MAPPINGS[symbol.upper()]
    else:
        trading_pair = f"{symbol.upper()}{currency}"
    
    # Check if this is a valid trading pair
    if trading_pair not in VALID_PAIRS:
        logger.warning(f"Trading pair {trading_pair} not found on Binance")
        return None
    
    # Get price from Binance
    params = {'symbol': trading_pair}
    data = make_api_request(BINANCE_PRICE_URL, params=params)
    
    if data and 'price' in data:
        price = float(data['price'])
        # Cache the price
        save_to_cache(cache_key, price, 'price')
        logger.info(f"Binance price for {symbol}: {price} {currency}")
        return price
    
    logger.warning(f"Failed to get price for {symbol} from Binance")
    return None

def get_price_cryptocompare(symbol, currency='USD'):
    """Get price from CryptoCompare API (fallback)."""
    # Check cache first
    cache_key = f"price_cryptocompare_{symbol}_{currency}"
    cached_price = load_from_cache(cache_key)
    if cached_price is not None:
        return cached_price
    
    # Add API key if available
    headers = {}
    try:
        from config import CRYPTOCOMPARE_API_KEY
        if CRYPTOCOMPARE_API_KEY:
            headers["authorization"] = f"Apikey {CRYPTOCOMPARE_API_KEY}"
    except ImportError:
        pass  # No API key available
    
    params = {
        'fsym': symbol.upper(),
        'tsyms': currency.upper()
    }
    
    data = make_api_request(CRYPTOCOMPARE_PRICE_URL, params=params, headers=headers)
    
    if data and currency.upper() in data:
        price = data[currency.upper()]
        # Cache the price
        save_to_cache(cache_key, price, 'price')
        logger.info(f"CryptoCompare price for {symbol}: {price} {currency}")
        return price
    
    logger.warning(f"Failed to get price for {symbol} from CryptoCompare")
    return None

def get_price(symbol, currency='USD'):
    """
    Get price from multiple sources with fallbacks.
    
    Args:
        symbol (str): Cryptocurrency symbol (e.g., 'BTC', 'ETH')
        currency (str): Currency to get price in (default: 'USD')
    
    Returns:
        float: Current price or None if all sources fail
    """
    # Convert currency to match Binance format if needed
    binance_currency = 'USDT' if currency.upper() == 'USD' else currency.upper()
    
    # Try Binance first
    price = get_price_binance(symbol, binance_currency)
    if price:
        return price
    
    # Try CryptoCompare as fallback
    price = get_price_cryptocompare(symbol, currency)
    if price:
        return price
    
    # Generate dummy price as last resort
    logger.error(f"Failed to get price for {symbol} from all sources, using dummy price")
    return get_dummy_price(symbol)

def get_market_data_binance(symbol, currency='USDT'):
    """Get detailed market data from Binance."""
    # Check cache first
    cache_key = f"market_data_binance_{symbol}_{currency}"
    cached_data = load_from_cache(cache_key)
    if cached_data is not None:
        return cached_data
    
    # Format trading pair
    if symbol.upper() in SYMBOL_MAPPINGS:
        trading_pair = SYMBOL_MAPPINGS[symbol.upper()]
    else:
        trading_pair = f"{symbol.upper()}{currency}"
    
    # Check if this is a valid trading pair
    if trading_pair not in VALID_PAIRS:
        logger.warning(f"Trading pair {trading_pair} not found on Binance")
        return None
    
    # Get 24hr ticker data from Binance
    data = make_api_request(BINANCE_TICKER_URL)
    
    if not data:
        return None
    
    for ticker in data:
        if ticker.get('symbol') == trading_pair:
            current_price = float(ticker.get('lastPrice', 0))
            price_change_24h = float(ticker.get('priceChangePercent', 0))
            
            # Get klines (candlesticks) for 7d and 30d price changes
            klines_7d = make_api_request(
                BINANCE_KLINES_URL, 
                params={'symbol': trading_pair, 'interval': '1d', 'limit': 7}
            )
            
            price_change_7d = 0
            if klines_7d and len(klines_7d) >= 7:
                open_7d_ago = float(klines_7d[0][1])  # Open price from 7 days ago
                if open_7d_ago > 0:
                    price_change_7d = ((current_price - open_7d_ago) / open_7d_ago) * 100
            
            klines_30d = make_api_request(
                BINANCE_KLINES_URL, 
                params={'symbol': trading_pair, 'interval': '1d', 'limit': 30}
            )
            
            price_change_30d = 0
            if klines_30d and len(klines_30d) >= 30:
                open_30d_ago = float(klines_30d[0][1])  # Open price from 30 days ago
                if open_30d_ago > 0:
                    price_change_30d = ((current_price - open_30d_ago) / open_30d_ago) * 100
            
            # Format the result
            result = {
                "symbol": symbol.upper(),
                "name": symbol.upper(),
                "current_price": current_price,
                "market_cap": 0,  # Not available from Binance
                "total_volume": float(ticker.get('quoteVolume', 0)),
                "high_24h": float(ticker.get('highPrice', 0)),
                "low_24h": float(ticker.get('lowPrice', 0)),
                "price_change_24h": price_change_24h,
                "price_change_7d": price_change_7d,
                "price_change_30d": price_change_30d,
                "market_cap_rank": 0,  # Not available from Binance
                "supply": {
                    "total": 0,
                    "circulating": 0,
                    "max": 0
                }
            }
            
            # Cache the result
            save_to_cache(cache_key, result, 'market_data')
            logger.info(f"Retrieved market data for {symbol} from Binance")
            return result
    
    logger.warning(f"No ticker data found for {symbol} on Binance")
    return None

def get_market_data_cryptocompare(symbol):
    """Get market data from CryptoCompare API (fallback)."""
    # Check cache first
    cache_key = f"market_data_cryptocompare_{symbol}"
    cached_data = load_from_cache(cache_key)
    if cached_data is not None:
        return cached_data
    
    # Add API key if available
    headers = {}
    try:
        from config import CRYPTOCOMPARE_API_KEY
        if CRYPTOCOMPARE_API_KEY:
            headers["authorization"] = f"Apikey {CRYPTOCOMPARE_API_KEY}"
    except ImportError:
        pass  # No API key available
    
    # Get full price data
    params = {
        "fsyms": symbol.upper(),
        "tsyms": "USD"
    }
    
    data = make_api_request(CRYPTOCOMPARE_MULTI_PRICE_URL, params=params, headers=headers)
    
    if data and "RAW" in data and symbol.upper() in data["RAW"] and "USD" in data["RAW"][symbol.upper()]:
        market_data = data["RAW"][symbol.upper()]["USD"]
        
        # Format the result
        result = {
            "symbol": symbol.upper(),
            "name": data.get("DISPLAY", {}).get(symbol.upper(), {}).get("USD", {}).get("FROMSYMBOL", symbol.upper()),
            "current_price": market_data.get("PRICE", 0),
            "market_cap": market_data.get("MKTCAP", 0),
            "total_volume": market_data.get("TOTALVOLUME24HTO", 0),
            "high_24h": market_data.get("HIGH24HOUR", 0),
            "low_24h": market_data.get("LOW24HOUR", 0),
            "price_change_24h": market_data.get("CHANGEPCT24HOUR", 0),
            "price_change_7d": 0,  # Not directly available
            "price_change_30d": 0,  # Not directly available
            "market_cap_rank": 0,  # Not directly available
            "supply": {
                "total": market_data.get("SUPPLY", 0),
                "circulating": market_data.get("CIRCULATINGSUPPLY", 0),
                "max": market_data.get("MAXSUPPLY", 0)
            }
        }
        
        # Cache the result
        save_to_cache(cache_key, result, 'market_data')
        logger.info(f"Retrieved market data for {symbol} from CryptoCompare")
        return result
    
    logger.warning(f"Failed to get market data for {symbol} from CryptoCompare")
    return None

def get_market_data(symbol):
    """
    Get comprehensive market data with fallbacks.
    
    Args:
        symbol (str): Cryptocurrency symbol (e.g., 'BTC', 'ETH')
    
    Returns:
        dict: Market data or None if all sources fail
    """
    # Try Binance first
    market_data = get_market_data_binance(symbol)
    if market_data:
        return market_data
    
    # Try CryptoCompare as fallback
    market_data = get_market_data_cryptocompare(symbol)
    if market_data:
        return market_data
    
    # If all sources fail, generate dummy data
    logger.error(f"Failed to get market data for {symbol} from all sources")
    return get_dummy_market_data(symbol)

def get_historical_data_binance(symbol, days=30):
    """Get historical data from Binance."""
    # Check cache first
    cache_key = f"historical_binance_{symbol}_{days}"
    cached_data = load_from_cache(cache_key)
    if cached_data is not None:
        return cached_data
    
    # Format trading pair
    if symbol.upper() in SYMBOL_MAPPINGS:
        trading_pair = SYMBOL_MAPPINGS[symbol.upper()]
    else:
        trading_pair = f"{symbol.upper()}USDT"
    
    # Check if this is a valid trading pair
    if trading_pair not in VALID_PAIRS:
        logger.warning(f"Trading pair {trading_pair} not found on Binance")
        return None
    
    # Get klines (candlesticks)
    params = {
        'symbol': trading_pair,
        'interval': '1d',
        'limit': min(1000, days)  # Binance limit is 1000 candles
    }
    
    data = make_api_request(BINANCE_KLINES_URL, params=params)
    
    if data:
        # Format the data
        result = []
        for kline in data:
            timestamp = int(kline[0] / 1000)  # Convert ms to s
            entry = {
                "timestamp": timestamp,
                "datetime": datetime.fromtimestamp(timestamp).isoformat(),
                "price": float(kline[4]),  # Close price
                "volume": float(kline[5]),  # Volume
                "market_cap": 0  # Not available from Binance
            }
            result.append(entry)
        
        # Cache the result
        save_to_cache(cache_key, result, 'historical')
        logger.info(f"Retrieved {len(result)} historical data points for {symbol} from Binance")
        return result
    
    logger.warning(f"Failed to get historical data for {symbol} from Binance")
    return None

def get_historical_data_cryptocompare(symbol, days=30):
    """Get historical data from CryptoCompare."""
    # Check cache first
    cache_key = f"historical_cryptocompare_{symbol}_{days}"
    cached_data = load_from_cache(cache_key)
    if cached_data is not None:
        return cached_data
    
    # Add API key if available
    headers = {}
    try:
        from config import CRYPTOCOMPARE_API_KEY
        if CRYPTOCOMPARE_API_KEY:
            headers["authorization"] = f"Apikey {CRYPTOCOMPARE_API_KEY}"
    except ImportError:
        pass  # No API key available
    
    params = {
        "fsym": symbol.upper(),
        "tsym": "USD",
        "limit": days
    }
    
    data = make_api_request(CRYPTOCOMPARE_HISTO_URL, params=params, headers=headers)
    
    if data and "Data" in data and "Data" in data["Data"]:
        historical_data = data["Data"]["Data"]
        
        # Format the data
        result = []
        for candle in historical_data:
            timestamp = candle["time"]
            entry = {
                "timestamp": timestamp,
                "datetime": datetime.fromtimestamp(timestamp).isoformat(),
                "price": candle["close"],
                "volume": candle["volumefrom"],
                "market_cap": 0  # Not directly available
            }
            result.append(entry)
        
        # Cache the result
        save_to_cache(cache_key, result, 'historical')
        logger.info(f"Retrieved {len(result)} historical data points for {symbol} from CryptoCompare")
        return result
    
    logger.warning(f"Failed to get historical data for {symbol} from CryptoCompare")
    return None

def get_historical_data(symbol, days=30):
    """
    Get historical price data with fallbacks.
    
    Args:
        symbol (str): Cryptocurrency symbol (e.g., 'BTC', 'ETH')
        days (int): Number of days of history
    
    Returns:
        list: Historical price data or dummy data if all sources fail
    """
    # Try Binance first
    historical_data = get_historical_data_binance(symbol, days)
    if historical_data:
        return historical_data
    
    # Try CryptoCompare next
    historical_data = get_historical_data_cryptocompare(symbol, days)
    if historical_data:
        return historical_data
    
    # If all sources fail, generate dummy data
    logger.error(f"Failed to get historical data for {symbol} from all sources")
    return get_dummy_historical_data(symbol, days)

def get_ohlc_data_binance(symbol, days=30):
    """Get OHLC data from Binance."""
    # Check cache first
    cache_key = f"ohlc_binance_{symbol}_{days}"
    cached_data = load_from_cache(cache_key)
    if cached_data is not None:
        return cached_data
    
    # Format trading pair
    if symbol.upper() in SYMBOL_MAPPINGS:
        trading_pair = SYMBOL_MAPPINGS[symbol.upper()]
    else:
        trading_pair = f"{symbol.upper()}USDT"
    
    # Check if this is a valid trading pair
    if trading_pair not in VALID_PAIRS:
        logger.warning(f"Trading pair {trading_pair} not found on Binance")
        return None
    
    # Get klines (candlesticks)
    params = {
        'symbol': trading_pair,
        'interval': '1d',
        'limit': min(1000, days)  # Binance limit is 1000 candles
    }
    
    data = make_api_request(BINANCE_KLINES_URL, params=params)
    
    if data:
        # Format the data
        result = []
        for kline in data:
            timestamp = int(kline[0] / 1000)  # Convert ms to s
            entry = {
                "timestamp": timestamp,
                "open": float(kline[1]),
                "high": float(kline[2]),
                "low": float(kline[3]),
                "close": float(kline[4]),
                "volume": float(kline[5])
            }
            result.append(entry)
        
        # Cache the result
        save_to_cache(cache_key, result, 'historical')
        logger.info(f"Retrieved {len(result)} OHLC data points for {symbol} from Binance")
        return result
    
    logger.warning(f"Failed to get OHLC data for {symbol} from Binance")
    return None

def get_ohlc_data_cryptocompare(symbol, days=30):
    """Get OHLC data from CryptoCompare."""
    # Check cache first
    cache_key = f"ohlc_cryptocompare_{symbol}_{days}"
    cached_data = load_from_cache(cache_key)
    if cached_data is not None:
        return cached_data
    
    # Add API key if available
    headers = {}
    try:
        from config import CRYPTOCOMPARE_API_KEY
        if CRYPTOCOMPARE_API_KEY:
            headers["authorization"] = f"Apikey {CRYPTOCOMPARE_API_KEY}"
    except ImportError:
        pass  # No API key available
    
    params = {
        "fsym": symbol.upper(),
        "tsym": "USD",
        "limit": days
    }
    
    data = make_api_request(CRYPTOCOMPARE_HISTO_URL, params=params, headers=headers)
    
    if data and "Data" in data and "Data" in data["Data"]:
        historical_data = data["Data"]["Data"]
        
        # Format the data
        result = []
        for candle in historical_data:
            timestamp = candle["time"]
            entry = {
                "timestamp": timestamp,
                "open": candle["open"],
                "high": candle["high"],
                "low": candle["low"],
                "close": candle["close"],
                "volume": candle["volumefrom"]
            }
            result.append(entry)
        
        # Cache the result
        save_to_cache(cache_key, result, 'historical')
        logger.info(f"Retrieved {len(result)} OHLC data points for {symbol} from CryptoCompare")
        return result
    
    logger.warning(f"Failed to get OHLC data for {symbol} from CryptoCompare")
    return None

def get_ohlc_data(symbol, days=30):
    """
    Get OHLC (Open, High, Low, Close) data with fallbacks.
    
    Args:
        symbol (str): Cryptocurrency symbol (e.g., 'BTC', 'ETH')
        days (int): Number of days of history
    
    Returns:
        list: OHLC data or dummy data if all sources fail
    """
    # Try Binance first
    ohlc_data = get_ohlc_data_binance(symbol, days)
    if ohlc_data:
        return ohlc_data
    
    # Try CryptoCompare next
    ohlc_data = get_ohlc_data_cryptocompare(symbol, days)
    if ohlc_data:
        return ohlc_data
    
    # If all sources fail, generate dummy data
    logger.error(f"Failed to get OHLC data for {symbol} from all sources")
    return get_dummy_ohlc_data(symbol, days)

def get_global_market_data_cryptocompare():
    """Get global market data from CryptoCompare."""
    # Check cache first
    cache_key = "global_market_cryptocompare"
    cached_data = load_from_cache(cache_key)
    if cached_data is not None:
        return cached_data
    
    # Add API key if available
    headers = {}
    try:
        from config import CRYPTOCOMPARE_API_KEY
        if CRYPTOCOMPARE_API_KEY:
            headers["authorization"] = f"Apikey {CRYPTOCOMPARE_API_KEY}"
    except ImportError:
        pass  # No API key available
    
    # Get top coins by volume to calculate global metrics
    params = {
        "limit": 20,
        "tsym": "USD"
    }
    
    data = make_api_request(CRYPTOCOMPARE_GLOBAL_URL, params=params, headers=headers)
    
    if data and "Data" in data:
        coins_data = data["Data"]
        
        # Calculate total market cap and volume
        total_market_cap = 0
        total_volume = 0
        btc_market_cap = 0
        eth_market_cap = 0
        
        for coin in coins_data:
            coin_info = coin.get("RAW", {}).get("USD", {})
            market_cap = coin_info.get("MKTCAP", 0)
            total_market_cap += market_cap
            total_volume += coin_info.get("TOTALVOLUME24HTO", 0)
            
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
        
        # Cache the result
        save_to_cache(cache_key, result, 'global')
        logger.info("Retrieved global market data from CryptoCompare")
        return result
    
    logger.warning("Failed to get global market data from CryptoCompare")
    return None

def get_global_market_data():
    """
    Get global cryptocurrency market data.
    
    Returns:
        dict: Global market data or estimated data if source fails
    """
    # Try CryptoCompare
    global_data = get_global_market_data_cryptocompare()
    if global_data:
        return global_data
    
    # If failed, generate estimated data based on top coins
    logger.warning("Failed to get global market data, generating estimated data")
    return get_dummy_global_market_data()

def get_trending_coins_cryptocompare():
    """Get trending coins from CryptoCompare (top volume as proxy)."""
    # Check cache first
    cache_key = "trending_coins_cryptocompare"
    cached_data = load_from_cache(cache_key)
    if cached_data is not None:
        return cached_data
    
    # Add API key if available
    headers = {}
    try:
        from config import CRYPTOCOMPARE_API_KEY
        if CRYPTOCOMPARE_API_KEY:
            headers["authorization"] = f"Apikey {CRYPTOCOMPARE_API_KEY}"
    except ImportError:
        pass  # No API key available
    
    # Use top volume coins as "trending"
    params = {
        "limit": 15,
        "tsym": "USD"
    }
    
    data = make_api_request(CRYPTOCOMPARE_GLOBAL_URL, params=params, headers=headers)
    
    if data and "Data" in data:
        coins_data = data["Data"]
        
        # Format the results
        result = []
        for i, coin in enumerate(coins_data):
            coin_info = coin.get("CoinInfo", {})
            
            result.append({
                "id": coin_info.get("Name", "").lower(),
                "name": coin_info.get("FullName", ""),
                "symbol": coin_info.get("Name", "").upper(),
                "market_cap_rank": i + 1,
                "thumb": coin_info.get("ImageUrl", ""),
                "score": 15 - i  # Simple score based on rank
            })
        
        # Cache the result
        save_to_cache(cache_key, result, 'trending')
        logger.info(f"Retrieved {len(result)} trending coins from CryptoCompare")
        return result
    
    logger.warning("Failed to get trending coins from CryptoCompare")
    return None

def get_trending_coins():
    """
    Get trending cryptocurrencies.
    
    Returns:
        list: Trending coins data or dummy data if source fails
    """
    # Try CryptoCompare
    trending_coins = get_trending_coins_cryptocompare()
    if trending_coins:
        return trending_coins
    
    # If failed, generate dummy data
    logger.warning("Failed to get trending coins, generating dummy data")
    return get_dummy_trending_coins()

# ----- Fallback dummy data functions -----

def get_dummy_price(symbol):
    """
    Generate a realistic dummy price for a cryptocurrency.
    
    Args:
        symbol (str): Cryptocurrency symbol
    
    Returns:
        float: Dummy price
    """
    # Base prices for common cryptocurrencies (approximate as of 2023)
    base_prices = {
        'BTC': 65000,  # Update to more recent values
        'ETH': 3500,
        'SOL': 150,
        'BNB': 550,
        'ADA': 0.45,
        'XRP': 0.65,
        'DOGE': 0.075,
        'DOT': 6.5,
        'AVAX': 30,
        'MATIC': 0.75,
        'LTC': 80,
        'LINK': 15,
        'UNI': 7,
        'SHIB': 0.000022,
        'TRX': 0.12,
    }
    
    # Use the base price if available, otherwise generate a random price
    base = base_prices.get(symbol.upper(), 10)
    
    # Add some randomness to the price (±5%)
    variation = random.uniform(0.95, 1.05)
    return round(base * variation, 8 if base < 0.01 else 6 if base < 1 else 2)

def get_dummy_market_data(symbol):
    """
    Generate dummy market data for a cryptocurrency.
    
    Args:
        symbol (str): Cryptocurrency symbol
        
    Returns:
        dict: Dummy market data
    """
    price = get_dummy_price(symbol)
    
    # Generate realistic market data
    return {
        "symbol": symbol.upper(),
        "name": symbol.upper(),
        "current_price": price,
        "market_cap": price * (10**9) if symbol.upper() in ['BTC', 'ETH'] else price * (10**8),
        "total_volume": price * (10**7) if symbol.upper() in ['BTC', 'ETH'] else price * (10**6),
        "high_24h": price * random.uniform(1.01, 1.05),
        "low_24h": price * random.uniform(0.95, 0.99),
        "price_change_24h": random.uniform(-5, 5),
        "price_change_7d": random.uniform(-10, 10),
        "price_change_30d": random.uniform(-20, 20),
        "market_cap_rank": random.randint(1, 100),
        "supply": {
            "total": random.randint(10**6, 10**9),
            "circulating": random.randint(10**6, 10**9),
            "max": random.randint(10**6, 10**9)
        }
    }

def get_dummy_historical_data(symbol, days=30):
    """
    Generate dummy historical data for a cryptocurrency.
    
    Args:
        symbol (str): Cryptocurrency symbol
        days (int): Number of days of history
        
    Returns:
        list: List of dummy historical data
    """
    current_price = get_dummy_price(symbol)
    
    result = []
    now = int(time.time())
    day_seconds = 86400  # Seconds in a day
    
    # Generate price with some random walk
    price = current_price * random.uniform(0.5, 1.5)  # Start with some variation
    
    for i in range(days):
        # Moving backwards in time
        day = days - i - 1
        timestamp = now - (day * day_seconds)
        
        # Add some price movement (more volatile for smaller coins)
        volatility = 0.02  # Default 2% volatility
        if symbol.upper() == 'BTC':
            volatility = 0.01  # 1% for BTC
        elif symbol.upper() == 'ETH':
            volatility = 0.015  # 1.5% for ETH
        
        # Random walk with slight upward bias
        price_change = price * random.uniform(-volatility, volatility * 1.1)
        price += price_change
        
        # Ensure price doesn't go below zero
        price = max(price, 0.00001)
        
        entry = {
            "timestamp": timestamp,
            "datetime": datetime.fromtimestamp(timestamp).isoformat(),
            "price": price,
            "volume": price * random.uniform(10**6, 10**8),
            "market_cap": price * random.uniform(10**8, 10**10)
        }
        
        result.append(entry)
    
    # Reverse so most recent is last (matching API format)
    result.reverse()
    
    return result

def get_dummy_ohlc_data(symbol, days=30):
    """
    Generate dummy OHLC data for a cryptocurrency.
    
    Args:
        symbol (str): Cryptocurrency symbol
        days (int): Number of days of history
        
    Returns:
        list: List of dummy OHLC data
    """
    current_price = get_dummy_price(symbol)
    
    result = []
    now = int(time.time())
    day_seconds = 86400  # Seconds in a day
    
    # Generate price with some random walk
    price = current_price * random.uniform(0.5, 1.5)  # Start with some variation
    
    for i in range(days):
        # Moving backwards in time
        day = days - i - 1
        timestamp = now - (day * day_seconds)
        
        # Add some random variation for open/high/low/close
        volatility = 0.02  # Default 2% volatility
        if symbol.upper() == 'BTC':
            volatility = 0.01  # 1% for BTC
        elif symbol.upper() == 'ETH':
            volatility = 0.015  # 1.5% for ETH
        
        # Random walk with slight upward bias
        price_change = price * random.uniform(-volatility, volatility * 1.1)
        price += price_change
        
        # Ensure price doesn't go below zero
        price = max(price, 0.00001)
        
        # Generate OHLC with realistic relationships
        daily_volatility = price * volatility
        open_price = price - price_change  # Previous day's close
        close_price = price
        high_price = max(open_price, close_price) + abs(random.uniform(0, daily_volatility))
        low_price = min(open_price, close_price) - abs(random.uniform(0, daily_volatility))
        
        entry = {
            "timestamp": timestamp,
            "open": open_price,
            "high": high_price,
            "low": low_price,
            "close": close_price,
            "volume": price * random.uniform(10**6, 10**8)
        }
        
        result.append(entry)
    
    # Reverse so most recent is last (matching API format)
    result.reverse()
    
    return result

def get_dummy_global_market_data():
    """
    Generate dummy global market data.
    
    Returns:
        dict: Dummy global market data
    """
    btc_price = get_dummy_price('BTC')
    eth_price = get_dummy_price('ETH')
    
    btc_market_cap = btc_price * 19_000_000  # Approximate BTC supply
    eth_market_cap = eth_price * 120_000_000  # Approximate ETH supply
    
    # Estimate total market cap (BTC + ETH is roughly 60% of total)
    total_market_cap = (btc_market_cap + eth_market_cap) / 0.6
    
    btc_dominance = (btc_market_cap / total_market_cap) * 100
    eth_dominance = (eth_market_cap / total_market_cap) * 100
    
    return {
        "market_cap_usd": total_market_cap,
        "volume_24h_usd": total_market_cap * 0.03,  # Typical volume is 2-4% of market cap
        "market_cap_change_24h": random.uniform(-5, 5),
        "active_cryptocurrencies": 10000,
        "markets": 600,
        "btc_dominance": btc_dominance,
        "eth_dominance": eth_dominance,
        "updated_at": int(time.time())
    }

def get_dummy_trending_coins():
    """
    Generate dummy trending coins data.
    
    Returns:
        list: Dummy trending coins
    """
    top_coins = [
        {"symbol": "BTC", "name": "Bitcoin"},
        {"symbol": "ETH", "name": "Ethereum"},
        {"symbol": "BNB", "name": "Binance Coin"},
        {"symbol": "SOL", "name": "Solana"},
        {"symbol": "XRP", "name": "XRP"},
        {"symbol": "ADA", "name": "Cardano"},
        {"symbol": "DOGE", "name": "Dogecoin"},
        {"symbol": "DOT", "name": "Polkadot"},
        {"symbol": "SHIB", "name": "Shiba Inu"},
        {"symbol": "TRX", "name": "TRON"},
        {"symbol": "AVAX", "name": "Avalanche"},
        {"symbol": "MATIC", "name": "Polygon"},
        {"symbol": "LTC", "name": "Litecoin"},
        {"symbol": "LINK", "name": "Chainlink"},
        {"symbol": "UNI", "name": "Uniswap"}
    ]
    
    # Shuffle to randomize "trending" status
    random.shuffle(top_coins)
    
    # Take the first 7 as trending
    trending = top_coins[:7]
    
    # Format the results
    result = []
    for i, coin in enumerate(trending):
        result.append({
            "id": coin["symbol"].lower(),
            "name": coin["name"],
            "symbol": coin["symbol"],
            "market_cap_rank": i + 1,
            "thumb": "",  # No image URL for dummy data
            "score": 7 - i  # Higher score for higher ranking
        })
    
    return result

# Initialize the cache
def init_cache():
    """Initialize the cache directory."""
    if not os.path.exists(CACHE_DIR):
        try:
            os.makedirs(CACHE_DIR, exist_ok=True)
            logger.info(f"Created cache directory at {CACHE_DIR}")
        except Exception as e:
            logger.warning(f"Failed to create cache directory: {str(e)}")

# Call init at module import
init_cache()