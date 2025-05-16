"""
Technical Indicator Fetcher

Retrieves technical indicators for cryptocurrencies.
Uses the CryptoCompare API for real data when available,
falls back to simulated data for the MVP.
"""

import requests
import logging
import random
import numpy as np
import time
from datetime import datetime, timedelta
import config
from core import crypto_symbol_mapper

# Set up logging
logger = logging.getLogger('crypto_bot.technical_indicators')

# CryptoCompare API endpoints
PRICE_URL = "https://min-api.cryptocompare.com/data/price"
HISTORICAL_URL = "https://min-api.cryptocompare.com/data/v2/histohour"

def get_price(symbol, currency='USD'):
    """
    Get the current price of a cryptocurrency.
    
    Args:
        symbol (str): Cryptocurrency symbol (e.g., 'BTC', 'ETH')
        currency (str): Currency to get price in (default: 'USD')
    
    Returns:
        float: Current price
    """
    headers = {}
    if config.CRYPTOCOMPARE_API_KEY:
        headers["authorization"] = f"Apikey {config.CRYPTOCOMPARE_API_KEY}"
    
    try:
        response = requests.get(
            PRICE_URL,
            params={"fsym": symbol, "tsyms": currency},
            headers=headers
        )
        response.raise_for_status()
        data = response.json()
        
        price = data.get(currency)
        if price:
            logger.info(f"Price for {symbol}: {price} {currency}")
            return price
        else:
            logger.error(f"Could not get price for {symbol}")
            return get_dummy_price(symbol)
            
    except Exception as e:
        logger.error(f"Error fetching price for {symbol}: {str(e)}")
        return get_dummy_price(symbol)

def get_historical_data(symbol, timeframe='1h', limit=100):
    """
    Get historical OHLCV data for a cryptocurrency.
    
    Args:
        symbol (str): Cryptocurrency symbol (e.g., 'BTC', 'ETH')
        timeframe (str): Time interval (e.g., '1h', '1d')
        limit (int): Number of data points to retrieve
        
    Returns:
        list: List of OHLCV dictionaries
    """
    headers = {}
    if config.CRYPTOCOMPARE_API_KEY:
        headers["authorization"] = f"Apikey {config.CRYPTOCOMPARE_API_KEY}"
    
    # Map timeframe to API parameter
    if timeframe == '1h':
        api_timeframe = 'hour'
    elif timeframe == '1d':
        api_timeframe = 'day'
    else:
        api_timeframe = 'hour'  # Default to hourly
    
    try:
        response = requests.get(
            HISTORICAL_URL,
            params={
                "fsym": symbol,
                "tsym": "USD",
                "limit": limit,
                "e": "CCCAGG"  # Crypto Compare aggregated data
            },
            headers=headers
        )
        response.raise_for_status()
        data = response.json()
        
        if data.get("Response") == "Success" or data.get("Response") == "success":
            ohlcv_data = data.get("Data", {}).get("Data", [])
            
            # Format the data into a more usable structure
            result = []
            for candle in ohlcv_data:
                result.append({
                    "timestamp": candle.get("time", 0),
                    "open": candle.get("open", 0),
                    "high": candle.get("high", 0),
                    "low": candle.get("low", 0),
                    "close": candle.get("close", 0),
                    "volume": candle.get("volumefrom", 0)
                })
            
            logger.info(f"Successfully fetched {len(result)} historical data points for {symbol}")
            return result
        else:
            logger.error(f"API error from CryptoCompare: {data.get('Message', 'Unknown error')}")
            return get_dummy_historical_data(symbol, limit)
            
    except Exception as e:
        logger.error(f"Error fetching historical data for {symbol}: {str(e)}")
        return get_dummy_historical_data(symbol, limit)

def get_rsi(symbol, timeframe='1h'):
    """
    Calculate the RSI (Relative Strength Index) for a cryptocurrency.
    
    Args:
        symbol (str): Cryptocurrency symbol (e.g., 'BTC', 'ETH')
        timeframe (str): Time interval (e.g., '1h', '4h', '1d')
    
    Returns:
        float: RSI value (0-100)
    """
    try:
        # Get historical data
        historical_data = get_historical_data(symbol, timeframe, limit=config.RSI_PERIOD * 3)
        
        if not historical_data or len(historical_data) < config.RSI_PERIOD + 1:
            logger.error(f"Not enough data to calculate RSI for {symbol}")
            return get_dummy_rsi()
        
        # Extract closing prices
        closes = [candle["close"] for candle in historical_data]
        
        # Calculate price changes
        deltas = np.diff(closes)
        
        # Calculate gains and losses
        gains = np.clip(deltas, 0, np.inf)
        losses = -np.clip(deltas, -np.inf, 0)
        
        # Calculate average gains and losses
        avg_gain = np.average(gains[-config.RSI_PERIOD:])
        avg_loss = np.average(losses[-config.RSI_PERIOD:])
        
        if avg_loss == 0:
            return 100.0
        
        # Calculate RS and RSI
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        logger.info(f"RSI for {symbol}: {rsi:.2f}")
        return round(rsi, 2)
    
    except Exception as e:
        logger.error(f"Error calculating RSI for {symbol}: {str(e)}")
        return get_dummy_rsi()

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
        'BTC': 42000,
        'ETH': 2200,
        'SOL': 140,
        'BNB': 320,
        'ADA': 0.4,
        'XRP': 0.55,
        'DOGE': 0.08,
        'DOT': 7.5,
        'AVAX': 35,
        'LTC': 70
    }
    
    # Use the base price if available, otherwise generate a random price
    base = base_prices.get(symbol.upper(), 10)
    
    # Add some randomness to the price (±5%)
    variation = random.uniform(0.95, 1.05)
    return round(base * variation, 2)

def get_dummy_historical_data(symbol, limit=100):
    """
    Generate dummy historical data for a cryptocurrency.
    
    Args:
        symbol (str): Cryptocurrency symbol
        limit (int): Number of data points to generate
        
    Returns:
        list: List of dummy OHLCV data
    """
    # Get a base price for the symbol
    current_price = get_dummy_price(symbol)
    
    # Generate random data points
    data = []
    timestamp = int(time.time()) - (limit * 3600)  # Start 'limit' hours ago
    
    price = current_price * random.uniform(0.8, 1.2)  # Start within ±20% of current price
    
    for i in range(limit):
        # Generate a random price movement (more volatile for smaller coins)
        volatility = 0.02  # Default 2% volatility
        if symbol.upper() == 'BTC':
            volatility = 0.01  # 1% for BTC
        elif symbol.upper() == 'ETH':
            volatility = 0.015  # 1.5% for ETH
        
        price_change = price * random.uniform(-volatility, volatility)
        price += price_change
        
        # Ensure price doesn't go negative
        price = max(price, 0.01)
        
        # Generate OHLCV data
        open_price = price
        close_price = price + price * random.uniform(-volatility/2, volatility/2)
        high_price = max(open_price, close_price) * random.uniform(1, 1 + volatility)
        low_price = min(open_price, close_price) * random.uniform(1 - volatility, 1)
        volume = price * random.uniform(100, 1000)  # Dummy volume
        
        data.append({
            "timestamp": timestamp,
            "open": round(open_price, 2),
            "high": round(high_price, 2),
            "low": round(low_price, 2),
            "close": round(close_price, 2),
            "volume": round(volume, 2)
        })
        
        # Move to next hour and update price
        timestamp += 3600
        price = close_price
    
    return data

def get_dummy_rsi():
    """
    Generate a dummy RSI value.
    
    Returns:
        float: Random RSI value between 20 and 80
    """
    return round(random.uniform(20, 80), 2)

def get_macd_signal(symbol, timeframe='1d'):
    """
    Get the MACD crossover signal for a cryptocurrency.
    
    Args:
        symbol (str): Cryptocurrency symbol
        timeframe (str): Time interval
        
    Returns:
        str: MACD signal ("bullish", "bearish", or "neutral")
    """
    try:
        # Get historical data
        historical_data = get_historical_data(symbol, timeframe, limit=35)  # Need at least 26 + a few more periods
        
        if not historical_data or len(historical_data) < 35:
            logger.error(f"Not enough data to calculate MACD for {symbol}")
            return get_dummy_macd_signal()
        
        # Extract closing prices
        closes = np.array([candle["close"] for candle in historical_data])
        
        # Calculate EMAs
        ema12 = calculate_ema(closes, 12)
        ema26 = calculate_ema(closes, 26)
        
        # Calculate MACD line
        macd_line = ema12 - ema26
        
        # Calculate signal line (9-day EMA of MACD line)
        signal_line = calculate_ema(macd_line, 9)
        
        # Determine cross-over condition
        if len(macd_line) < 2 or len(signal_line) < 2:
            return "neutral"
        
        # Check if MACD line crossed above signal line (bullish)
        if macd_line[-2] < signal_line[-2] and macd_line[-1] > signal_line[-1]:
            return "bullish"
        
        # Check if MACD line crossed below signal line (bearish)
        if macd_line[-2] > signal_line[-2] and macd_line[-1] < signal_line[-1]:
            return "bearish"
        
        # No recent crossover
        return "neutral"
        
    except Exception as e:
        logger.error(f"Error calculating MACD for {symbol}: {str(e)}")
        return get_dummy_macd_signal()

def calculate_ema(data, period):
    """
    Calculate Exponential Moving Average.
    
    Args:
        data (numpy.array): Data array
        period (int): EMA period
        
    Returns:
        numpy.array: EMA values
    """
    alpha = 2 / (period + 1)
    ema = np.zeros_like(data)
    ema[0] = data[0]
    
    for i in range(1, len(data)):
        ema[i] = data[i] * alpha + ema[i-1] * (1 - alpha)
    
    return ema

def get_dummy_macd_signal():
    """
    Generate a dummy MACD signal.
    
    Returns:
        str: Random signal ("bullish", "bearish", or "neutral")
    """
    signals = ["bullish", "bearish", "neutral"]
    return random.choice(signals)

def check_volume_spike(symbol, timeframe='1h'):
    """
    Check if there's a significant volume spike for a cryptocurrency.
    
    Args:
        symbol (str): Cryptocurrency symbol
        timeframe (str): Time interval
        
    Returns:
        str: "yes" if there's a volume spike, "no" otherwise
    """
    try:
        # Get historical data
        historical_data = get_historical_data(symbol, timeframe, limit=24)  # Last 24 periods
        
        if not historical_data or len(historical_data) < 10:
            logger.error(f"Not enough data to check volume for {symbol}")
            return "no"
        
        # Extract volume data
        volumes = [candle["volume"] for candle in historical_data]
        
        # Calculate average volume (excluding the latest)
        avg_volume = sum(volumes[:-1]) / len(volumes[:-1])
        
        # Check if the latest volume is 2x the average
        if volumes[-1] > avg_volume * 2:
            logger.info(f"Volume spike detected for {symbol}")
            return "yes"
        else:
            return "no"
            
    except Exception as e:
        logger.error(f"Error checking volume for {symbol}: {str(e)}")
        return random.choice(["yes", "no", "no", "no"])  # 25% chance of "yes"

def get_price_trend(symbol, timeframe='1h'):
    """
    Determine the price trend for a cryptocurrency.
    
    Args:
        symbol (str): Cryptocurrency symbol
        timeframe (str): Time interval
        
    Returns:
        str: Price trend description
    """
    try:
        # Get historical data
        historical_data = get_historical_data(symbol, timeframe, limit=6)  # Last 6 periods
        
        if not historical_data or len(historical_data) < 3:
            logger.error(f"Not enough data to determine trend for {symbol}")
            return get_dummy_price_trend()
        
        # Extract closing prices
        closes = [candle["close"] for candle in historical_data]
        
        # Calculate price changes
        latest_price = closes[-1]
        previous_price = closes[-2]
        
        # Calculate short-term trend (latest vs previous)
        short_term_change = (latest_price - previous_price) / previous_price
        
        # Calculate longer-term trend
        start_price = closes[0]
        long_term_change = (latest_price - start_price) / start_price
        
        # Determine trend description
        if short_term_change > 0.02:  # 2% up
            if long_term_change > 0.05:  # 5% up
                return "strongly rising"
            else:
                return "currently rising"
        elif short_term_change < -0.02:  # 2% down
            if long_term_change < -0.05:  # 5% down
                return "strongly falling"
            else:
                return "currently falling"
        else:
            return "sideways movement"
            
    except Exception as e:
        logger.error(f"Error determining price trend for {symbol}: {str(e)}")
        return get_dummy_price_trend()

def get_dummy_price_trend():
    """
    Generate a dummy price trend.
    
    Returns:
        str: Random price trend
    """
    trends = ["currently rising", "strongly rising", "currently falling", "strongly falling", "sideways movement"]
    return random.choice(trends)