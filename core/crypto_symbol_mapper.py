"""
Crypto Symbol Mapper

Maps cryptocurrency names to their symbols and vice versa.
Also handles detection of crypto mentions in text.
"""

import logging
import re

# Set up logging
logger = logging.getLogger('crypto_bot.symbol_mapper')

# Common cryptocurrency mappings
CRYPTO_MAPPINGS = {
    "bitcoin": "BTC",
    "ethereum": "ETH",
    "solana": "SOL",
    "cardano": "ADA",
    "binance coin": "BNB",
    "ripple": "XRP",
    "dogecoin": "DOGE",
    "polkadot": "DOT",
    "avalanche": "AVAX",
    "litecoin": "LTC",
    "chainlink": "LINK",
    "polygon": "MATIC",
    "stellar": "XLM",
    "tron": "TRX",
    "uniswap": "UNI",
    "algorand": "ALGO",
    "cosmos": "ATOM",
    "monero": "XMR",
    "filecoin": "FIL"
}

# Add lowercase symbol mappings (e.g., "btc" -> "BTC")
SYMBOL_MAPPINGS = {k.lower(): v for k, v in CRYPTO_MAPPINGS.items()}
for symbol in CRYPTO_MAPPINGS.values():
    SYMBOL_MAPPINGS[symbol.lower()] = symbol

def name_to_symbol(name):
    """
    Convert a cryptocurrency name to its symbol.
    
    Args:
        name (str): Cryptocurrency name (e.g., 'bitcoin')
    
    Returns:
        str: Cryptocurrency symbol (e.g., 'BTC') or None if not found
    """
    return SYMBOL_MAPPINGS.get(name.lower())

def symbol_to_name(symbol):
    """
    Convert a cryptocurrency symbol to its name.
    
    Args:
        symbol (str): Cryptocurrency symbol (e.g., 'BTC')
    
    Returns:
        str: Cryptocurrency name (e.g., 'Bitcoin') or None if not found
    """
    symbol = symbol.upper()
    for name, sym in CRYPTO_MAPPINGS.items():
        if sym == symbol:
            return name.title()
    return None

def extract_crypto_mentions(text):
    """
    Extract cryptocurrency mentions from text.
    
    Args:
        text (str): Text to analyze
    
    Returns:
        list: List of cryptocurrency symbols mentioned in the text
    """
    text = text.lower()
    mentioned = set()
    
    # Check for cryptocurrency names and symbols
    for term, symbol in SYMBOL_MAPPINGS.items():
        pattern = r'\b' + re.escape(term) + r'\b'
        if re.search(pattern, text):
            mentioned.add(symbol)
    
    return list(mentioned)

def get_exchange_symbol(symbol, exchange='binance'):
    """
    Get the exchange-specific symbol format.
    
    Args:
        symbol (str): Base cryptocurrency symbol (e.g., 'BTC')
        exchange (str): Exchange name (default: 'binance')
    
    Returns:
        str: Exchange-specific symbol (e.g., 'BTCUSDT' for Binance)
    """
    exchange = exchange.lower()
    symbol = symbol.upper()  # Ensure uppercase for consistency
    
    # Exchange-specific symbol formats
    exchange_formats = {
        'binance': f"{symbol}USDT",
        'coinbase': f"{symbol}-USD",  # Coinbase uses hyphens
        'coinbasepro': f"{symbol}-USD",
        'kraken': f"{symbol}/USD",
        'bitfinex': f"{symbol}USD",
        'kucoin': f"{symbol}-USDT"
    }
    
    # Return exchange-specific format, or default format if not recognized
    return exchange_formats.get(exchange, f"{symbol}USDT")

def is_valid_symbol(symbol):
    """
    Check if a symbol is a recognized cryptocurrency symbol.
    
    Args:
        symbol (str): Cryptocurrency symbol to check
    
    Returns:
        bool: True if valid, False otherwise
    """
    return symbol.upper() in [s.upper() for s in CRYPTO_MAPPINGS.values()]