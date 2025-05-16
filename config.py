"""
Configuration settings for crypto_bot_fusion_v1

This module contains all configurable parameters for the trading bot,
loaded from environment variables where appropriate.
"""

import os

# API Keys (loaded from environment variables)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
CRYPTOCOMPARE_API_KEY = os.getenv("CRYPTOCOMPARE_API_KEY", "")

# Exchange API Keys & Configuration
EXCHANGE_API_KEY = os.getenv("EXCHANGE_API_KEY", "")
EXCHANGE_API_SECRET = os.getenv("EXCHANGE_API_SECRET", "")
EXCHANGE_ID = os.getenv("EXCHANGE_ID", "coinbase")  # Options: binance, coinbase, kraken, etc.
EXCHANGE_SANDBOX_MODE = os.getenv("EXCHANGE_SANDBOX_MODE", "True").lower() == "true"

# Trading Mode
LIVE_TRADING_ENABLED = os.getenv("LIVE_TRADING_ENABLED", "False").lower() == "true"

# Enhanced API Keys
STOCKGEIST_API_KEY = os.getenv("STOCKGEIST_API_KEY", "")
THE_TIE_API_KEY = os.getenv("THE_TIE_API_KEY", "")
SANTIMENT_API_KEY = os.getenv("SANTIMENT_API_KEY", "")
WHALE_ALERT_API_KEY = os.getenv("WHALE_ALERT_API_KEY", "")

# GPT Model Configuration
GPT_MODEL = os.getenv("GPT_MODEL", "gpt-3.5-turbo")
GPT_TEMPERATURE = float(os.getenv("GPT_TEMPERATURE", "0.2"))
GPT_MAX_TOKENS = int(os.getenv("GPT_MAX_TOKENS", "500"))

# Trading Parameters
CONVICTION_THRESHOLD = int(os.getenv("CONVICTION_THRESHOLD", "40"))  # Minimum conviction to execute trade
SYMBOLS_TO_MONITOR = os.getenv("SYMBOLS_TO_MONITOR", "BTC,ETH,SOL").split(",")

# Paper Trading Initial Balance
PAPER_TRADING_INITIAL_BALANCE = float(os.getenv("PAPER_TRADING_INITIAL_BALANCE", "1000"))

# Default position sizes for each symbol (used when dynamic sizing fails)
DEFAULT_POSITION_SIZE = {
    "BTC": float(os.getenv("BTC_POSITION_SIZE", "0.001")),
    "ETH": float(os.getenv("ETH_POSITION_SIZE", "0.01")),
    "SOL": float(os.getenv("SOL_POSITION_SIZE", "0.1")),
}

# Minimum position sizes for each symbol
MIN_POSITION_SIZE = {
    "BTC": float(os.getenv("BTC_MIN_POSITION_SIZE", "0.0005")),
    "ETH": float(os.getenv("ETH_MIN_POSITION_SIZE", "0.005")),
    "SOL": float(os.getenv("SOL_MIN_POSITION_SIZE", "0.05")),
}

# Risk management settings
MAX_RISK_PER_TRADE = float(os.getenv("MAX_RISK_PER_TRADE", "10"))  # Max percentage of portfolio per trade
USE_STOP_LOSS = os.getenv("USE_STOP_LOSS", "True").lower() == "true"
STOP_LOSS_PERCENTAGE = float(os.getenv("STOP_LOSS_PERCENTAGE", "5"))  # Default stop loss percentage

# Technical indicator parameters
RSI_PERIOD = int(os.getenv("RSI_PERIOD", "14"))
RSI_OVERBOUGHT = int(os.getenv("RSI_OVERBOUGHT", "70"))
RSI_OVERSOLD = int(os.getenv("RSI_OVERSOLD", "30"))

# Scheduling
CYCLE_INTERVAL_SECONDS = int(os.getenv("CYCLE_INTERVAL_SECONDS", "600"))  # 10 minutes

# Logging and Debug
DEBUG_MODE = os.getenv("DEBUG_MODE", "False").lower() == "true"

# Reddit API credentials
REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID", "")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET", "")

# Twitter API credentials
TWITTER_API_KEY = os.getenv("TWITTER_API_KEY", "")
TWITTER_API_SECRET = os.getenv("TWITTER_API_SECRET", "")
TWITTER_ACCESS_TOKEN = os.getenv("TWITTER_ACCESS_TOKEN", "")
TWITTER_ACCESS_SECRET = os.getenv("TWITTER_ACCESS_SECRET", "")
TWITTER_API_ENABLED = os.getenv("TWITTER_API_ENABLED", "False").lower() == "true"

# Dynamic position sizing
DYNAMIC_POSITION_SIZING = os.getenv("DYNAMIC_POSITION_SIZING", "True").lower() == "true"
BASE_RISK_PERCENTAGE = float(os.getenv("BASE_RISK_PERCENTAGE", "2"))  # Base percentage of portfolio to risk per trade

EXCHANGE_ADVANCED_TRADE = os.getenv("EXCHANGE_ADVANCED_TRADE", "False").lower() == "true"

# ----- GPT Integration Settings -----

# Enable/disable GPT analysis (set to False to use only rule-based decisions)
GPT_ENABLED = os.getenv("GPT_ENABLED", "True").lower() == "true"

# Budget management
GPT_DAILY_BUDGET_USD = float(os.getenv("GPT_DAILY_BUDGET_USD", "1.0"))  # $1 per day by default

# Model selection
GPT_MODEL = os.getenv("GPT_MODEL", "gpt-3.5-turbo")  # Default to GPT-3.5 Turbo
GPT_TEMPERATURE = float(os.getenv("GPT_TEMPERATURE", "0.2"))  # Low temperature for more consistent results
GPT_MAX_TOKENS = int(os.getenv("GPT_MAX_TOKENS", "500"))  # Limit response size
GPT4_ENABLED = os.getenv("GPT4_ENABLED", "False").lower() == "true"  # Set to True to allow GPT-4 for important decisions

# Reporting
GENERATE_REPORTS = os.getenv("GENERATE_REPORTS", "True").lower() == "true"  # Enable trading reports
REPORT_FREQUENCY_HOURS = float(os.getenv("REPORT_FREQUENCY_HOURS", "6"))  # Generate reports every 6 hours