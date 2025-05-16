"""
Trade Executor with Dynamic Position Sizing

Executes trades based on signals from the GPT Fusion Analyzer.
Handles both paper trading and live trading (when configured).
Uses CCXT library for exchange compatibility.
Includes dynamic position sizing and risk management.
"""

import logging
import uuid
import json
import ccxt
from datetime import datetime
import config
import os
import math

# Set up logging
logger = logging.getLogger('crypto_bot.trade_executor')

# Store paper trading history
paper_trades = []
paper_balance = {
    'USD': 1000.00,  # Starting with $1000 for paper trading
    'BTC': 0,
    'ETH': 0,
    'SOL': 0
}

"""
Coinbase API fix for trade_executor.py

This update fixes the Coinbase API integration by implementing:
1. Proper Coinbase authentication
2. Using the Coinbase REST API instead of Advanced Trade
3. Implementing more robust fallback mechanisms
"""

def get_exchange():
    """
    Initialize the exchange API client.
    
    Returns:
        ccxt.Exchange: Exchange API client
    """
    try:
        # Get exchange ID from config
        exchange_id = config.EXCHANGE_ID.lower()
        
        # Check if the exchange is supported by ccxt
        if not hasattr(ccxt, exchange_id):
            logger.error(f"Exchange {exchange_id} is not supported by ccxt")
            return None
        
        # Create exchange instance with proper parameters
        exchange_class = getattr(ccxt, exchange_id)
        
        # Prepare options based on exchange type
        options = {
            'apiKey': config.EXCHANGE_API_KEY,
            'secret': config.EXCHANGE_API_SECRET,
            'enableRateLimit': True,
        }
        
        # Special handling for Coinbase
        if exchange_id == 'coinbase':
            # Instead of using Advanced Trade API, use the regular REST API
            # This works more reliably with the current CCXT version
            options['options'] = {
                'createMarketBuyOrderRequiresPrice': False,
                'sandboxMode': False,  # Don't use sandbox mode with Coinbase
                'version': 'v2',       # Use the v2 API endpoint
            }
        
        # Initialize the exchange
        exchange = exchange_class(options)
        
        # Use testnet if in sandbox mode, but skip for exchanges that don't support it
        if config.EXCHANGE_SANDBOX_MODE and exchange_id != 'coinbase':
            try:
                exchange.set_sandbox_mode(True)
                logger.info(f"Connected to {exchange_id} in sandbox mode")
            except Exception as e:
                logger.warning(f"Sandbox mode not supported for {exchange_id}: {str(e)}")
                logger.info(f"Connected to {exchange_id} in {'paper trading' if not getattr(config, 'LIVE_TRADING_ENABLED', False) else 'live'} mode")
        else:
            logger.info(f"Connected to {exchange_id} in {'paper trading' if not getattr(config, 'LIVE_TRADING_ENABLED', False) else 'live'} mode")
        
        return exchange
    
    except Exception as e:
        logger.error(f"Error initializing exchange: {str(e)}")
        return None

def get_market_price(symbol):
    """
    Get the current market price of a cryptocurrency.
    
    Args:
        symbol (str): Cryptocurrency symbol
        
    Returns:
        float: Current market price
    """
    # First try using CryptoCompare (more reliable)
    try:
        from core import crypto_compare_api
        market_data = crypto_compare_api.get_market_data(symbol)
        if market_data and 'current_price' in market_data:
            price = market_data.get('current_price')
            if price:
                logger.info(f"Price for {symbol} (CryptoCompare): {price} USD")
                return price
    except Exception as e:
        logger.warning(f"Error getting price from CryptoCompare: {str(e)}")
    
    # If CryptoCompare fails, try using exchange
    try:
        exchange = get_exchange()
        
        if exchange:
            try:
                # For Coinbase, use a different approach to get price
                if config.EXCHANGE_ID.lower() == 'coinbase':
                    # Directly get the spot price - this endpoint works more reliably
                    import requests
                    
                    # Coinbase public API doesn't need authentication
                    price_endpoint = f"https://api.coinbase.com/v2/prices/{symbol}-USD/spot"
                    
                    response = requests.get(price_endpoint)
                    if response.status_code == 200:
                        data = response.json()
                        if 'data' in data and 'amount' in data['data']:
                            price = float(data['data']['amount'])
                            logger.info(f"Price for {symbol} (Coinbase direct API): {price} USD")
                            return price
                
                # Standard approach using CCXT
                from core import crypto_symbol_mapper
                market_symbol = crypto_symbol_mapper.get_exchange_symbol(symbol, exchange=config.EXCHANGE_ID)
                
                # Try to get ticker
                try:
                    ticker = exchange.fetch_ticker(market_symbol)
                    
                    # Check if ticker has last price
                    if isinstance(ticker, dict) and 'last' in ticker:
                        price = ticker['last']
                        logger.info(f"Price for {symbol} (exchange): {price} USD")
                        return price
                except Exception as e:
                    logger.warning(f"Error with fetch_ticker: {str(e)}")
            
            except Exception as e:
                logger.warning(f"Error preparing exchange request: {str(e)}")
    
    except Exception as e:
        logger.warning(f"Error getting price from exchange: {str(e)}")
    
    # If all else fails, use technical_indicator_fetcher
    from core import technical_indicator_fetcher
    price = technical_indicator_fetcher.get_price(symbol)
    logger.info(f"Price for {symbol} (fallback): {price} USD")
    return price

def get_account_balance(paper_trading=True):
    """
    Get the current account balance.
    
    Args:
        paper_trading (bool): Whether to use paper trading balance or live exchange
    
    Returns:
        dict: Account balance information
    """
    if paper_trading:
        # Return the paper trading balance
        return paper_balance
    
    try:
        exchange = get_exchange()
        
        if not exchange:
            logger.error("Failed to initialize exchange for balance check")
            return None
        
        # Fetch balance from exchange
        balance = exchange.fetch_balance()
        
        # Format and return the balance
        formatted_balance = {}
        for currency, amount in balance['total'].items():
            if amount > 0:
                formatted_balance[currency] = {
                    'total': amount,
                    'free': balance['free'].get(currency, 0),
                    'used': balance['used'].get(currency, 0)
                }
        
        logger.info(f"Account balance retrieved: {formatted_balance}")
        return formatted_balance
    
    except Exception as e:
        logger.error(f"Error getting account balance: {str(e)}")
        return None

def calculate_position_size(symbol, action, conviction, context, paper_trading=True):
    """
    Calculate dynamic position size based on market conditions and conviction.
    
    Args:
        symbol (str): Cryptocurrency symbol
        action (str): Trade action ('buy' or 'sell')
        conviction (int): Conviction score (-100 to +100)
        context (dict): Market context with sentiment and technical data
        paper_trading (bool): Whether to use paper trading balance or live exchange
    
    Returns:
        float: Position size to trade
    """
    # Get the current price of the symbol
    price = get_market_price(symbol)
    if not price:
        logger.error(f"Could not get price for {symbol}, using default position size")
        return getattr(config, 'DEFAULT_POSITION_SIZE', {}).get(symbol, 0.01)
    
    # Get account balance
    balance = get_account_balance(paper_trading)
    if not balance:
        logger.error("Could not get account balance, using default position size")
        return getattr(config, 'DEFAULT_POSITION_SIZE', {}).get(symbol, 0.01)
    
    # For sell orders, check if we have enough of the asset
    if action == 'sell':
        asset_balance = balance.get(symbol, 0)
        if paper_trading:
            asset_balance = balance.get(symbol, 0)
        else:
            asset_balance = balance.get(symbol, {}).get('free', 0)
        
        if asset_balance <= 0:
            logger.warning(f"No {symbol} balance available for selling")
            return 0
    
    # For buy orders, check available cash (USD)
    available_cash = 0
    if paper_trading:
        available_cash = balance.get('USD', 0)
    else:
        available_cash = balance.get('USD', {}).get('free', 0)
    
    if available_cash <= 0 and action == 'buy':
        logger.warning("No USD balance available for buying")
        return 0
    
    # Calculate risk percentage based on conviction and market conditions
    # Higher conviction = higher risk percentage (1-10%)
    base_risk = min(10, max(1, abs(conviction) / 10))
    
    # Adjust risk based on market conditions
    
    # 1. Fear & Greed Index (contrarian)
    fear_greed = context.get('fear_greed_index', 'Neutral')
    if fear_greed == "Extreme Fear":
        # Buy more in extreme fear (contrarian)
        if action == 'buy':
            base_risk *= 1.5
        else:
            base_risk *= 0.7
    elif fear_greed == "Extreme Greed":
        # Sell more in extreme greed (contrarian)
        if action == 'sell':
            base_risk *= 1.5
        else:
            base_risk *= 0.7
    
    # 2. Market trend (from price data)
    if 'crypto_data' in context:
        price_change_24h = context['crypto_data'].get('price_change_24h', 0)
        price_change_7d = context['crypto_data'].get('price_change_7d', 0)
        
        # In strong uptrend
        if price_change_24h > 5 and price_change_7d > 10:
            if action == 'buy':
                base_risk *= 1.2  # More aggressive buying in uptrends
            else:
                base_risk *= 0.8  # More cautious selling in uptrends
        
        # In strong downtrend
        elif price_change_24h < -5 and price_change_7d < -10:
            if action == 'sell':
                base_risk *= 1.2  # More aggressive selling in downtrends
            else:
                base_risk *= 0.5  # Much more cautious buying in downtrends
    
    # 3. Technical indicators
    if 'rsi' in context:
        rsi = context.get('rsi', 50)
        if rsi < 30 and action == 'buy':
            # Oversold condition - good buying opportunity
            base_risk *= 1.3
        elif rsi > 70 and action == 'sell':
            # Overbought condition - good selling opportunity
            base_risk *= 1.3
    
    # Cap the risk percentage at 10%
    risk_percentage = min(10, base_risk)
    
    # Calculate amount to trade in USD
    if action == 'buy':
        # For buy orders, calculate the USD amount to use
        amount_usd = available_cash * (risk_percentage / 100)
        
        # Ensure we have a minimum trade size
        min_trade_usd = 10  # Minimum $10 trade
        amount_usd = max(min_trade_usd, min(amount_usd, available_cash * 0.95))  # Don't use more than 95% of available cash
        
        # Convert to coin amount
        position_size = amount_usd / price
        
        # Round to appropriate decimal places based on price
        if price > 1000:  # Like BTC
            position_size = round(position_size, 5)
        elif price > 100:  # Like ETH
            position_size = round(position_size, 4)
        else:  # Like most altcoins
            position_size = round(position_size, 3)
    else:
        # For sell orders, calculate percentage of holdings to sell
        if paper_trading:
            max_position = balance.get(symbol, 0)
        else:
            max_position = balance.get(symbol, {}).get('free', 0)
        
        # Use a percentage of holdings based on conviction
        sell_percentage = risk_percentage
        position_size = max_position * (sell_percentage / 100)
        
        # Ensure we're not selling more than available
        position_size = min(position_size, max_position * 0.95)  # Don't sell more than 95% at once
    
    # Ensure the position size is not too small
    min_position = getattr(config, 'MIN_POSITION_SIZE', {}).get(symbol, 0.001)
    if position_size < min_position:
        logger.warning(f"Calculated position size {position_size} {symbol} is below minimum, using {min_position}")
        position_size = min_position
    
    # Log the decision
    logger.info(f"Dynamic position sizing: {position_size} {symbol} (Risk: {risk_percentage:.1f}%, " + 
                f"Conviction: {conviction}, Action: {action})")
    
    return position_size

def update_paper_balance(symbol, action, amount, price):
    """
    Update the paper trading balance after a trade.
    
    Args:
        symbol (str): Cryptocurrency symbol
        action (str): Trade action ('buy' or 'sell')
        amount (float): Amount of cryptocurrency traded
        price (float): Trade price
    """
    global paper_balance
    
    # Calculate the USD value
    usd_value = amount * price
    
    if action == 'buy':
        # Deduct USD, add crypto
        if paper_balance.get('USD', 0) >= usd_value:
            paper_balance['USD'] -= usd_value
            paper_balance[symbol] = paper_balance.get(symbol, 0) + amount
            logger.info(f"Paper balance updated: Bought {amount} {symbol} for ${usd_value:.2f}")
        else:
            logger.error(f"Insufficient paper USD balance for trade. Required: ${usd_value:.2f}, Available: ${paper_balance.get('USD', 0):.2f}")
    elif action == 'sell':
        # Add USD, deduct crypto
        if paper_balance.get(symbol, 0) >= amount:
            paper_balance['USD'] += usd_value
            paper_balance[symbol] -= amount
            logger.info(f"Paper balance updated: Sold {amount} {symbol} for ${usd_value:.2f}")
        else:
            logger.error(f"Insufficient paper {symbol} balance for trade. Required: {amount}, Available: {paper_balance.get(symbol, 0)}")
    
    # Log the new balance
    logger.info(f"Current paper balance: {paper_balance}")

def execute_paper_trade(symbol, action, conviction, amount=None, context=None):
    """
    Execute a paper trade and log the details.
    
    Args:
        symbol (str): Cryptocurrency symbol (e.g., 'BTC', 'ETH')
        action (str): Trade action ('buy' or 'sell')
        conviction (int): Conviction score (-100 to +100)
        amount (float, optional): Position size in the cryptocurrency. If None, calculated dynamically.
        context (dict, optional): Market context for dynamic position sizing
        
    Returns:
        dict: Trade details
    """
    # Generate a unique trade ID
    trade_id = str(uuid.uuid4())
    
    # Get the current market price
    price = get_market_price(symbol)
    
    # Determine position size dynamically if not provided and context is available
    if amount is None and context is not None:
        amount = calculate_position_size(symbol, action, conviction, context, paper_trading=True)
    elif amount is None:
        # Fallback to default position size
        amount = getattr(config, 'DEFAULT_POSITION_SIZE', {}).get(symbol, 0.01)
    
    # Calculate the trade size in USD
    usd_value = price * amount
    
    # Update paper balance
    update_paper_balance(symbol, action, amount, price)
    
    # Record the trade details
    trade = {
        "trade_id": trade_id,
        "timestamp": datetime.now().isoformat(),
        "symbol": symbol,
        "action": action,
        "amount": amount,
        "price": price,
        "usd_value": usd_value,
        "conviction": conviction,
        "type": "paper",
        "status": "executed"
    }
    
    # Add to paper trading history
    paper_trades.append(trade)
    
    # Log the trade
    logger.info(f"PAPER TRADE: {action.upper()} {amount} {symbol} at ${price} (${usd_value:.2f} USD)")
    
    return trade

def execute_live_trade(symbol, action, conviction, amount=None, context=None):
    """
    Execute a live trade on the configured exchange.
    
    Args:
        symbol (str): Cryptocurrency symbol (e.g., 'BTC', 'ETH')
        action (str): Trade action ('buy' or 'sell')
        conviction (int): Conviction score (-100 to +100)
        amount (float, optional): Position size in the cryptocurrency. If None, calculated dynamically.
        context (dict, optional): Market context for dynamic position sizing
        
    Returns:
        dict: Trade details or None if failed
    """
    # Check if we're allowed to do live trading
    if not getattr(config, 'LIVE_TRADING_ENABLED', False):
        logger.warning("Live trading not enabled in config. Executing paper trade instead.")
        return execute_paper_trade(symbol, action, conviction, amount, context)
    
    try:
        exchange = get_exchange()
        
        if not exchange:
            logger.error("Failed to initialize exchange for live trading")
            return None
        
        # Format the symbol for the exchange
        from core import crypto_symbol_mapper
        market_symbol = crypto_symbol_mapper.get_exchange_symbol(symbol, exchange=config.EXCHANGE_ID)
        
        # Determine position size dynamically if not provided and context is available
        if amount is None and context is not None:
            amount = calculate_position_size(symbol, action, conviction, context, paper_trading=False)
        elif amount is None:
            # Fallback to default position size
            amount = getattr(config, 'DEFAULT_POSITION_SIZE', {}).get(symbol, 0.01)
        
        # Set a stop-loss price if buying
        stop_loss_price = None
        if action == 'buy':
            # Set stop-loss 5-15% below entry depending on conviction
            stop_loss_percentage = 15 - min(10, abs(conviction) / 10)  # Higher conviction = tighter stop
            current_price = get_market_price(symbol)
            stop_loss_price = current_price * (1 - stop_loss_percentage / 100)
        
        # Create the order
        order_type = 'market'  # Use market orders for simplicity
        side = 'buy' if action == 'buy' else 'sell'
        
        logger.info(f"Placing {order_type} {side} order for {amount} {symbol}")
        
        # Place the order
        order = exchange.create_order(
            symbol=market_symbol,
            type=order_type,
            side=side,
            amount=amount
        )
        
        # Extract order details
        order_id = order.get('id', 'unknown')
        price = order.get('price', 0)
        
        if price == 0:  # Market orders might not have a price immediately
            # Get filled price from order status
            order_status = exchange.fetch_order(order_id, market_symbol)
            price = order_status.get('price', 0)
            if price == 0:
                # If still no price, use current market price
                price = get_market_price(symbol)
        
        # Calculate USD value
        usd_amount = price * amount
        
        # Set stop-loss if configured
        if stop_loss_price and action == 'buy' and getattr(config, 'USE_STOP_LOSS', True):
            try:
                stop_order = exchange.create_order(
                    symbol=market_symbol,
                    type='stop_loss',
                    side='sell',
                    amount=amount,
                    price=stop_loss_price
                )
                logger.info(f"Stop-loss set for {symbol} at ${stop_loss_price:.2f} (order ID: {stop_order.get('id', 'unknown')})")
            except Exception as e:
                logger.error(f"Failed to set stop-loss: {str(e)}")
        
        # Record the trade details
        trade = {
            "trade_id": order_id,
            "timestamp": datetime.now().isoformat(),
            "symbol": symbol,
            "action": action,
            "amount": amount,
            "price": price,
            "usd_value": usd_amount,
            "conviction": conviction,
            "type": "live",
            "status": "executed",
            "order": order
        }
        
        # Log the trade
        logger.info(f"LIVE TRADE: {action.upper()} {amount} {symbol} at ${price} (${usd_amount:.2f} USD)")
        
        # Save trade history
        save_trade_history()
        
        return trade
    
    except Exception as e:
        logger.error(f"Error executing live trade: {str(e)}")
        return None

def execute_trade(symbol, action, conviction, amount=None, context=None):
    """
    Execute a trade, either paper or live based on configuration.
    
    Args:
        symbol (str): Cryptocurrency symbol (e.g., 'BTC', 'ETH')
        action (str): Trade action ('buy' or 'sell')
        conviction (int): Conviction score (-100 to +100)
        amount (float, optional): Position size in the cryptocurrency. If None, calculated dynamically.
        context (dict, optional): Market context for dynamic position sizing
        
    Returns:
        dict: Trade details
    """
    if getattr(config, 'LIVE_TRADING_ENABLED', False):
        return execute_live_trade(symbol, action, conviction, amount, context)
    else:
        return execute_paper_trade(symbol, action, conviction, amount, context)

def get_trading_history(type="paper"):
    """
    Get the trading history.
    
    Args:
        type (str): Type of trades to get ('paper', 'live', 'all')
        
    Returns:
        list: List of trades
    """
    if type == "paper":
        return paper_trades
    elif type == "live":
        # In a real implementation, this would fetch live trade history from the exchange
        return []
    else:  # 'all'
        # In a real implementation, this would combine paper and live trades
        return paper_trades

def save_trade_history():
    """
    Save the paper trading history to a file.
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Create the data directory if it doesn't exist
        if not os.path.exists('data'):
            os.makedirs('data')
            
        # Save paper trades
        with open('data/paper_trades.json', 'w') as f:
            json.dump(paper_trades, f, indent=2)
            
        # Save paper balance
        with open('data/paper_balance.json', 'w') as f:
            json.dump(paper_balance, f, indent=2)
            
        return True
    except Exception as e:
        logger.error(f"Error saving trade history: {str(e)}")
        return False

def load_trade_history():
    """
    Load the paper trading history from a file.
    
    Returns:
        bool: True if successful, False otherwise
    """
    global paper_trades, paper_balance
    
    try:
        # Load paper trades
        if os.path.exists('data/paper_trades.json'):
            with open('data/paper_trades.json', 'r') as f:
                paper_trades = json.load(f)
                
        # Load paper balance
        if os.path.exists('data/paper_balance.json'):
            with open('data/paper_balance.json', 'r') as f:
                paper_balance = json.load(f)
        else:
            # Initialize with default balance
            paper_balance = {
                'USD': 1000.00,  # Default starting balance
                'BTC': 0,
                'ETH': 0,
                'SOL': 0
            }
            
        return True
    except FileNotFoundError:
        logger.info("No trade history file found. Starting with empty history.")
        return False
    except Exception as e:
        logger.error(f"Error loading trade history: {str(e)}")
        return False

def get_portfolio_value():
    """
    Calculate the current portfolio value based on paper trades.
    
    Returns:
        dict: Portfolio value breakdown
    """
    # Use the paper balance for a more accurate portfolio value
    portfolio_usd = {}
    total_usd = paper_balance.get('USD', 0)
    
    # Calculate value of each crypto holding
    for symbol in ['BTC', 'ETH', 'SOL']:  # Add more symbols as needed
        amount = paper_balance.get(symbol, 0)
        if amount > 0:
            try:
                price = get_market_price(symbol)
                usd_value = amount * price
                
                portfolio_usd[symbol] = {
                    'amount': amount,
                    'price': price,
                    'usd_value': usd_value
                }
                
                total_usd += usd_value
            except Exception as e:
                logger.error(f"Error getting price for {symbol}: {str(e)}")
    
    return {
        'holdings': portfolio_usd,
        'cash_balance': paper_balance.get('USD', 0),
        'total_usd_value': total_usd,
        'portfolio_breakdown': {
            'cash_percentage': (paper_balance.get('USD', 0) / total_usd * 100) if total_usd > 0 else 0,
            'crypto_percentage': ((total_usd - paper_balance.get('USD', 0)) / total_usd * 100) if total_usd > 0 else 0
        }
    }

def calculate_performance_metrics():
    """
    Calculate performance metrics for the trading strategy.
    
    Returns:
        dict: Performance metrics
    """
    if not paper_trades:
        return {
            'total_trades': 0,
            'win_rate': 0,
            'profit_loss': 0,
            'roi': 0
        }
    
    total_trades = len(paper_trades)
    profitable_trades = 0
    total_profit_loss = 0
    
    # Initial investment (assumed to be the paper balance starting amount)
    initial_investment = 1000.0  # Default starting value
    
    # Current portfolio value
    current_value = get_portfolio_value()['total_usd_value']
    
    # Calculate ROI
    roi = ((current_value - initial_investment) / initial_investment) * 100
    
    return {
        'total_trades': total_trades,
        'win_rate': 0,  # Need more sophisticated analysis to determine this
        'profit_loss': current_value - initial_investment,
        'roi': roi,
        'current_value': current_value
    }

# Initialize by loading trade history on module import
load_trade_history()