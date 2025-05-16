"""
crypto_bot_fusion_v1 - Enhanced Main with GPT and CryptoCompare Integration

This version integrates GPT analysis with budget management to keep costs under $1/day
while providing intelligent trading decisions combining multiple data sources.
"""

import time
import logging
import json
import os
import random
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Load environment variables and configuration
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger('crypto_bot')

# Import the modules that are working
from core import enhanced_news_fetcher
logger.info("✅ Using enhanced news data")

from core import enhanced_social_sentiment
logger.info("✅ Using enhanced social sentiment data")

from core import data_aggregator
logger.info("✅ Using centralized data aggregator")

from core import unified_whale_data
logger.info("✅ Using unified whale data")

from core import fear_greed_fetcher
logger.info("✅ Using fear & greed index")

from core import crypto_compare_api
logger.info("✅ Using CryptoCompare market data")

from core import technical_indicator_fetcher
logger.info("✅ Using technical indicators")

from core import trade_executor
logger.info("✅ Using enhanced trade executor with dynamic position sizing")

# Critical component - abort if not available
try:
    from core import enhanced_gpt_fusion_analyzer as gpt_analyzer
    logger.info("✅ Using Enhanced GPT Fusion Analyzer")
except Exception as e:
    logger.critical(f"❌ CRITICAL ERROR: Enhanced GPT Fusion Analyzer not available: {str(e)}")
    logger.critical("❌ Trading bot cannot operate without GPT analyzer! Aborting.")
    # Exit with error code
    import sys
    sys.exit(1)

# Import configuration
import config

def validate_critical_components():
    """
    Validate that all critical components are functioning.
    Exits the program if any critical check fails.
    """
    logger.info("Performing critical component validation...")
    
    # Check OpenAI API key
    if not os.environ.get("OPENAI_API_KEY"):
        logger.critical("❌ CRITICAL ERROR: OpenAI API key not found in environment")
        logger.critical("❌ Set OPENAI_API_KEY environment variable before running")
        sys.exit(1)
    
    # Check GPT analyzer core functions
    try:
        budget_status = gpt_analyzer.get_budget_status()
        logger.info(f"GPT budget check passed: ${budget_status['remaining_budget']:.2f} available")
    except Exception as e:
        logger.critical(f"❌ CRITICAL ERROR: GPT analyzer budget check failed: {str(e)}")
        sys.exit(1)

    # Check other critical APIs - for example, price data
    try:
        btc_price = crypto_compare_api.get_price("BTC")
        if not btc_price:
            raise ValueError("Could not get BTC price")
        logger.info(f"Market data check passed: BTC price = ${btc_price}")
    except Exception as e:
        logger.critical(f"❌ CRITICAL ERROR: Market data check failed: {str(e)}")
        sys.exit(1)
    
    logger.info("✅ All critical component checks passed")

# Rule-based trading decision (original logic as fallback)
def rule_based_trading_decision(context):
    """
    Make a trading decision based on sentiment and market data.
    
    Args:
        context (dict): Market context with sentiment and market data
        
    Returns:
        dict: Trading decision
    """
    # Convert sentiment strings to numeric values for scoring
    sentiment_scores = {
        "bullish": 1,
        "neutral": 0,
        "bearish": -1,
        "large_buy": 1,
        "large_sell": -1
    }
    
    # Initialize score
    score = 0
    confidence_points = 0
    
    # Process news sentiment (weight: 2)
    news_sentiment = context['news_sentiment']
    if news_sentiment in sentiment_scores:
        score += sentiment_scores[news_sentiment] * 2
        confidence_points += 1
    
    # Process Social Media Sentiment (weight: 1)
    social = context['social_media_sentiment']
    if social in sentiment_scores:
        score += sentiment_scores[social] * 1
        confidence_points += 0.5
    
    # Process Whale Activity (weight: 2)
    whale = context['whale_transactions']
    if whale in sentiment_scores:  # Only include if we have real data
        score += sentiment_scores[whale] * 2
        confidence_points += 1
    elif whale == "unavailable":
        # If whale data is unavailable, we don't adjust the score
        # but we also don't increase confidence
        logger.info(f"Whale activity data unavailable for {context['symbol']}, excluding from analysis")
    
    # Process Fear & Greed (contrarian indicator, weight: 1)
    fear_greed = context['fear_greed_index']
    if fear_greed == "Extreme Fear":
        score += 1  # Contrarian signal - buy when others are fearful
        confidence_points += 0.5
    elif fear_greed == "Extreme Greed":
        score -= 1  # Contrarian signal - sell when others are greedy
        confidence_points += 0.5
    
    # Process RSI (weight: 1.5)
    if 'rsi' in context:
        rsi = context['rsi']
        if rsi <= 30:  # Oversold
            score += 1 * 1.5
        elif rsi >= 70:  # Overbought
            score -= 1 * 1.5
        confidence_points += 0.75
    
    # Process MACD (weight: 1.5)
    if 'macd_crossover' in context:
        macd = context['macd_crossover']
        if macd == "bullish":
            score += 1 * 1.5
        elif macd == "bearish":
            score -= 1 * 1.5
        confidence_points += 0.75
    
    # Process CryptoCompare price data (if available)
    if 'crypto_data' in context:
        coin_data = context['crypto_data']
        price_change_24h = coin_data.get('price_change_24h', 0)
        
        # Add price momentum to scoring (weight: 2)
        if price_change_24h > 5:  # Strong positive momentum
            score += 1 * 2
        elif price_change_24h < -5:  # Strong negative momentum
            score -= 1 * 2
        elif price_change_24h > 2:  # Moderate positive momentum
            score += 0.5 * 2
        elif price_change_24h < -2:  # Moderate negative momentum
            score -= 0.5 * 2
            
        confidence_points += 1
    
    # Normalize score to -100 to +100 range
    max_possible_score = 10  # Sum of all weights
    conviction_score = int((score / max_possible_score) * 100)
    conviction_score = max(-100, min(100, conviction_score))  # Clamp to range
    
    # Determine action based on conviction score
    if conviction_score >= 20:
        action = "buy"
    elif conviction_score <= -20:
        action = "sell"
    else:
        action = "hold"
    
    # Determine confidence level
    max_confidence = 5  # Total number of indicators
    confidence_ratio = confidence_points / max_confidence
    
    if confidence_ratio >= 0.7:
        confidence_level = "high"
    elif confidence_ratio >= 0.4:
        confidence_level = "medium"
    else:
        confidence_level = "low"
    
    # Generate rationale
    rationales = []
    
    if news_sentiment == "bullish":
        rationales.append("News sentiment is positive")
    elif news_sentiment == "bearish":
        rationales.append("News sentiment is negative")
    
    if social == "bullish":
        rationales.append("Social media sentiment is bullish")
    elif social == "bearish":
        rationales.append("Social media sentiment is bearish")
    
    if whale == "large_buy":
        rationales.append("Significant whale buying detected")
    elif whale == "large_sell":
        rationales.append("Significant whale selling detected")
    
    if fear_greed == "Extreme Fear":
        rationales.append("Market in Extreme Fear (contrarian bullish)")
    elif fear_greed == "Extreme Greed":
        rationales.append("Market in Extreme Greed (contrarian bearish)")
    
    if 'rsi' in context:
        rsi = context['rsi']
        if rsi <= 30:
            rationales.append(f"RSI is oversold at {rsi}")
        elif rsi >= 70:
            rationales.append(f"RSI is overbought at {rsi}")
    
    if 'macd_crossover' in context and context['macd_crossover'] != "neutral":
        rationales.append(f"MACD shows {context['macd_crossover']} momentum")
    
    if 'crypto_data' in context:
        price_change_24h = context['crypto_data'].get('price_change_24h', 0)
        if price_change_24h > 5:
            rationales.append(f"Strong price momentum (+{price_change_24h:.1f}% in 24h)")
        elif price_change_24h < -5:
            rationales.append(f"Significant price drop ({price_change_24h:.1f}% in 24h)")
    
    # Select up to 2 rationale points
    if rationales:
        rationale = " and ".join(random.sample(rationales, min(2, len(rationales)))) + "."
    else:
        if action == "buy":
            rationale = "Multiple indicators suggest positive momentum."
        elif action == "sell":
            rationale = "Multiple indicators suggest negative momentum."
        else:
            rationale = "Mixed signals suggest caution."
    
    # Build the response
    return {
        "action": action,
        "conviction_score": conviction_score,
        "confidence_level": confidence_level,
        "rationale": rationale,
        "decision_method": "rule_based"  # Add marker for which method was used
    }

# Unified trading decision function that uses GPT if available, rule-based otherwise
def trading_decision(context):
    """
    Make a trading decision using GPT if available, falling back to rule-based method.
    
    Args:
        context (dict): Market context with sentiment and market data
        
    Returns:
        dict: Trading decision
    """
    # If we have the GPT analyzer module and GPT is enabled
    if HAS_GPT_ANALYZER and getattr(config, 'GPT_ENABLED', True):
        try:
            # Check budget status
            budget_status = gpt_analyzer.get_budget_status()
            # If we have budget remaining (more than 5 cents to be safe)
            if budget_status['remaining_budget'] > 0.05:
                # Try to get GPT-enhanced trading decision
                decision = gpt_analyzer.get_trading_recommendation(context, cost_conscious=True)
                
                # Log budget information after the call
                updated_budget = gpt_analyzer.get_budget_status()
                remaining = updated_budget['remaining_budget']
                used = updated_budget['total_cost']
                logger.info(f"GPT budget status: ${used:.2f} used, ${remaining:.2f} remaining today")
                
                # If we got a valid decision, return it
                if decision:
                    # Add marker for which method was used
                    decision["decision_method"] = "gpt_enhanced"
                    if "used_fallback" in decision and decision["used_fallback"]:
                        decision["decision_method"] = f"gpt_fallback_{decision.get('fallback_reason', 'unknown')}"
                    return decision
            else:
                logger.warning(f"GPT budget low (${budget_status['remaining_budget']:.2f} remaining). Using rule-based decision.")
        except Exception as e:
            logger.error(f"Error using GPT analyzer: {str(e)}")
    
    # If we get here, either GPT isn't available, or we're out of budget, or there was an error
    # Fall back to rule-based method
    return rule_based_trading_decision(context)

def gather_data_for_symbol(symbol):
    """
    Gather market data for a specific cryptocurrency symbol.
    
    Args:
        symbol (str): Cryptocurrency symbol (e.g., 'BTC', 'ETH')
        
    Returns:
        dict: Context object with gathered data
    """
    logger.info(f"Gathering data for {symbol}")
    
    # Get news headlines
    news_headlines = news_fetcher.get_latest_headlines(limit=5)
    news_sentiment = news_fetcher.analyze_sentiment(news_headlines)
    
    # Get market sentiment indicators
    # Replace whale_data_provider with unified_whale_data
    whale_signal = unified_whale_data.get_recent_activity(symbol)
    
    # Create the base context object first to avoid reference before assignment
    context = {
        "timestamp": datetime.now().isoformat(),
        "symbol": symbol,
        "news_headlines": news_headlines,
        "news_sentiment": news_sentiment,
        "whale_transactions": whale_signal,
        "fear_greed_index": fear_greed_fetcher.get_current_index(),
        "social_media_sentiment": social_sentiment_fetcher.get_sentiment(symbol),
        "rsi": technical_indicator_fetcher.get_rsi(symbol),
        "macd_crossover": technical_indicator_fetcher.get_macd_signal(symbol),
        "volume_spike": technical_indicator_fetcher.check_volume_spike(symbol),
        "price_action": technical_indicator_fetcher.get_price_trend(symbol),
        "time_of_day": get_time_of_day()
    }
    
    # Optional: Add more whale data to the context
    try:
        # Replace whale_data_provider with unified_whale_data
        whale_transactions = unified_whale_data.get_whale_transactions(symbol, count=3)
        if whale_transactions and len(whale_transactions) > 0:
            context["recent_whale_transactions"] = whale_transactions
    except Exception as e:
        logger.warning(f"Couldn't retrieve whale transactions for {symbol}: {str(e)}")
    
    # Rest of the function remains unchanged
    try:
        # Get comprehensive market data
        market_data = crypto_compare_api.get_market_data(symbol)
        if market_data:
            logger.info(f"CryptoCompare data retrieved for {symbol}")
            crypto_data = {
                "price": market_data.get("current_price"),
                "market_cap": market_data.get("market_cap"),
                "total_volume": market_data.get("total_volume"),
                "price_change_24h": market_data.get("price_change_24h"),
                "price_change_7d": market_data.get("price_change_7d"),
                "price_change_30d": market_data.get("price_change_30d"),
                "market_cap_rank": market_data.get("market_cap_rank"),
                "ath": 0,  # Not provided by CryptoCompare
                "ath_change_percentage": 0  # Not provided by CryptoCompare
            }
            context["crypto_data"] = crypto_data
        else:
            logger.warning(f"Failed to get CryptoCompare market data for {symbol}")
    except Exception as e:
        logger.error(f"Error retrieving CryptoCompare data: {str(e)}")
    
    # If we have portfolio information, add it for enhanced GPT analysis
    try:
        portfolio = trade_executor.get_portfolio_value()
        # Get current position in this symbol
        current_position = 0
        position_value = 0
        if symbol in portfolio['holdings']:
            current_position = portfolio['holdings'][symbol]['amount']
            position_value = portfolio['holdings'][symbol]['usd_value']
        
        # Calculate allocation percentage
        allocation_percentage = 0
        if portfolio['total_usd_value'] > 0:
            allocation_percentage = (position_value / portfolio['total_usd_value']) * 100
        
        # Add portfolio context
        context['portfolio'] = {
            'holdings': portfolio['holdings'],
            'cash_balance': portfolio['cash_balance'],
            'current_position': current_position,
            'position_value': position_value,
            'allocation_percentage': allocation_percentage
        }
    except Exception as e:
        logger.warning(f"Couldn't add portfolio context: {str(e)}")
    
    return context

def get_time_of_day():
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

def generate_trading_report(portfolio, symbols_to_monitor, global_market_context):
    """Generate a trading report with GPT if available"""
    if not HAS_GPT_ANALYZER or not getattr(config, 'GENERATE_REPORTS', False):
        return None
        
    try:
        # Check if we have enough budget for a report
        budget_status = gpt_analyzer.get_budget_status() 
        if budget_status['remaining_budget'] < 0.1:  # Need at least 10 cents
            logger.warning("Insufficient budget for GPT trading report")
            return None
            
        # Get recent trades
        recent_trades = trade_executor.get_trading_history()[:10]  # Last 10 trades
        
        # Add some performance metrics
        metrics = trade_executor.calculate_performance_metrics()
        
        # Generate report context
        report_context = {
            "portfolio": portfolio,
            "monitored_symbols": symbols_to_monitor,
            "global_market": global_market_context,
            "recent_trades": recent_trades,
            "metrics": metrics
        }
        
        # Use GPT to generate a report
        if hasattr(gpt_analyzer, 'generate_trading_report'):
            return gpt_analyzer.generate_trading_report(portfolio, recent_trades, global_market_context)
        return None
        
    except Exception as e:
        logger.error(f"Error generating trading report: {str(e)}")
        return None

def main_loop():
    """Main bot execution loop with centralized data aggregation."""
    logger.info("Starting enhanced crypto trading bot with GPT as main brain")
    
    # Load trade history
    trade_executor.load_trade_history()
    
    # Get the list of symbols to monitor
    symbols_to_monitor = getattr(config, 'SYMBOLS_TO_MONITOR', ['BTC', 'ETH', 'SOL'])
    
    # Initialize the data aggregator
    data_mgr = data_aggregator.DataAggregator()
    logger.info(f"Initialized centralized data aggregator")
    
    # Log initial portfolio value
    portfolio = trade_executor.get_portfolio_value()
    logger.info(f"Initial portfolio value: ${portfolio['total_usd_value']:.2f} (Cash: ${portfolio['cash_balance']:.2f})")
    for symbol, details in portfolio['holdings'].items():
        logger.info(f"  {symbol}: {details['amount']} (${details['usd_value']:.2f})")
    
    # Log GPT budget status if available
    if HAS_GPT_ANALYZER:
        try:
            budget_status = gpt_analyzer.get_budget_status()
            logger.info(f"GPT budget status: ${budget_status['total_cost']:.2f} used, " + 
                        f"${budget_status['remaining_budget']:.2f} remaining today")
        except Exception as e:
            logger.warning(f"Couldn't get GPT budget status: {str(e)}")
    
    # Track reports timing
    last_report_time = datetime.now()
    report_frequency_hours = getattr(config, 'REPORT_FREQUENCY_HOURS', 6)
    
    try:
        # Try to get global market data
        try:
            global_data = crypto_compare_api.get_global_market_data()
            if global_data:
                market_cap = global_data.get('market_cap_usd', 0) / 1e9
                btc_dom = global_data.get('btc_dominance', 0)
                eth_dom = global_data.get('eth_dominance', 0)
                logger.info(f"Global crypto market cap: ${market_cap:.2f}B")
                logger.info(f"BTC dominance: {btc_dom:.2f}%, ETH dominance: {eth_dom:.2f}%")
                
                # Store global market data for use in analysis
                global_market_context = {
                    'market_cap_usd': global_data.get('market_cap_usd', 0),
                    'btc_dominance': global_data.get('btc_dominance', 0),
                    'eth_dominance': global_data.get('eth_dominance', 0),
                    'updated_at': global_data.get('updated_at', 0)
                }
        except Exception as e:
            logger.warning(f"Could not retrieve global market data: {str(e)}")
            global_market_context = None
            
        # Try to get trending coins
        try:
            trending = crypto_compare_api.get_trending_coins()
            if trending and len(trending) > 0:
                trending_names = [f"{coin.get('name')} ({coin.get('symbol')})" for coin in trending[:3]]
                logger.info(f"Trending coins: {', '.join(trending_names)}")
        except Exception as e:
            logger.warning(f"Could not retrieve trending coins: {str(e)}")
        
        # Bot execution loop
        cycle_count = 0
        while True:
            cycle_count += 1
            logger.info(f"Beginning data collection cycle #{cycle_count}")
            
            # Reset the Twitter API request counter at the start of each cycle
            if hasattr(enhanced_social_sentiment, 'reset_twitter_request_counter'):
                enhanced_social_sentiment.reset_twitter_request_counter()
            
            # Check if we should generate a trading report
            current_time = datetime.now()
            hours_since_last_report = (current_time - last_report_time).total_seconds() / 3600
            
            if hours_since_last_report >= report_frequency_hours:
                logger.info("Generating periodic trading report...")
                portfolio = trade_executor.get_portfolio_value()
                report = generate_trading_report(portfolio, symbols_to_monitor, global_market_context)
                if report:
                    logger.info("\n----- TRADING REPORT -----\n" + report + "\n--------------------------")
                last_report_time = current_time
            
            for symbol in symbols_to_monitor:
                logger.info(f"Processing {symbol}")
                
                try:
                    # 1. Generate context with all signals using data aggregator
                    context = data_mgr.gather_all_data(symbol)
                    
                    # Add global market data if available
                    if global_market_context:
                        context['global_market'] = global_market_context
                    
                    logger.info(f"Context generated for {symbol}")
                    
                    # 2. Get trading decision (will use GPT if available, rule-based otherwise)
                    analysis_result = trading_decision(context)
                    
                    # 3. Log the decision
                    action = analysis_result["action"]
                    conviction = analysis_result["conviction_score"]
                    confidence = analysis_result["confidence_level"]
                    rationale = analysis_result["rationale"]
                    decision_method = analysis_result.get("decision_method", "unknown")
                    
                    logger.info(f"TRADING DECISION ({decision_method}): {action.upper()} {symbol} with {conviction} conviction ({confidence} confidence)")
                    logger.info(f"RATIONALE: {rationale}")
                    
                    # Log CryptoCompare price data if available
                    if "crypto_data" in context:
                        price = context["crypto_data"]["price"]
                        change_24h = context["crypto_data"]["price_change_24h"]
                        change_7d = context["crypto_data"]["price_change_7d"]
                        logger.info(f"{symbol} PRICE: ${price:.2f} ({change_24h:.2f}% 24h, {change_7d:.2f}% 7d)")
                    
                    # 4. Execute trade if conviction is high enough
                    conviction_threshold = getattr(config, 'CONVICTION_THRESHOLD', 40)
                    
                    if action != "hold" and abs(conviction) >= conviction_threshold:
                        # Execute trade with dynamic position sizing
                        trade_result = trade_executor.execute_trade(
                            symbol=symbol,
                            action=action,
                            conviction=conviction,
                            context=context  # Pass full context for dynamic sizing
                        )
                        
                        logger.info(f"TRADE EXECUTED: {trade_result}")
                        
                        # Save trade history after each trade
                        trade_executor.save_trade_history()
                    else:
                        logger.info(f"HOLDING POSITION: Conviction {conviction} below threshold {conviction_threshold}")
                    
                except Exception as e:
                    logger.error(f"Error processing {symbol}: {str(e)}", exc_info=True)
            
            # Log portfolio value and performance metrics after each cycle
            portfolio = trade_executor.get_portfolio_value()
            logger.info(f"Current portfolio value: ${portfolio['total_usd_value']:.2f} (Cash: ${portfolio['cash_balance']:.2f})")
            
            # Calculate and log performance metrics
            metrics = trade_executor.calculate_performance_metrics()
            logger.info(f"Performance metrics: ROI {metrics['roi']:.2f}%, P&L ${metrics['profit_loss']:.2f}")
            
            # Log GPT budget status if available
            if HAS_GPT_ANALYZER:
                try:
                    budget_status = gpt_analyzer.get_budget_status()
                    logger.info(f"GPT budget status: ${budget_status['total_cost']:.2f} used, " + 
                                f"${budget_status['remaining_budget']:.2f} remaining today")
                except Exception as e:
                    logger.warning(f"Couldn't get GPT budget status: {str(e)}")
            
            # Update global market data for the next cycle
            try:
                updated_global_data = crypto_compare_api.get_global_market_data()
                if updated_global_data:
                    global_market_context = {
                        'market_cap_usd': updated_global_data.get('market_cap_usd', 0),
                        'btc_dominance': updated_global_data.get('btc_dominance', 0),
                        'eth_dominance': updated_global_data.get('eth_dominance', 0),
                        'updated_at': updated_global_data.get('updated_at', 0)
                    }
            except Exception as e:
                logger.warning(f"Could not update global market data: {str(e)}")
            
            # Wait for next cycle
            cycle_interval = getattr(config, 'CYCLE_INTERVAL_SECONDS', 600)
            logger.info(f"Cycle complete. Waiting {cycle_interval} seconds until next cycle")
            time.sleep(cycle_interval)
            
    except KeyboardInterrupt:
        logger.info("Bot execution terminated by user")
        
        # Show final portfolio value
        final_portfolio = trade_executor.get_portfolio_value()
        logger.info(f"Final portfolio value: ${final_portfolio['total_usd_value']:.2f}")
        for symbol, details in final_portfolio['holdings'].items():
            logger.info(f"  {symbol}: {details['amount']} (${details['usd_value']:.2f})")
        logger.info(f"  Cash: ${final_portfolio['cash_balance']:.2f}")
        
        # Show final performance metrics
        final_metrics = trade_executor.calculate_performance_metrics()
        logger.info(f"Final performance: ROI {final_metrics['roi']:.2f}%, P&L ${final_metrics['profit_loss']:.2f}")
        
        # Generate a final trading report if GPT is available
        if HAS_GPT_ANALYZER:
            logger.info("Generating final trading report...")
            final_report = generate_trading_report(final_portfolio, symbols_to_monitor, global_market_context)
            if final_report:
                logger.info("\n----- FINAL TRADING REPORT -----\n" + final_report + "\n-----------------------------")
        
    except Exception as e:
        logger.error(f"Error in main bot loop: {str(e)}", exc_info=True)
        raise
    finally:
        # Save trade history before exiting
        trade_executor.save_trade_history()
        # Save GPT usage data if available
        if HAS_GPT_ANALYZER and hasattr(gpt_analyzer, 'save_usage_stats'):
            gpt_analyzer.save_usage_stats()
        logger.info("Bot execution completed")