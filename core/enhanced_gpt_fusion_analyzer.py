"""
Enhanced GPT Fusion Analyzer v2 - Main Brain Integration

This module extends the original GPT Fusion Analyzer with:
1. Cost management features to stay under budget
2. Selective GPT usage based on opportunity importance 
3. Dynamic model selection (3.5 vs 4) based on decision complexity
4. Integration with multiple data sources as a central brain

Modified to function as the "main brain" that integrates all system components.
"""

try:
    import talib
except ImportError:
    # If TA-Lib is not available, use our compatibility layer
    from core import talib_compatibility as talib

import logging
import json
import random
import requests
import os
import time
from datetime import datetime, timedelta
import tiktoken
import config
from core import predictive_technical_analysis

# Set up logging
logger = logging.getLogger('crypto_bot.enhanced_gpt_fusion_v2')

# System prompt templates
SYSTEM_PROMPT_BASIC = """
You are CryptoFusionAI, an advanced cryptocurrency trading assistant with expertise in technical analysis.
Analyze both the market data and technical patterns provided to make an optimal trading recommendation.

Your output must be in the following JSON format:
{
  "action": "buy" or "sell" or "hold",
  "conviction_score": number between -100 (strong sell) and +100 (strong buy),
  "confidence_level": "low" or "medium" or "high",
  "rationale": "Detailed explanation of your reasoning"
}

The conviction score should represent how strongly you believe in the recommendation:
- Values from -100 to -60: Strong sell
- Values from -59 to -20: Moderate sell
- Values from -19 to +19: Hold
- Values from +20 to +59: Moderate buy
- Values from +60 to +100: Strong buy

Be objective and consider all available data points to make the most accurate recommendation possible.
"""

SYSTEM_PROMPT_ADVANCED = """
You are CryptoFusionAI, an advanced cryptocurrency trading expert with deep expertise in technical analysis, market psychology, and macro trends.
You serve as the central decision-making brain for a sophisticated crypto trading system that combines multiple data sources.

MISSION: Analyze all available data comprehensively to make the optimal trading recommendation.

Your output must be in the following JSON format:
{
  "action": "buy" or "sell" or "hold",
  "conviction_score": number between -100 (strong sell) and +100 (strong buy),
  "confidence_level": "low" or "medium" or "high",
  "position_size_recommendation": percentage of available capital to deploy (1-100),
  "stop_loss_percentage": recommended stop loss percentage,
  "take_profit_percentage": recommended take profit percentage,
  "rationale": "Detailed explanation of your reasoning",
  "key_factors": ["factor1", "factor2", ...],
  "risk_assessment": "Description of potential risks and how they factor into decision",
  "market_sentiment_analysis": "Analysis of overall market sentiment from multiple sources",
  "technical_summary": "Summary of key technical indicators and patterns"
}

Conviction score reference:
- Values from -100 to -60: Strong sell
- Values from -59 to -20: Moderate sell
- Values from -19 to +19: Hold
- Values from +20 to +59: Moderate buy
- Values from +60 to +100: Strong buy

Be thorough in your analysis. Prioritize capital preservation and risk management.
Consider possible conflicting signals between different indicators and explain how you resolve these conflicts.
"""

# Token usage tracking
daily_token_usage = {
    'date': datetime.now().strftime('%Y-%m-%d'),
    'input_tokens': 0,
    'output_tokens': 0,
    'total_cost': 0,
    'total_calls': 0,
    'gpt4_calls': 0,
    'gpt35_calls': 0
}

# Token pricing (as of 2023)
TOKEN_COSTS = {
    'gpt-3.5-turbo': {
        'input': 0.0015 / 1000,  # per 1K tokens
        'output': 0.002 / 1000    # per 1K tokens
    },
    'gpt-4': {
        'input': 0.03 / 1000,     # per 1K tokens
        'output': 0.06 / 1000     # per 1K tokens
    }
}

# Budget management
DAILY_BUDGET = float(os.getenv("GPT_DAILY_BUDGET_USD", "1.0"))  # Default $1 per day
USAGE_DATA_FILE = "data/gpt_usage_stats.json"

def count_tokens(text, model="gpt-3.5-turbo"):
    """
    Count the number of tokens in a text string.
    
    Args:
        text (str): The text to count tokens for
        model (str): The model to use for token counting
        
    Returns:
        int: The number of tokens
    """
    try:
        # Use tiktoken for accurate token counting
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    except Exception as e:
        # Fallback to approximate counting if tiktoken fails
        logger.warning(f"Token counting error: {str(e)}")
        # Very rough approximation: ~4 chars per token
        return len(text) // 4

def estimate_cost(input_text, model="gpt-3.5-turbo", estimated_output_tokens=500):
    """
    Estimate the cost of a GPT API call.
    
    Args:
        input_text (str): The input text
        model (str): The model to use
        estimated_output_tokens (int): Estimated number of output tokens
        
    Returns:
        float: Estimated cost in USD
    """
    input_tokens = count_tokens(input_text, model)
    
    # Calculate cost
    input_cost = input_tokens * TOKEN_COSTS[model]['input']
    output_cost = estimated_output_tokens * TOKEN_COSTS[model]['output']
    
    return input_cost + output_cost, input_tokens, estimated_output_tokens

def check_budget():
    """
    Check if we're within our daily budget.
    
    Returns:
        bool: True if within budget, False otherwise
    """
    global daily_token_usage
    
    # Reset tracking if it's a new day
    current_date = datetime.now().strftime('%Y-%m-%d')
    if daily_token_usage['date'] != current_date:
        # Save the old data before resetting
        save_usage_stats()
        
        # Reset for the new day
        daily_token_usage = {
            'date': current_date,
            'input_tokens': 0,
            'output_tokens': 0,
            'total_cost': 0,
            'total_calls': 0,
            'gpt4_calls': 0,
            'gpt35_calls': 0
        }
    
    # Check if we're under budget
    return daily_token_usage['total_cost'] < DAILY_BUDGET

def update_usage_stats(model, input_tokens, output_tokens, cost):
    """
    Update the token usage statistics.
    
    Args:
        model (str): The model used
        input_tokens (int): Number of input tokens
        output_tokens (int): Number of output tokens
        cost (float): Cost of the API call
    """
    global daily_token_usage
    
    daily_token_usage['input_tokens'] += input_tokens
    daily_token_usage['output_tokens'] += output_tokens
    daily_token_usage['total_cost'] += cost
    daily_token_usage['total_calls'] += 1
    
    if model == 'gpt-4':
        daily_token_usage['gpt4_calls'] += 1
    else:
        daily_token_usage['gpt35_calls'] += 1
    
    # Log current usage
    logger.info(f"GPT usage today: ${daily_token_usage['total_cost']:.4f} of ${DAILY_BUDGET:.2f} budget " +
                f"({daily_token_usage['total_calls']} calls, {daily_token_usage['input_tokens'] + daily_token_usage['output_tokens']} tokens)")
    
    # Save usage stats periodically
    if daily_token_usage['total_calls'] % 5 == 0:
        save_usage_stats()

def save_usage_stats():
    """Save the current usage statistics to a file."""
    try:
        # Create data directory if it doesn't exist
        os.makedirs(os.path.dirname(USAGE_DATA_FILE), exist_ok=True)
        
        # Load existing data if available
        usage_history = []
        if os.path.exists(USAGE_DATA_FILE):
            try:
                with open(USAGE_DATA_FILE, 'r') as f:
                    usage_history = json.load(f)
            except json.JSONDecodeError:
                usage_history = []
        
        # Check if we already have data for today
        for i, entry in enumerate(usage_history):
            if entry.get('date') == daily_token_usage['date']:
                # Update existing entry
                usage_history[i] = daily_token_usage
                break
        else:
            # Add new entry
            usage_history.append(daily_token_usage)
        
        # Keep only last 30 days
        usage_history = usage_history[-30:]
        
        # Save to file
        with open(USAGE_DATA_FILE, 'w') as f:
            json.dump(usage_history, f, indent=2)
            
    except Exception as e:
        logger.error(f"Error saving usage stats: {str(e)}")

def select_model(context, technical_analysis=None):
    """
    Dynamically select which GPT model to use based on trade opportunity importance.
    
    Args:
        context (dict): The context data
        technical_analysis (dict): Technical analysis data
        
    Returns:
        str: The model to use ('gpt-3.5-turbo' or 'gpt-4')
    """
    # Default to GPT-3.5 to save costs
    default_model = "gpt-3.5-turbo"
    
    # Check if GPT-4 is enabled in config
    if not getattr(config, 'GPT4_ENABLED', False):
        return default_model
    
    # If we're close to budget limit, use cheaper model
    if daily_token_usage['total_cost'] > DAILY_BUDGET * 0.8:
        return default_model
    
    # Use GPT-4 for high-impact situations:
    
    # 1. Portfolio significance check - use GPT-4 for large positions
    if 'portfolio' in context and 'allocation_percentage' in context['portfolio']:
        allocation = context['portfolio']['allocation_percentage']
        if allocation > 20:  # If this asset represents >20% of portfolio
            logger.info(f"Using GPT-4 due to significant portfolio allocation ({allocation}%)")
            return "gpt-4"
    
    # 2. Strong technical signals that need careful interpretation
    if technical_analysis and technical_analysis.get('status') == 'success':
        conviction = technical_analysis.get('conviction', 0)
        if abs(conviction) > 70:  # Strong conviction
            logger.info(f"Using GPT-4 due to strong technical conviction ({conviction})")
            return "gpt-4"
    
    # 3. Large market movements or high volatility
    if 'crypto_data' in context:
        price_change_24h = context['crypto_data'].get('price_change_24h', 0)
        if abs(price_change_24h) > 8:  # Large price movement
            logger.info(f"Using GPT-4 due to significant price change ({price_change_24h}%)")
            return "gpt-4"
    
    # 4. Conflicting signals that need more nuanced analysis
    signals = []
    sentiment_fields = ['news_sentiment', 'social_media_sentiment', 'whale_transactions']
    
    # Count conflicting signals
    signal_conflict = False
    bullish_count = 0
    bearish_count = 0
    
    for field in sentiment_fields:
        if field in context:
            if context[field] in ['bullish', 'large_buy']:
                bullish_count += 1
            elif context[field] in ['bearish', 'large_sell']:
                bearish_count += 1
    
    # Add technical signals
    if 'rsi' in context:
        if context['rsi'] < 30:
            bullish_count += 1  # Oversold = bullish
        elif context['rsi'] > 70:
            bearish_count += 1  # Overbought = bearish
    
    if 'macd_crossover' in context:
        if context['macd_crossover'] == 'bullish':
            bullish_count += 1
        elif context['macd_crossover'] == 'bearish':
            bearish_count += 1
    
    # Check for significant conflict
    if bullish_count >= 2 and bearish_count >= 2:
        logger.info(f"Using GPT-4 due to conflicting signals (bullish: {bullish_count}, bearish: {bearish_count})")
        return "gpt-4"
    
    # 5. Major cryptocurrencies during high activity
    if context['symbol'] in ['BTC', 'ETH']:
        # Use GPT-4 more often for major coins
        if context['symbol'] == 'BTC' and random.random() < 0.3:  # 30% chance for BTC
            return "gpt-4"
        elif context['symbol'] == 'ETH' and random.random() < 0.2:  # 20% chance for ETH
            return "gpt-4"
    
    # Otherwise, use default model
    return default_model

def build_enhanced_prompt(context, technical_analysis=None, advanced=True):
    """
    Build a comprehensive prompt with all available market data.
    
    Args:
        context (dict): The context data containing all market indicators
        technical_analysis (dict, optional): Predictive technical analysis data
        advanced (bool): Whether to use the advanced prompt format (default: True)
        
    Returns:
        str: The formatted prompt for GPT
    """
    # Format the context into a structured format for GPT
    prompt = f"""
# COMPREHENSIVE MARKET ANALYSIS FOR {context['symbol']}
Analysis timestamp: {context['timestamp']}

## NEWS SENTIMENT
Recent Headlines:
"""
    
    # Add headlines
    for headline in context['news_headlines']:
        prompt += f"- {headline['headline']}"
        if 'timestamp' in headline:
            prompt += f" ({headline['timestamp']})"
        prompt += "\n"
    
    prompt += f"Overall News Sentiment: {context['news_sentiment']}\n\n"
    
    # Add CryptoCompare market data if available
    if 'crypto_data' in context:
        coin_data = context['crypto_data']
        prompt += f"""
## MARKET FUNDAMENTALS
- Current Price: ${coin_data.get('price', 'N/A')}
- 24h Price Change: {coin_data.get('price_change_24h', 'N/A')}%
- 7d Price Change: {coin_data.get('price_change_7d', 'N/A')}%
- 30d Price Change: {coin_data.get('price_change_30d', 'N/A')}%
- Market Cap: ${coin_data.get('market_cap', 'N/A'):,}
- 24h Trading Volume: ${coin_data.get('total_volume', 'N/A'):,}
- Market Cap Rank: #{coin_data.get('market_cap_rank', 'N/A')}
"""

    # Add global market data if available
    if 'global_market' in context:
        global_data = context['global_market']
        prompt += f"""
## GLOBAL MARKET CONDITIONS
- Total Crypto Market Cap: ${global_data.get('market_cap_usd', 0) / 1e9:.2f}B
- Bitcoin Dominance: {global_data.get('btc_dominance', 'N/A')}%
- Ethereum Dominance: {global_data.get('eth_dominance', 'N/A')}%
- Market Cycle Phase: {global_data.get('market_cycle', 'Unknown')}
"""
    
    # Add market sentiment indicators
    prompt += f"""
## MARKET SENTIMENT INDICATORS
- Fear & Greed Index: {context['fear_greed_index']}
- Social Media Sentiment: {context['social_media_sentiment']}
- Whale Activity: {context['whale_transactions']}
"""

    # Add technical indicators
    prompt += f"""
## TECHNICAL INDICATORS
- RSI ({context['symbol']}): {context.get('rsi', 'N/A')}
- MACD Signal: {context.get('macd_crossover', 'N/A')}
- Price Action: {context.get('price_action', 'N/A')}
- Volume Spike: {context.get('volume_spike', 'N/A')}
"""

    # Add whale transactions if available
    if 'recent_whale_transactions' in context:
        prompt += "\n## RECENT WHALE TRANSACTIONS\n"
        for tx in context['recent_whale_transactions'][:3]:
            tx_type = tx.get('type', 'unknown')
            amount = tx.get('amount', 'unknown')
            value_usd = tx.get('value_usd', 'unknown')
            timestamp = tx.get('timestamp', 'unknown')
            
            prompt += f"- {tx_type.upper()} {amount} {context['symbol']} (${value_usd:,}) at {timestamp}\n"

    # Add predictive technical analysis if available
    if technical_analysis and technical_analysis.get('status') == 'success':
        prompt += f"""
## PREDICTIVE TECHNICAL ANALYSIS
Current Price: ${technical_analysis['current_price']:.2f} ({technical_analysis['day_change_pct']:.2f}% 24h change)

Multi-Timeframe Analysis:
- 1-Day Trend: {technical_analysis.get('trend', 'neutral').upper()} (Strength: {technical_analysis.get('trend_strength', 0)})
- Volatility: {technical_analysis.get('volatility', 'medium').upper()}

Key Support Levels: {', '.join([f"${level:.2f}" for level in technical_analysis.get('support_levels', [])])}
Key Resistance Levels: {', '.join([f"${level:.2f}" for level in technical_analysis.get('resistance_levels', [])])}

Bullish Patterns Detected: {', '.join(technical_analysis.get('patterns', {}).get('bullish', ['None'])).replace('_', ' ').title()}
Bearish Patterns Detected: {', '.join(technical_analysis.get('patterns', {}).get('bearish', ['None'])).replace('_', ' ').title()}

Technical Analysis Recommendation: {technical_analysis.get('recommendation', 'hold').upper()} with conviction {technical_analysis.get('conviction', 0)} out of 100
"""

    # Add portfolio context if available
    if 'portfolio' in context:
        portfolio = context['portfolio']
        prompt += f"""
## PORTFOLIO CONTEXT
- Current Holdings: {portfolio.get('holdings', 'N/A')}
- Available Cash: ${portfolio.get('cash_balance', 0):,.2f}
- Current Position in {context['symbol']}: {portfolio.get('current_position', 0)} ({portfolio.get('position_value', 0):,.2f} USD)
- Portfolio Allocation: {portfolio.get('allocation_percentage', 0)}% in {context['symbol']}
"""

    # Add trading performance history if available
    if 'performance_history' in context:
        perf = context['performance_history']
        prompt += f"""
## TRADING PERFORMANCE
- Win Rate: {perf.get('win_rate', 0)}%
- Average Profit per Trade: {perf.get('avg_profit', 0)}%
- Average Loss per Trade: {perf.get('avg_loss', 0)}%
- Recent Trades in {context['symbol']}: {perf.get('recent_symbol_trades', 'None')}
"""

    # Add specific analysis instructions
    prompt += """
## ANALYSIS REQUIRED

Based on ALL the data provided above, please analyze:

1. MARKET SENTIMENT: What is the overall market sentiment across news, social, fear/greed, and whale indicators? Are there any conflicts between different sentiment sources?

2. TECHNICAL ANALYSIS: What do the technical indicators collectively suggest? Are there reliable patterns or signals? How strong are the current support/resistance levels?

3. CROSS-VALIDATION: Do sentiment indicators confirm or contradict the technical analysis? How do you resolve any conflicts?

4. RISK ASSESSMENT: What is the current risk level for this asset? What are the main risks to be aware of?

5. TRADING DECISION: Based on your comprehensive analysis, should we buy, sell, or hold? At what conviction level?

6. POSITION SIZING: If trading is recommended, what percentage of available capital should be deployed? How should we manage this position?

Provide your complete analysis in the required JSON format. Be specific about key factors influencing your decision and include clear rationale.
"""
    
    return prompt

def analyze(context, cost_conscious=True):
    """
    Send context data to GPT for enhanced analysis.
    Includes budget management and selective model usage.
    
    Args:
        context (dict): The context data containing all market indicators
        cost_conscious (bool): Whether to optimize for lower costs
        
    Returns:
        dict: GPT's analysis and recommendation
    """
    # Get API key directly from environment
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    
    if not openai_api_key:
        logger.warning("No OpenAI API key found in environment.")
        return fallback_analysis(context)
    
    try:
        # Get the symbol from context
        symbol = context['symbol']
        
        # Run the predictive technical analysis
        logger.info(f"Running predictive technical analysis for {symbol}")
        technical_analysis = predictive_technical_analysis.get_analysis(symbol)
        
        # Check if predictive analysis succeeded
        if technical_analysis.get('status') != 'success':
            logger.warning(f"Predictive analysis failed: {technical_analysis.get('message', 'Unknown error')}")
        else:
            logger.info(f"Predictive analysis recommends: {technical_analysis.get('recommendation')} with conviction {technical_analysis.get('conviction')}")
        
        # Check if we're over budget for the day
        if not check_budget() and cost_conscious:
            logger.warning("Daily GPT budget exceeded. Using fallback analysis.")
            return fallback_to_technical_analysis(technical_analysis, context)
        
        # Determine which model to use based on importance
        # Only use advanced features for important trades if cost-conscious
        use_advanced = False
        if not cost_conscious or (
            technical_analysis and 
            technical_analysis.get('status') == 'success' and 
            abs(technical_analysis.get('conviction', 0)) > 60
        ):
            use_advanced = True
        
        # Select the appropriate model
        model = select_model(context, technical_analysis)
        logger.info(f"Selected model for analysis: {model}")
        
        # Choose appropriate system prompt
        system_prompt = SYSTEM_PROMPT_ADVANCED if use_advanced else SYSTEM_PROMPT_BASIC
        
        # Build the enhanced prompt
        prompt = build_enhanced_prompt(context, technical_analysis, advanced=use_advanced)
        
        # Estimate cost before making the API call
        estimated_cost, input_tokens, estimated_output_tokens = estimate_cost(
            system_prompt + prompt, 
            model=model
        )
        
        # Check if this specific call would put us over budget
        if daily_token_usage['total_cost'] + estimated_cost > DAILY_BUDGET and cost_conscious:
            logger.warning(f"This analysis would exceed daily budget (est. ${estimated_cost:.4f}). Using fallback.")
            return fallback_to_technical_analysis(technical_analysis, context)
        
        logger.info(f"Estimated cost for this analysis: ${estimated_cost:.4f}")
        
        # Prepare the API request
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {openai_api_key}"
        }
        
        data = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            "temperature": config.GPT_TEMPERATURE,
            "max_tokens": config.GPT_MAX_TOKENS
        }
        
        logger.info(f"Calling OpenAI API with model: {model}")
        
        # Call the OpenAI API
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=data
        )
        
        # Check if the request was successful
        if response.status_code != 200:
            logger.error(f"OpenAI API error: {response.status_code} - {response.text}")
            return fallback_to_technical_analysis(technical_analysis, context)
            
        result = response.json()
        
        # Extract the response content
        gpt_response = result['choices'][0]['message']['content'].strip()
        
        # Get actual usage for tracking
        actual_input_tokens = result['usage']['prompt_tokens']
        actual_output_tokens = result['usage']['completion_tokens']
        actual_cost = (
            actual_input_tokens * TOKEN_COSTS[model]['input'] + 
            actual_output_tokens * TOKEN_COSTS[model]['output']
        )
        
        # Update usage tracking
        update_usage_stats(model, actual_input_tokens, actual_output_tokens, actual_cost)
        
        logger.debug(f"GPT Response: {gpt_response}")
        
        try:
            # Parse the JSON response
            analysis = json.loads(gpt_response)
            
            # Check for required fields based on whether we used advanced prompt
            if use_advanced:
                required_fields = ["action", "conviction_score", "confidence_level", 
                                  "position_size_recommendation", "stop_loss_percentage", 
                                  "take_profit_percentage", "rationale"]
            else:
                required_fields = ["action", "conviction_score", "confidence_level", "rationale"]
            
            if all(field in analysis for field in required_fields):
                logger.info(f"Successfully received analysis from OpenAI API: {analysis['action']}")
                
                # Add metadata about the analysis
                analysis['gpt_model_used'] = model
                analysis['cost'] = actual_cost
                analysis['timestamp'] = datetime.now().isoformat()
                
                # Blend with technical analysis for higher quality
                if technical_analysis and technical_analysis.get('status') == 'success':
                    # If GPT and technical analysis agree, boost conviction
                    if analysis['action'] == technical_analysis.get('recommendation'):
                        # Boost conviction by up to 20%
                        boost = min(20, abs(analysis['conviction_score']) * 0.2)
                        if analysis['conviction_score'] > 0:
                            analysis['conviction_score'] = min(100, analysis['conviction_score'] + boost)
                        else:
                            analysis['conviction_score'] = max(-100, analysis['conviction_score'] - boost)
                        
                        analysis['confidence_level'] = "high"  # Increase confidence when both agree
                        
                    # If they disagree significantly, temper the conviction
                    elif analysis['action'] != technical_analysis.get('recommendation') and \
                         (analysis['action'] == 'buy' and technical_analysis.get('recommendation') == 'sell' or \
                         analysis['action'] == 'sell' and technical_analysis.get('recommendation') == 'buy'):
                         
                        # Reduce conviction by 30%
                        analysis['conviction_score'] = int(analysis['conviction_score'] * 0.7)
                        analysis['confidence_level'] = "low"  # Decrease confidence when strong disagreement
                        
                        # Add a note about the conflicting signals
                        analysis['rationale'] += f" However, technical analysis suggests a conflicting {technical_analysis.get('recommendation')} signal, which reduces overall conviction."
                
                return analysis
            else:
                missing = [field for field in required_fields if field not in analysis]
                logger.error(f"GPT response missing required fields: {missing}")
                return fallback_to_technical_analysis(technical_analysis, context)
                
        except json.JSONDecodeError:
            logger.error(f"Failed to parse GPT response as JSON: {gpt_response}")
            return fallback_to_technical_analysis(technical_analysis, context)
            
    except Exception as e:
        logger.error(f"Error in enhanced analysis: {str(e)}")
        return fallback_analysis(context)

def fallback_to_technical_analysis(technical_analysis, context):
    """
    Fall back to using just technical analysis when GPT is unavailable or over budget.
    
    Args:
        technical_analysis (dict): Technical analysis data
        context (dict): Context data
        
    Returns:
        dict: Analysis based on technical indicators
    """
    if technical_analysis and technical_analysis.get('status') == 'success':
        logger.info("Falling back to predictive technical analysis recommendation")
        
        # Convert the predictive analysis recommendation to our format
        recommendation = technical_analysis.get('recommendation', 'hold')
        conviction = technical_analysis.get('conviction', 0)
        confidence = "medium"
        
        if abs(conviction) > 70:
            confidence = "high"
        elif abs(conviction) < 30:
            confidence = "low"
            
        rationale = "Based on technical analysis"
        
        if technical_analysis.get('trend'):
            rationale += f", with a {technical_analysis.get('trend')} trend"
            
        if technical_analysis.get('patterns', {}).get('bullish') and recommendation == 'buy':
            bullish_patterns = technical_analysis.get('patterns', {}).get('bullish', [])
            if bullish_patterns:
                rationale += f" and bullish patterns including {', '.join(p.replace('_', ' ') for p in bullish_patterns[:2])}"
                
        if technical_analysis.get('patterns', {}).get('bearish') and recommendation == 'sell':
            bearish_patterns = technical_analysis.get('patterns', {}).get('bearish', [])
            if bearish_patterns:
                rationale += f" and bearish patterns including {', '.join(p.replace('_', ' ') for p in bearish_patterns[:2])}"
        
        return {
            "action": recommendation,
            "conviction_score": conviction,
            "confidence_level": confidence,
            "rationale": rationale,
            "used_fallback": True,
            "fallback_reason": "budget_exceeded"
        }
    else:
        return fallback_analysis(context)

def fallback_analysis(context):
    """
    Generate a fallback analysis when both GPT and technical analysis are unavailable.
    
    Args:
        context (dict): The context data
        
    Returns:
        dict: Simulated analysis
    """
    logger.info("Using fallback decision logic")
    
    # Convert sentiment strings to numeric values for scoring
    sentiment_scores = {
        "bullish": 1,
        "neutral": 0,
        "bearish": -1,
        "large_buy": 1,
        "large_sell": -1,
        "currently rising": 0.5,
        "strongly rising": 1,
        "sideways movement": 0,
        "currently falling": -0.5,
        "strongly falling": -1,
        "yes": 0.5,  # For volume spike
        "no": 0
    }
    
    # Initialize scoring
    score = 0
    confidence_points = 0
    
    # Process news sentiment (weight: 2)
    news_sentiment = context['news_sentiment']
    if news_sentiment in sentiment_scores:
        score += sentiment_scores[news_sentiment] * 2
        confidence_points += 1
    
    # Process RSI (weight: 2)
    rsi = context.get('rsi')
    if rsi is not None:
        if rsi <= 30:  # Oversold
            score += 1 * 2
        elif rsi >= 70:  # Overbought
            score -= 1 * 2
        confidence_points += 1
    
    # Process MACD (weight: 1.5)
    macd = context.get('macd_crossover')
    if macd in sentiment_scores:
        score += sentiment_scores[macd] * 1.5
        confidence_points += 1
    
    # Process Fear & Greed Index (weight: 1, contrarian)
    fear_greed = context['fear_greed_index']
    if fear_greed == "Extreme Fear":
        score += 1  # Contrarian indicator
    elif fear_greed == "Extreme Greed":
        score -= 1  # Contrarian indicator
    elif fear_greed == "Fear":
        score += 0.5
    elif fear_greed == "Greed":
        score -= 0.5
    confidence_points += 1
    
    # Process Social Media Sentiment (weight: 1)
    social = context['social_media_sentiment']
    if social in sentiment_scores:
        score += sentiment_scores[social] * 1
        confidence_points += 0.5
    
    # Process Whale Activity (weight: 2)
    whale = context['whale_transactions']
    if whale in sentiment_scores:
        score += sentiment_scores[whale] * 2
        confidence_points += 1
    
    # Process Price Action (weight: 1)
    price_action = context.get('price_action')
    if price_action in sentiment_scores:
        score += sentiment_scores[price_action] * 1
        confidence_points += 0.5
    
    # Process Volume Spike (weight: 1)
    volume = context.get('volume_spike')
    if volume in sentiment_scores:
        score += sentiment_scores[volume] * 1
        confidence_points += 0.5
    
    # Process CryptoCompare data if available (weight: 2)
    if 'crypto_data' in context:
        crypto_data = context['crypto_data']
        price_change_24h = crypto_data.get('price_change_24h', 0)
        
        # Use 24h price change as a signal
        if price_change_24h > 5:  # Strong positive change
            score += 1 * 2
        elif price_change_24h > 2:  # Moderate positive change
            score += 0.5 * 2
        elif price_change_24h < -5:  # Strong negative change
            score -= 1 * 2
        elif price_change_24h < -2:  # Moderate negative change
            score -= 0.5 * 2
        
        confidence_points += 1
    
    # Normalize score to -100 to +100 range
    max_possible_score = 12  # Sum of all weights
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
    max_confidence = 8  # Total number of indicators
    confidence_ratio = confidence_points / max_confidence
    
    if confidence_ratio >= 0.7:
        confidence_level = "high"
    elif confidence_ratio >= 0.4:
        confidence_level = "medium"
    else:
        confidence_level = "low"
    
    # Generate rationale
    rationales = []
    
    if rsi is not None:
        if rsi <= 30:
            rationales.append(f"RSI at {rsi} indicates oversold conditions")
        elif rsi >= 70:
            rationales.append(f"RSI at {rsi} indicates overbought conditions")
    
    if news_sentiment == "bullish":
        rationales.append("News sentiment is positive")
    elif news_sentiment == "bearish":
        rationales.append("News sentiment is negative")
    
    if macd == "bullish":
        rationales.append("MACD shows bullish momentum")
    elif macd == "bearish":
        rationales.append("MACD shows bearish momentum")
    
    if whale == "large_buy":
        rationales.append("Significant whale buying detected")
    elif whale == "large_sell":
        rationales.append("Significant whale selling detected")
    
    if fear_greed == "Extreme Fear":
        rationales.append("Market in Extreme Fear (contrarian bullish)")
    elif fear_greed == "Extreme Greed":
        rationales.append("Market in Extreme Greed (contrarian bearish)")
    
    if 'crypto_data' in context:
        price_change = context['crypto_data'].get('price_change_24h', 0)
        if price_change > 5:
            rationales.append(f"Strong positive price movement in last 24h ({price_change:.1f}%)")
        elif price_change < -5:
            rationales.append(f"Strong negative price movement in last 24h ({price_change:.1f}%)")
    
    # Select up to 3 rationale points
    if rationales:
        rationale = " and ".join(random.sample(rationales, min(3, len(rationales)))) + "."
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
        "used_fallback": True,
        "fallback_reason": "no_gpt_and_no_technical_analysis"
    }

def get_trading_recommendation(context, use_gpt=True, cost_conscious=True):
    """
    Main function to get a trading recommendation based on context.
    Entry point for other modules to use this service.
    
    Args:
        context (dict): The context data containing all market indicators
        use_gpt (bool): Whether to try using GPT if available
        cost_conscious (bool): Whether to optimize for lower costs
        
    Returns:
        dict: Trading recommendation
    """
    # Check if OpenAI API key is available
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    
    if not openai_api_key:
        logger.warning("No OpenAI API key found in environment, using fallback analysis.")
        return fallback_analysis(context)
    
    # Early check if GPT is disabled
    if not use_gpt or not getattr(config, 'GPT_ENABLED', True):
        logger.info("GPT analysis disabled, using rule-based analysis.")
        return fallback_analysis(context)
        
    try:
        # Check budget status
        budget_status = get_budget_status()
        remaining_budget = budget_status['remaining_budget']
        
        # If budget is critically low and we're being cost conscious
        if remaining_budget < 0.05 and cost_conscious:  # Less than 5 cents
            logger.warning(f"GPT budget critically low (${remaining_budget:.2f}). Using fallback.")
            return fallback_to_technical_analysis(
                predictive_technical_analysis.get_analysis(context['symbol']), 
                context
            )
        
        # Run GPT analysis
        analysis_result = analyze(context, cost_conscious)
        
        # Log the recommendation
        action = analysis_result.get('action', 'hold')
        conviction = analysis_result.get('conviction_score', 0)
        confidence = analysis_result.get('confidence_level', 'medium')
        
        logger.info(f"GPT recommends: {action.upper()} {context['symbol']} with conviction {conviction} ({confidence} confidence)")
        
        return analysis_result
        
    except Exception as e:
        logger.error(f"Error in GPT analysis: {str(e)}", exc_info=True)
        logger.info("Falling back to rule-based analysis due to error")
        
        # Use technical analysis as fallback if possible
        try:
            technical_analysis = predictive_technical_analysis.get_analysis(context['symbol'])
            if technical_analysis and technical_analysis.get('status') == 'success':
                return fallback_to_technical_analysis(technical_analysis, context)
        except Exception:
            pass
            
        # Ultimate fallback
        return fallback_analysis(context)

def get_budget_status():
    """
    Get the current budget status.
    
    Returns:
        dict: Budget status
    """
    global daily_token_usage
    
    # Calculate remaining budget
    remaining_budget = DAILY_BUDGET - daily_token_usage['total_cost']
    usage_percent = (daily_token_usage['total_cost'] / DAILY_BUDGET) * 100 if DAILY_BUDGET > 0 else 100
    
    return {
        'date': daily_token_usage['date'],
        'total_cost': daily_token_usage['total_cost'],
        'daily_budget': DAILY_BUDGET,
        'remaining_budget': remaining_budget,
        'usage_percent': usage_percent,
        'total_calls': daily_token_usage['total_calls'],
        'gpt35_calls': daily_token_usage['gpt35_calls'],
        'gpt4_calls': daily_token_usage['gpt4_calls'],
        'input_tokens': daily_token_usage['input_tokens'],
        'output_tokens': daily_token_usage['output_tokens']
    }

def get_recent_usage_stats(days=7):
    """
    Get usage statistics for recent days.
    
    Args:
        days (int): Number of days to include
        
    Returns:
        list: Usage statistics for recent days
    """
    try:
        if os.path.exists(USAGE_DATA_FILE):
            with open(USAGE_DATA_FILE, 'r') as f:
                usage_history = json.load(f)
                
            # Return the most recent N days
            return usage_history[-days:]
        else:
            return []
    except Exception as e:
        logger.error(f"Error getting usage stats: {str(e)}")
        return []
        
def generate_trading_report(portfolio, recent_trades, global_market):
    """
    Generate a comprehensive trading report using GPT.
    
    Args:
        portfolio (dict): Current portfolio information
        recent_trades (list): Recent trade history
        global_market (dict): Global market conditions
        
    Returns:
        str: Formatted trading report
    """
    # Check budget status
    budget_status = get_budget_status()
    if budget_status['remaining_budget'] < 0.15:  # Need at least 15 cents
        logger.warning("Insufficient budget for GPT trading report")
        return generate_basic_report(portfolio, recent_trades, global_market)
    
    # Build report prompt
    prompt = f"""
# TRADING REPORT REQUEST

Please generate a comprehensive trading report based on the following data:

## PORTFOLIO SUMMARY
{json.dumps(portfolio, indent=2)}

## RECENT TRADING ACTIVITY
{json.dumps(recent_trades, indent=2)}

## MARKET CONDITIONS
{json.dumps(global_market, indent=2)}

The report should include:
1. Portfolio Performance Summary
2. Analysis of Recent Trades - patterns of success and failure
3. Market Outlook based on current conditions
4. Recommendations for portfolio adjustments
5. Key risks to monitor

Format the report in a professional style with clear sections and bullet points where appropriate.
"""
    
    # Select model (always use GPT-3.5 for reports to save budget)
    model = "gpt-3.5-turbo"
    
    try:
        # Estimate cost
        estimated_cost, input_tokens, _ = estimate_cost(prompt, model=model, estimated_output_tokens=1000)
        
        # Check budget
        if budget_status['remaining_budget'] < estimated_cost:
            logger.warning(f"Insufficient budget for report (est. ${estimated_cost:.2f})")
            return generate_basic_report(portfolio, recent_trades, global_market)
        
        # Make API request
        openai_api_key = os.environ.get("OPENAI_API_KEY")
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {openai_api_key}"
        }
        
        data = {
            "model": model,
            "messages": [
                {"role": "system", "content": "You are a professional crypto trading analyst."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.3,
            "max_tokens": 1500
        }
        
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=data
        )
        
        if response.status_code != 200:
            logger.error(f"OpenAI API error: {response.status_code} - {response.text}")
            return generate_basic_report(portfolio, recent_trades, global_market)
            
        result = response.json()
        report = result['choices'][0]['message']['content'].strip()
        
        # Update token usage
        actual_input_tokens = result['usage']['prompt_tokens']
        actual_output_tokens = result['usage']['completion_tokens']
        actual_cost = (
            actual_input_tokens * TOKEN_COSTS[model]['input'] + 
            actual_output_tokens * TOKEN_COSTS[model]['output']
        )
        
        update_usage_stats(model, actual_input_tokens, actual_output_tokens, actual_cost)
        
        return report
        
    except Exception as e:
        logger.error(f"Error generating report: {str(e)}")
        return generate_basic_report(portfolio, recent_trades, global_market)

def generate_basic_report(portfolio, recent_trades, global_market):
    """Generate a basic report without using GPT"""
    # Simple non-GPT report generation logic
    total_value = portfolio.get('total_usd_value', 0)
    cash = portfolio.get('cash_balance', 0)
    
    report = f"""
# TRADING REPORT

## PORTFOLIO SUMMARY
- Total Value: ${total_value:.2f}
- Cash: ${cash:.2f} ({(cash/total_value*100):.1f}% of portfolio)
- Holdings: {len(portfolio.get('holdings', {}))} assets

## RECENT ACTIVITY
- {len(recent_trades)} recent trades
"""
    
    # Add basic market info
    if global_market:
        report += f"""
## MARKET CONDITIONS
- Market Cap: ${global_market.get('market_cap_usd', 0)/1e9:.2f}B
- BTC Dominance: {global_market.get('btc_dominance', 0):.1f}%
"""
    
    return report

def optimize_budget_allocation(remaining_budget, priority_coins):
    """
    Optimize how to allocate remaining daily budget.
    
    Args:
        remaining_budget (float): Remaining budget in USD
        priority_coins (list): List of coins to prioritize
        
    Returns:
        dict: Budget allocation recommendations
    """
    # If budget is very low, reserve for high priority coins only
    if remaining_budget < 0.1:  # Less than 10 cents
        return {
            'recommendation': 'emergency_only',
            'message': 'Budget critically low. Use GPT only for emergency signals.',
            'use_gpt': False,
            'default_model': 'gpt-3.5-turbo',
            'max_calls_remaining': 0
        }
    
    # Calculate typical cost per analysis
    avg_cost_per_call = 0.05  # Estimated average
    if daily_token_usage['total_calls'] > 0:
        avg_cost_per_call = daily_token_usage['total_cost'] / daily_token_usage['total_calls']
    
    # Estimate how many calls we can still make
    max_calls_remaining = int(remaining_budget / avg_cost_per_call)
    
    # Determine which model to use based on remaining budget
    default_model = 'gpt-3.5-turbo'
    use_gpt = True
    
    if remaining_budget < 0.25:  # Less than 25 cents
        return {
            'recommendation': 'high_priority_only',
            'message': 'Budget low. Use GPT only for highest priority coins and signals.',
            'use_gpt': True,
            'default_model': 'gpt-3.5-turbo',
            'priority_coins': priority_coins[:1],  # Only top priority coin
            'max_calls_remaining': max_calls_remaining
        }
    elif remaining_budget < 0.5:  # Less than 50 cents
        return {
            'recommendation': 'selective_use',
            'message': 'Budget moderate. Use GPT selectively.',
            'use_gpt': True,
            'default_model': 'gpt-3.5-turbo',
            'priority_coins': priority_coins[:2],  # Top two priority coins
            'max_calls_remaining': max_calls_remaining
        }
    else:  # More than 50 cents
        return {
            'recommendation': 'normal_use',
            'message': 'Budget healthy. Use GPT normally.',
            'use_gpt': True,
            'default_model': 'gpt-3.5-turbo',
            'priority_coins': priority_coins,
            'max_calls_remaining': max_calls_remaining
        }

# Load usage data on module initialization
try:
    if os.path.exists(USAGE_DATA_FILE):
        with open(USAGE_DATA_FILE, 'r') as f:
            usage_history = json.load(f)
            
        # Find today's entry if it exists
        today = datetime.now().strftime('%Y-%m-%d')
        for entry in usage_history:
            if entry.get('date') == today:
                daily_token_usage = entry
                break
except Exception as e:
    logger.error(f"Error loading usage data: {str(e)}")