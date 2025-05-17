"""
Predictive Technical Analysis Module

This module provides advanced technical analysis capabilities including:
1. Advanced technical indicators beyond basic RSI/MACD
2. Chart pattern recognition
3. Support/resistance level detection
4. Trend strength analysis
5. Volatility analysis
"""

import numpy as np
import pandas as pd
import logging
import talib
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from scipy.signal import argrelextrema
import config

# Set up logging
logger = logging.getLogger('crypto_bot.predictive_analysis')

def get_ohlcv_data(symbol, timeframe='1h', limit=200):
    """
    Get OHLCV data for a cryptocurrency.
    Uses technical_indicator_fetcher as the data source.
    
    Args:
        symbol (str): Cryptocurrency symbol
        timeframe (str): Time interval
        limit (int): Number of candles to retrieve
        
    Returns:
        pd.DataFrame: DataFrame with OHLCV data
    """
    from core import technical_indicator_fetcher
    import pandas as pd
    import numpy as np
    import logging
    
    logger = logging.getLogger('crypto_bot.predictive_analysis')
    
    # Get historical data from the technical indicator fetcher
    historical_data = technical_indicator_fetcher.get_historical_data(symbol, timeframe, limit)
    
    if not historical_data or len(historical_data) < 50:
        logger.error(f"Not enough historical data for {symbol} to perform predictive analysis")
        return None
    
    # Convert to DataFrame
    df = pd.DataFrame(historical_data)
    
    # Rename columns to standard OHLCV names if needed
    if not all(col in df.columns for col in ['open', 'high', 'low', 'close', 'volume']):
        df = df.rename(columns={
            'timestamp': 'timestamp',
            'open': 'open',
            'high': 'high',
            'low': 'low',
            'close': 'close',
            'volume': 'volume'
        })
    
    # Sort by timestamp (oldest first)
    df = df.sort_values('timestamp')
    
    # Set timestamp as index
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
    df = df.set_index('datetime')
    
    return df

def calculate_advanced_indicators(df):
    """
    Calculate advanced technical indicators.
    
    Args:
        df (pd.DataFrame): DataFrame with OHLCV data
        
    Returns:
        pd.DataFrame: DataFrame with additional indicators
    """
    if df is None or len(df) < 50:
        return None
    
    # Convert to numpy arrays for TALib
    open_prices = df['open'].values
    high_prices = df['high'].values
    low_prices = df['low'].values
    close_prices = df['close'].values
    volume = df['volume'].values
    
    try:
        # Basic indicators
        df['sma_20'] = talib.SMA(close_prices, timeperiod=20)
        df['sma_50'] = talib.SMA(close_prices, timeperiod=50)
        df['sma_200'] = talib.SMA(close_prices, timeperiod=200)
        df['ema_20'] = talib.EMA(close_prices, timeperiod=20)
        
        # RSI
        df['rsi_14'] = talib.RSI(close_prices, timeperiod=14)
        
        # MACD
        macd, macd_signal, macd_hist = talib.MACD(close_prices, 
                                                 fastperiod=12, 
                                                 slowperiod=26, 
                                                 signalperiod=9)
        df['macd'] = macd
        df['macd_signal'] = macd_signal
        df['macd_hist'] = macd_hist
        
        # Bollinger Bands
        upper, middle, lower = talib.BBANDS(close_prices, 
                                           timeperiod=20, 
                                           nbdevup=2, 
                                           nbdevdn=2)
        df['bb_upper'] = upper
        df['bb_middle'] = middle
        df['bb_lower'] = lower
        
        # Calculate Bollinger Band width (volatility indicator)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        
        # Stochastic Oscillator
        slowk, slowd = talib.STOCH(high_prices, 
                                   low_prices, 
                                   close_prices, 
                                   fastk_period=14, 
                                   slowk_period=3, 
                                   slowk_matype=0, 
                                   slowd_period=3, 
                                   slowd_matype=0)
        df['stoch_k'] = slowk
        df['stoch_d'] = slowd
        
        # Average Directional Index (ADX) - Trend Strength
        df['adx'] = talib.ADX(high_prices, low_prices, close_prices, timeperiod=14)
        
        # On-Balance Volume (OBV)
        df['obv'] = talib.OBV(close_prices, volume)
        
        # Ichimoku Cloud
        df['tenkan_sen'] = (df['high'].rolling(window=9).max() + df['low'].rolling(window=9).min()) / 2
        df['kijun_sen'] = (df['high'].rolling(window=26).max() + df['low'].rolling(window=26).min()) / 2
        df['senkou_span_a'] = ((df['tenkan_sen'] + df['kijun_sen']) / 2).shift(26)
        df['senkou_span_b'] = ((df['high'].rolling(window=52).max() + df['low'].rolling(window=52).min()) / 2).shift(26)
        df['chikou_span'] = df['close'].shift(-26)
        
        # Average True Range (ATR) - Volatility
        df['atr'] = talib.ATR(high_prices, low_prices, close_prices, timeperiod=14)
        
        # Rate of Change (ROC)
        df['roc'] = talib.ROC(close_prices, timeperiod=10)
        
        # Williams %R
        df['willr'] = talib.WILLR(high_prices, low_prices, close_prices, timeperiod=14)
        
        # Money Flow Index (MFI)
        df['mfi'] = talib.MFI(high_prices, low_prices, close_prices, volume, timeperiod=14)
        
        # Commodity Channel Index (CCI)
        df['cci'] = talib.CCI(high_prices, low_prices, close_prices, timeperiod=14)
        
        # Relative Volatility Index (RVI) - simplified calculation
        std_14 = df['close'].rolling(window=14).std()
        df['rvi'] = std_14.rolling(window=14).mean() / std_14
        
        # Hull Moving Average (HMA)
        wma_half_length = talib.WMA(close_prices, timeperiod=10)
        wma_full_length = talib.WMA(close_prices, timeperiod=20)
        sqrt_length = int(np.sqrt(20))
        raw_hma = 2 * wma_half_length - wma_full_length
        df['hma'] = pd.Series(talib.WMA(raw_hma, timeperiod=sqrt_length), index=df.index)
        
        # Parabolic SAR
        df['sar'] = talib.SAR(high_prices, low_prices, acceleration=0.02, maximum=0.2)
        
        # Chaikin Money Flow (CMF)
        df['cmf'] = talib.ADOSC(high_prices, low_prices, close_prices, volume, fastperiod=3, slowperiod=10)
        
        logger.info(f"Successfully calculated advanced indicators")
        return df
        
    except Exception as e:
        logger.error(f"Error calculating advanced indicators: {str(e)}")
        return df

def identify_chart_patterns(df):
    """
    Identify common chart patterns.
    
    Args:
        df (pd.DataFrame): DataFrame with OHLCV and indicator data
        
    Returns:
        dict: Dictionary of identified patterns and their strengths
    """
    if df is None or len(df) < 50:
        return {}
    
    patterns = {}
    
    try:
        close_prices = df['close'].values
        high_prices = df['high'].values
        low_prices = df['low'].values
        open_prices = df['open'].values
        
        # Candlestick patterns using TALib
        patterns['hammer'] = talib.CDLHAMMER(open_prices, high_prices, low_prices, close_prices)[-1]
        patterns['doji'] = talib.CDLDOJI(open_prices, high_prices, low_prices, close_prices)[-1]
        patterns['engulfing'] = talib.CDLENGULFING(open_prices, high_prices, low_prices, close_prices)[-1]
        patterns['evening_star'] = talib.CDLEVENINGSTAR(open_prices, high_prices, low_prices, close_prices)[-1]
        patterns['morning_star'] = talib.CDLMORNINGSTAR(open_prices, high_prices, low_prices, close_prices)[-1]
        patterns['shooting_star'] = talib.CDLSHOOTINGSTAR(open_prices, high_prices, low_prices, close_prices)[-1]
        patterns['three_white_soldiers'] = talib.CDL3WHITESOLDIERS(open_prices, high_prices, low_prices, close_prices)[-1]
        patterns['three_black_crows'] = talib.CDL3BLACKCROWS(open_prices, high_prices, low_prices, close_prices)[-1]
        
        # Moving Average Crossovers
        if len(df) >= 50:
            # Golden Cross (SMA 50 crosses above SMA 200)
            if df['sma_50'].iloc[-2] < df['sma_200'].iloc[-2] and df['sma_50'].iloc[-1] > df['sma_200'].iloc[-1]:
                patterns['golden_cross'] = 100
            # Death Cross (SMA 50 crosses below SMA 200)
            elif df['sma_50'].iloc[-2] > df['sma_200'].iloc[-2] and df['sma_50'].iloc[-1] < df['sma_200'].iloc[-1]:
                patterns['death_cross'] = -100
            
            # EMA and SMA Crossovers
            if df['ema_20'].iloc[-2] < df['sma_50'].iloc[-2] and df['ema_20'].iloc[-1] > df['sma_50'].iloc[-1]:
                patterns['ema_cross_above_sma'] = 70
            elif df['ema_20'].iloc[-2] > df['sma_50'].iloc[-2] and df['ema_20'].iloc[-1] < df['sma_50'].iloc[-1]:
                patterns['ema_cross_below_sma'] = -70
        
        # MACD Crossovers
        if 'macd' in df.columns and 'macd_signal' in df.columns:
            if df['macd'].iloc[-2] < df['macd_signal'].iloc[-2] and df['macd'].iloc[-1] > df['macd_signal'].iloc[-1]:
                patterns['macd_bullish_cross'] = 60
            elif df['macd'].iloc[-2] > df['macd_signal'].iloc[-2] and df['macd'].iloc[-1] < df['macd_signal'].iloc[-1]:
                patterns['macd_bearish_cross'] = -60
        
        # RSI Conditions
        if 'rsi_14' in df.columns:
            # Oversold
            if df['rsi_14'].iloc[-1] < 30:
                patterns['rsi_oversold'] = 75
            # Overbought
            elif df['rsi_14'].iloc[-1] > 70:
                patterns['rsi_overbought'] = -75
            
            # RSI Divergence (simplified)
            if len(df) >= 14:
                # Bullish divergence (price making lower lows, RSI making higher lows)
                price_making_lower_low = df['close'].iloc[-1] < df['close'].iloc[-5] and df['close'].iloc[-5] < df['close'].iloc[-10]
                rsi_making_higher_low = df['rsi_14'].iloc[-1] > df['rsi_14'].iloc[-5] and df['rsi_14'].iloc[-5] > df['rsi_14'].iloc[-10]
                
                if price_making_lower_low and rsi_making_higher_low:
                    patterns['bullish_divergence'] = 80
                
                # Bearish divergence (price making higher highs, RSI making lower highs)
                price_making_higher_high = df['close'].iloc[-1] > df['close'].iloc[-5] and df['close'].iloc[-5] > df['close'].iloc[-10]
                rsi_making_lower_high = df['rsi_14'].iloc[-1] < df['rsi_14'].iloc[-5] and df['rsi_14'].iloc[-5] < df['rsi_14'].iloc[-10]
                
                if price_making_higher_high and rsi_making_lower_high:
                    patterns['bearish_divergence'] = -80
        
        # Bollinger Band Signals
        if all(col in df.columns for col in ['bb_upper', 'bb_middle', 'bb_lower']):
            # Price breaking out above upper band
            if df['close'].iloc[-1] > df['bb_upper'].iloc[-1]:
                patterns['bollinger_breakout_up'] = 65
            # Price breaking below lower band
            elif df['close'].iloc[-1] < df['bb_lower'].iloc[-1]:
                patterns['bollinger_breakout_down'] = -65
            
            # Bollinger Band Squeeze (volatility contraction)
            current_bb_width = (df['bb_upper'].iloc[-1] - df['bb_lower'].iloc[-1]) / df['bb_middle'].iloc[-1]
            avg_bb_width = df['bb_width'].tail(20).mean()
            
            if current_bb_width < avg_bb_width * 0.8:
                patterns['bollinger_squeeze'] = 50  # Neutral but significant
        
        # Ichimoku Cloud Analysis
        if all(col in df.columns for col in ['tenkan_sen', 'kijun_sen', 'senkou_span_a', 'senkou_span_b']):
            # Price above the cloud
            if df['close'].iloc[-1] > df['senkou_span_a'].iloc[-1] and df['close'].iloc[-1] > df['senkou_span_b'].iloc[-1]:
                patterns['price_above_cloud'] = 60
            # Price below the cloud
            elif df['close'].iloc[-1] < df['senkou_span_a'].iloc[-1] and df['close'].iloc[-1] < df['senkou_span_b'].iloc[-1]:
                patterns['price_below_cloud'] = -60
                
            # Tenkan-sen / Kijun-sen Cross (TK Cross)
            if df['tenkan_sen'].iloc[-2] < df['kijun_sen'].iloc[-2] and df['tenkan_sen'].iloc[-1] > df['kijun_sen'].iloc[-1]:
                patterns['bullish_tk_cross'] = 70
            elif df['tenkan_sen'].iloc[-2] > df['kijun_sen'].iloc[-2] and df['tenkan_sen'].iloc[-1] < df['kijun_sen'].iloc[-1]:
                patterns['bearish_tk_cross'] = -70
                
        # Filter out zero values
        patterns = {k: v for k, v in patterns.items() if v != 0}
        
        logger.info(f"Identified {len(patterns)} chart patterns")
        return patterns
        
    except Exception as e:
        logger.error(f"Error identifying chart patterns: {str(e)}")
        return {}

def detect_support_resistance(df, n_levels=3):
    """
    Detect key support and resistance levels.
    
    Args:
        df (pd.DataFrame): DataFrame with OHLCV data
        n_levels (int): Number of levels to identify
        
    Returns:
        dict: Dictionary with support and resistance levels
    """
    if df is None or len(df) < 50:
        return {'support': [], 'resistance': []}
    
    try:
        # Get local maxima and minima
        close_prices = df['close'].values
        
        # Find local maxima and minima using scipy
        order = 5  # Adjust as needed
        max_idx = argrelextrema(close_prices, np.greater, order=order)[0]
        min_idx = argrelextrema(close_prices, np.less, order=order)[0]
        
        # Get the prices at these points
        resistance_levels = close_prices[max_idx]
        support_levels = close_prices[min_idx]
        
        # Function to find clusters
        def find_clusters(levels, threshold_pct=0.02):
            if len(levels) == 0:
                return []
                
            # Sort levels
            sorted_levels = np.sort(levels)
            
            # Cluster nearby levels
            clusters = []
            current_cluster = [sorted_levels[0]]
            
            for i in range(1, len(sorted_levels)):
                # If this level is close to the previous one
                if sorted_levels[i] <= current_cluster[-1] * (1 + threshold_pct):
                    current_cluster.append(sorted_levels[i])
                else:
                    # Take the average of the cluster
                    clusters.append(np.mean(current_cluster))
                    current_cluster = [sorted_levels[i]]
            
            # Don't forget the last cluster
            if current_cluster:
                clusters.append(np.mean(current_cluster))
            
            return clusters
        
        # Identify clusters of support and resistance
        resistance_clusters = find_clusters(resistance_levels)
        support_clusters = find_clusters(support_levels)
        
        # Sort by strength (frequency of touches)
        resistance_clusters = sorted(resistance_clusters, reverse=True)
        support_clusters = sorted(support_clusters)
        
        # Get current price
        current_price = df['close'].iloc[-1]
        
        # Filter levels - keep only relevant ones
        resistance_clusters = [level for level in resistance_clusters if level > current_price]
        support_clusters = [level for level in support_clusters if level < current_price]
        
        # Take only the closest n levels
        resistance_clusters = resistance_clusters[:n_levels]
        support_clusters = support_clusters[:n_levels]
        
        logger.info(f"Detected {len(support_clusters)} support and {len(resistance_clusters)} resistance levels")
        
        return {
            'support': support_clusters,
            'resistance': resistance_clusters
        }
    
    except Exception as e:
        logger.error(f"Error detecting support/resistance levels: {str(e)}")
        return {'support': [], 'resistance': []}

def calculate_trend_strength(df):
    """
    Calculate the strength and direction of the current trend.
    
    Args:
        df (pd.DataFrame): DataFrame with OHLCV and indicator data
        
    Returns:
        dict: Trend strength and direction metrics
    """
    if df is None or len(df) < 50:
        return {'trend_direction': 'neutral', 'trend_strength': 0}
    
    try:
        # Initialize trend metrics
        trend_metrics = {}
        
        # ADX for trend strength
        if 'adx' in df.columns:
            adx_value = df['adx'].iloc[-1]
            
            if adx_value < 20:
                trend_metrics['adx_trend_strength'] = 'weak'
            elif adx_value < 40:
                trend_metrics['adx_trend_strength'] = 'moderate'
            else:
                trend_metrics['adx_trend_strength'] = 'strong'
                
            trend_metrics['adx_value'] = adx_value
        
        # Determine trend direction based on moving averages
        if all(col in df.columns for col in ['sma_20', 'sma_50']):
            sma_20 = df['sma_20'].iloc[-1]
            sma_50 = df['sma_50'].iloc[-1]
            
            if sma_20 > sma_50 * 1.02:  # 2% above
                trend_metrics['ma_trend_direction'] = 'bullish'
            elif sma_20 < sma_50 * 0.98:  # 2% below
                trend_metrics['ma_trend_direction'] = 'bearish'
            else:
                trend_metrics['ma_trend_direction'] = 'neutral'
        
        # Linear regression slope for trend direction and strength
        if len(df) >= 20:
            # Use last 20 periods for regression
            close_prices = df['close'].values[-20:]
            x = np.array(range(len(close_prices))).reshape(-1, 1)
            y = close_prices
            
            model = LinearRegression()
            model.fit(x, y)
            
            # Slope of the line
            slope = model.coef_[0]
            
            # R-squared as a measure of trend consistency
            r_squared = model.score(x, y)
            
            # Normalize the slope relative to price
            normalized_slope = slope / np.mean(close_prices) * 100
            
            trend_metrics['slope'] = normalized_slope
            trend_metrics['r_squared'] = r_squared
            
            if normalized_slope > 0.5 and r_squared > 0.6:
                trend_metrics['regression_trend'] = 'strong_bullish'
            elif normalized_slope > 0.2 and r_squared > 0.4:
                trend_metrics['regression_trend'] = 'moderate_bullish'
            elif normalized_slope < -0.5 and r_squared > 0.6:
                trend_metrics['regression_trend'] = 'strong_bearish'
            elif normalized_slope < -0.2 and r_squared > 0.4:
                trend_metrics['regression_trend'] = 'moderate_bearish'
            else:
                trend_metrics['regression_trend'] = 'neutral'
        
        # Overall trend determination
        # Simple algorithm: combine ADX strength with regression direction
        if 'adx_trend_strength' in trend_metrics and 'regression_trend' in trend_metrics:
            strength_map = {'weak': 0.3, 'moderate': 0.6, 'strong': 1.0}
            adx_strength = strength_map.get(trend_metrics['adx_trend_strength'], 0.3)
            
            direction_map = {
                'strong_bullish': 1.0,
                'moderate_bullish': 0.5,
                'neutral': 0,
                'moderate_bearish': -0.5,
                'strong_bearish': -1.0
            }
            
            direction = direction_map.get(trend_metrics['regression_trend'], 0)
            
            # Combine for overall trend score (-100 to +100)
            trend_score = direction * adx_strength * 100
            
            # Determine overall trend
            if trend_score > 30:
                trend_direction = 'bullish'
            elif trend_score < -30:
                trend_direction = 'bearish'
            else:
                trend_direction = 'neutral'
                
            return {
                'trend_direction': trend_direction,
                'trend_strength': trend_score,
                'details': trend_metrics
            }
        
        # Fallback if we couldn't compute the advanced metrics
        return {'trend_direction': 'neutral', 'trend_strength': 0, 'details': trend_metrics}
    
    except Exception as e:
        logger.error(f"Error calculating trend strength: {str(e)}")
        return {'trend_direction': 'neutral', 'trend_strength': 0}

def analyze_volatility(df):
    """
    Analyze the volatility of the market.
    
    Args:
        df (pd.DataFrame): DataFrame with OHLCV and indicator data
        
    Returns:
        dict: Volatility metrics
    """
    if df is None or len(df) < 50:
        return {'volatility': 'unknown', 'volatility_score': 0}
    
    try:
        volatility_metrics = {}
        
        # Calculate historical volatility (standard deviation of returns)
        returns = df['close'].pct_change().dropna()
        hist_vol = returns.std() * np.sqrt(365 * 24)  # Annualized for hourly data
        volatility_metrics['historical_volatility'] = hist_vol
        
        # Average True Range
        if 'atr' in df.columns:
            # Normalize ATR by price
            atr_pct = df['atr'].iloc[-1] / df['close'].iloc[-1]
            volatility_metrics['atr_pct'] = atr_pct
        
        # Bollinger Band Width
        if 'bb_width' in df.columns:
            current_bb_width = df['bb_width'].iloc[-1]
            avg_bb_width = df['bb_width'].tail(20).mean()
            
            volatility_metrics['bb_width'] = current_bb_width
            volatility_metrics['bb_width_relative'] = current_bb_width / avg_bb_width
        
        # Calculate volatility score (0-100)
        vol_score = 0
        
        if 'historical_volatility' in volatility_metrics:
            # Map historical volatility to a score (0-40)
            # These thresholds can be adjusted based on the asset
            if hist_vol < 0.3:  # Low volatility
                vol_score += hist_vol / 0.3 * 10
            elif hist_vol < 0.6:  # Medium volatility
                vol_score += 10 + (hist_vol - 0.3) / 0.3 * 10
            else:  # High volatility
                vol_score += 20 + min((hist_vol - 0.6) / 0.4 * 20, 20)
        
        if 'atr_pct' in volatility_metrics:
            # Map ATR to a score (0-30)
            atr_pct = volatility_metrics['atr_pct']
            if atr_pct < 0.01:  # Very low
                vol_score += atr_pct / 0.01 * 10
            elif atr_pct < 0.03:  # Low to medium
                vol_score += 10 + (atr_pct - 0.01) / 0.02 * 10
            else:  # High
                vol_score += 20 + min((atr_pct - 0.03) / 0.03 * 10, 10)
        
        if 'bb_width_relative' in volatility_metrics:
            # Map BB width to a score (0-30)
            bb_width_rel = volatility_metrics['bb_width_relative']
            if bb_width_rel < 0.7:  # Compressed bands (low volatility)
                vol_score += bb_width_rel / 0.7 * 10
            elif bb_width_rel < 1.3:  # Normal bands
                vol_score += 10 + (bb_width_rel - 0.7) / 0.6 * 10
            else:  # Expanded bands (high volatility)
                vol_score += 20 + min((bb_width_rel - 1.3) / 0.7 * 10, 10)
        
        # Determine volatility category
        if vol_score < 30:
            volatility_category = 'low'
        elif vol_score < 60:
            volatility_category = 'medium'
        else:
            volatility_category = 'high'
        
        return {
            'volatility': volatility_category,
            'volatility_score': vol_score,
            'details': volatility_metrics
        }
    
    except Exception as e:
        logger.error(f"Error analyzing volatility: {str(e)}")
        return {'volatility': 'unknown', 'volatility_score': 0}

def predict_price_direction(df):
    """
    Predict the likely price direction based on all technical indicators.
    
    Args:
        df (pd.DataFrame): DataFrame with OHLCV and indicator data
        
    Returns:
        dict: Prediction with confidence score
    """
    if df is None or len(df) < 50:
        return {'direction': 'neutral', 'confidence': 0, 'conviction': 0}
    
    try:
        # Calculate all the technical analysis components
        patterns = identify_chart_patterns(df)
        support_resistance = detect_support_resistance(df)
        trend = calculate_trend_strength(df)
        volatility = analyze_volatility(df)
        
        # Aggregate bullish and bearish signals with their strengths
        bullish_score = 0
        bearish_score = 0
        
        # Add pattern signals
        for pattern, value in patterns.items():
            if value > 0:
                bullish_score += value
            else:
                bearish_score += abs(value)
        
        # Add trend signal
        trend_strength = abs(trend.get('trend_strength', 0))
        if trend.get('trend_direction') == 'bullish':
            bullish_score += trend_strength
        elif trend.get('trend_direction') == 'bearish':
            bearish_score += trend_strength
        
        # Consider support/resistance proximity
        current_price = df['close'].iloc[-1]
        
        # Check if price is near support (bullish)
        for support in support_resistance.get('support', []):
            # Calculate distance to support
            distance_pct = (current_price - support) / current_price
            if 0 <= distance_pct <= 0.03:  # Within 3% of support
                bullish_score += 50 * (1 - distance_pct / 0.03)  # Higher score for closer support
        
        # Check if price is near resistance (bearish)
        for resistance in support_resistance.get('resistance', []):
            # Calculate distance to resistance
            distance_pct = (resistance - current_price) / current_price
            if 0 <= distance_pct <= 0.03:  # Within 3% of resistance
                bearish_score += 50 * (1 - distance_pct / 0.03)  # Higher score for closer resistance
        
        # Add basic indicator signals
        
        # RSI
        if 'rsi_14' in df.columns:
            rsi = df['rsi_14'].iloc[-1]
            if rsi < 30:  # Oversold
                bullish_score += 60 * (1 - rsi / 30)
            elif rsi > 70:  # Overbought
                bearish_score += 60 * ((rsi - 70) / 30)
        
                    # MACD
        if all(col in df.columns for col in ['macd', 'macd_signal']):
            macd = df['macd'].iloc[-1]
            macd_signal = df['macd_signal'].iloc[-1]
            macd_hist = df['macd_hist'].iloc[-1] if 'macd_hist' in df.columns else macd - macd_signal
            
            if macd > macd_signal:
                bullish_score += 40 * min(1, (macd - macd_signal) / abs(macd))
            else:
                bearish_score += 40 * min(1, (macd_signal - macd) / abs(macd))
            
            # MACD histogram direction
            if len(df) > 1 and 'macd_hist' in df.columns:
                prev_hist = df['macd_hist'].iloc[-2]
                if macd_hist > prev_hist:  # Increasing histogram (momentum)
                    bullish_score += 20
                else:  # Decreasing histogram
                    bearish_score += 20
        
        # Bollinger Bands
        if all(col in df.columns for col in ['bb_upper', 'bb_middle', 'bb_lower']):
            close = df['close'].iloc[-1]
            upper = df['bb_upper'].iloc[-1]
            lower = df['bb_lower'].iloc[-1]
            middle = df['bb_middle'].iloc[-1]
            
            # Distance from bands as percentage
            upper_dist_pct = (upper - close) / close
            lower_dist_pct = (close - lower) / close
            
            # Close to lower band (potential bounce)
            if close < middle and lower_dist_pct < 0.02:
                bullish_score += 30 * (1 - lower_dist_pct / 0.02)
            # Close to upper band (potential resistance)
            elif close > middle and upper_dist_pct < 0.02:
                bearish_score += 30 * (1 - upper_dist_pct / 0.02)
            
            # Price relative to middle band
            if close > middle:
                bullish_score += 20 * min(1, (close - middle) / (upper - middle))
            else:
                bearish_score += 20 * min(1, (middle - close) / (middle - lower))
        
        # Consider volatility for confidence
        volatility_score = volatility.get('volatility_score', 50)
        confidence_modifier = 1.0
        
        if volatility_score < 30:  # Low volatility
            confidence_modifier = 0.8  # Lower confidence in low volatility
        elif volatility_score > 70:  # High volatility
            confidence_modifier = 0.7  # Even lower confidence in high volatility
        
        # Calculate net score and confidence
        total_score = bullish_score + bearish_score
        if total_score == 0:
            return {'direction': 'neutral', 'confidence': 0, 'conviction': 0}
        
        bullish_weight = bullish_score / total_score
        bearish_weight = bearish_score / total_score
        
        # Determine direction and conviction
        if bullish_weight > 0.6:  # >60% bullish signals
            direction = 'bullish'
            confidence = bullish_weight * confidence_modifier
            # Conviction from -100 to +100
            conviction = int((bullish_score - bearish_score) / max(1, total_score) * 100)
        elif bearish_weight > 0.6:  # >60% bearish signals
            direction = 'bearish'
            confidence = bearish_weight * confidence_modifier
            # Conviction from -100 to +100
            conviction = int((bearish_score - bullish_score) / max(1, total_score) * -100)
        else:
            direction = 'neutral'
            confidence = (1 - abs(bullish_weight - 0.5) * 2) * confidence_modifier
            # Conviction near 0 for neutral
            conviction = int((bullish_score - bearish_score) / max(1, total_score) * 50)
        
        logger.info(f"Price direction prediction: {direction} with {conviction} conviction")
        
        return {
            'direction': direction,
            'confidence': confidence,
            'conviction': conviction,
            'bullish_score': bullish_score,
            'bearish_score': bearish_score
        }
    
    except Exception as e:
        logger.error(f"Error predicting price direction: {str(e)}")
        return {'direction': 'neutral', 'confidence': 0, 'conviction': 0}

def get_analysis(symbol, timeframe='1h'):
    """
    Get comprehensive technical analysis for a symbol.
    
    Args:
        symbol (str): Cryptocurrency symbol
        timeframe (str): Time interval
        
    Returns:
        dict: Complete technical analysis
    """
    try:
        # Get OHLCV data
        df = get_ohlcv_data(symbol, timeframe)
        if df is None or len(df) < 50:
            logger.error(f"Not enough data for {symbol} to perform analysis")
            return {
                'status': 'error',
                'message': 'Not enough historical data',
                'recommendation': 'hold',
                'conviction': 0
            }
        
        # Calculate indicators
        df = calculate_advanced_indicators(df)
        
        # Perform analysis
        patterns = identify_chart_patterns(df)
        support_resistance = detect_support_resistance(df)
        trend = calculate_trend_strength(df)
        volatility = analyze_volatility(df)
        prediction = predict_price_direction(df)
        
        # Get current price and recent performance
        current_price = df['close'].iloc[-1]
        prev_day_price = df['close'].iloc[-24] if len(df) >= 24 else df['close'].iloc[0]
        
        day_change_pct = (current_price - prev_day_price) / prev_day_price * 100
        
        # Extract key RSI and MACD values
        rsi = df['rsi_14'].iloc[-1] if 'rsi_14' in df.columns else None
        macd = df['macd'].iloc[-1] if 'macd' in df.columns else None
        macd_signal = df['macd_signal'].iloc[-1] if 'macd_signal' in df.columns else None
        
        # Determine overall recommendation
        conviction = prediction.get('conviction', 0)
        
        if conviction > 40:
            recommendation = 'buy'
        elif conviction < -40:
            recommendation = 'sell'
        else:
            recommendation = 'hold'
        
        # Format support/resistance levels for readability
        formatted_support = [round(level, 2) for level in support_resistance.get('support', [])]
        formatted_resistance = [round(level, 2) for level in support_resistance.get('resistance', [])]
        
        # Summarize key patterns
        bullish_patterns = [k for k, v in patterns.items() if v > 0]
        bearish_patterns = [k for k, v in patterns.items() if v < 0]
        
        # Prepare final analysis
        analysis = {
            'status': 'success',
            'symbol': symbol,
            'timeframe': timeframe,
            'current_price': current_price,
            'day_change_pct': day_change_pct,
            'technical_indicators': {
                'rsi': rsi,
                'macd': macd,
                'macd_signal': macd_signal
            },
            'support_levels': formatted_support,
            'resistance_levels': formatted_resistance,
            'trend': trend.get('trend_direction', 'neutral'),
            'trend_strength': trend.get('trend_strength', 0),
            'volatility': volatility.get('volatility', 'medium'),
            'patterns': {
                'bullish': bullish_patterns,
                'bearish': bearish_patterns
            },
            'recommendation': recommendation,
            'conviction': conviction,
            'confidence': prediction.get('confidence', 0)
        }
        
        logger.info(f"Completed technical analysis for {symbol} with recommendation: {recommendation}")
        
        return analysis
    
    except Exception as e:
        logger.error(f"Error performing analysis for {symbol}: {str(e)}")
        return {
            'status': 'error',
            'message': str(e),
            'recommendation': 'hold',
            'conviction': 0
        }

def get_multi_timeframe_analysis(symbol):
    """
    Get technical analysis across multiple timeframes.
    
    Args:
        symbol (str): Cryptocurrency symbol
        
    Returns:
        dict: Analysis for multiple timeframes
    """
    timeframes = ['1h', '4h', '1d']
    analysis = {}
    
    try:
        for tf in timeframes:
            analysis[tf] = get_analysis(symbol, tf)
        
        # Calculate combined conviction score across timeframes
        # Weight longer timeframes more heavily
        weights = {'1h': 0.2, '4h': 0.3, '1d': 0.5}
        
        combined_conviction = 0
        for tf, weight in weights.items():
            if tf in analysis and 'conviction' in analysis[tf]:
                combined_conviction += analysis[tf]['conviction'] * weight
        
        # Determine overall recommendation
        if combined_conviction > 40:
            recommendation = 'buy'
        elif combined_conviction < -40:
            recommendation = 'sell'
        else:
            recommendation = 'hold'
        
        result = {
            'symbol': symbol,
            'timeframes': analysis,
            'combined_conviction': int(combined_conviction),
            'recommendation': recommendation
        }
        
        logger.info(f"Completed multi-timeframe analysis for {symbol} with combined conviction: {int(combined_conviction)}")
        
        return result
    
    except Exception as e:
        logger.error(f"Error performing multi-timeframe analysis for {symbol}: {str(e)}")
        return {
            'status': 'error',
            'message': str(e),
            'recommendation': 'hold',
            'conviction': 0
        }