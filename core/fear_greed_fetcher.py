"""
Fear & Greed Index Fetcher

Retrieves the current Fear & Greed Index value, which measures market sentiment.
Uses the Alternative.me API for real data when available,
falls back to simulated data for the MVP.
"""

import requests
import logging
import random
from datetime import datetime, timedelta

# Set up logging
logger = logging.getLogger('crypto_bot.fear_greed')

# Alternative.me Fear & Greed Index API endpoint
FEAR_GREED_API_URL = "https://api.alternative.me/fng/"

# Map of index values to classifications
FEAR_GREED_MAP = {
    (0, 10): "Extreme Fear",
    (11, 25): "Fear",
    (26, 40): "Mild Fear",
    (41, 59): "Neutral",
    (60, 74): "Mild Greed",
    (75, 89): "Greed", 
    (90, 100): "Extreme Greed"
}

def get_current_index():
    """
    Get the current Fear & Greed Index value.
    
    Returns:
        str: Fear & Greed Index classification
    """
    try:
        # Make API request
        response = requests.get(
            FEAR_GREED_API_URL,
            params={"limit": 1}
        )
        response.raise_for_status()
        data = response.json()
        
        # Check if API returned valid data
        if data.get("metadata", {}).get("error") is None:
            index_data = data.get("data", [])
            
            if index_data and len(index_data) > 0:
                # Get the latest index value
                index_value = int(index_data[0].get("value", 0))
                classification = get_classification(index_value)
                
                logger.info(f"Fear & Greed Index: {index_value} ({classification})")
                return classification
        
        # If we didn't return by now, fall back to dummy data
        logger.error("Failed to get Fear & Greed Index from API, using simulated data")
        return get_dummy_index()
        
    except Exception as e:
        logger.error(f"Error fetching Fear & Greed Index: {str(e)}")
        return get_dummy_index()

def get_classification(index_value):
    """
    Map an index value to its classification.
    
    Args:
        index_value (int): Fear & Greed Index value (0-100)
    
    Returns:
        str: Classification description
    """
    for range_tuple, classification in FEAR_GREED_MAP.items():
        if range_tuple[0] <= index_value <= range_tuple[1]:
            return classification
    
    return "Neutral"  # Default if somehow out of range

def get_dummy_index():
    """
    Generate a dummy Fear & Greed Index classification.
    
    Returns:
        str: Random classification
    """
    # Use weighted random choice to simulate realistic distribution
    # Market tends to be in fear or greed more often than extremes
    classifications = [
        "Extreme Fear", 
        "Fear", 
        "Mild Fear", 
        "Neutral", 
        "Mild Greed", 
        "Greed", 
        "Extreme Greed"
    ]
    weights = [0.1, 0.2, 0.15, 0.1, 0.15, 0.2, 0.1]
    
    return random.choices(classifications, weights=weights, k=1)[0]

def get_historical_index(days=30):
    """
    Get historical Fear & Greed Index data.
    
    Args:
        days (int): Number of days of history to retrieve
    
    Returns:
        list: Historical index data
    """
    try:
        # Make API request
        response = requests.get(
            FEAR_GREED_API_URL,
            params={"limit": days}
        )
        response.raise_for_status()
        data = response.json()
        
        # Check if API returned valid data
        if data.get("metadata", {}).get("error") is None:
            index_data = data.get("data", [])
            
            if index_data and len(index_data) > 0:
                # Format the data
                historical = []
                for entry in index_data:
                    value = int(entry.get("value", 0))
                    timestamp = int(entry.get("timestamp", 0))
                    date = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d")
                    
                    historical.append({
                        "date": date,
                        "value": value,
                        "classification": get_classification(value)
                    })
                
                logger.info(f"Retrieved {len(historical)} days of Fear & Greed Index history")
                return historical
        
        # If we didn't return by now, fall back to dummy data
        logger.error("Failed to get historical Fear & Greed Index, using simulated data")
        return get_dummy_historical_index(days)
        
    except Exception as e:
        logger.error(f"Error fetching historical Fear & Greed Index: {str(e)}")
        return get_dummy_historical_index(days)

def get_dummy_historical_index(days=30):
    """
    Generate dummy historical Fear & Greed Index data.
    
    Args:
        days (int): Number of days of history to generate
    
    Returns:
        list: Dummy historical data
    """
    historical = []
    
    # Start with a random value between 25 and 75
    value = random.randint(25, 75)
    
    # Generate data for each day
    for i in range(days):
        # Move the index slightly each day (with a bit of mean reversion)
        if value < 30:
            # More likely to go up if very low
            change = random.randint(-3, 7)
        elif value > 70:
            # More likely to go down if very high
            change = random.randint(-7, 3)
        else:
            # Otherwise random movement
            change = random.randint(-5, 5)
        
        value += change
        
        # Ensure value stays within bounds
        value = max(0, min(100, value))
        
        # Calculate the date
        date = (datetime.now() - timedelta(days=days-i)).strftime("%Y-%m-%d")
        
        historical.append({
            "date": date,
            "value": value,
            "classification": get_classification(value)
        })
    
    return historical