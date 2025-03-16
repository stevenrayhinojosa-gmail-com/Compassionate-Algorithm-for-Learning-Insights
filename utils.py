import pandas as pd
import numpy as np

def validate_data(df):
    """Validate input data format and content"""
    required_columns = ['Date']
    
    # Check required columns
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Check date format
    try:
        pd.to_datetime(df['Date'])
    except:
        raise ValueError("Invalid date format in Date column")
    
    # Check behavior values
    behavior_columns = df.iloc[:, 2:35]
    valid_values = {'r', 'y', 'g', '0', ''}
    invalid_values = set(behavior_columns.values.flatten()) - valid_values
    
    if invalid_values:
        raise ValueError(f"Invalid behavior values found: {invalid_values}")
        
    return True

def calculate_rolling_stats(df, window=7):
    """Calculate rolling statistics for behavior scores"""
    return {
        'mean': df['behavior_score'].rolling(window=window).mean(),
        'std': df['behavior_score'].rolling(window=window).std()
    }
