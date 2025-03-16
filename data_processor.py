import pandas as pd
import numpy as np
from datetime import datetime

class DataProcessor:
    def __init__(self, data):
        self.data = data
        
    def process_data(self):
        """Process and clean the behavioral data"""
        df = self.data.copy()
        
        # Convert date column
        df['date'] = pd.to_datetime(df['Date'])
        
        # Create behavior score columns
        behavior_columns = df.iloc[:, 2:35]  # Time slots 7:30 AM - 4:00 PM
        
        # Convert behavior markers to numerical values
        behavior_map = {'r': 0, 'y': 1, 'g': 2, '0': np.nan}
        
        # Calculate daily average behavior score
        df['behavior_score'] = behavior_columns.replace(behavior_map).mean(axis=1)
        
        # Extract temporal features
        df['day_of_week'] = df['date'].dt.dayofweek
        df['month'] = df['date'].dt.month
        df['week'] = df['date'].dt.isocalendar().week
        
        # Handle missing values
        df['behavior_score'] = df['behavior_score'].fillna(df['behavior_score'].mean())
        
        # Select relevant features
        processed_df = df[['date', 'day_of_week', 'month', 'week', 'behavior_score']].copy()
        
        return processed_df
    
    def get_behavior_distribution(self):
        """Calculate behavior distribution"""
        behavior_columns = self.data.iloc[:, 2:35]
        behavior_counts = behavior_columns.values.flatten()
        return pd.Series(behavior_counts).value_counts().drop('0')
