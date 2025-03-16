import pandas as pd
import numpy as np
from datetime import datetime

class DataProcessor:
    def __init__(self, data):
        self.data = data

    def process_data(self):
        """Process and clean the behavioral data"""
        try:
            df = self.data.copy()

            # Clean date column - handle various formats
            df['Date'] = pd.to_datetime(df['Date'], format='mixed', errors='coerce')
            df['date'] = df['Date']  # Keep consistent naming

            # Get behavior columns (from 7:30 AM to 3:45 PM)
            time_columns = df.columns[2:35]  # Columns with r/y/g values

            # Convert behavior markers to numerical values
            behavior_map = {
                'r': 0,  # Red
                'y': 1,  # Yellow
                'g': 2,  # Green
                'G': 2,  # Alternate Green format
                '0': np.nan,
                '': np.nan
            }

            # Calculate daily behavior score
            behavior_scores = df[time_columns].replace(behavior_map)
            df['behavior_score'] = behavior_scores.mean(axis=1)

            # Extract temporal features
            df['day_of_week'] = df['date'].dt.dayofweek
            df['month'] = df['date'].dt.month
            df['week'] = df['date'].dt.isocalendar().week

            # Handle missing values
            df['behavior_score'] = df['behavior_score'].fillna(method='ffill').fillna(method='bfill')

            # Add behavior counts
            df['red_count'] = behavior_scores.apply(lambda x: (x == 0).sum(), axis=1)
            df['yellow_count'] = behavior_scores.apply(lambda x: (x == 1).sum(), axis=1)
            df['green_count'] = behavior_scores.apply(lambda x: (x == 2).sum(), axis=1)

            # Select relevant features
            processed_df = df[[
                'date', 'day_of_week', 'month', 'week', 
                'behavior_score', 'red_count', 'yellow_count', 'green_count'
            ]].copy()

            return processed_df

        except Exception as e:
            raise Exception(f"Error processing data: {str(e)}")

    def get_behavior_distribution(self):
        """Calculate behavior distribution"""
        time_columns = self.data.columns[2:35]
        behavior_counts = self.data[time_columns].values.flatten()
        valid_counts = pd.Series(behavior_counts).value_counts().drop(['0', '', 'G'], errors='ignore')
        return valid_counts.reindex(['g', 'y', 'r']).fillna(0)