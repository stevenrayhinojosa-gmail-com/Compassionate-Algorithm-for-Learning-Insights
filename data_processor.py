import pandas as pd
import numpy as np
from datetime import datetime

class DataProcessor:
    def __init__(self, data):
        self.data = data
        self.time_slots = self.data.columns[2:35]  # Time slots from 7:30 AM to 3:45 PM

    def clean_date(self, date_str):
        """Clean and standardize date format"""
        if pd.isna(date_str) or date_str == '':
            return None
        # Remove extra slashes and clean the date string
        clean_str = '/'.join(filter(None, date_str.split('/')))
        try:
            return pd.to_datetime(clean_str)
        except:
            return None

    def calculate_weekly_stats(self, df):
        """Calculate weekly behavior statistics"""
        weekly_stats = df.groupby('week').agg({
            'behavior_score': ['mean', 'std'],
            'red_count': 'sum',
            'yellow_count': 'sum',
            'green_count': 'sum'
        }).reset_index()
        weekly_stats.columns = ['week', 'weekly_avg_score', 'weekly_score_std', 
                              'weekly_red_total', 'weekly_yellow_total', 'weekly_green_total']
        return weekly_stats

    def process_data(self):
        """Process and clean the behavioral data"""
        try:
            df = self.data.copy()

            # Clean date column
            df['date'] = df['Date'].apply(self.clean_date)
            df = df.dropna(subset=['date'])  # Remove rows with invalid dates

            # Convert behavior markers to numerical values
            behavior_map = {
                'r': 0,  # Red - needs significant support
                'y': 1,  # Yellow - needs some support
                'g': 2,  # Green - meeting expectations
                'G': 2,  # Alternate Green format
                '0': np.nan,
                '': np.nan
            }

            # Calculate behavior scores
            behavior_scores = df[self.time_slots].replace(behavior_map)

            # Calculate daily metrics
            df['behavior_score'] = behavior_scores.mean(axis=1)
            df['red_count'] = behavior_scores.apply(lambda x: (x == 0).sum(), axis=1)
            df['yellow_count'] = behavior_scores.apply(lambda x: (x == 1).sum(), axis=1)
            df['green_count'] = behavior_scores.apply(lambda x: (x == 2).sum(), axis=1)

            # Extract temporal features
            df['day_of_week'] = df['date'].dt.dayofweek
            df['month'] = df['date'].dt.month
            df['week'] = df['date'].dt.isocalendar().week
            df['is_monday'] = (df['day_of_week'] == 0).astype(int)
            df['is_friday'] = (df['day_of_week'] == 4).astype(int)

            # Calculate rolling statistics
            df['rolling_avg_7d'] = df['behavior_score'].rolling(window=7, min_periods=1).mean()
            df['rolling_std_7d'] = df['behavior_score'].rolling(window=7, min_periods=1).std()

            # Calculate behavioral trends
            df['behavior_trend'] = df['behavior_score'].diff().fillna(0)
            df['weekly_improvement'] = df['rolling_avg_7d'] - df['rolling_avg_7d'].shift(7)

            # Get weekly statistics
            weekly_stats = self.calculate_weekly_stats(df)
            df = df.merge(weekly_stats, on='week', how='left')

            # Select features for the final dataset
            processed_df = df[[
                'date', 'day_of_week', 'month', 'week',
                'behavior_score', 'red_count', 'yellow_count', 'green_count',
                'rolling_avg_7d', 'rolling_std_7d', 'behavior_trend',
                'weekly_improvement', 'weekly_avg_score', 'weekly_score_std',
                'is_monday', 'is_friday'
            ]].copy()

            return processed_df

        except Exception as e:
            raise Exception(f"Error processing data: {str(e)}")

    def get_behavior_distribution(self):
        """Calculate behavior distribution"""
        behavior_counts = self.data[self.time_slots].values.flatten()
        valid_counts = pd.Series(behavior_counts).value_counts().drop(['0', '', 'G'], errors='ignore')
        return valid_counts.reindex(['g', 'y', 'r']).fillna(0)

    def get_summary_stats(self):
        """Get summary statistics for the processed data"""
        processed_data = self.process_data()
        return {
            'total_days': len(processed_data),
            'avg_score': processed_data['behavior_score'].mean(),
            'std_score': processed_data['behavior_score'].std(),
            'best_day': processed_data.loc[processed_data['behavior_score'].idxmax(), 'date'],
            'challenging_day': processed_data.loc[processed_data['behavior_score'].idxmin(), 'date'],
            'weekly_trend': processed_data['weekly_improvement'].mean()
        }