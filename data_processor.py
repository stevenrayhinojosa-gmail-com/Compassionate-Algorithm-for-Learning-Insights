import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from models import (
    EnvironmentalFactor, SeasonalPattern, StaffChange,
    LearningEnvironment, NutritionLog, RoutineChange
)

class DataProcessor:
    def __init__(self, data, db: Session = None):
        self.data = data
        self.db = db
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

    def get_environmental_factors(self, student_id: int, date: datetime) -> dict:
        """Get environmental factors for a specific date"""
        if not self.db:
            return {}

        factors = {}

        # Get active environmental factors
        env_factors = self.db.query(EnvironmentalFactor).filter(
            EnvironmentalFactor.student_id == student_id,
            EnvironmentalFactor.date == date.date()
        ).all()
        factors['environmental_impact'] = sum(f.impact_level for f in env_factors) / len(env_factors) if env_factors else 0

        # Get seasonal pattern
        season = self._get_season(date)
        seasonal_pattern = self.db.query(SeasonalPattern).filter(
            SeasonalPattern.student_id == student_id,
            SeasonalPattern.season == season,
            SeasonalPattern.year == date.year
        ).first()
        factors['seasonal_score'] = seasonal_pattern.avg_behavior_score if seasonal_pattern else None

        # Get active staff changes
        staff_changes = self.db.query(StaffChange).filter(
            StaffChange.student_id == student_id,
            StaffChange.change_date <= date.date(),
            StaffChange.change_date + timedelta(days=StaffChange.adjustment_period) >= date.date()
        ).all()
        factors['active_staff_changes'] = len(staff_changes)

        # Get learning environment
        learning_env = self.db.query(LearningEnvironment).filter(
            LearningEnvironment.student_id == student_id,
            LearningEnvironment.start_date <= date.date(),
            (LearningEnvironment.end_date >= date.date()) | (LearningEnvironment.end_date.is_(None))
        ).first()
        if learning_env:
            factors['noise_level'] = learning_env.noise_level
            factors['temperature'] = learning_env.temperature

        # Get nutrition info
        nutrition = self.db.query(NutritionLog).filter(
            NutritionLog.student_id == student_id,
            NutritionLog.date == date.date()
        ).all()
        if nutrition:
            factors['high_sugar_meals'] = sum(1 for n in nutrition if n.sugar_intake_level == 'high')
            factors['high_protein_meals'] = sum(1 for n in nutrition if n.protein_intake_level == 'high')

        # Get routine changes
        routine_changes = self.db.query(RoutineChange).filter(
            RoutineChange.student_id == student_id,
            RoutineChange.change_date <= date.date(),
            RoutineChange.change_date + timedelta(days=RoutineChange.duration) >= date.date()
        ).all()
        factors['active_routine_changes'] = len(routine_changes)
        factors['routine_adaptation'] = min([r.adaptation_level for r in routine_changes]) if routine_changes else 5

        return factors

    def _get_season(self, date: datetime) -> str:
        """Determine season based on date"""
        month = date.month
        if month in [12, 1, 2]:
            return 'Winter'
        elif month in [3, 4, 5]:
            return 'Spring'
        elif month in [6, 7, 8]:
            return 'Summer'
        else:
            return 'Fall'

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

    def process_data(self, student_id: int = None):
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
            df['season'] = df['date'].apply(lambda x: self._get_season(x))

            # Calculate rolling statistics
            df['rolling_avg_7d'] = df['behavior_score'].rolling(window=7, min_periods=1).mean()
            df['rolling_std_7d'] = df['behavior_score'].rolling(window=7, min_periods=1).std()

            # Calculate behavioral trends
            df['behavior_trend'] = df['behavior_score'].diff().fillna(0)
            df['weekly_improvement'] = df['rolling_avg_7d'] - df['rolling_avg_7d'].shift(7)

            # Get weekly statistics
            weekly_stats = self.calculate_weekly_stats(df)
            df = df.merge(weekly_stats, on='week', how='left')

            # Add environmental factors if database session is available
            if self.db and student_id:
                env_features = df['date'].apply(
                    lambda x: self.get_environmental_factors(student_id, x)
                ).tolist()

                env_df = pd.DataFrame(env_features)
                if not env_df.empty:
                    df = pd.concat([df, env_df], axis=1)

            # Select features for the final dataset
            processed_df = df[[
                'date', 'day_of_week', 'month', 'week', 'season',
                'behavior_score', 'red_count', 'yellow_count', 'green_count',
                'rolling_avg_7d', 'rolling_std_7d', 'behavior_trend',
                'weekly_improvement', 'weekly_avg_score', 'weekly_score_std',
                'is_monday', 'is_friday'
            ]].copy()

            # Add environmental columns if they exist
            env_columns = ['environmental_impact', 'seasonal_score', 'active_staff_changes',
                         'noise_level', 'temperature', 'high_sugar_meals',
                         'high_protein_meals', 'active_routine_changes', 'routine_adaptation']

            for col in env_columns:
                if col in df.columns:
                    processed_df[col] = df[col]

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