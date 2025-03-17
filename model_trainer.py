import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import pandas as pd
from datetime import datetime, timedelta

class ModelTrainer:
    def __init__(self, data):
        self.data = data
        self.model = None
        self.scaler = StandardScaler()

    def prepare_features(self):
        """Prepare features for model training"""
        # Create time-based features
        X = pd.DataFrame({
            'day_of_week': self.data['day_of_week'],
            'month': self.data['month'],
            'week': self.data['week'],
            'is_monday': self.data['is_monday'],
            'is_friday': self.data['is_friday'],
            'prev_day_score': self.data['behavior_score'].shift(1),
            'prev_week_avg': self.data['rolling_avg_7d'],
            'prev_week_std': self.data['rolling_std_7d'],
            'behavior_trend': self.data['behavior_trend']
        }).fillna(method='ffill')

        y = self.data['behavior_score']

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        return X_scaled, y

    def generate_future_features(self, last_date, periods=24):
        """Generate feature matrix for future predictions"""
        future_dates = pd.date_range(
            start=last_date + timedelta(hours=1),
            periods=periods,
            freq='H'
        )

        # Create future feature matrix
        future_features = pd.DataFrame({
            'day_of_week': future_dates.dayofweek,
            'month': future_dates.month,
            'week': future_dates.isocalendar().week,
            'is_monday': (future_dates.dayofweek == 0).astype(int),
            'is_friday': (future_dates.dayofweek == 4).astype(int),
            'prev_day_score': [self.data['behavior_score'].iloc[-1]] * periods,
            'prev_week_avg': [self.data['rolling_avg_7d'].iloc[-1]] * periods,
            'prev_week_std': [self.data['rolling_std_7d'].iloc[-1]] * periods,
            'behavior_trend': [self.data['behavior_trend'].iloc[-1]] * periods
        })

        # Scale features
        future_features_scaled = self.scaler.transform(future_features)

        return future_features_scaled, future_dates

    def train_and_predict(self):
        """Train XGBoost model and generate predictions"""
        X, y = self.prepare_features()

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Train model
        self.model = xgb.XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=42
        )
        self.model.fit(X_train, y_train)

        # Generate predictions for test set
        y_pred = self.model.predict(X_test)

        # Calculate metrics
        metrics = {
            'mae': mean_absolute_error(y_test, y_pred),
            'mse': mean_squared_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred)
        }

        # Generate future predictions
        future_features, future_dates = self.generate_future_features(
            self.data['date'].iloc[-1]
        )
        future_predictions = self.model.predict(future_features)

        # Prepare prediction results
        historical_predictions = pd.DataFrame({
            'date': self.data['date'].iloc[-len(y_test):],
            'actual': y_test,
            'predicted': y_pred
        })

        future_predictions_df = pd.DataFrame({
            'date': future_dates,
            'predicted': future_predictions
        })

        predictions = pd.concat([
            historical_predictions,
            future_predictions_df
        ]).reset_index(drop=True)

        return metrics, predictions