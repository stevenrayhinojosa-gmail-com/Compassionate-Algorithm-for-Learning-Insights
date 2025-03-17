import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import pandas as pd
from datetime import datetime, timedelta

class ModelTrainer:
    def __init__(self, data):
        self.data = data
        self.model = None
        self.preprocessor = None

        # Define feature groups
        self.numeric_features = [
            'day_of_week', 'month', 'week',
            'rolling_avg_7d', 'rolling_std_7d',
            'behavior_trend', 'weekly_improvement'
        ]

        self.categorical_features = [
            'season', 'noise_level'
        ]

        # Add environmental features if they exist
        self.environmental_features = [
            'environmental_impact', 'seasonal_score', 'active_staff_changes',
            'temperature', 'high_sugar_meals', 'high_protein_meals',
            'active_routine_changes', 'routine_adaptation'
        ]

        # Add available environmental features to numeric features
        self.numeric_features.extend([f for f in self.environmental_features 
                                   if f in data.columns])

        # Binary features
        self.binary_features = ['is_monday', 'is_friday']

    def prepare_features(self):
        """Prepare features for model training"""
        # Create feature preprocessing pipeline
        numeric_transformer = Pipeline([
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline([
            ('onehot', OneHotEncoder(drop='first', sparse=False))
        ])

        # Get available categorical features
        available_cat_features = [f for f in self.categorical_features 
                                if f in self.data.columns]

        # Create column transformer
        transformers = [
            ('num', numeric_transformer, self.numeric_features),
            ('binary', 'passthrough', self.binary_features)
        ]

        # Add categorical transformer if categorical features exist
        if available_cat_features:
            transformers.append(
                ('cat', categorical_transformer, available_cat_features)
            )

        self.preprocessor = ColumnTransformer(transformers)

        # Prepare feature matrix
        X = self.data[self.numeric_features + self.binary_features + available_cat_features]
        y = self.data['behavior_score']

        # Fit and transform features
        X_transformed = self.preprocessor.fit_transform(X)

        return X_transformed, y

    def generate_next_day_features(self, last_date):
        """Generate feature matrix for next school day predictions"""
        next_day = last_date + timedelta(days=1)
        while next_day.weekday() > 4:  # Skip weekends
            next_day += timedelta(days=1)

        # Create time slots for school day (7:30 AM to 3:45 PM)
        time_slots = pd.date_range(
            start=datetime.combine(next_day, datetime.strptime("7:30", "%H:%M").time()),
            end=datetime.combine(next_day, datetime.strptime("15:45", "%H:%M").time()),
            freq="15min"
        )

        # Create base features for each time slot
        future_features = pd.DataFrame({
            'day_of_week': next_day.weekday(),
            'month': next_day.month,
            'week': next_day.isocalendar()[1],
            'is_monday': int(next_day.weekday() == 0),
            'is_friday': int(next_day.weekday() == 4),
            'season': self._get_season(next_day),
            'rolling_avg_7d': self.data['rolling_avg_7d'].iloc[-1],
            'rolling_std_7d': self.data['rolling_std_7d'].iloc[-1],
            'behavior_trend': self.data['behavior_trend'].iloc[-1],
            'weekly_improvement': self.data['weekly_improvement'].iloc[-1]
        }, index=range(len(time_slots)))

        # Add environmental features if they exist in training data
        for feature in self.environmental_features:
            if feature in self.data.columns:
                future_features[feature] = self.data[feature].iloc[-1]

        # Transform features
        future_features_transformed = self.preprocessor.transform(future_features)

        return future_features_transformed, time_slots

    def _get_season(self, date):
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

    def train_and_predict_next_day(self):
        """Train XGBoost model and generate next day predictions"""
        X, y = self.prepare_features()

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Train model with environmental features
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

        # Generate next day predictions
        future_features, time_slots = self.generate_next_day_features(
            self.data['date'].iloc[-1]
        )
        next_day_predictions = self.model.predict(future_features)

        # Prepare prediction results
        next_day_df = pd.DataFrame({
            'time': time_slots,
            'predicted_score': next_day_predictions,
            'confidence': [
                1 - (metrics['mae'] / (self.data['behavior_score'].max() - self.data['behavior_score'].min()))
            ] * len(next_day_predictions)
        })

        # Add behavior category predictions
        next_day_df['predicted_category'] = pd.cut(
            next_day_df['predicted_score'],
            bins=[-float('inf'), 0.6, 1.4, float('inf')],
            labels=['Red', 'Yellow', 'Green']
        )

        # Add risk levels
        next_day_df['risk_level'] = pd.cut(
            next_day_df['predicted_score'],
            bins=[-float('inf'), 0.4, 0.8, 1.2, 1.6, float('inf')],
            labels=['Very High', 'High', 'Moderate', 'Low', 'Very Low']
        )

        return metrics, next_day_df