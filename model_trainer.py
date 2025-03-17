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

    def generate_future_features(self, last_date, periods=24):
        """Generate feature matrix for future predictions"""
        future_dates = pd.date_range(
            start=last_date + timedelta(hours=1),
            periods=periods,
            freq='H'
        )

        # Create base features
        future_features = pd.DataFrame({
            'day_of_week': future_dates.dayofweek,
            'month': future_dates.month,
            'week': future_dates.isocalendar().week,
            'is_monday': (future_dates.dayofweek == 0).astype(int),
            'is_friday': (future_dates.dayofweek == 4).astype(int),
            'season': [self._get_season(d) for d in future_dates],
            'rolling_avg_7d': [self.data['rolling_avg_7d'].iloc[-1]] * periods,
            'rolling_std_7d': [self.data['rolling_std_7d'].iloc[-1]] * periods,
            'behavior_trend': [self.data['behavior_trend'].iloc[-1]] * periods,
            'weekly_improvement': [self.data['weekly_improvement'].iloc[-1]] * periods
        })

        # Add environmental features if they exist in training data
        for feature in self.environmental_features:
            if feature in self.data.columns:
                future_features[feature] = self.data[feature].iloc[-1]

        # Transform features
        future_features_transformed = self.preprocessor.transform(future_features)

        return future_features_transformed, future_dates

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

    def train_and_predict(self):
        """Train XGBoost model and generate predictions"""
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