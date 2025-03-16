import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import pandas as pd

class ModelTrainer:
    def __init__(self, data):
        self.data = data
        self.model = None
        self.scaler = StandardScaler()
        
    def prepare_features(self):
        """Prepare features for model training"""
        X = self.data[['day_of_week', 'month', 'week']].copy()
        y = self.data['behavior_score'].copy()
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        return X_scaled, y
        
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
        
        # Generate predictions
        y_pred = self.model.predict(X_test)
        
        # Calculate metrics
        metrics = {
            'mae': mean_absolute_error(y_test, y_pred),
            'mse': mean_squared_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred)
        }
        
        # Prepare prediction results
        predictions = pd.DataFrame({
            'actual': y_test,
            'predicted': y_pred
        })
        
        return metrics, predictions
