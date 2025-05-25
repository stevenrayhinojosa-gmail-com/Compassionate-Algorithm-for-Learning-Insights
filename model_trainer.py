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
        """Initialize with processed data"""
        self.data = data
        self.model = None
        self.preprocessor = None

        # Define feature groups - core behavioral features
        self.numeric_features = [
            'day_of_week', 'month', 'week',
            'rolling_avg_7d', 'rolling_std_7d',
            'behavior_trend', 'weekly_improvement'
        ]

        # Add environmental factors if available
        env_features = ['environmental_impact', 'seasonal_score', 
                       'active_staff_changes', 'noise_level', 
                       'temperature', 'high_sugar_meals',
                       'high_protein_meals', 'active_routine_changes']
        
        for feature in env_features:
            if feature in data.columns:
                self.numeric_features.append(feature)

        self.categorical_features = ['season']

    def prepare_features(self):
        """Prepare features for model training"""
        # Create feature preprocessing pipeline
        numeric_transformer = Pipeline([
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline([
            ('onehot', OneHotEncoder(drop='first', sparse_output=False))
        ])

        # Create column transformer
        self.preprocessor = ColumnTransformer([
            ('num', numeric_transformer, self.numeric_features),
            ('cat', categorical_transformer, self.categorical_features)
        ])

        # Prepare feature matrix
        X = self.data[self.numeric_features + self.categorical_features]
        y = self.data['behavior_score']

        # Clean data to remove NaN and infinite values
        X = X.fillna(0)
        y = y.fillna(0)
        
        # Replace infinite values
        X = X.replace([np.inf, -np.inf], 0)
        y = y.replace([np.inf, -np.inf], 0)

        # Fit and transform features
        X_transformed = self.preprocessor.fit_transform(X)

        return X_transformed, y

    def train_and_predict_next_day(self):
        """Train model and generate predictions for next school day"""
        try:
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

            # Calculate metrics
            y_pred = self.model.predict(X_test)
            metrics = {
                'mae': mean_absolute_error(y_test, y_pred),
                'mse': mean_squared_error(y_test, y_pred),
                'r2': r2_score(y_test, y_pred)
            }

            # Generate next day predictions
            last_date = self.data['date'].max()
            future_features, time_slots = self.generate_next_day_features(last_date)
            next_day_predictions = self.model.predict(future_features)

            # Create prediction dataframe
            predictions_df = pd.DataFrame({
                'time': time_slots,
                'predicted_score': next_day_predictions
            })

            # Add behavior categories
            predictions_df['predicted_category'] = pd.cut(
                predictions_df['predicted_score'],
                bins=[-float('inf'), 0.6, 1.4, float('inf')],
                labels=['Red', 'Yellow', 'Green']
            )

            # Add risk levels
            predictions_df['risk_level'] = pd.cut(
                predictions_df['predicted_score'],
                bins=[-float('inf'), 0.4, 0.8, 1.2, 1.6, float('inf')],
                labels=['Very High', 'High', 'Moderate', 'Low', 'Very Low']
            )

            return metrics, predictions_df

        except Exception as e:
            print(f"Error in training and prediction: {str(e)}")
            raise

    def generate_next_day_features(self, last_date):
        """Generate features for next school day predictions"""
        try:
            # Get next school day
            next_day = last_date + timedelta(days=1)
            while next_day.weekday() > 4:  # Skip weekends
                next_day += timedelta(days=1)

            # Create time slots for school day (7:30 AM to 3:45 PM)
            time_slots = pd.date_range(
                start=datetime.combine(next_day, datetime.strptime("7:30", "%H:%M").time()),
                end=datetime.combine(next_day, datetime.strptime("15:45", "%H:%M").time()),
                freq="15min"
            )

            # Create base features
            features = pd.DataFrame(index=range(len(time_slots)))

            # Add temporal features
            features['day_of_week'] = next_day.weekday()
            features['month'] = next_day.month
            features['week'] = next_day.isocalendar()[1]

            # Get season
            month = next_day.month
            if month in [12, 1, 2]:
                season = 'Winter'
            elif month in [3, 4, 5]:
                season = 'Spring'
            elif month in [6, 7, 8]:
                season = 'Summer'
            else:
                season = 'Fall'
            features['season'] = season

            # Add rolling statistics from last known data
            features['rolling_avg_7d'] = self.data['rolling_avg_7d'].iloc[-1]
            features['rolling_std_7d'] = self.data['rolling_std_7d'].iloc[-1]
            features['behavior_trend'] = self.data['behavior_trend'].iloc[-1]
            features['weekly_improvement'] = self.data['weekly_improvement'].iloc[-1]
            
            # Add all environmental factors that were used in training
            env_features = ['environmental_impact', 'seasonal_score', 
                           'active_staff_changes', 'noise_level', 
                           'temperature', 'high_sugar_meals',
                           'high_protein_meals', 'active_routine_changes']
            
            for feature in env_features:
                if feature in self.data.columns:
                    # Use the most recent value for environmental factors
                    features[feature] = self.data[feature].iloc[-1]
            
            # Add time-of-day effects (morning, midday, afternoon patterns)
            hour_of_day = [t.hour + t.minute/60 for t in time_slots]
            
            # Morning dip (8:30-10:00), midday slump (12:30-1:30), end-of-day fatigue (after 2:30)
            # These are common patterns in student behavior
            for i, hour in enumerate(hour_of_day):
                # Adjust behavior trend based on time of day
                if 8.5 <= hour < 10:  # Morning adjustment
                    features.loc[i, 'behavior_trend'] = features.loc[i, 'behavior_trend'] - 0.1
                elif 12.5 <= hour < 13.5:  # Lunch/midday adjustment
                    features.loc[i, 'behavior_trend'] = features.loc[i, 'behavior_trend'] - 0.15
                elif hour >= 14.5:  # End of day adjustment
                    features.loc[i, 'behavior_trend'] = features.loc[i, 'behavior_trend'] - 0.2

            # Transform features
            features_transformed = self.preprocessor.transform(features)

            return features_transformed, time_slots

        except Exception as e:
            print(f"Error generating future features: {str(e)}")
            raise