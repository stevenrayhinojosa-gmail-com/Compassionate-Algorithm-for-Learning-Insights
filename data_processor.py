import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from models import (
    EnvironmentalFactor, SeasonalPattern, StaffChange,
    LearningEnvironment, NutritionLog, RoutineChange
)

class DataProcessor:
    def __init__(self, file_path, db: Session = None):
        """Initialize with file path and database session"""
        self.db = db
        # Skip the first two metadata rows and use the third row as header
        # Simple approach: read the CSV and manually fix the Date column
        self.data = pd.read_csv(file_path, skiprows=2, on_bad_lines='skip', dtype=str)
        
        # Debug: Check what we actually loaded
        print("CSV columns loaded:", list(self.data.columns)[:5])
        print("First few rows of first column:", self.data.iloc[:5, 0].tolist())
        
        # If the first column is still just weekday names, try reading differently
        if all(day in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday', 'Date'] 
               for day in self.data.iloc[:10, 0].fillna('').astype(str)):
            print("Detected split date issue, attempting to read raw file...")
            
            # Read raw lines and manually parse
            with open(file_path, 'r') as f:
                lines = f.readlines()
            
            # Find lines with actual dates (containing both weekday and date)
            data_rows = []
            for line in lines[3:]:  # Skip first 3 rows
                if line.strip() and ',' in line:
                    parts = line.strip().split(',')
                    # Look for pattern like "Tuesday" followed by "8/16/2022"
                    if len(parts) >= 2 and '/' in parts[1] and any(c.isdigit() for c in parts[1]):
                        # Reconstruct the proper date format
                        full_date = f"{parts[0]},{parts[1]}"
                        row_data = [full_date] + parts[2:]
                        data_rows.append(row_data)
            
            if data_rows:
                # Find maximum row length to handle inconsistent column counts
                max_cols = max(len(row) for row in data_rows)
                
                # Pad shorter rows with empty strings
                for row in data_rows:
                    while len(row) < max_cols:
                        row.append('')
                
                # Create DataFrame from properly parsed data
                headers = ['Date'] + [f'Col_{i}' for i in range(max_cols-1)]
                self.data = pd.DataFrame(data_rows, columns=headers)
                print(f"Successfully reconstructed {len(data_rows)} rows with proper dates")
        # Get only the time slot columns (from 7:30 AM to 3:45 PM)
        self.time_slots = [col for col in self.data.columns if ':' in col and ('AM' in col or 'PM' in col)][:33]
        print(f"Found {len(self.time_slots)} time slots")  # Debug log

    def clean_date(self, date_str):
        """Clean and standardize date format"""
        if pd.isna(date_str) or date_str == '' or str(date_str).strip() == '':
            return None
        try:
            date_str = str(date_str).strip()
<<<<<<< HEAD

            # Special date format parsing for "Day, M/D/YYYY" or "Day, M//D/YYYY" formats
            if ',' in date_str:
                # Extract just the date part after the comma
                date_part = date_str.split(',')[1].strip()
            else:
                date_part = date_str

            # Handle special case where there are double slashes
            date_part = date_part.replace('//', '/')

            # Fix missing slashes (like "8/182022" should be "8/18/2022")
            if '/' in date_part and date_part.count('/') == 1:
                parts = date_part.split('/')
                if len(parts) == 2 and len(parts[1]) > 4:
                    # Likely format like "8/182022" - split the second part
                    month = parts[0]
                    day_year = parts[1]
                    if len(day_year) == 6:  # DDYYYY
                        day = day_year[:2]
                        year = day_year[2:]
                        date_part = f"{month}/{day}/{year}"
                    elif len(day_year) == 5:  # DYYYY
                        day = day_year[:1]
                        year = day_year[1:]
                        date_part = f"{month}/{day}/{year}"

=======
            
            # Handle the specific format: "Tuesday,8/16/2022" or "Tuesday, 8/16/2022"
            if ',' in date_str:
                # Split by comma and take the date part
                parts = date_str.split(',')
                if len(parts) >= 2:
                    date_part = parts[1].strip()
                else:
                    date_part = parts[0].strip()
            else:
                date_part = date_str
                
            # Handle special case where there are double slashes like "8//19/2022"
            date_part = date_part.replace('//', '/')
            
            # Skip non-date entries
            if date_part.lower() in ['date', ':', '', 'nan']:
                return None
                
>>>>>>> 56c348cdb7964770c039deace67dab7eebfd2236
            # Try to parse the date
            if date_part.count('/') == 2:
                return pd.to_datetime(date_part, format='%m/%d/%Y', errors='coerce')
            else:
                return pd.to_datetime(date_part, errors='coerce')

        except Exception as e:
            print(f"Error parsing date '{date_str}': {str(e)}")
            return None

    def process_data(self, student_id: int = None):
        """Process and clean the behavioral data"""
        try:
            print("Starting data processing...")  # Debug log

<<<<<<< HEAD
            # Clean date column first - the actual dates are in the second column (Unnamed: 1)
            # The first column just contains day names
            date_column = 'Unnamed: 1' if 'Unnamed: 1' in self.data.columns else 'Date'
            self.data['date'] = self.data[date_column].apply(self.clean_date)
=======
            # Debug: Check the actual data structure
            print("First 10 rows of Date column:")
            print(self.data['Date'].head(10).tolist())
            
            # Look for rows containing actual dates (format: "Tuesday,8/16/2022")
            # The CSV has weekday names followed by comma and date (no space after comma)
            date_pattern = r'[A-Za-z]+,\d+/+\d+/\d+'
            date_mask = self.data['Date'].astype(str).str.contains(date_pattern, na=False)
            valid_data = self.data[date_mask].copy()
            print(f"Found {len(valid_data)} rows with weekday+date patterns")

            if len(valid_data) == 0:
                # Fallback: Look for any rows with MM/DD/YYYY pattern
                date_mask = self.data['Date'].astype(str).str.contains(r'\d+/+\d+/\d+', na=False)
                valid_data = self.data[date_mask].copy()
                print(f"Fallback: Found {len(valid_data)} rows with date patterns")

            if len(valid_data) == 0:
                raise ValueError("No valid date rows found in the data")

            # Clean date column
            valid_data['date'] = valid_data['Date'].apply(self.clean_date)
>>>>>>> 56c348cdb7964770c039deace67dab7eebfd2236

            # Print the first few dates and dtype for verification
            print("Date column head:", valid_data['date'].head())
            print("Date column dtype:", valid_data['date'].dtype)

<<<<<<< HEAD
            # Remove rows with invalid dates
            df = self.data.dropna(subset=['date']).copy()
=======
            # Remove rows with invalid dates after parsing
            df = valid_data.dropna(subset=['date'])
>>>>>>> 56c348cdb7964770c039deace67dab7eebfd2236
            print(f"Rows after date cleaning: {len(df)}")  # Debug log

            # Convert behavior markers to numerical values
            behavior_map = {
                'r': 0,  # Red - needs significant support
                'y': 1,  # Yellow - needs some support
                'g': 2,  # Green - meeting expectations
                'G': 2,  # Alternate Green format
                'a': np.nan,  # Absent
                '0': np.nan,
                '': np.nan
            }

            # Calculate behavior scores for time slots
            behavior_scores = df[self.time_slots].replace(behavior_map)
            print("Behavior scores calculated")  # Debug log

            # Calculate daily metrics
            df['behavior_score'] = behavior_scores.mean(axis=1)
            df['red_count'] = behavior_scores.apply(lambda x: (x == 0).sum(), axis=1)
            df['yellow_count'] = behavior_scores.apply(lambda x: (x == 1).sum(), axis=1)
            df['green_count'] = behavior_scores.apply(lambda x: (x == 2).sum(), axis=1)

            # Extract temporal features
            df['day_of_week'] = df['date'].dt.dayofweek
            df['month'] = df['date'].dt.month
            df['week'] = df['date'].dt.isocalendar().week
            df['season'] = df['date'].apply(lambda x: self._get_season(x))

            # Calculate rolling statistics
            df['rolling_avg_7d'] = df['behavior_score'].rolling(window=7, min_periods=1).mean()
            df['rolling_std_7d'] = df['behavior_score'].rolling(window=7, min_periods=1).std()

            # Calculate behavioral trends
            df['behavior_trend'] = df['behavior_score'].diff().fillna(0)
            df['weekly_improvement'] = df['rolling_avg_7d'] - df['rolling_avg_7d'].shift(7)

            # Remove rows with NaN behavior scores (days with no valid behavior data)
            df = df.dropna(subset=['behavior_score'])
            print(f"Rows after removing NaN behavior scores: {len(df)}")  # Debug log

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
                'weekly_improvement'
            ]].copy()

            # Add environmental columns if they exist
            env_columns = ['environmental_impact', 'seasonal_score', 'active_staff_changes',
                         'noise_level', 'temperature', 'high_sugar_meals',
                         'high_protein_meals', 'active_routine_changes', 'routine_adaptation']

            for col in env_columns:
                if col in df.columns:
                    processed_df[col] = df[col]

            print("Data processing completed successfully")  # Debug log
            return processed_df

        except Exception as e:
            print(f"Error processing data: {str(e)}")
            raise

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

        return factors

    def get_behavior_distribution(self):
        """Calculate behavior distribution"""
        behavior_counts = self.data[self.time_slots].values.flatten()
        valid_counts = pd.Series(behavior_counts).value_counts().drop(['0', '', 'a', 'G'], errors='ignore')
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