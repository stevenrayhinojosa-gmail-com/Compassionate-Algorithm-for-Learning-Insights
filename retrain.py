#!/usr/bin/env python3
"""
CALI Model Retraining Script
Automatically updates the model with new data and improved performance
"""

import os
import pickle
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
from sqlalchemy import create_engine, text
from data_processor import DataProcessor
from model_trainer import ModelTrainer
from logger import cali_logger
import shutil

class ModelRetrainer:
    def __init__(self):
        self.model_versions_dir = "model_versions"
        self.current_model_path = "current_model.pkl"
        self.backup_threshold = 0.05  # Retrain if R² improves by 5%
        
        # Create directories
        os.makedirs(self.model_versions_dir, exist_ok=True)
        os.makedirs("logs", exist_ok=True)
    
    def should_retrain(self):
        """Determine if model should be retrained based on data freshness and performance"""
        try:
            # Check if we have enough new data
            engine = create_engine(os.environ.get('DATABASE_URL'))
            
            # Get latest model training date
            last_retrain = self._get_last_retrain_date()
            
            # Count new behavior records since last retrain
            query = text("""
                SELECT COUNT(*) as new_records 
                FROM behavior_records 
                WHERE created_at > :last_retrain
            """)
            
            with engine.connect() as conn:
                result = conn.execute(query, {"last_retrain": last_retrain})
                new_records = result.fetchone()[0]
            
            # Retrain if we have 50+ new records or haven't retrained in 30 days
            days_since_retrain = (datetime.now() - last_retrain).days
            
            should_retrain = new_records >= 50 or days_since_retrain >= 30
            
            cali_logger.logger.info(f"Retrain check: {new_records} new records, {days_since_retrain} days since last retrain")
            return should_retrain, new_records, days_since_retrain
            
        except Exception as e:
            cali_logger.log_error("retrain_check", str(e))
            return False, 0, 0
    
    def _get_last_retrain_date(self):
        """Get the date of the last model retraining"""
        try:
            if os.path.exists(self.metrics_log_file):
                df = pd.read_csv("model_metrics.csv")
                if len(df) > 0:
                    return pd.to_datetime(df['timestamp'].iloc[-1])
            
            # Default to 30 days ago if no previous training
            return datetime.now() - timedelta(days=30)
            
        except:
            return datetime.now() - timedelta(days=30)
    
    def backup_current_model(self):
        """Backup the current model before retraining"""
        try:
            if os.path.exists(self.current_model_path):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_path = os.path.join(self.model_versions_dir, f"model_backup_{timestamp}.pkl")
                shutil.copy2(self.current_model_path, backup_path)
                cali_logger.logger.info(f"Model backed up to {backup_path}")
                return backup_path
            return None
            
        except Exception as e:
            cali_logger.log_error("model_backup", str(e))
            return None
    
    def retrain_model(self, min_data_points=100):
        """Retrain the model with all available data"""
        try:
            cali_logger.logger.info("Starting model retraining process...")
            start_time = datetime.now()
            
            # Get all behavior data from database
            engine = create_engine(os.environ.get('DATABASE_URL'))
            
            query = text("""
                SELECT br.*, s.name as student_name
                FROM behavior_records br
                JOIN students s ON br.student_id = s.id
                ORDER BY br.date DESC
            """)
            
            with engine.connect() as conn:
                df = pd.read_sql(query, conn)
            
            if len(df) < min_data_points:
                cali_logger.logger.warning(f"Insufficient data for retraining: {len(df)} records (minimum: {min_data_points})")
                return False, "insufficient_data"
            
            # Process data for training
            combined_data = []
            for student_name in df['student_name'].unique():
                student_data = df[df['student_name'] == student_name].copy()
                
                # Create DataProcessor instance with this student's data
                processor = DataProcessor(None)  # No file path needed
                processor.data = student_data
                
                # Process the data
                processed_data = processor.process_data()
                processed_data['student_name'] = student_name
                combined_data.append(processed_data)
            
            # Combine all student data
            if not combined_data:
                return False, "no_valid_data"
            
            full_dataset = pd.concat(combined_data, ignore_index=True)
            
            # Train new model
            trainer = ModelTrainer(full_dataset)
            metrics, predictions = trainer.train_and_predict_next_day()
            
            training_time = (datetime.now() - start_time).total_seconds()
            
            # Evaluate model performance
            model_version = datetime.now().strftime("%Y%m%d_%H%M%S")
            features_used = len(trainer.numeric_features) + len(trainer.categorical_features)
            
            # Log metrics
            cali_logger.log_model_metrics(
                model_version=model_version,
                mse=metrics.get('mse', 0),
                mae=metrics.get('mae', 0),
                r2_score=metrics.get('r2', 0),
                data_points=len(full_dataset),
                training_time=training_time,
                features_used=features_used
            )
            
            # Save new model
            model_data = {
                'model': trainer.model,
                'preprocessor': trainer.preprocessor,
                'version': model_version,
                'trained_on': datetime.now().isoformat(),
                'data_points': len(full_dataset),
                'features': trainer.numeric_features + trainer.categorical_features,
                'metrics': metrics
            }
            
            with open(self.current_model_path, 'wb') as f:
                pickle.dump(model_data, f)
            
            # Save versioned copy
            version_path = os.path.join(self.model_versions_dir, f"model_{model_version}.pkl")
            with open(version_path, 'wb') as f:
                pickle.dump(model_data, f)
            
            cali_logger.logger.info(f"Model retrained successfully: R² = {metrics.get('r2', 0):.3f}")
            return True, model_version
            
        except Exception as e:
            cali_logger.log_error("model_retrain", str(e))
            return False, str(e)
    
    def validate_new_model(self, test_size=0.2):
        """Validate the newly trained model against test data"""
        try:
            # Load the current model
            with open(self.current_model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            # Get validation metrics
            metrics = model_data.get('metrics', {})
            r2_score = metrics.get('r2', 0)
            
            # Simple validation: R² should be above 0.3 for acceptable performance
            is_valid = r2_score > 0.3
            
            if is_valid:
                cali_logger.logger.info(f"Model validation passed: R² = {r2_score:.3f}")
            else:
                cali_logger.logger.warning(f"Model validation failed: R² = {r2_score:.3f}")
            
            return is_valid, metrics
            
        except Exception as e:
            cali_logger.log_error("model_validation", str(e))
            return False, {}
    
    def rollback_model(self, backup_path):
        """Rollback to a previous model version"""
        try:
            if os.path.exists(backup_path):
                shutil.copy2(backup_path, self.current_model_path)
                cali_logger.logger.info(f"Model rolled back from {backup_path}")
                return True
            return False
            
        except Exception as e:
            cali_logger.log_error("model_rollback", str(e))
            return False
    
    def run_retraining_pipeline(self):
        """Complete retraining pipeline with validation and rollback"""
        try:
            # Check if retraining is needed
            should_retrain, new_records, days_since = self.should_retrain()
            
            if not should_retrain:
                cali_logger.logger.info("Retraining not needed at this time")
                return False, "not_needed"
            
            # Backup current model
            backup_path = self.backup_current_model()
            
            # Retrain model
            success, result = self.retrain_model()
            
            if not success:
                cali_logger.logger.error(f"Retraining failed: {result}")
                return False, result
            
            # Validate new model
            is_valid, metrics = self.validate_new_model()
            
            if not is_valid and backup_path:
                # Rollback to previous version
                self.rollback_model(backup_path)
                cali_logger.logger.warning("New model failed validation, rolled back to previous version")
                return False, "validation_failed"
            
            cali_logger.logger.info(f"Model retraining completed successfully: version {result}")
            return True, result
            
        except Exception as e:
            cali_logger.log_error("retrain_pipeline", str(e))
            return False, str(e)

def main():
    """Main retraining script entry point"""
    print("CALI Model Retraining Script")
    print("=" * 40)
    
    retrainer = ModelRetrainer()
    
    # Run the complete retraining pipeline
    success, result = retrainer.run_retraining_pipeline()
    
    if success:
        print(f"✅ Model retrained successfully: {result}")
        print("🔄 New model is now active")
    else:
        print(f"❌ Retraining failed: {result}")
        print("🔒 Previous model remains active")
    
    # Display current model info
    try:
        if os.path.exists("current_model.pkl"):
            with open("current_model.pkl", 'rb') as f:
                model_data = pickle.load(f)
            
            print("\n📊 Current Model Info:")
            print(f"   Version: {model_data.get('version', 'unknown')}")
            print(f"   Trained: {model_data.get('trained_on', 'unknown')}")
            print(f"   Data Points: {model_data.get('data_points', 0)}")
            print(f"   R² Score: {model_data.get('metrics', {}).get('r2', 0):.3f}")
    except:
        print("\n⚠️  No current model found")

if __name__ == "__main__":
    main()