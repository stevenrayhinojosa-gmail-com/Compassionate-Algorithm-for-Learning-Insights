import logging
import os
from datetime import datetime
import pandas as pd

class CALILogger:
    def __init__(self, log_dir="logs"):
        """Initialize CALI logging system"""
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        # Setup main logger
        self.setup_logger()
        
        # CSV log files
        self.usage_log_file = "usage_log.csv"
        self.metrics_log_file = "model_metrics.csv"
        
        # Initialize CSV files if they don't exist
        self.init_csv_logs()
    
    def setup_logger(self):
        """Setup structured logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(self.log_dir, 'cali.log')),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('CALI')
    
    def init_csv_logs(self):
        """Initialize CSV log files with headers"""
        # Usage log
        if not os.path.exists(self.usage_log_file):
            usage_df = pd.DataFrame(columns=[
                'timestamp', 'action', 'student_name', 'file_name', 
                'file_size', 'rows_processed', 'success', 'error_msg'
            ])
            usage_df.to_csv(self.usage_log_file, index=False)
        
        # Model metrics log
        if not os.path.exists(self.metrics_log_file):
            metrics_df = pd.DataFrame(columns=[
                'timestamp', 'model_version', 'mse', 'mae', 'r2_score',
                'data_points', 'training_time', 'features_used'
            ])
            metrics_df.to_csv(self.metrics_log_file, index=False)
    
    def log_upload(self, student_name, file_name, file_size, rows_processed, success=True, error_msg=None):
        """Log file upload events"""
        try:
            new_row = {
                'timestamp': datetime.now().isoformat(),
                'action': 'file_upload',
                'student_name': student_name,
                'file_name': file_name,
                'file_size': file_size,
                'rows_processed': rows_processed,
                'success': success,
                'error_msg': error_msg or ''
            }
            
            # Append to CSV
            df = pd.read_csv(self.usage_log_file)
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            df.to_csv(self.usage_log_file, index=False)
            
            # Log to main logger
            if success:
                self.logger.info(f"File uploaded successfully: {file_name} for {student_name}")
            else:
                self.logger.error(f"File upload failed: {file_name} - {error_msg}")
                
        except Exception as e:
            self.logger.error(f"Failed to log upload event: {str(e)}")
    
    def log_prediction(self, student_name, prediction_type, success=True, error_msg=None):
        """Log prediction generation events"""
        try:
            new_row = {
                'timestamp': datetime.now().isoformat(),
                'action': f'prediction_{prediction_type}',
                'student_name': student_name,
                'file_name': '',
                'file_size': 0,
                'rows_processed': 0,
                'success': success,
                'error_msg': error_msg or ''
            }
            
            # Append to CSV
            df = pd.read_csv(self.usage_log_file)
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            df.to_csv(self.usage_log_file, index=False)
            
            if success:
                self.logger.info(f"Prediction generated: {prediction_type} for {student_name}")
            else:
                self.logger.error(f"Prediction failed: {prediction_type} - {error_msg}")
                
        except Exception as e:
            self.logger.error(f"Failed to log prediction event: {str(e)}")
    
    def log_model_metrics(self, model_version, mse, mae, r2_score, data_points, training_time, features_used):
        """Log model performance metrics"""
        try:
            new_row = {
                'timestamp': datetime.now().isoformat(),
                'model_version': model_version,
                'mse': mse,
                'mae': mae,
                'r2_score': r2_score,
                'data_points': data_points,
                'training_time': training_time,
                'features_used': features_used
            }
            
            # Append to CSV
            df = pd.read_csv(self.metrics_log_file)
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            df.to_csv(self.metrics_log_file, index=False)
            
            self.logger.info(f"Model metrics logged: v{model_version} - R²: {r2_score:.3f}")
            
        except Exception as e:
            self.logger.error(f"Failed to log model metrics: {str(e)}")
    
    def log_error(self, component, error_msg, student_name=None):
        """Log system errors"""
        try:
            new_row = {
                'timestamp': datetime.now().isoformat(),
                'action': f'error_{component}',
                'student_name': student_name or 'system',
                'file_name': '',
                'file_size': 0,
                'rows_processed': 0,
                'success': False,
                'error_msg': error_msg
            }
            
            # Append to CSV
            df = pd.read_csv(self.usage_log_file)
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            df.to_csv(self.usage_log_file, index=False)
            
            self.logger.error(f"System error in {component}: {error_msg}")
            
        except Exception as e:
            self.logger.error(f"Failed to log error: {str(e)}")
    
    def get_usage_stats(self, days=30):
        """Get usage statistics for the last N days"""
        try:
            df = pd.read_csv(self.usage_log_file)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Filter last N days
            cutoff_date = datetime.now() - pd.Timedelta(days=days)
            recent_df = df[df['timestamp'] >= cutoff_date]
            
            stats = {
                'total_uploads': len(recent_df[recent_df['action'] == 'file_upload']),
                'successful_uploads': len(recent_df[(recent_df['action'] == 'file_upload') & (recent_df['success'] == True)]),
                'total_predictions': len(recent_df[recent_df['action'].str.contains('prediction')]),
                'unique_students': recent_df['student_name'].nunique(),
                'error_rate': (len(recent_df[recent_df['success'] == False]) / len(recent_df) * 100) if len(recent_df) > 0 else 0
            }
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Failed to get usage stats: {str(e)}")
            return {}
    
    def get_model_performance_trend(self):
        """Get model performance trends over time"""
        try:
            df = pd.read_csv(self.metrics_log_file)
            if len(df) == 0:
                return {}
            
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp')
            
            latest = df.iloc[-1]
            trend = {
                'latest_r2': latest['r2_score'],
                'latest_mse': latest['mse'],
                'latest_mae': latest['mae'],
                'improvement_trend': 'improving' if len(df) > 1 and latest['r2_score'] > df.iloc[-2]['r2_score'] else 'stable'
            }
            
            return trend
            
        except Exception as e:
            self.logger.error(f"Failed to get model trends: {str(e)}")
            return {}

# Global logger instance
cali_logger = CALILogger()