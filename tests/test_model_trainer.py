import unittest
import pandas as pd
import numpy as np
from model_trainer import ModelTrainer

class TestModelTrainer(unittest.TestCase):
    def setUp(self):
        # Create sample training data
        np.random.seed(42)
        dates = pd.date_range(start='2022-01-01', periods=100, freq='D')
        self.test_data = pd.DataFrame({
            'date': dates,
            'day_of_week': dates.dayofweek,
            'month': dates.month,
            'week': dates.isocalendar().week,
            'behavior_score': np.random.uniform(0, 2, 100)
        })
        self.trainer = ModelTrainer(self.test_data)

    def test_prepare_features(self):
        """Test feature preparation"""
        X, y = self.trainer.prepare_features()
        self.assertEqual(X.shape[1], 3)  # Should have 3 features
        self.assertEqual(len(y), len(self.test_data))

    def test_train_and_predict(self):
        """Test model training and prediction"""
        metrics, predictions = self.trainer.train_and_predict()
        
        # Check metrics
        self.assertIn('mae', metrics)
        self.assertIn('mse', metrics)
        self.assertIn('r2', metrics)
        
        # Check predictions
        self.assertEqual(len(predictions), len(self.test_data) // 5)  # 20% test size
        self.assertIn('actual', predictions.columns)
        self.assertIn('predicted', predictions.columns)

if __name__ == '__main__':
    unittest.main()
