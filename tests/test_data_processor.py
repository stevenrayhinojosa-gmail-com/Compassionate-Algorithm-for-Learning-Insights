import unittest
import pandas as pd
import numpy as np
from data_processor import DataProcessor

class TestDataProcessor(unittest.TestCase):
    def setUp(self):
        # Create sample test data
        self.test_data = pd.DataFrame({
            'Date': ['8/16/2022', '8/17/2022', '8/18/2022'],
            'Time1': ['g', 'r', 'y'],
            'Time2': ['y', 'g', 'g'],
            'Time3': ['r', 'y', 'g']
        })
        self.processor = DataProcessor(self.test_data)

    def test_clean_date(self):
        """Test date cleaning functionality"""
        clean_date = self.processor.clean_date('8/16/2022')
        self.assertIsNotNone(clean_date)
        self.assertEqual(clean_date.month, 8)
        self.assertEqual(clean_date.day, 16)
        self.assertEqual(clean_date.year, 2022)

    def test_behavior_mapping(self):
        """Test behavior score calculation"""
        processed_data = self.processor.process_data()
        self.assertIn('behavior_score', processed_data.columns)
        self.assertTrue(all(processed_data['behavior_score'].between(0, 2)))

    def test_get_behavior_distribution(self):
        """Test behavior distribution calculation"""
        distribution = self.processor.get_behavior_distribution()
        self.assertEqual(len(distribution), 3)  # Should have r, y, g counts
        self.assertTrue(all(distribution >= 0))  # All counts should be non-negative

if __name__ == '__main__':
    unittest.main()
