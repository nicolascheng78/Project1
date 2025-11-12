"""
Basic tests for IDX trading system
"""

import unittest
import pandas as pd
import numpy as np
import yaml
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from idx_trading_system.data_loader import DataLoader
from idx_trading_system.features import FeatureEngine, create_labels
from idx_trading_system.models import create_model, LinearModel
from idx_trading_system.portfolio import PortfolioConstructor


class TestDataLoader(unittest.TestCase):
    """Test data loading functionality"""
    
    def test_initialization(self):
        """Test DataLoader initialization"""
        loader = DataLoader()
        self.assertIsNotNone(loader)
        self.assertTrue(loader.cache_dir.exists())


class TestFeatureEngine(unittest.TestCase):
    """Test feature engineering"""
    
    def setUp(self):
        """Set up test config"""
        self.config = {
            'features': {
                'momentum_windows': [5, 20],
                'volatility_windows': [20],
                'rsi_period': 14,
                'macd': [12, 26, 9],
                'volume_zscore_window': 20,
                'normalization_window': 60
            }
        }
        self.engine = FeatureEngine(self.config)
    
    def test_compute_returns(self):
        """Test return computation"""
        df = pd.DataFrame({
            'adj_close': [100, 102, 101, 103, 105]
        })
        
        result = self.engine.compute_returns(df)
        self.assertIn('returns', result.columns)
        self.assertEqual(len(result), 5)
        # First return should be NaN
        self.assertTrue(pd.isna(result['returns'].iloc[0]))
    
    def test_compute_momentum(self):
        """Test momentum feature computation"""
        df = pd.DataFrame({
            'returns': np.random.randn(100) * 0.01
        })
        
        result = self.engine.compute_momentum(df, [5, 20])
        self.assertIn('momentum_5d', result.columns)
        self.assertIn('momentum_20d', result.columns)
    
    def test_compute_rsi(self):
        """Test RSI computation"""
        df = pd.DataFrame({
            'adj_close': 100 + np.cumsum(np.random.randn(100))
        })
        
        result = self.engine.compute_rsi(df)
        self.assertIn('rsi', result.columns)
        # RSI should be between 0 and 100
        valid_rsi = result['rsi'].dropna()
        self.assertTrue((valid_rsi >= 0).all())
        self.assertTrue((valid_rsi <= 100).all())


class TestLabels(unittest.TestCase):
    """Test label creation"""
    
    def test_create_labels(self):
        """Test forward return label creation"""
        df = pd.DataFrame({
            'ticker': ['A'] * 50,
            'returns': np.random.randn(50) * 0.01
        })
        
        result = create_labels(df, forward_periods=[5, 10])
        self.assertIn('target_return_5d', result.columns)
        self.assertIn('target_return_10d', result.columns)
        self.assertIn('target_direction_5d', result.columns)


class TestModels(unittest.TestCase):
    """Test model implementations"""
    
    def test_linear_model(self):
        """Test Linear model"""
        model = LinearModel(model_type='regression')
        
        # Generate synthetic data
        X_train = np.random.randn(100, 5)
        y_train = np.random.randn(100)
        X_test = np.random.randn(20, 5)
        
        # Train
        model.fit(X_train, y_train)
        
        # Predict
        predictions = model.predict(X_test)
        self.assertEqual(len(predictions), 20)
    
    def test_create_model(self):
        """Test model factory"""
        config = {
            'models': {
                'xgboost': {
                    'max_depth': 6,
                    'learning_rate': 0.05,
                    'n_estimators': 10
                }
            }
        }
        
        # Test linear model
        linear_model = create_model('linear', model_type='regression', config=config)
        self.assertIsNotNone(linear_model)
        
        # Test XGBoost model (if available)
        try:
            xgb_model = create_model('xgboost', model_type='regression', config=config)
            self.assertIsNotNone(xgb_model)
        except ImportError:
            pass  # XGBoost not installed


class TestPortfolio(unittest.TestCase):
    """Test portfolio construction"""
    
    def setUp(self):
        """Set up test config"""
        self.config = {
            'portfolio': {
                'type': 'long_only',
                'rebalance_frequency': 'daily',
                'top_k': 3,
                'volatility_target': 0.10,
                'position_cap': 0.10,
                'turnover_cap': 2.0
            }
        }
        self.constructor = PortfolioConstructor(self.config)
    
    def test_apply_position_cap(self):
        """Test position cap application"""
        weights = pd.Series({
            'A': 0.5,
            'B': 0.3,
            'C': 0.2
        })
        
        capped = self.constructor._apply_position_cap(weights)
        
        # All weights should be <= position_cap
        self.assertTrue((capped <= self.config['portfolio']['position_cap']).all())
        # Weights should sum to 1
        self.assertAlmostEqual(capped.sum(), 1.0, places=6)
    
    def test_compute_metrics(self):
        """Test metric computation"""
        # Generate synthetic returns
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        returns = pd.Series(np.random.randn(100) * 0.01, index=dates)
        
        metrics = self.constructor._compute_metrics_from_returns(returns)
        
        self.assertIn('cagr', metrics)
        self.assertIn('sharpe_ratio', metrics)
        self.assertIn('max_drawdown', metrics)
        self.assertIsInstance(metrics['cagr'], float)


if __name__ == '__main__':
    unittest.main()
