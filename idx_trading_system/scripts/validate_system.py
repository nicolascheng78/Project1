#!/usr/bin/env python
"""
Validation script to test the trading system components
This script can run even with missing optional dependencies
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_imports():
    """Test that all modules can be imported"""
    print("Testing module imports...")
    
    try:
        from idx_trading_system import data_loader
        print("✓ data_loader module imported")
    except ImportError as e:
        print(f"✗ data_loader import failed: {e}")
        return False
    
    try:
        from idx_trading_system import features
        print("✓ features module imported")
    except ImportError as e:
        print(f"✗ features import failed: {e}")
        return False
    
    try:
        from idx_trading_system import models
        print("✓ models module imported")
    except ImportError as e:
        print(f"✗ models import failed: {e}")
        return False
    
    try:
        from idx_trading_system import portfolio
        print("✓ portfolio module imported")
    except ImportError as e:
        print(f"✗ portfolio import failed: {e}")
        return False
    
    try:
        from idx_trading_system import backtest
        print("✓ backtest module imported")
    except ImportError as e:
        print(f"✗ backtest import failed: {e}")
        return False
    
    return True


def test_config():
    """Test configuration loading"""
    print("\nTesting configuration...")
    
    try:
        import yaml
        config_path = Path(__file__).parent.parent / 'config.yaml'
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        print(f"✓ Config loaded successfully")
        print(f"  - Universe: {len(config['universe']['tickers'])} tickers")
        print(f"  - Backtest splits: {len(config['backtest']['splits'])}")
        print(f"  - Target CAGR: {config['targets']['cagr']:.1%}")
        return True
        
    except Exception as e:
        print(f"✗ Config loading failed: {e}")
        return False


def test_data_loader():
    """Test DataLoader initialization"""
    print("\nTesting DataLoader...")
    
    try:
        from idx_trading_system.data_loader import DataLoader
        
        loader = DataLoader()
        print("✓ DataLoader initialized")
        print(f"  - Cache dir: {loader.cache_dir}")
        print(f"  - Cache dir exists: {loader.cache_dir.exists()}")
        return True
        
    except Exception as e:
        print(f"✗ DataLoader test failed: {e}")
        return False


def test_feature_engine():
    """Test FeatureEngine initialization"""
    print("\nTesting FeatureEngine...")
    
    try:
        import yaml
        from idx_trading_system.features import FeatureEngine
        
        config_path = Path(__file__).parent.parent / 'config.yaml'
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        engine = FeatureEngine(config)
        print("✓ FeatureEngine initialized")
        print(f"  - Momentum windows: {engine.momentum_windows}")
        print(f"  - RSI period: {engine.rsi_period}")
        return True
        
    except Exception as e:
        print(f"✗ FeatureEngine test failed: {e}")
        return False


def test_models():
    """Test model creation"""
    print("\nTesting Models...")
    
    try:
        import yaml
        from idx_trading_system.models import create_model, LinearModel
        
        config_path = Path(__file__).parent.parent / 'config.yaml'
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Test Linear model (no dependencies)
        linear_model = create_model('linear', model_type='regression', config=config)
        print("✓ Linear model created")
        
        # Test other models if dependencies available
        try:
            xgb_model = create_model('xgboost', model_type='regression', config=config)
            print("✓ XGBoost model created")
        except ImportError:
            print("⚠ XGBoost not available (optional)")
        
        try:
            lgb_model = create_model('lightgbm', model_type='regression', config=config)
            print("✓ LightGBM model created")
        except ImportError:
            print("⚠ LightGBM not available (optional)")
        
        try:
            lstm_model = create_model('lstm', model_type='regression', config=config, input_size=10)
            print("✓ LSTM model created")
        except ImportError:
            print("⚠ PyTorch not available (optional)")
        
        return True
        
    except Exception as e:
        print(f"✗ Models test failed: {e}")
        return False


def test_portfolio():
    """Test PortfolioConstructor initialization"""
    print("\nTesting PortfolioConstructor...")
    
    try:
        import yaml
        from idx_trading_system.portfolio import PortfolioConstructor
        
        config_path = Path(__file__).parent.parent / 'config.yaml'
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        constructor = PortfolioConstructor(config)
        print("✓ PortfolioConstructor initialized")
        print(f"  - Portfolio type: {constructor.portfolio_type}")
        print(f"  - Top K: {constructor.top_k}")
        print(f"  - Position cap: {constructor.position_cap:.1%}")
        return True
        
    except Exception as e:
        print(f"✗ PortfolioConstructor test failed: {e}")
        return False


def test_backtester():
    """Test WalkForwardBacktest initialization"""
    print("\nTesting WalkForwardBacktest...")
    
    try:
        import yaml
        from idx_trading_system.backtest import WalkForwardBacktest
        
        config_path = Path(__file__).parent.parent / 'config.yaml'
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        backtester = WalkForwardBacktest(config)
        print("✓ WalkForwardBacktest initialized")
        print(f"  - Number of splits: {len(backtester.splits)}")
        return True
        
    except Exception as e:
        print(f"✗ WalkForwardBacktest test failed: {e}")
        return False


def check_dependencies():
    """Check which dependencies are installed"""
    print("\nChecking dependencies...")
    
    dependencies = [
        'numpy',
        'pandas',
        'yfinance',
        'sklearn',
        'xgboost',
        'lightgbm',
        'torch',
        'yaml',
        'matplotlib',
        'seaborn',
        'joblib'
    ]
    
    installed = []
    missing = []
    
    for dep in dependencies:
        try:
            __import__(dep)
            installed.append(dep)
            print(f"✓ {dep}")
        except ImportError:
            missing.append(dep)
            print(f"✗ {dep}")
    
    print(f"\nInstalled: {len(installed)}/{len(dependencies)}")
    
    if missing:
        print(f"\nTo install missing dependencies:")
        print(f"pip install {' '.join(missing)}")
    
    return len(missing) == 0


def main():
    """Run all validation tests"""
    print("="*60)
    print("IDX Trading System Validation")
    print("="*60)
    
    results = {
        'Dependencies': check_dependencies(),
        'Imports': test_imports(),
        'Config': test_config(),
        'DataLoader': test_data_loader(),
        'FeatureEngine': test_feature_engine(),
        'Models': test_models(),
        'Portfolio': test_portfolio(),
        'Backtester': test_backtester()
    }
    
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{test_name:20s} {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! The system is ready to use.")
        return 0
    else:
        print("\n⚠️  Some tests failed. Please install missing dependencies:")
        print("pip install -r requirements.txt")
        return 1


if __name__ == '__main__':
    sys.exit(main())
