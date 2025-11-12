#!/usr/bin/env python
"""
Simple example demonstrating the trading system
This shows the basic workflow without running a full backtest
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def simple_demo():
    """Run a simple demonstration"""
    print("="*60)
    print("IDX Trading System - Simple Demo")
    print("="*60)
    
    # 1. Load configuration
    print("\n1. Loading configuration...")
    try:
        import yaml
        config_path = Path(__file__).parent.parent / 'config.yaml'
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        print(f"✓ Configuration loaded")
        print(f"  Universe: {config['universe']['tickers']}")
        print(f"  Index: {config['universe']['index']}")
        print(f"  Target CAGR: {config['targets']['cagr']:.1%}")
    except ImportError:
        print("✗ PyYAML not installed. Run: pip install pyyaml")
        return
    except Exception as e:
        print(f"✗ Error loading config: {e}")
        return
    
    # 2. Initialize components
    print("\n2. Initializing components...")
    try:
        from idx_trading_system.data_loader import DataLoader
        from idx_trading_system.features import FeatureEngine
        from idx_trading_system.portfolio import PortfolioConstructor
        from idx_trading_system.backtest import WalkForwardBacktest
        
        data_loader = DataLoader(config['paths']['cache_dir'])
        feature_engine = FeatureEngine(config)
        portfolio_constructor = PortfolioConstructor(config)
        backtester = WalkForwardBacktest(config)
        
        print("✓ All components initialized")
    except ImportError as e:
        print(f"✗ Missing dependency: {e}")
        print("  Run: pip install -r requirements.txt")
        return
    except Exception as e:
        print(f"✗ Error initializing components: {e}")
        return
    
    # 3. Show what features will be computed
    print("\n3. Feature engineering configuration...")
    print(f"  Momentum windows: {feature_engine.momentum_windows}")
    print(f"  Volatility windows: {feature_engine.volatility_windows}")
    print(f"  RSI period: {feature_engine.rsi_period}")
    print(f"  MACD params: {feature_engine.macd_params}")
    
    # 4. Show portfolio configuration
    print("\n4. Portfolio construction configuration...")
    print(f"  Type: {portfolio_constructor.portfolio_type}")
    print(f"  Top K: {portfolio_constructor.top_k}")
    print(f"  Position cap: {portfolio_constructor.position_cap:.1%}")
    print(f"  Volatility target: {portfolio_constructor.vol_target:.1%}")
    print(f"  Turnover cap: {portfolio_constructor.turnover_cap:.1f}x")
    
    # 5. Show backtest configuration
    print("\n5. Backtesting configuration...")
    print(f"  Number of splits: {len(backtester.splits)}")
    for i, split in enumerate(backtester.splits):
        print(f"\n  Split {i+1}:")
        print(f"    Train: {split['train'][0]} to {split['train'][1]}")
        print(f"    Val:   {split['val'][0]} to {split['val'][1]}")
        print(f"    Test:  {split['test'][0]} to {split['test'][1]}")
    
    # 6. Show transaction cost configuration
    print("\n6. Transaction cost configuration...")
    print(f"  Commission: {config['costs']['commission']:.2%}")
    print(f"  Slippage: {config['costs']['slippage']:.2%}")
    print(f"  Round-trip: {config['costs']['round_trip']:.2%}")
    
    # 7. Show model options
    print("\n7. Available models...")
    from idx_trading_system import models
    
    available_models = ['linear']
    
    if models.HAS_XGBOOST:
        available_models.append('xgboost')
        print("  ✓ XGBoost")
    else:
        print("  ✗ XGBoost (not installed)")
    
    if models.HAS_LIGHTGBM:
        available_models.append('lightgbm')
        print("  ✓ LightGBM")
    else:
        print("  ✗ LightGBM (not installed)")
    
    if models.HAS_PYTORCH:
        available_models.extend(['lstm', 'transformer'])
        print("  ✓ PyTorch (LSTM, Transformer)")
    else:
        print("  ✗ PyTorch (not installed)")
    
    print(f"  ✓ Linear Regression (always available)")
    
    # 8. Instructions for running backtest
    print("\n" + "="*60)
    print("To run a backtest:")
    print("="*60)
    print("\nOption 1: Command line")
    print("  cd idx_trading_system/scripts")
    print(f"  python run_backtest.py --model {available_models[0]}")
    
    print("\nOption 2: Jupyter notebook")
    print("  jupyter notebook idx_trading_system/notebooks/demo_trading_system.ipynb")
    
    print("\nOption 3: Python API")
    print("""
from idx_trading_system.data_loader import DataLoader
from idx_trading_system.features import FeatureEngine
from idx_trading_system.models import create_model
from idx_trading_system.portfolio import PortfolioConstructor
from idx_trading_system.backtest import WalkForwardBacktest
import yaml

with open('idx_trading_system/config.yaml') as f:
    config = yaml.safe_load(f)

data_loader = DataLoader(config['paths']['cache_dir'])
feature_engine = FeatureEngine(config)
portfolio_constructor = PortfolioConstructor(config)
backtester = WalkForwardBacktest(config)

data_dict = data_loader.load_universe(config)
model = create_model('xgboost', model_type='regression', config=config)

results = backtester.run_backtest(
    model,
    feature_engine,
    portfolio_constructor,
    data_dict
)

print(f"CAGR: {results['average_metrics_after_costs']['cagr']:.2%}")
""")
    
    print("\n" + "="*60)
    print("Demo complete!")
    print("="*60)


if __name__ == '__main__':
    simple_demo()
