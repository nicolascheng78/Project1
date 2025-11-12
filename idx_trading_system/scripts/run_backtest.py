#!/usr/bin/env python
"""
CLI script for running IDX trading system backtests
"""

import argparse
import yaml
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from idx_trading_system.data_loader import DataLoader
from idx_trading_system.features import FeatureEngine
from idx_trading_system.models import create_model
from idx_trading_system.portfolio import PortfolioConstructor
from idx_trading_system.backtest import WalkForwardBacktest


def main():
    parser = argparse.ArgumentParser(description='Run IDX trading system backtest')
    parser.add_argument('--config', type=str, default='../idx_trading_system/config.yaml',
                       help='Path to config file')
    parser.add_argument('--model', type=str, default='xgboost',
                       choices=['linear', 'xgboost', 'lightgbm', 'lstm', 'transformer'],
                       help='Model to use')
    parser.add_argument('--clear-cache', action='store_true',
                       help='Clear data cache before running')
    parser.add_argument('--output', type=str, default='../results/backtest_results.json',
                       help='Output path for results')
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    print("="*60)
    print("IDX Neural Network Trading System")
    print("="*60)
    print(f"Model: {args.model}")
    print(f"Config: {args.config}")
    print("="*60)
    
    # Initialize components
    data_loader = DataLoader(config['paths']['cache_dir'])
    
    if args.clear_cache:
        print("Clearing cache...")
        data_loader.clear_cache()
    
    # Load data
    print("\nLoading data...")
    data_dict = data_loader.load_universe(config)
    
    if data_dict['equities'].empty:
        print("ERROR: No equity data loaded!")
        return
    
    print(f"Loaded {len(data_dict['equities']['ticker'].unique())} tickers")
    print(f"Date range: {data_dict['equities'].index.min()} to {data_dict['equities'].index.max()}")
    
    # Initialize feature engine
    feature_engine = FeatureEngine(config)
    
    # Initialize portfolio constructor
    portfolio_constructor = PortfolioConstructor(config)
    
    # Initialize backtester
    backtester = WalkForwardBacktest(config)
    
    # Create model
    print(f"\nInitializing {args.model} model...")
    model = create_model(args.model, model_type='regression', config=config)
    
    # Run backtest
    print("\nRunning backtest...")
    results = backtester.run_backtest(
        model,
        feature_engine,
        portfolio_constructor,
        data_dict
    )
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    backtester.save_results(results, str(output_path))
    
    # Print summary
    print("\n" + "="*60)
    print("BACKTEST COMPLETE")
    print("="*60)
    
    avg_metrics = results['average_metrics_after_costs']
    print(f"Average CAGR: {avg_metrics['cagr']:.2%}")
    print(f"Average Sharpe: {avg_metrics['sharpe_ratio']:.2f}")
    print(f"Average Max Drawdown: {avg_metrics['max_drawdown']:.2%}")
    
    target_cagr = config['targets']['cagr']
    if avg_metrics['cagr'] >= target_cagr:
        print(f"\n✓ TARGET ACHIEVED! CAGR {avg_metrics['cagr']:.2%} >= {target_cagr:.2%}")
    else:
        print(f"\n✗ Target not met. CAGR {avg_metrics['cagr']:.2%} < {target_cagr:.2%}")
        print("Consider hyperparameter tuning or feature engineering.")
    
    print("="*60)


if __name__ == '__main__':
    main()
