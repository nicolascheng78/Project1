"""
Backtesting module for IDX trading system
Implements walk-forward backtesting with strict time-series splits
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WalkForwardBacktest:
    """
    Walk-forward backtesting framework
    """
    
    def __init__(self, config: dict):
        """
        Initialize backtester
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.splits = config['backtest']['splits']
        
    def create_splits(
        self,
        data: pd.DataFrame
    ) -> List[Dict]:
        """
        Create train/val/test splits
        
        Args:
            data: Full dataset
            
        Returns:
            List of split dictionaries
        """
        splits = []
        
        for split_config in self.splits:
            train_start = pd.to_datetime(split_config['train'][0])
            train_end = pd.to_datetime(split_config['train'][1])
            val_start = pd.to_datetime(split_config['val'][0])
            val_end = pd.to_datetime(split_config['val'][1])
            test_start = pd.to_datetime(split_config['test'][0])
            test_end = pd.to_datetime(split_config['test'][1])
            
            train_data = data[(data.index >= train_start) & (data.index <= train_end)]
            val_data = data[(data.index >= val_start) & (data.index <= val_end)]
            test_data = data[(data.index >= test_start) & (data.index <= test_end)]
            
            splits.append({
                'train': train_data,
                'val': val_data,
                'test': test_data,
                'train_period': (train_start, train_end),
                'val_period': (val_start, val_end),
                'test_period': (test_start, test_end)
            })
        
        return splits
    
    def run_backtest(
        self,
        model,
        feature_engine,
        portfolio_constructor,
        data_dict: Dict,
        target_col: str = 'target_return_20d',
        feature_cols: Optional[List[str]] = None
    ) -> Dict:
        """
        Run complete walk-forward backtest
        
        Args:
            model: Model instance
            feature_engine: FeatureEngine instance
            portfolio_constructor: PortfolioConstructor instance
            data_dict: Dictionary with 'equities', 'index', 'macro' data
            target_col: Name of target column
            feature_cols: List of feature columns (auto-detected if None)
            
        Returns:
            Dictionary with results
        """
        equity_data = data_dict['equities']
        index_data = data_dict['index']
        
        # Engineer features
        logger.info("Engineering features for backtesting...")
        from .features import create_labels
        
        features = feature_engine.engineer_features(equity_data, index_data, shift_features=True)
        features = create_labels(features, self.config['labels']['forward_periods'])
        
        # Auto-detect feature columns if not provided
        if feature_cols is None:
            exclude_cols = ['ticker', 'open', 'high', 'low', 'close', 'adj_close', 'volume', 'returns']
            exclude_cols += [col for col in features.columns if 'target' in col]
            feature_cols = [col for col in features.columns if col not in exclude_cols]
        
        logger.info(f"Using {len(feature_cols)} features")
        
        # Create splits
        splits = self.create_splits(features)
        
        all_results = []
        
        for i, split in enumerate(splits):
            logger.info(f"\n{'='*60}")
            logger.info(f"Running backtest split {i+1}/{len(splits)}")
            logger.info(f"Train: {split['train_period'][0]} to {split['train_period'][1]}")
            logger.info(f"Val: {split['val_period'][0]} to {split['val_period'][1]}")
            logger.info(f"Test: {split['test_period'][0]} to {split['test_period'][1]}")
            logger.info(f"{'='*60}\n")
            
            result = self._run_single_split(
                model,
                feature_engine,
                portfolio_constructor,
                split,
                target_col,
                feature_cols
            )
            
            all_results.append(result)
        
        # Aggregate results
        aggregated = self._aggregate_results(all_results)
        
        return aggregated
    
    def _run_single_split(
        self,
        model,
        feature_engine,
        portfolio_constructor,
        split: Dict,
        target_col: str,
        feature_cols: List[str]
    ) -> Dict:
        """
        Run backtest on a single split
        
        Args:
            model: Model instance
            feature_engine: FeatureEngine instance
            portfolio_constructor: PortfolioConstructor instance
            split: Split dictionary
            target_col: Target column name
            feature_cols: Feature column names
            
        Returns:
            Results dictionary
        """
        train_data = split['train']
        val_data = split['val']
        test_data = split['test']
        
        # Prepare data
        X_train = train_data[feature_cols].fillna(0).values
        y_train = train_data[target_col].fillna(0).values
        
        X_val = val_data[feature_cols].fillna(0).values
        y_val = val_data[target_col].fillna(0).values
        
        X_test = test_data[feature_cols].fillna(0).values
        y_test = test_data[target_col].fillna(0).values
        
        # Train model
        logger.info("Training model...")
        model.fit(X_train, y_train, X_val, y_val)
        
        # Predict on test set
        logger.info("Generating predictions...")
        predictions = model.predict(X_test)
        
        # Create predictions DataFrame
        test_data_copy = test_data.copy()
        test_data_copy['prediction'] = predictions
        
        # Pivot predictions for portfolio construction (date x ticker)
        pred_pivot = test_data_copy.pivot_table(
            index=test_data_copy.index,
            columns='ticker',
            values='prediction'
        )
        
        # Construct portfolios
        logger.info("Constructing portfolios...")
        weights_history = []
        dates = pred_pivot.index.unique()
        
        previous_weights = None
        for date in dates:
            weights = portfolio_constructor.construct_portfolio(
                pred_pivot,
                test_data_copy,
                date,
                previous_weights
            )
            
            if len(weights) > 0:
                weights_history.append(weights)
                previous_weights = weights
            else:
                weights_history.append(pd.Series())
        
        weights_df = pd.DataFrame(weights_history, index=dates)
        
        # Compute metrics
        logger.info("Computing metrics...")
        metrics = portfolio_constructor.compute_portfolio_metrics(weights_df, test_data_copy)
        
        # Compute transaction costs
        costs = portfolio_constructor.compute_transaction_costs(
            weights_df,
            test_data_copy,
            self.config['costs']['commission'],
            self.config['costs']['slippage']
        )
        
        # Adjust returns for costs
        portfolio_returns = self._compute_portfolio_returns(weights_df, test_data_copy)
        portfolio_returns_after_costs = portfolio_returns - costs
        
        # Recompute metrics after costs
        metrics_after_costs = self._compute_metrics_from_returns(portfolio_returns_after_costs)
        
        result = {
            'split_period': split['test_period'],
            'metrics_before_costs': metrics,
            'metrics_after_costs': metrics_after_costs,
            'weights_history': weights_df,
            'returns': portfolio_returns,
            'returns_after_costs': portfolio_returns_after_costs,
            'costs': costs
        }
        
        # Log results
        logger.info(f"\nResults for test period {split['test_period'][0]} to {split['test_period'][1]}:")
        logger.info(f"CAGR (before costs): {metrics['cagr']:.2%}")
        logger.info(f"CAGR (after costs): {metrics_after_costs['cagr']:.2%}")
        logger.info(f"Sharpe (before costs): {metrics['sharpe_ratio']:.2f}")
        logger.info(f"Sharpe (after costs): {metrics_after_costs['sharpe_ratio']:.2f}")
        logger.info(f"Max Drawdown (before costs): {metrics['max_drawdown']:.2%}")
        logger.info(f"Max Drawdown (after costs): {metrics_after_costs['max_drawdown']:.2%}")
        logger.info(f"Annual Turnover: {metrics['annual_turnover']:.2f}")
        
        return result
    
    def _compute_portfolio_returns(
        self,
        weights_history: pd.DataFrame,
        returns_data: pd.DataFrame
    ) -> pd.Series:
        """
        Compute portfolio returns from weights
        
        Args:
            weights_history: Weights over time
            returns_data: Returns data
            
        Returns:
            Series of portfolio returns
        """
        portfolio_returns = []
        
        for date in weights_history.index:
            weights = weights_history.loc[date]
            weights = weights.dropna()
            
            if len(weights) == 0:
                portfolio_returns.append(0.0)
                continue
            
            day_return = 0.0
            for ticker, weight in weights.items():
                ticker_ret = returns_data[
                    (returns_data['ticker'] == ticker) &
                    (returns_data.index == date)
                ]
                
                if len(ticker_ret) > 0:
                    day_return += weight * ticker_ret['returns'].iloc[0]
            
            portfolio_returns.append(day_return)
        
        return pd.Series(portfolio_returns, index=weights_history.index)
    
    def _compute_metrics_from_returns(self, returns: pd.Series) -> Dict:
        """
        Compute metrics from return series
        
        Args:
            returns: Return series
            
        Returns:
            Metrics dict
        """
        total_return = (1 + returns).prod() - 1
        n_years = len(returns) / 252
        cagr = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0
        
        annual_vol = returns.std() * np.sqrt(252)
        sharpe = cagr / annual_vol if annual_vol > 0 else 0
        
        downside_returns = returns[returns < 0]
        downside_vol = downside_returns.std() * np.sqrt(252)
        sortino = cagr / downside_vol if downside_vol > 0 else 0
        
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        calmar = cagr / abs(max_drawdown) if max_drawdown != 0 else 0
        hit_rate = (returns > 0).sum() / len(returns) if len(returns) > 0 else 0
        
        return {
            'total_return': total_return,
            'cagr': cagr,
            'annual_volatility': annual_vol,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar,
            'hit_rate': hit_rate
        }
    
    def _aggregate_results(self, results: List[Dict]) -> Dict:
        """
        Aggregate results across splits
        
        Args:
            results: List of result dictionaries
            
        Returns:
            Aggregated results
        """
        # Average metrics
        metrics_before = []
        metrics_after = []
        
        for result in results:
            metrics_before.append(result['metrics_before_costs'])
            metrics_after.append(result['metrics_after_costs'])
        
        # Compute averages
        avg_metrics_before = self._average_metrics(metrics_before)
        avg_metrics_after = self._average_metrics(metrics_after)
        
        aggregated = {
            'individual_results': results,
            'average_metrics_before_costs': avg_metrics_before,
            'average_metrics_after_costs': avg_metrics_after,
            'num_splits': len(results)
        }
        
        logger.info(f"\n{'='*60}")
        logger.info("AGGREGATED RESULTS ACROSS ALL SPLITS")
        logger.info(f"{'='*60}")
        logger.info(f"Number of splits: {len(results)}")
        logger.info(f"\nAverage metrics (after costs):")
        logger.info(f"CAGR: {avg_metrics_after['cagr']:.2%}")
        logger.info(f"Sharpe Ratio: {avg_metrics_after['sharpe_ratio']:.2f}")
        logger.info(f"Sortino Ratio: {avg_metrics_after['sortino_ratio']:.2f}")
        logger.info(f"Max Drawdown: {avg_metrics_after['max_drawdown']:.2%}")
        logger.info(f"Calmar Ratio: {avg_metrics_after['calmar_ratio']:.2f}")
        logger.info(f"Hit Rate: {avg_metrics_after['hit_rate']:.2%}")
        logger.info(f"{'='*60}\n")
        
        return aggregated
    
    def _average_metrics(self, metrics_list: List[Dict]) -> Dict:
        """
        Average metrics across splits
        
        Args:
            metrics_list: List of metric dictionaries
            
        Returns:
            Averaged metrics
        """
        if not metrics_list:
            return {}
        
        avg_metrics = {}
        keys = metrics_list[0].keys()
        
        for key in keys:
            values = [m[key] for m in metrics_list]
            avg_metrics[key] = np.mean(values)
        
        return avg_metrics
    
    def save_results(self, results: Dict, path: str):
        """
        Save results to file
        
        Args:
            results: Results dictionary
            path: Output path
        """
        # Convert to JSON-serializable format
        serializable_results = {
            'average_metrics_before_costs': results['average_metrics_before_costs'],
            'average_metrics_after_costs': results['average_metrics_after_costs'],
            'num_splits': results['num_splits']
        }
        
        with open(path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Results saved to {path}")
