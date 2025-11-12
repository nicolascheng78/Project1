"""
Portfolio construction module for IDX trading system
Implements long-only portfolio with constraints (position caps, volatility targeting, turnover limits)
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PortfolioConstructor:
    """
    Construct long-only portfolios with constraints
    """
    
    def __init__(self, config: dict):
        """
        Initialize PortfolioConstructor
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.portfolio_type = config['portfolio']['type']
        self.rebalance_freq = config['portfolio']['rebalance_frequency']
        self.top_k = config['portfolio']['top_k']
        self.vol_target = config['portfolio']['volatility_target']
        self.position_cap = config['portfolio']['position_cap']
        self.turnover_cap = config['portfolio']['turnover_cap']
        
    def construct_portfolio(
        self,
        predictions: pd.DataFrame,
        returns_data: pd.DataFrame,
        date: pd.Timestamp,
        previous_weights: Optional[pd.Series] = None
    ) -> pd.Series:
        """
        Construct portfolio for a given date
        
        Args:
            predictions: DataFrame with predictions (index=date, columns=tickers)
            returns_data: DataFrame with historical returns for vol estimation
            date: Current date
            previous_weights: Previous portfolio weights
            
        Returns:
            Series with portfolio weights (ticker -> weight)
        """
        # Get predictions for this date
        if date not in predictions.index:
            logger.warning(f"No predictions for {date}")
            return pd.Series()
        
        scores = predictions.loc[date]
        
        # Filter valid scores
        scores = scores.dropna()
        
        if len(scores) == 0:
            logger.warning(f"No valid scores for {date}")
            return pd.Series()
        
        # Rank-based portfolio: select top K
        top_stocks = scores.nlargest(self.top_k)
        
        # Equal weight or inverse volatility weight
        weights = self._compute_weights(top_stocks.index, returns_data, date)
        
        # Apply position cap
        weights = self._apply_position_cap(weights)
        
        # Apply turnover constraint if previous weights exist
        if previous_weights is not None:
            weights = self._apply_turnover_constraint(weights, previous_weights)
        
        # Normalize to sum to 1
        weights = weights / weights.sum()
        
        return weights
    
    def _compute_weights(
        self,
        tickers: pd.Index,
        returns_data: pd.DataFrame,
        date: pd.Timestamp,
        lookback: int = 60
    ) -> pd.Series:
        """
        Compute weights using inverse volatility
        
        Args:
            tickers: Selected tickers
            returns_data: Historical returns
            date: Current date
            lookback: Lookback period for volatility
            
        Returns:
            Series with weights
        """
        weights = {}
        
        for ticker in tickers:
            # Get historical returns
            ticker_returns = returns_data[
                (returns_data['ticker'] == ticker) & 
                (returns_data.index < date)
            ]['returns'].tail(lookback)
            
            if len(ticker_returns) > 0:
                vol = ticker_returns.std()
                # Inverse volatility weight
                weights[ticker] = 1.0 / (vol + 1e-8)
            else:
                weights[ticker] = 1.0
        
        weights = pd.Series(weights)
        
        # Normalize
        weights = weights / weights.sum()
        
        return weights
    
    def _apply_position_cap(self, weights: pd.Series) -> pd.Series:
        """
        Apply position cap constraint
        
        Args:
            weights: Portfolio weights
            
        Returns:
            Capped weights
        """
        # Cap individual positions
        weights = weights.clip(upper=self.position_cap)
        
        # Renormalize
        weights = weights / weights.sum()
        
        return weights
    
    def _apply_turnover_constraint(
        self,
        new_weights: pd.Series,
        old_weights: pd.Series
    ) -> pd.Series:
        """
        Apply turnover constraint
        
        Args:
            new_weights: Proposed new weights
            old_weights: Current weights
            
        Returns:
            Adjusted weights
        """
        # Align weights
        all_tickers = new_weights.index.union(old_weights.index)
        new_w = new_weights.reindex(all_tickers, fill_value=0)
        old_w = old_weights.reindex(all_tickers, fill_value=0)
        
        # Compute turnover
        turnover = (new_w - old_w).abs().sum()
        
        # If turnover exceeds cap, blend old and new weights
        if turnover > self.turnover_cap / 252:  # Daily turnover limit
            # Reduce turnover by blending
            blend_factor = (self.turnover_cap / 252) / turnover
            adjusted_w = old_w + blend_factor * (new_w - old_w)
            
            # Renormalize
            adjusted_w = adjusted_w.clip(lower=0)
            adjusted_w = adjusted_w / adjusted_w.sum()
            
            return adjusted_w
        
        return new_weights
    
    def apply_volatility_targeting(
        self,
        weights: pd.Series,
        returns_data: pd.DataFrame,
        date: pd.Timestamp,
        lookback: int = 60
    ) -> pd.Series:
        """
        Scale portfolio to target volatility
        
        Args:
            weights: Portfolio weights
            returns_data: Historical returns
            date: Current date
            lookback: Lookback for vol estimation
            
        Returns:
            Scaled weights
        """
        # Compute portfolio historical returns
        portfolio_returns = []
        
        for i in range(1, lookback + 1):
            hist_date = date - pd.Timedelta(days=i)
            day_return = 0.0
            
            for ticker, weight in weights.items():
                ticker_data = returns_data[
                    (returns_data['ticker'] == ticker) &
                    (returns_data.index == hist_date)
                ]
                
                if len(ticker_data) > 0:
                    day_return += weight * ticker_data['returns'].iloc[0]
            
            portfolio_returns.append(day_return)
        
        if len(portfolio_returns) > 0:
            portfolio_vol = np.std(portfolio_returns) * np.sqrt(252)
            
            # Scale weights to target vol
            if portfolio_vol > 0:
                scale_factor = self.vol_target / portfolio_vol
                scaled_weights = weights * scale_factor
                
                # Ensure we don't over-leverage (max 100% allocation for long-only)
                if scaled_weights.sum() > 1.0:
                    scaled_weights = scaled_weights / scaled_weights.sum()
                
                return scaled_weights
        
        return weights
    
    def compute_portfolio_metrics(
        self,
        weights_history: pd.DataFrame,
        returns_data: pd.DataFrame
    ) -> Dict:
        """
        Compute portfolio performance metrics
        
        Args:
            weights_history: DataFrame of weights over time (date x ticker)
            returns_data: Returns data
            
        Returns:
            Dict of metrics
        """
        # Compute portfolio returns
        portfolio_returns = []
        
        for date in weights_history.index:
            weights = weights_history.loc[date]
            weights = weights.dropna()
            
            if len(weights) == 0:
                portfolio_returns.append(0.0)
                continue
            
            # Get returns for this date
            day_return = 0.0
            for ticker, weight in weights.items():
                ticker_ret = returns_data[
                    (returns_data['ticker'] == ticker) &
                    (returns_data.index == date)
                ]
                
                if len(ticker_ret) > 0:
                    day_return += weight * ticker_ret['returns'].iloc[0]
            
            portfolio_returns.append(day_return)
        
        portfolio_returns = pd.Series(portfolio_returns, index=weights_history.index)
        
        # Compute metrics
        metrics = self._compute_metrics(portfolio_returns, weights_history)
        
        return metrics
    
    def _compute_metrics(self, returns: pd.Series, weights_history: pd.DataFrame) -> Dict:
        """
        Compute performance metrics
        
        Args:
            returns: Portfolio returns series
            weights_history: Weights over time
            
        Returns:
            Dict of metrics
        """
        # Basic statistics
        total_return = (1 + returns).prod() - 1
        n_years = len(returns) / 252
        cagr = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0
        
        # Volatility
        annual_vol = returns.std() * np.sqrt(252)
        
        # Sharpe ratio (assuming 0 risk-free rate)
        sharpe = cagr / annual_vol if annual_vol > 0 else 0
        
        # Sortino ratio
        downside_returns = returns[returns < 0]
        downside_vol = downside_returns.std() * np.sqrt(252)
        sortino = cagr / downside_vol if downside_vol > 0 else 0
        
        # Maximum drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Calmar ratio
        calmar = cagr / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Hit rate
        hit_rate = (returns > 0).sum() / len(returns) if len(returns) > 0 else 0
        
        # Turnover
        turnover = 0.0
        for i in range(1, len(weights_history)):
            prev_weights = weights_history.iloc[i-1].fillna(0)
            curr_weights = weights_history.iloc[i].fillna(0)
            
            # Align
            all_tickers = prev_weights.index.union(curr_weights.index)
            prev_w = prev_weights.reindex(all_tickers, fill_value=0)
            curr_w = curr_weights.reindex(all_tickers, fill_value=0)
            
            turnover += (curr_w - prev_w).abs().sum()
        
        avg_turnover = turnover / len(weights_history) if len(weights_history) > 0 else 0
        annual_turnover = avg_turnover * 252
        
        metrics = {
            'total_return': total_return,
            'cagr': cagr,
            'annual_volatility': annual_vol,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar,
            'hit_rate': hit_rate,
            'annual_turnover': annual_turnover
        }
        
        return metrics
    
    def compute_transaction_costs(
        self,
        weights_history: pd.DataFrame,
        returns_data: pd.DataFrame,
        commission: float = 0.002,
        slippage: float = 0.002
    ) -> pd.Series:
        """
        Compute transaction costs
        
        Args:
            weights_history: Portfolio weights over time
            returns_data: Returns data
            commission: Commission rate
            slippage: Slippage rate
            
        Returns:
            Series of transaction costs over time
        """
        costs = []
        
        for i in range(1, len(weights_history)):
            prev_weights = weights_history.iloc[i-1].fillna(0)
            curr_weights = weights_history.iloc[i].fillna(0)
            
            # Align
            all_tickers = prev_weights.index.union(curr_weights.index)
            prev_w = prev_weights.reindex(all_tickers, fill_value=0)
            curr_w = curr_weights.reindex(all_tickers, fill_value=0)
            
            # Turnover
            turnover = (curr_w - prev_w).abs().sum()
            
            # Cost = turnover * (commission + slippage)
            cost = turnover * (commission + slippage)
            costs.append(cost)
        
        return pd.Series([0] + costs, index=weights_history.index)
