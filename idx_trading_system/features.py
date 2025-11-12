"""
Feature engineering module for IDX trading system
All features are engineered to avoid lookahead bias using rolling windows and shift(1)
"""

import pandas as pd
import numpy as np
from typing import List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureEngine:
    """
    Feature engineering for trading system
    Creates technical indicators, cross-asset features, and regime features
    """
    
    def __init__(self, config: dict):
        """
        Initialize FeatureEngine
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.momentum_windows = config['features']['momentum_windows']
        self.volatility_windows = config['features']['volatility_windows']
        self.rsi_period = config['features']['rsi_period']
        self.macd_params = config['features']['macd']
        self.volume_zscore_window = config['features']['volume_zscore_window']
        self.norm_window = config['features']['normalization_window']
        
    def compute_returns(self, df: pd.DataFrame, price_col: str = 'adj_close') -> pd.DataFrame:
        """
        Compute log returns
        
        Args:
            df: DataFrame with price data
            price_col: Column name for price
            
        Returns:
            DataFrame with returns
        """
        df = df.copy()
        df['returns'] = np.log(df[price_col] / df[price_col].shift(1))
        return df
    
    def compute_momentum(self, df: pd.DataFrame, windows: List[int]) -> pd.DataFrame:
        """
        Compute momentum features for multiple windows
        
        Args:
            df: DataFrame with returns
            windows: List of window sizes
            
        Returns:
            DataFrame with momentum features
        """
        df = df.copy()
        
        for window in windows:
            # Cumulative return over window
            df[f'momentum_{window}d'] = df['returns'].rolling(window).sum()
            
            # Average return over window
            df[f'avg_return_{window}d'] = df['returns'].rolling(window).mean()
        
        return df
    
    def compute_volatility(self, df: pd.DataFrame, windows: List[int]) -> pd.DataFrame:
        """
        Compute volatility features
        
        Args:
            df: DataFrame with returns
            windows: List of window sizes
            
        Returns:
            DataFrame with volatility features
        """
        df = df.copy()
        
        for window in windows:
            df[f'volatility_{window}d'] = df['returns'].rolling(window).std()
        
        return df
    
    def compute_atr(self, df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
        """
        Compute Average True Range
        
        Args:
            df: DataFrame with OHLC data
            window: ATR window
            
        Returns:
            DataFrame with ATR
        """
        df = df.copy()
        
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift(1))
        low_close = np.abs(df['low'] - df['close'].shift(1))
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr'] = true_range.rolling(window).mean()
        
        return df
    
    def compute_rsi(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """
        Compute Relative Strength Index
        
        Args:
            df: DataFrame with price data
            period: RSI period
            
        Returns:
            DataFrame with RSI
        """
        df = df.copy()
        
        delta = df['adj_close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        return df
    
    def compute_macd(self, df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
        """
        Compute MACD indicator
        
        Args:
            df: DataFrame with price data
            fast: Fast EMA period
            slow: Slow EMA period
            signal: Signal line period
            
        Returns:
            DataFrame with MACD features
        """
        df = df.copy()
        
        ema_fast = df['adj_close'].ewm(span=fast, adjust=False).mean()
        ema_slow = df['adj_close'].ewm(span=slow, adjust=False).mean()
        
        df['macd'] = ema_fast - ema_slow
        df['macd_signal'] = df['macd'].ewm(span=signal, adjust=False).mean()
        df['macd_diff'] = df['macd'] - df['macd_signal']
        
        return df
    
    def compute_volume_features(self, df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
        """
        Compute volume-based features
        
        Args:
            df: DataFrame with volume data
            window: Rolling window for statistics
            
        Returns:
            DataFrame with volume features
        """
        df = df.copy()
        
        # Volume z-score
        volume_mean = df['volume'].rolling(window).mean()
        volume_std = df['volume'].rolling(window).std()
        df['volume_zscore'] = (df['volume'] - volume_mean) / (volume_std + 1e-8)
        
        # Volume ratio
        df['volume_ratio'] = df['volume'] / (volume_mean + 1e-8)
        
        return df
    
    def compute_gap(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute overnight gap
        
        Args:
            df: DataFrame with OHLC data
            
        Returns:
            DataFrame with gap feature
        """
        df = df.copy()
        df['gap'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
        return df
    
    def compute_relative_strength(self, stock_df: pd.DataFrame, index_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute stock vs index relative strength
        
        Args:
            stock_df: Stock DataFrame with returns
            index_df: Index DataFrame with returns
            
        Returns:
            DataFrame with relative strength
        """
        stock_df = stock_df.copy()
        
        # Align dates
        common_dates = stock_df.index.intersection(index_df.index)
        stock_aligned = stock_df.loc[common_dates]
        index_aligned = index_df.loc[common_dates, 'returns']
        
        # Relative strength = stock return - index return
        stock_aligned['rel_strength'] = stock_aligned['returns'] - index_aligned
        
        # Rolling relative strength
        for window in self.momentum_windows:
            stock_aligned[f'rel_strength_{window}d'] = stock_aligned['rel_strength'].rolling(window).sum()
        
        return stock_aligned
    
    def compute_regime_features(self, index_df: pd.DataFrame, windows: List[int] = [20, 60]) -> pd.DataFrame:
        """
        Compute market regime features based on index
        
        Args:
            index_df: Index DataFrame
            windows: Windows for regime computation
            
        Returns:
            DataFrame with regime features
        """
        index_df = index_df.copy()
        
        for window in windows:
            # Rolling volatility regime
            index_df[f'index_vol_{window}d'] = index_df['returns'].rolling(window).std()
            
            # Drawdown state
            rolling_max = index_df['adj_close'].rolling(window).max()
            index_df[f'drawdown_{window}d'] = (index_df['adj_close'] - rolling_max) / rolling_max
        
        return index_df
    
    def engineer_features(
        self, 
        equity_data: pd.DataFrame, 
        index_data: pd.DataFrame,
        shift_features: bool = True
    ) -> pd.DataFrame:
        """
        Engineer all features for equities
        
        Args:
            equity_data: Equity OHLCV data
            index_data: Index OHLCV data
            shift_features: Whether to shift features by 1 to avoid lookahead
            
        Returns:
            DataFrame with all features
        """
        logger.info("Engineering features...")
        
        # Group by ticker
        features_list = []
        
        for ticker in equity_data['ticker'].unique():
            logger.info(f"Processing {ticker}")
            ticker_data = equity_data[equity_data['ticker'] == ticker].copy()
            
            # Compute returns
            ticker_data = self.compute_returns(ticker_data)
            
            # Technical features
            ticker_data = self.compute_momentum(ticker_data, self.momentum_windows)
            ticker_data = self.compute_volatility(ticker_data, self.volatility_windows)
            ticker_data = self.compute_atr(ticker_data)
            ticker_data = self.compute_rsi(ticker_data, self.rsi_period)
            ticker_data = self.compute_macd(
                ticker_data, 
                self.macd_params[0], 
                self.macd_params[1], 
                self.macd_params[2]
            )
            ticker_data = self.compute_volume_features(ticker_data, self.volume_zscore_window)
            ticker_data = self.compute_gap(ticker_data)
            
            # Cross-asset features (relative to index)
            index_with_returns = self.compute_returns(index_data)
            ticker_data = self.compute_relative_strength(ticker_data, index_with_returns)
            
            features_list.append(ticker_data)
        
        # Combine all tickers
        all_features = pd.concat(features_list, axis=0)
        
        # Add regime features from index
        index_regime = self.compute_regime_features(index_with_returns)
        
        # Merge regime features to equity data
        regime_cols = [col for col in index_regime.columns if 'index_vol' in col or 'drawdown' in col]
        for col in regime_cols:
            all_features[col] = all_features.index.map(index_regime[col])
        
        # Shift features by 1 to avoid lookahead bias
        if shift_features:
            feature_cols = [col for col in all_features.columns 
                          if col not in ['ticker', 'open', 'high', 'low', 'close', 'adj_close', 'volume']]
            all_features[feature_cols] = all_features.groupby('ticker')[feature_cols].shift(1)
        
        logger.info(f"Feature engineering complete. Shape: {all_features.shape}")
        
        return all_features
    
    def normalize_features(
        self, 
        train_data: pd.DataFrame, 
        test_data: Optional[pd.DataFrame] = None,
        feature_cols: Optional[List[str]] = None
    ) -> tuple:
        """
        Normalize features using rolling statistics from training data
        
        Args:
            train_data: Training DataFrame
            test_data: Test DataFrame (optional)
            feature_cols: List of feature columns to normalize
            
        Returns:
            Tuple of (normalized_train, normalized_test) or just normalized_train
        """
        if feature_cols is None:
            # Auto-detect feature columns
            exclude_cols = ['ticker', 'open', 'high', 'low', 'close', 'adj_close', 'volume', 'returns']
            feature_cols = [col for col in train_data.columns if col not in exclude_cols]
        
        train_normalized = train_data.copy()
        
        # Compute rolling mean and std on training data
        for col in feature_cols:
            rolling_mean = train_data.groupby('ticker')[col].transform(
                lambda x: x.rolling(self.norm_window, min_periods=1).mean()
            )
            rolling_std = train_data.groupby('ticker')[col].transform(
                lambda x: x.rolling(self.norm_window, min_periods=1).std()
            )
            
            # Normalize
            train_normalized[col] = (train_data[col] - rolling_mean) / (rolling_std + 1e-8)
        
        if test_data is not None:
            test_normalized = test_data.copy()
            
            # Use last values from training for test normalization
            for ticker in test_data['ticker'].unique():
                train_ticker = train_data[train_data['ticker'] == ticker]
                test_ticker_mask = test_data['ticker'] == ticker
                
                if len(train_ticker) > 0:
                    for col in feature_cols:
                        last_mean = train_ticker[col].rolling(self.norm_window, min_periods=1).mean().iloc[-1]
                        last_std = train_ticker[col].rolling(self.norm_window, min_periods=1).std().iloc[-1]
                        
                        test_normalized.loc[test_ticker_mask, col] = (
                            test_data.loc[test_ticker_mask, col] - last_mean
                        ) / (last_std + 1e-8)
            
            return train_normalized, test_normalized
        
        return train_normalized


def create_labels(
    df: pd.DataFrame, 
    forward_periods: List[int] = [5, 10, 20],
    classification_threshold: float = 0.0
) -> pd.DataFrame:
    """
    Create forward-looking labels for prediction
    
    Args:
        df: DataFrame with returns
        forward_periods: List of forward periods to predict
        classification_threshold: Threshold for binary classification
        
    Returns:
        DataFrame with labels
    """
    df = df.copy()
    
    for period in forward_periods:
        # Regression target: forward log return
        df[f'target_return_{period}d'] = df.groupby('ticker')['returns'].shift(-period).rolling(period).sum()
        
        # Classification target: directional
        df[f'target_direction_{period}d'] = (df[f'target_return_{period}d'] > classification_threshold).astype(int)
    
    return df
