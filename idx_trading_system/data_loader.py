"""
Data loader module for IDX trading system
Handles downloading and caching OHLCV data from Yahoo Finance with proper auto_adjust handling
"""

import os
import pandas as pd
import numpy as np
import yfinance as yf
from pathlib import Path
from typing import List, Union, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataLoader:
    """
    Download and cache historical OHLCV data for Indonesian equities
    
    Handles Yahoo Finance auto_adjust behavior:
    - When auto_adjust=True: use 'Close' (already adjusted)
    - When auto_adjust=False: use 'Adj Close'
    """
    
    def __init__(self, cache_dir: str = "./idx_trading_system/data/cache"):
        """
        Initialize DataLoader
        
        Args:
            cache_dir: Directory to cache downloaded data
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def download_ticker(
        self, 
        ticker: str, 
        start_date: str, 
        end_date: Optional[str] = None,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Download data for a single ticker
        
        Args:
            ticker: Yahoo Finance ticker symbol (e.g., TLKM.JK)
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format (None = today)
            use_cache: Whether to use cached data if available
            
        Returns:
            DataFrame with OHLCV data and adjusted close
        """
        cache_file = self.cache_dir / f"{ticker.replace('.', '_')}_{start_date}_{end_date}.parquet"
        
        # Try to load from cache
        if use_cache and cache_file.exists():
            logger.info(f"Loading {ticker} from cache")
            return pd.read_parquet(cache_file)
        
        logger.info(f"Downloading {ticker} from Yahoo Finance")
        
        try:
            # Download with auto_adjust=True to get adjusted prices in 'Close'
            data = yf.download(
                ticker, 
                start=start_date, 
                end=end_date,
                auto_adjust=False,  # Use False to get both Close and Adj Close
                progress=False
            )
            
            if data.empty:
                logger.warning(f"No data downloaded for {ticker}")
                return pd.DataFrame()
            
            # Ensure we have the adjusted close
            if 'Adj Close' in data.columns:
                data['AdjClose'] = data['Adj Close']
            elif 'Close' in data.columns:
                data['AdjClose'] = data['Close']
            else:
                logger.error(f"No Close price found for {ticker}")
                return pd.DataFrame()
            
            # Standardize column names
            columns_map = {
                'Open': 'open',
                'High': 'high', 
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume',
                'AdjClose': 'adj_close'
            }
            
            data = data.rename(columns=columns_map)
            data = data[['open', 'high', 'low', 'close', 'adj_close', 'volume']]
            data['ticker'] = ticker
            
            # Save to cache
            data.to_parquet(cache_file)
            logger.info(f"Cached {ticker} to {cache_file}")
            
            return data
            
        except Exception as e:
            logger.error(f"Error downloading {ticker}: {e}")
            return pd.DataFrame()
    
    def download_multiple(
        self,
        tickers: List[str],
        start_date: str,
        end_date: Optional[str] = None,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Download data for multiple tickers
        
        Args:
            tickers: List of ticker symbols
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            use_cache: Whether to use cached data
            
        Returns:
            Combined DataFrame with all tickers
        """
        all_data = []
        
        for ticker in tickers:
            data = self.download_ticker(ticker, start_date, end_date, use_cache)
            if not data.empty:
                all_data.append(data)
        
        if not all_data:
            logger.error("No data downloaded for any ticker")
            return pd.DataFrame()
        
        # Combine all data
        combined = pd.concat(all_data, axis=0)
        combined = combined.sort_index()
        
        return combined
    
    def load_universe(
        self,
        config: dict,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Load complete universe based on config
        
        Args:
            config: Configuration dictionary
            use_cache: Whether to use cached data
            
        Returns:
            DataFrame with all universe data
        """
        tickers = config['universe']['tickers']
        index = config['universe']['index']
        start_date = config['universe']['start_date']
        end_date = config['universe'].get('end_date', None)
        
        # Download equity tickers
        logger.info("Loading equity universe...")
        equity_data = self.download_multiple(tickers, start_date, end_date, use_cache)
        
        # Download index
        logger.info("Loading index...")
        index_data = self.download_ticker(index, start_date, end_date, use_cache)
        
        # Download macro data if configured
        macro_data = {}
        if 'macro' in config:
            logger.info("Loading macro data...")
            
            if 'fx' in config['macro'] and config['macro']['fx']:
                fx_data = self.download_ticker(
                    config['macro']['fx'], start_date, end_date, use_cache
                )
                if not fx_data.empty:
                    macro_data['fx'] = fx_data
            
            if 'global_indices' in config['macro']:
                for name, ticker in config['macro']['global_indices'].items():
                    if ticker:
                        idx_data = self.download_ticker(ticker, start_date, end_date, use_cache)
                        if not idx_data.empty:
                            macro_data[name] = idx_data
        
        return {
            'equities': equity_data,
            'index': index_data,
            'macro': macro_data
        }
    
    def clear_cache(self):
        """Clear all cached data"""
        import shutil
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info("Cache cleared")


def align_data(df: pd.DataFrame, reference_dates: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Align data to reference dates (e.g., trading calendar)
    Forward-fill missing values
    
    Args:
        df: DataFrame to align
        reference_dates: Target date index
        
    Returns:
        Aligned DataFrame
    """
    df = df.reindex(reference_dates)
    df = df.fillna(method='ffill')  # Forward fill for non-trading days
    return df


def handle_corporate_actions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure corporate actions (splits, dividends) are properly handled
    This is done via adjusted prices from Yahoo Finance
    
    Args:
        df: DataFrame with 'close' and 'adj_close'
        
    Returns:
        DataFrame with adjustment factor
    """
    if 'adj_close' in df.columns and 'close' in df.columns:
        df['adjustment_factor'] = df['adj_close'] / df['close']
        df['adjustment_factor'] = df['adjustment_factor'].fillna(1.0)
    else:
        df['adjustment_factor'] = 1.0
    
    return df
