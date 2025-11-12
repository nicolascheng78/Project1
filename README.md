# IDX Neural Network Trading System

A research-grade neural network trading system for Indonesian equities (IDX) targeting >15% CAGR after realistic transaction costs.

## Overview

This system implements a complete quantitative trading pipeline for Indonesian stocks (Jakarta Stock Exchange), featuring:

- **Data Pipeline**: Robust Yahoo Finance integration with proper handling of corporate actions and adjusted prices
- **Feature Engineering**: 50+ technical indicators, cross-asset features, and market regime indicators with no lookahead bias
- **Multiple Models**: Baseline (Linear, XGBoost, LightGBM) and neural network models (LSTM, Transformer)
- **Portfolio Construction**: Long-only portfolios with volatility targeting, position caps, and turnover constraints
- **Rigorous Backtesting**: Walk-forward methodology with strict time-series splits
- **Transaction Costs**: Realistic modeling of commissions, slippage, and liquidity constraints

## Features

### Data & Universe
- **Liquid Universe**: LQ45 constituents (TLKM.JK, BBCA.JK, BBRI.JK, BMRI.JK, ASII.JK, etc.)
- **Benchmark**: Jakarta Composite Index (^JKSE)
- **Historical Data**: 2005+ with daily OHLCV
- **Macro Data**: USDIDR FX, global indices (SPX, HSCEI), commodities
- **Corporate Actions**: Properly handled via Yahoo Finance adjusted prices
- **Caching**: Local parquet caching for fast iteration

### Feature Engineering (No Lookahead Bias)
- **Momentum**: Multiple windows (5/20/60/120 days)
- **Volatility**: Rolling standard deviation, ATR
- **Technical Indicators**: RSI, MACD, volume z-scores
- **Cross-Asset**: Relative strength vs index
- **Regime Features**: Index volatility, drawdown states
- **Normalization**: Rolling statistics fit on training data only

### Models
1. **Baseline Models**
   - Linear/Logistic Regression
   - XGBoost with early stopping
   - LightGBM with early stopping

2. **Neural Networks**
   - LSTM with 2 layers and dropout
   - Transformer with causal masking
   - Configurable architecture via config.yaml

### Portfolio Construction
- **Long-only** (Indonesia restricts shorting)
- **Top-K selection** based on model predictions
- **Inverse volatility weighting**
- **Position caps**: Max 10% per stock
- **Volatility targeting**: 10% annualized
- **Turnover constraints**: Cap on daily turnover

### Transaction Costs
- **Commission**: 20 bps per trade
- **Slippage**: 20 bps per trade
- **Round-trip**: 40 bps total
- **Realistic modeling** of execution costs

## Project Structure

```
idx_trading_system/
├── __init__.py           # Package initialization
├── config.yaml           # Main configuration file
├── data_loader.py        # Yahoo Finance data loader with caching
├── features.py           # Feature engineering pipeline
├── models.py            # Model implementations (baseline + neural nets)
├── portfolio.py         # Portfolio construction and risk management
├── backtest.py          # Walk-forward backtesting framework
├── data/                # Data cache directory
├── models/              # Saved model checkpoints
├── notebooks/           # Jupyter notebooks
│   └── demo_trading_system.ipynb
└── scripts/             # CLI scripts
    └── run_backtest.py  # Command-line backtesting
```

## Installation

### Requirements
- Python 3.8+
- Required packages listed in `requirements.txt`

### Setup

```bash
# Clone repository
git clone https://github.com/nicolascheng78/Project1.git
cd Project1

# Install dependencies
pip install -r requirements.txt

# Optional: Install PyTorch with CUDA support for GPU acceleration
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Quick Start

### 1. Command Line Interface

Run a backtest with default configuration:

```bash
cd idx_trading_system/scripts
python run_backtest.py
```

With custom options:

```bash
python run_backtest.py --model xgboost --clear-cache --output ../../results/xgb_results.json
```

Available models: `linear`, `xgboost`, `lightgbm`, `lstm`, `transformer`

### 2. Jupyter Notebook

Open the demo notebook for interactive exploration:

```bash
jupyter notebook idx_trading_system/notebooks/demo_trading_system.ipynb
```

The notebook demonstrates:
- Data loading and visualization
- Feature engineering
- Model training
- Walk-forward backtesting
- Performance analysis with plots

### 3. Python API

```python
import yaml
from idx_trading_system.data_loader import DataLoader
from idx_trading_system.features import FeatureEngine
from idx_trading_system.models import create_model
from idx_trading_system.portfolio import PortfolioConstructor
from idx_trading_system.backtest import WalkForwardBacktest

# Load config
with open('idx_trading_system/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Initialize components
data_loader = DataLoader(config['paths']['cache_dir'])
feature_engine = FeatureEngine(config)
portfolio_constructor = PortfolioConstructor(config)
backtester = WalkForwardBacktest(config)

# Load data
data_dict = data_loader.load_universe(config)

# Create model
model = create_model('xgboost', model_type='regression', config=config)

# Run backtest
results = backtester.run_backtest(
    model,
    feature_engine,
    portfolio_constructor,
    data_dict
)

# View results
print(f"CAGR: {results['average_metrics_after_costs']['cagr']:.2%}")
print(f"Sharpe: {results['average_metrics_after_costs']['sharpe_ratio']:.2f}")
```

## Configuration

Edit `idx_trading_system/config.yaml` to customize:

- **Universe**: Add/remove tickers
- **Date Range**: Adjust start/end dates
- **Features**: Modify momentum windows, indicators
- **Models**: Tune hyperparameters
- **Portfolio**: Change position limits, volatility target
- **Costs**: Adjust transaction costs
- **Backtest**: Define train/val/test splits

## Methodology

### Walk-Forward Backtesting

Strict time-series methodology to avoid lookahead bias:

1. **Split 1**: Train 2005-2014, Validate 2015-2016, Test 2017-2019
2. **Split 2**: Train 2006-2015, Validate 2016-2017, Test 2018-2020

Each split:
- Train model on training period
- Tune hyperparameters on validation period
- Generate predictions on test period
- Construct portfolios and compute returns
- Calculate metrics after transaction costs

### Performance Metrics

- **CAGR**: Compound annual growth rate
- **Sharpe Ratio**: Risk-adjusted returns (0% risk-free rate)
- **Sortino Ratio**: Downside risk-adjusted returns
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Calmar Ratio**: CAGR / |Max Drawdown|
- **Hit Rate**: Percentage of positive return days
- **Annual Turnover**: Portfolio turnover rate

### Target Performance

- **Primary Goal**: CAGR > 15% after costs
- **Risk Management**: Max Drawdown < 30%
- **Sharpe Ratio**: Target > 1.0

## Dataset Caveats

### Survivorship Bias
Currently using live tickers only. For production, consider:
- Tracking delisted companies
- Maintaining historical LQ45 membership by quarter
- Including stocks that were removed from the index

### Corporate Actions
- Yahoo Finance provides adjusted prices for splits and dividends
- The `adj_close` field reflects all corporate actions
- Always use adjusted prices for backtesting

### Data Quality
- Some tickers may have limited history (post-2010)
- Early data (pre-2008) may be sparse for some stocks
- Index data (^JKSE) generally available from 1990s

### Market Constraints
- **No Shorting**: Indonesian market restricts short selling
- **T+2 Settlement**: Trades settle two days after execution
- **Tick Size**: Varies by price level (not modeled)
- **Trading Halts**: Not modeled in backtest

## Hyperparameter Optimization

If target CAGR not achieved, run systematic optimization:

```python
from sklearn.model_selection import ParameterGrid

# Define parameter grid
param_grid = {
    'max_depth': [4, 6, 8],
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [50, 100, 200]
}

results_list = []

for params in ParameterGrid(param_grid):
    model = create_model('xgboost', model_type='regression', config=config, **params)
    result = backtester.run_backtest(model, feature_engine, portfolio_constructor, data_dict)
    results_list.append({
        'params': params,
        'cagr': result['average_metrics_after_costs']['cagr'],
        'sharpe': result['average_metrics_after_costs']['sharpe_ratio']
    })

# Sort by CAGR
results_df = pd.DataFrame(results_list).sort_values('cagr', ascending=False)
print(results_df.head(10))
```

## Stress Testing

Test robustness across crisis periods:

```python
crisis_periods = {
    'AFC_1997': ('1997-07-01', '1998-12-31'),  # Asian Financial Crisis
    'GFC_2008': ('2008-01-01', '2009-12-31'),  # Global Financial Crisis
    'COVID_2020': ('2020-01-01', '2020-12-31') # COVID-19 Pandemic
}

for name, (start, end) in crisis_periods.items():
    # Filter data for crisis period
    # Run backtest
    # Analyze drawdown and recovery
```

## Advanced Features

### Probability Calibration
For classification models, calibrate probabilities:

```python
from sklearn.calibration import CalibratedClassifierCV

calibrated_model = CalibratedClassifierCV(base_model, method='isotonic', cv=3)
calibrated_model.fit(X_train, y_train)
```

### Feature Attribution
For tree-based models:

```python
import shap

explainer = shap.TreeExplainer(xgb_model.model)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test, feature_names=feature_cols)
```

### Ensemble Methods

```python
# Train multiple models
models = {
    'xgb': create_model('xgboost', ...),
    'lgb': create_model('lightgbm', ...),
    'lstm': create_model('lstm', ...)
}

# Average predictions
ensemble_pred = np.mean([m.predict(X_test) for m in models.values()], axis=0)
```

## Limitations & Future Work

### Current Limitations
1. **Survivorship Bias**: Only current constituents included
2. **Market Impact**: Not modeled (assumes unlimited liquidity)
3. **Regime Changes**: Static model doesn't adapt to regime shifts
4. **Factor Exposure**: No explicit factor risk management

### Future Enhancements
1. **Alternative Data**: News sentiment, social media
2. **Intraday Data**: Higher frequency signals
3. **Options Overlay**: Protective puts, covered calls
4. **Multi-Asset**: Include bonds, commodities
5. **Online Learning**: Continuous model updates
6. **Reinforcement Learning**: Model-free portfolio optimization

## Contributing

Contributions welcome! Areas for improvement:
- Additional data sources
- New feature engineering techniques
- Advanced models (e.g., Graph Neural Networks)
- Better risk management
- Performance optimization

## License

MIT License - see LICENSE file for details

## Citation

If you use this system in your research, please cite:

```
@software{idx_trading_system,
  title={IDX Neural Network Trading System},
  author={Project1 Contributors},
  year={2024},
  url={https://github.com/nicolascheng78/Project1}
}
```

## Disclaimer

This system is for educational and research purposes only. It is not financial advice. Trading stocks involves substantial risk of loss. Past performance does not guarantee future results. Always consult with a qualified financial advisor before making investment decisions.

The authors are not responsible for any financial losses incurred from using this system.

## Contact

For questions, issues, or suggestions:
- Open an issue on GitHub
- Email: [your-email@example.com]

---

**Built with Python 🐍 | Powered by Deep Learning 🧠 | Trading Indonesian Equities 🇮🇩**