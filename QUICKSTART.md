# Quick Start Guide

## Installation

### Option 1: Install from source (recommended)

```bash
# Clone the repository
git clone https://github.com/nicolascheng78/Project1.git
cd Project1

# Install in development mode
pip install -e .

# Or install dependencies separately
pip install -r requirements.txt
```

### Option 2: Install with minimal dependencies

If you only want to test the system without neural networks:

```bash
pip install numpy pandas yfinance scikit-learn xgboost lightgbm pyyaml matplotlib seaborn joblib
```

## Quick Test

### 1. Validate Installation

```bash
cd idx_trading_system/scripts
python validate_system.py
```

This will check if all dependencies are installed and components can be initialized.

### 2. Run Simple Demo

```bash
python simple_demo.py
```

This displays the system configuration and available models.

### 3. Run Backtest

```bash
python run_backtest.py --model xgboost
```

This will:
- Download data from Yahoo Finance (cached locally)
- Engineer features
- Train XGBoost model
- Run walk-forward backtest
- Display results

**Note:** First run will download ~15-20 years of data for 10+ tickers, which may take 5-10 minutes.

## Usage Examples

### Example 1: Run with Different Models

```bash
# Linear regression baseline
python run_backtest.py --model linear

# XGBoost (recommended)
python run_backtest.py --model xgboost

# LightGBM
python run_backtest.py --model lightgbm

# LSTM (requires PyTorch)
python run_backtest.py --model lstm

# Transformer (requires PyTorch)
python run_backtest.py --model transformer
```

### Example 2: Clear Cache and Redownload Data

```bash
python run_backtest.py --model xgboost --clear-cache
```

### Example 3: Custom Output Path

```bash
python run_backtest.py --model xgboost --output ../../results/my_backtest.json
```

### Example 4: Jupyter Notebook (Interactive)

```bash
cd idx_trading_system/notebooks
jupyter notebook demo_trading_system.ipynb
```

The notebook provides:
- Step-by-step walkthrough
- Data visualization
- Feature analysis
- Model comparison
- Performance plots

### Example 5: Python API

```python
import yaml
from idx_trading_system.data_loader import DataLoader
from idx_trading_system.features import FeatureEngine
from idx_trading_system.models import create_model
from idx_trading_system.portfolio import PortfolioConstructor
from idx_trading_system.backtest import WalkForwardBacktest

# Load configuration
with open('idx_trading_system/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Initialize components
data_loader = DataLoader(config['paths']['cache_dir'])
feature_engine = FeatureEngine(config)
portfolio_constructor = PortfolioConstructor(config)
backtester = WalkForwardBacktest(config)

# Load data
print("Loading data...")
data_dict = data_loader.load_universe(config)

# Create and run model
print("Running backtest...")
model = create_model('xgboost', model_type='regression', config=config)
results = backtester.run_backtest(
    model,
    feature_engine,
    portfolio_constructor,
    data_dict
)

# Display results
avg_metrics = results['average_metrics_after_costs']
print(f"\nResults:")
print(f"  CAGR: {avg_metrics['cagr']:.2%}")
print(f"  Sharpe: {avg_metrics['sharpe_ratio']:.2f}")
print(f"  Max Drawdown: {avg_metrics['max_drawdown']:.2%}")

# Check if target achieved
target_cagr = config['targets']['cagr']
if avg_metrics['cagr'] >= target_cagr:
    print(f"\n✓ Target CAGR {target_cagr:.1%} achieved!")
else:
    print(f"\n✗ Target CAGR {target_cagr:.1%} not met. Consider tuning.")
```

## Customization

### Modify Universe

Edit `idx_trading_system/config.yaml`:

```yaml
universe:
  tickers:
    - TLKM.JK
    - BBCA.JK
    - BBRI.JK
    # Add more Indonesian stocks
```

### Adjust Backtest Periods

```yaml
backtest:
  splits:
    - train: ["2005-01-01", "2014-12-31"]
      val: ["2015-01-01", "2016-12-31"]
      test: ["2017-01-01", "2019-12-31"]
```

### Tune Model Hyperparameters

```yaml
models:
  xgboost:
    max_depth: 8  # Increase for more complex trees
    learning_rate: 0.03  # Lower for more conservative learning
    n_estimators: 200  # More trees
```

### Adjust Portfolio Constraints

```yaml
portfolio:
  top_k: 5  # Number of positions
  position_cap: 0.15  # Max 15% per position
  volatility_target: 0.12  # 12% target volatility
```

### Change Transaction Costs

```yaml
costs:
  commission: 0.0025  # 25 bps
  slippage: 0.0025  # 25 bps
  round_trip: 0.0050  # 50 bps total
```

## Troubleshooting

### Issue: Data download fails

**Solution:** Check internet connection and Yahoo Finance availability. Some tickers may have limited history.

```python
# Test single ticker download
from idx_trading_system.data_loader import DataLoader
loader = DataLoader()
data = loader.download_ticker('BBCA.JK', '2020-01-01')
print(data.head())
```

### Issue: "No module named 'idx_trading_system'"

**Solution:** Install the package or add to Python path:

```bash
# Option 1: Install package
pip install -e .

# Option 2: Add to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:/path/to/Project1"
```

### Issue: Out of memory with neural networks

**Solution:** Reduce batch size or sequence length in config:

```yaml
models:
  lstm:
    sequence_length: 30  # Reduce from 60
```

### Issue: Backtest takes too long

**Solution:** 
1. Use smaller universe (fewer tickers)
2. Use shorter time periods
3. Use XGBoost/LightGBM instead of neural networks
4. Enable caching (default)

### Issue: Target CAGR not achieved

**Solution:**
1. Run hyperparameter optimization
2. Add more features
3. Try different models
4. Ensemble multiple models
5. Adjust portfolio construction

## Performance Tips

1. **Enable caching**: Data downloads are cached by default in `idx_trading_system/data/cache/`
2. **Use XGBoost first**: Faster than neural networks for initial testing
3. **Parallel execution**: Run multiple backtests in parallel with different configs
4. **GPU acceleration**: For neural networks, install PyTorch with CUDA support

## Next Steps

1. **Run initial backtest** to establish baseline
2. **Analyze feature importance** to understand drivers
3. **Test multiple models** to find best performer
4. **Optimize hyperparameters** if target not met
5. **Validate on holdout period** for production readiness

## Support

For issues or questions:
- Check the main README.md
- Open an issue on GitHub
- Review the Jupyter notebook for detailed examples
