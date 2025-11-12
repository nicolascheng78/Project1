"""
Setup script for IDX Trading System
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="idx_trading_system",
    version="1.0.0",
    author="Project1 Contributors",
    description="Research-grade neural network trading system for Indonesian equities",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/nicolascheng78/Project1",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "Topic :: Office/Business :: Financial :: Investment",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "yfinance>=0.2.0",
        "scikit-learn>=1.0.0",
        "xgboost>=1.5.0",
        "lightgbm>=3.3.0",
        "torch>=1.10.0",
        "pyyaml>=6.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "joblib>=1.1.0",
        "ta>=0.10.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=3.0",
            "black>=22.0",
            "flake8>=4.0",
            "mypy>=0.950",
            "jupyter>=1.0",
            "notebook>=6.4",
        ],
    },
    entry_points={
        "console_scripts": [
            "idx-backtest=idx_trading_system.scripts.run_backtest:main",
        ],
    },
    include_package_data=True,
    package_data={
        "idx_trading_system": ["config.yaml"],
    },
)
