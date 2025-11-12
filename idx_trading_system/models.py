"""
Models module for IDX trading system
Implements baseline (Linear, XGBoost, LightGBM) and neural network models (LSTM, Transformer)
"""

import numpy as np
import pandas as pd
from typing import List, Optional, Dict, Tuple
import logging
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, accuracy_score, roc_auc_score
import joblib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Optional imports with error handling
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    logger.warning("XGBoost not available")

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    logger.warning("LightGBM not available")

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    HAS_PYTORCH = True
except ImportError:
    HAS_PYTORCH = False
    logger.warning("PyTorch not available")


class BaseModel:
    """Base class for all models"""
    
    def __init__(self, model_type: str = 'regression'):
        self.model_type = model_type
        self.model = None
        
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        raise NotImplementedError
        
    def predict(self, X):
        raise NotImplementedError
        
    def save(self, path: str):
        joblib.dump(self.model, path)
        
    def load(self, path: str):
        self.model = joblib.load(path)


class LinearModel(BaseModel):
    """Linear/Logistic Regression baseline"""
    
    def __init__(self, model_type: str = 'regression'):
        super().__init__(model_type)
        if model_type == 'regression':
            self.model = LinearRegression()
        else:
            self.model = LogisticRegression(max_iter=1000)
    
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        logger.info(f"Training Linear {self.model_type} model...")
        self.model.fit(X_train, y_train)
        
        if X_val is not None and y_val is not None:
            val_pred = self.predict(X_val)
            if self.model_type == 'regression':
                val_mse = mean_squared_error(y_val, val_pred)
                logger.info(f"Validation MSE: {val_mse:.6f}")
            else:
                val_acc = accuracy_score(y_val, val_pred > 0.5)
                logger.info(f"Validation Accuracy: {val_acc:.4f}")
    
    def predict(self, X):
        if self.model_type == 'regression':
            return self.model.predict(X)
        else:
            return self.model.predict_proba(X)[:, 1]


class XGBoostModel(BaseModel):
    """XGBoost model"""
    
    def __init__(self, model_type: str = 'regression', **kwargs):
        super().__init__(model_type)
        
        if not HAS_XGBOOST:
            raise ImportError("XGBoost not installed")
        
        default_params = {
            'max_depth': 6,
            'learning_rate': 0.05,
            'n_estimators': 100,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42
        }
        default_params.update(kwargs)
        
        if model_type == 'regression':
            self.model = xgb.XGBRegressor(**default_params)
        else:
            self.model = xgb.XGBClassifier(**default_params)
    
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        logger.info(f"Training XGBoost {self.model_type} model...")
        
        eval_set = [(X_train, y_train)]
        if X_val is not None and y_val is not None:
            eval_set.append((X_val, y_val))
        
        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            verbose=False
        )
        
        if X_val is not None and y_val is not None:
            val_pred = self.predict(X_val)
            if self.model_type == 'regression':
                val_mse = mean_squared_error(y_val, val_pred)
                logger.info(f"Validation MSE: {val_mse:.6f}")
            else:
                val_acc = accuracy_score(y_val, val_pred > 0.5)
                logger.info(f"Validation Accuracy: {val_acc:.4f}")
    
    def predict(self, X):
        if self.model_type == 'regression':
            return self.model.predict(X)
        else:
            return self.model.predict_proba(X)[:, 1]
    
    def get_feature_importance(self):
        """Get feature importance scores"""
        return self.model.feature_importances_


class LightGBMModel(BaseModel):
    """LightGBM model"""
    
    def __init__(self, model_type: str = 'regression', **kwargs):
        super().__init__(model_type)
        
        if not HAS_LIGHTGBM:
            raise ImportError("LightGBM not installed")
        
        default_params = {
            'max_depth': 6,
            'learning_rate': 0.05,
            'n_estimators': 100,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'verbose': -1
        }
        default_params.update(kwargs)
        
        if model_type == 'regression':
            self.model = lgb.LGBMRegressor(**default_params)
        else:
            self.model = lgb.LGBMClassifier(**default_params)
    
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        logger.info(f"Training LightGBM {self.model_type} model...")
        
        eval_set = [(X_train, y_train)]
        if X_val is not None and y_val is not None:
            eval_set.append((X_val, y_val))
        
        self.model.fit(
            X_train, y_train,
            eval_set=eval_set if X_val is not None else None,
        )
        
        if X_val is not None and y_val is not None:
            val_pred = self.predict(X_val)
            if self.model_type == 'regression':
                val_mse = mean_squared_error(y_val, val_pred)
                logger.info(f"Validation MSE: {val_mse:.6f}")
            else:
                val_acc = accuracy_score(y_val, val_pred > 0.5)
                logger.info(f"Validation Accuracy: {val_acc:.4f}")
    
    def predict(self, X):
        if self.model_type == 'regression':
            return self.model.predict(X)
        else:
            return self.model.predict_proba(X)[:, 1]
    
    def get_feature_importance(self):
        """Get feature importance scores"""
        return self.model.feature_importances_


# Neural Network Models (PyTorch)
if HAS_PYTORCH:
    
    class TimeSeriesDataset(Dataset):
        """Dataset for sequence models"""
        
        def __init__(self, X, y, sequence_length=60):
            self.X = X
            self.y = y
            self.sequence_length = sequence_length
            
        def __len__(self):
            return len(self.X) - self.sequence_length
            
        def __getitem__(self, idx):
            X_seq = self.X[idx:idx + self.sequence_length]
            y_val = self.y[idx + self.sequence_length - 1]
            return torch.FloatTensor(X_seq), torch.FloatTensor([y_val])
    
    
    class LSTMModel(nn.Module):
        """LSTM model for time series prediction"""
        
        def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
            super(LSTMModel, self).__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            
            self.lstm = nn.LSTM(
                input_size, 
                hidden_size, 
                num_layers, 
                dropout=dropout,
                batch_first=True
            )
            self.fc = nn.Linear(hidden_size, 1)
            
        def forward(self, x):
            # x shape: (batch, seq_len, input_size)
            lstm_out, _ = self.lstm(x)
            # Take last output
            out = self.fc(lstm_out[:, -1, :])
            return out
    
    
    class TransformerModel(nn.Module):
        """Transformer model with causal masking"""
        
        def __init__(self, input_size, d_model=64, nhead=4, num_layers=2, dropout=0.2):
            super(TransformerModel, self).__init__()
            self.d_model = d_model
            
            self.embedding = nn.Linear(input_size, d_model)
            
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dropout=dropout,
                batch_first=True
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            
            self.fc = nn.Linear(d_model, 1)
            
        def forward(self, x):
            # x shape: (batch, seq_len, input_size)
            x = self.embedding(x)
            
            # Create causal mask
            seq_len = x.size(1)
            mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
            mask = mask.to(x.device)
            
            x = self.transformer(x, mask=mask)
            out = self.fc(x[:, -1, :])
            return out
    
    
    class NeuralNetworkModel(BaseModel):
        """Wrapper for PyTorch neural network models"""
        
        def __init__(
            self, 
            model_type: str = 'regression',
            architecture: str = 'lstm',
            input_size: int = 10,
            sequence_length: int = 60,
            **kwargs
        ):
            super().__init__(model_type)
            
            self.architecture = architecture
            self.input_size = input_size
            self.sequence_length = sequence_length
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # Create model
            if architecture == 'lstm':
                self.model = LSTMModel(
                    input_size=input_size,
                    hidden_size=kwargs.get('hidden_size', 64),
                    num_layers=kwargs.get('num_layers', 2),
                    dropout=kwargs.get('dropout', 0.2)
                ).to(self.device)
            elif architecture == 'transformer':
                self.model = TransformerModel(
                    input_size=input_size,
                    d_model=kwargs.get('d_model', 64),
                    nhead=kwargs.get('nhead', 4),
                    num_layers=kwargs.get('num_layers', 2),
                    dropout=kwargs.get('dropout', 0.2)
                ).to(self.device)
            else:
                raise ValueError(f"Unknown architecture: {architecture}")
            
            self.criterion = nn.MSELoss() if model_type == 'regression' else nn.BCEWithLogitsLoss()
            self.optimizer = optim.Adam(self.model.parameters(), lr=kwargs.get('learning_rate', 0.001))
            
        def fit(self, X_train, y_train, X_val=None, y_val=None, epochs=50, batch_size=32):
            logger.info(f"Training {self.architecture} {self.model_type} model...")
            
            # Create datasets
            train_dataset = TimeSeriesDataset(X_train, y_train, self.sequence_length)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            
            if X_val is not None and y_val is not None:
                val_dataset = TimeSeriesDataset(X_val, y_val, self.sequence_length)
                val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            
            # Training loop
            best_val_loss = float('inf')
            patience = 10
            patience_counter = 0
            
            for epoch in range(epochs):
                self.model.train()
                train_loss = 0.0
                
                for X_batch, y_batch in train_loader:
                    X_batch = X_batch.to(self.device)
                    y_batch = y_batch.to(self.device)
                    
                    self.optimizer.zero_grad()
                    outputs = self.model(X_batch)
                    loss = self.criterion(outputs, y_batch)
                    loss.backward()
                    self.optimizer.step()
                    
                    train_loss += loss.item()
                
                train_loss /= len(train_loader)
                
                # Validation
                if X_val is not None and y_val is not None:
                    self.model.eval()
                    val_loss = 0.0
                    
                    with torch.no_grad():
                        for X_batch, y_batch in val_loader:
                            X_batch = X_batch.to(self.device)
                            y_batch = y_batch.to(self.device)
                            outputs = self.model(X_batch)
                            loss = self.criterion(outputs, y_batch)
                            val_loss += loss.item()
                    
                    val_loss /= len(val_loader)
                    
                    if (epoch + 1) % 10 == 0:
                        logger.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
                    
                    # Early stopping
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        if patience_counter >= patience:
                            logger.info(f"Early stopping at epoch {epoch+1}")
                            break
                else:
                    if (epoch + 1) % 10 == 0:
                        logger.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f}")
        
        def predict(self, X):
            self.model.eval()
            
            dataset = TimeSeriesDataset(X, np.zeros(len(X)), self.sequence_length)
            loader = DataLoader(dataset, batch_size=32, shuffle=False)
            
            predictions = []
            with torch.no_grad():
                for X_batch, _ in loader:
                    X_batch = X_batch.to(self.device)
                    outputs = self.model(X_batch)
                    if self.model_type == 'classification':
                        outputs = torch.sigmoid(outputs)
                    predictions.append(outputs.cpu().numpy())
            
            return np.concatenate(predictions).flatten()
        
        def save(self, path: str):
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'architecture': self.architecture,
                'input_size': self.input_size,
                'sequence_length': self.sequence_length
            }, path)
        
        def load(self, path: str):
            checkpoint = torch.load(path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


def create_model(model_name: str, model_type: str = 'regression', config: dict = None, **kwargs):
    """
    Factory function to create models
    
    Args:
        model_name: Name of model (linear, xgboost, lightgbm, lstm, transformer)
        model_type: regression or classification
        config: Configuration dict
        **kwargs: Additional model parameters
        
    Returns:
        Model instance
    """
    if model_name == 'linear':
        return LinearModel(model_type)
    elif model_name == 'xgboost':
        params = config['models']['xgboost'] if config else {}
        params.update(kwargs)
        return XGBoostModel(model_type, **params)
    elif model_name == 'lightgbm':
        params = config['models']['lightgbm'] if config else {}
        params.update(kwargs)
        return LightGBMModel(model_type, **params)
    elif model_name == 'lstm':
        if not HAS_PYTORCH:
            raise ImportError("PyTorch not installed")
        params = config['models']['lstm'] if config else {}
        params.update(kwargs)
        return NeuralNetworkModel(model_type, architecture='lstm', **params)
    elif model_name == 'transformer':
        if not HAS_PYTORCH:
            raise ImportError("PyTorch not installed")
        params = config['models']['transformer'] if config else {}
        params.update(kwargs)
        return NeuralNetworkModel(model_type, architecture='transformer', **params)
    else:
        raise ValueError(f"Unknown model: {model_name}")
