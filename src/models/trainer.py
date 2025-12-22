"""
Model Training Pipeline

Trains ML models for game outcome prediction.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, 
    log_loss, 
    roc_auc_score,
    brier_score_loss,
    classification_report
)
import xgboost as xgb
import joblib
from pathlib import Path
from datetime import datetime
from typing import Tuple, Optional, Dict, Any
import yaml


class ModelTrainer:
    """Trains and evaluates prediction models."""
    
    def __init__(
        self,
        model_type: str = "xgboost",
        model_dir: str = "./models",
        config_path: Optional[str] = None
    ):
        """
        Initialize trainer.
        
        Args:
            model_type: "logistic" or "xgboost"
            model_dir: Directory to save trained models
            config_path: Path to config.yaml for hyperparameters
        """
        self.model_type = model_type
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.config = self._load_config(config_path)
        self.model = None
        self.feature_names = None
        self.training_metrics = {}
    
    def _load_config(self, config_path: Optional[str]) -> dict:
        """Load configuration from YAML file."""
        default_config = {
            'test_size': 0.2,
            'random_state': 42,
            'xgboost': {
                'n_estimators': 100,
                'max_depth': 5,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'objective': 'binary:logistic',
                'eval_metric': 'logloss'
            },
            'logistic': {
                'C': 1.0,
                'max_iter': 1000
            }
        }
        
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                user_config = yaml.safe_load(f)
                if 'model' in user_config:
                    default_config.update(user_config['model'])
        
        return default_config
    
    def _create_model(self):
        """Create the base model."""
        if self.model_type == "xgboost":
            params = self.config.get('xgboost', {})
            return xgb.XGBClassifier(
                n_estimators=params.get('n_estimators', 100),
                max_depth=params.get('max_depth', 5),
                learning_rate=params.get('learning_rate', 0.1),
                subsample=params.get('subsample', 0.8),
                colsample_bytree=params.get('colsample_bytree', 0.8),
                objective='binary:logistic',
                eval_metric='logloss',
                random_state=self.config.get('random_state', 42),
                verbosity=0
            )
        else:
            params = self.config.get('logistic', {})
            return LogisticRegression(
                C=params.get('C', 1.0),
                max_iter=params.get('max_iter', 1000),
                random_state=self.config.get('random_state', 42)
            )
    
    def prepare_data(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        time_based_split: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split data into training and test sets.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            time_based_split: If True, use chronological split (recommended)
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        # Get feature columns only (exclude metadata)
        feature_cols = [c for c in X.columns if c not in ['GAME_ID', 'GAME_DATE']]
        self.feature_names = feature_cols
        
        X_features = X[feature_cols].copy()
        
        # Handle missing values
        X_features = X_features.fillna(0)
        
        test_size = self.config.get('test_size', 0.2)
        
        if time_based_split and 'GAME_DATE' in X.columns:
            # Sort by date and split chronologically
            sorted_idx = X['GAME_DATE'].argsort()
            X_sorted = X_features.iloc[sorted_idx]
            y_sorted = y.iloc[sorted_idx]
            
            split_idx = int(len(X_sorted) * (1 - test_size))
            
            X_train = X_sorted.iloc[:split_idx]
            X_test = X_sorted.iloc[split_idx:]
            y_train = y_sorted.iloc[:split_idx]
            y_test = y_sorted.iloc[split_idx:]
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X_features, y,
                test_size=test_size,
                random_state=self.config.get('random_state', 42)
            )
        
        return X_train, X_test, y_train, y_test
    
    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        calibrate: bool = True
    ):
        """
        Train the model.
        
        Args:
            X_train: Training features
            y_train: Training targets
            calibrate: Whether to calibrate probabilities
        """
        print(f"Training {self.model_type} model on {len(X_train)} samples...")
        
        base_model = self._create_model()
        
        if calibrate:
            # Use calibration for better probability estimates
            self.model = CalibratedClassifierCV(
                base_model,
                method='isotonic',
                cv=5
            )
            self.model.fit(X_train, y_train)
        else:
            base_model.fit(X_train, y_train)
            self.model = base_model
        
        print("Training complete!")
    
    def evaluate(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series
    ) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Args:
            X_test: Test features
            y_test: Test targets
            
        Returns:
            Dictionary of metrics
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        y_pred = self.model.predict(X_test)
        y_prob = self.model.predict_proba(X_test)[:, 1]
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'log_loss': log_loss(y_test, y_prob),
            'roc_auc': roc_auc_score(y_test, y_prob),
            'brier_score': brier_score_loss(y_test, y_prob),
            'n_test_samples': len(y_test)
        }
        
        self.training_metrics = metrics
        
        print("\n" + "="*50)
        print("Model Evaluation Results")
        print("="*50)
        print(f"Accuracy:     {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.1f}%)")
        print(f"ROC AUC:      {metrics['roc_auc']:.4f}")
        print(f"Log Loss:     {metrics['log_loss']:.4f}")
        print(f"Brier Score:  {metrics['brier_score']:.4f}")
        print("="*50)
        
        # Detailed classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, 
                                    target_names=['Away Win', 'Home Win']))
        
        return metrics
    
    def cross_validate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_splits: int = 5
    ) -> Dict[str, float]:
        """
        Perform time-series cross-validation.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            n_splits: Number of CV splits
            
        Returns:
            Dictionary of averaged metrics
        """
        feature_cols = [c for c in X.columns if c not in ['GAME_ID', 'GAME_DATE']]
        X_features = X[feature_cols].fillna(0)
        
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        metrics_list = []
        
        for fold, (train_idx, test_idx) in enumerate(tscv.split(X_features)):
            X_train = X_features.iloc[train_idx]
            X_test = X_features.iloc[test_idx]
            y_train = y.iloc[train_idx]
            y_test = y.iloc[test_idx]
            
            model = self._create_model()
            model.fit(X_train, y_train)
            
            y_prob = model.predict_proba(X_test)[:, 1]
            y_pred = model.predict(X_test)
            
            fold_metrics = {
                'fold': fold + 1,
                'accuracy': accuracy_score(y_test, y_pred),
                'roc_auc': roc_auc_score(y_test, y_prob),
                'brier_score': brier_score_loss(y_test, y_prob)
            }
            metrics_list.append(fold_metrics)
            
            print(f"Fold {fold+1}: Accuracy={fold_metrics['accuracy']:.4f}, "
                  f"AUC={fold_metrics['roc_auc']:.4f}")
        
        # Average metrics
        avg_metrics = {
            'cv_accuracy_mean': np.mean([m['accuracy'] for m in metrics_list]),
            'cv_accuracy_std': np.std([m['accuracy'] for m in metrics_list]),
            'cv_auc_mean': np.mean([m['roc_auc'] for m in metrics_list]),
            'cv_auc_std': np.std([m['roc_auc'] for m in metrics_list]),
        }
        
        print(f"\nCV Accuracy: {avg_metrics['cv_accuracy_mean']:.4f} "
              f"(+/- {avg_metrics['cv_accuracy_std']:.4f})")
        
        return avg_metrics
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Get win probabilities for new games.
        
        Args:
            X: Feature DataFrame for prediction
            
        Returns:
            Array of [away_win_prob, home_win_prob] for each game
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Use only the features the model was trained on
        X_pred = X[self.feature_names].fillna(0)
        return self.model.predict_proba(X_pred)
    
    def save(self, filename: str = None):
        """Save the trained model."""
        if self.model is None:
            raise ValueError("No model to save. Train a model first.")
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.model_type}_{timestamp}.joblib"
        
        filepath = self.model_dir / filename
        
        model_data = {
            'model': self.model,
            'feature_names': self.feature_names,
            'model_type': self.model_type,
            'training_metrics': self.training_metrics,
            'created_at': datetime.now().isoformat()
        }
        
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
        
        return filepath
    
    def load(self, filepath: str):
        """Load a trained model."""
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.feature_names = model_data['feature_names']
        self.model_type = model_data['model_type']
        self.training_metrics = model_data.get('training_metrics', {})
        
        print(f"Loaded {self.model_type} model from {filepath}")
        print(f"Features: {len(self.feature_names)}")


def main():
    """Demo the model trainer."""
    trainer = ModelTrainer(model_type="xgboost")
    
    print("Model Trainer initialized")
    print(f"Model type: {trainer.model_type}")
    print(f"Model directory: {trainer.model_dir}")
    
    # Create synthetic demo data
    print("\nGenerating synthetic training data for demo...")
    np.random.seed(42)
    
    n_samples = 1000
    X = pd.DataFrame({
        'PTS_DIFF_L5': np.random.randn(n_samples) * 5,
        'PTS_DIFF_L10': np.random.randn(n_samples) * 5,
        'REB_DIFF_L5': np.random.randn(n_samples) * 3,
        'REB_DIFF_L10': np.random.randn(n_samples) * 3,
        'AST_DIFF_L5': np.random.randn(n_samples) * 2,
        'AST_DIFF_L10': np.random.randn(n_samples) * 2,
        'HOME_WIN_RATE': np.random.uniform(0.3, 0.7, n_samples),
        'AWAY_WIN_RATE': np.random.uniform(0.3, 0.7, n_samples),
        'WIN_RATE_DIFF': np.random.randn(n_samples) * 0.2,
        'HOME_REST': np.random.randint(1, 5, n_samples),
        'AWAY_REST': np.random.randint(1, 5, n_samples),
        'REST_ADVANTAGE': np.random.randint(-3, 4, n_samples),
        'HOME_STREAK': np.random.randint(-5, 6, n_samples),
        'AWAY_STREAK': np.random.randint(-5, 6, n_samples),
        'STREAK_DIFF': np.random.randint(-8, 9, n_samples),
        'HOME_B2B': np.random.randint(0, 2, n_samples),
        'AWAY_B2B': np.random.randint(0, 2, n_samples),
        'GAME_DATE': pd.date_range('2024-01-01', periods=n_samples, freq='3h')
    })
    
    # Synthetic target with some signal
    home_advantage = 0.05
    y = (
        (X['PTS_DIFF_L5'] > 0).astype(float) * 0.3 +
        (X['HOME_WIN_RATE'] > X['AWAY_WIN_RATE']).astype(float) * 0.3 +
        home_advantage + 
        np.random.randn(n_samples) * 0.2
    ) > 0.5
    y = y.astype(int)
    
    print(f"Home win rate in data: {y.mean():.1%}")
    
    # Split and train
    X_train, X_test, y_train, y_test = trainer.prepare_data(X, pd.Series(y))
    
    print(f"\nTrain size: {len(X_train)}, Test size: {len(X_test)}")
    
    trainer.train(X_train, y_train, calibrate=True)
    metrics = trainer.evaluate(X_test, y_test)
    
    # Save model
    trainer.save("demo_model.joblib")


if __name__ == "__main__":
    main()
