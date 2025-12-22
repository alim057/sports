"""
Spread Prediction Model

Predicts point differentials for spread betting.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import joblib
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Tuple
import yaml


class SpreadPredictor:
    """
    Predicts point differential for spread betting.
    
    Positive value = home team favored
    Negative value = away team favored
    """
    
    def __init__(
        self,
        model_type: str = "xgboost",
        model_dir: str = "./models",
        config_path: Optional[str] = None
    ):
        """
        Initialize spread predictor.
        
        Args:
            model_type: "ridge" or "xgboost"
            model_dir: Directory to save models
            config_path: Path to config.yaml
        """
        self.model_type = model_type
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.model = None
        self.feature_names = []
        self.config = self._load_config(config_path)
    
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load configuration."""
        default_config = {
            'test_size': 0.2,
            'xgb_params': {
                'n_estimators': 100,
                'max_depth': 5,
                'learning_rate': 0.1,
                'reg_alpha': 0.1,
                'reg_lambda': 1.0
            }
        }
        
        if config_path and Path(config_path).exists():
            try:
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                    return {**default_config, **config.get('spread_model', {})}
            except:
                pass
        
        return default_config
    
    def _create_model(self):
        """Create the regression model."""
        if self.model_type == "xgboost":
            params = self.config.get('xgb_params', {})
            self.model = xgb.XGBRegressor(
                objective='reg:squarederror',
                random_state=42,
                **params
            )
        else:
            self.model = Ridge(alpha=1.0)
    
    def prepare_data(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        time_based_split: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Prepare and split data for training.
        
        Args:
            X: Features DataFrame
            y: Target (point differential)
            time_based_split: Use chronological split
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        feature_cols = [c for c in X.columns if c not in ['GAME_ID', 'GAME_DATE']]
        self.feature_names = feature_cols
        
        X_features = X[feature_cols].fillna(0)
        test_size = self.config.get('test_size', 0.2)
        
        if time_based_split and 'GAME_DATE' in X.columns:
            sorted_idx = X['GAME_DATE'].argsort()
            X_sorted = X_features.iloc[sorted_idx]
            y_sorted = y.iloc[sorted_idx]
            
            split_idx = int(len(X_sorted) * (1 - test_size))
            
            return (
                X_sorted.iloc[:split_idx],
                X_sorted.iloc[split_idx:],
                y_sorted.iloc[:split_idx],
                y_sorted.iloc[split_idx:]
            )
        else:
            return train_test_split(
                X_features, y,
                test_size=test_size,
                random_state=42
            )
    
    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series
    ) -> Dict:
        """
        Train the spread prediction model.
        
        Args:
            X_train: Training features
            y_train: Training targets (point differential)
            
        Returns:
            Training metrics
        """
        self._create_model()
        
        print(f"Training {self.model_type} spread model...")
        print(f"  Features: {len(self.feature_names)}")
        print(f"  Training samples: {len(X_train)}")
        
        self.model.fit(X_train, y_train)
        
        train_pred = self.model.predict(X_train)
        
        metrics = {
            'mae': mean_absolute_error(y_train, train_pred),
            'rmse': np.sqrt(mean_squared_error(y_train, train_pred)),
            'r2': r2_score(y_train, train_pred)
        }
        
        print(f"  Train MAE: {metrics['mae']:.2f} points")
        print(f"  Train RMSE: {metrics['rmse']:.2f} points")
        
        return metrics
    
    def evaluate(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        book_spreads: pd.Series = None
    ) -> Dict:
        """
        Evaluate model on test data.
        
        Args:
            X_test: Test features
            y_test: Actual point differentials
            book_spreads: Bookmaker spread lines (optional)
            
        Returns:
            Evaluation metrics
        """
        predictions = self.model.predict(X_test)
        
        metrics = {
            'mae': mean_absolute_error(y_test, predictions),
            'rmse': np.sqrt(mean_squared_error(y_test, predictions)),
            'r2': r2_score(y_test, predictions)
        }
        
        # Calculate spread betting accuracy if book lines provided
        if book_spreads is not None:
            # Did we correctly pick the ATS (against the spread) winner?
            model_pick_home = predictions > -book_spreads
            actual_covered_home = y_test > -book_spreads
            
            spread_accuracy = (model_pick_home == actual_covered_home).mean()
            metrics['spread_accuracy'] = spread_accuracy
        
        print(f"Test Results:")
        print(f"  MAE: {metrics['mae']:.2f} points")
        print(f"  RMSE: {metrics['rmse']:.2f} points")
        print(f"  R²: {metrics['r2']:.3f}")
        if 'spread_accuracy' in metrics:
            print(f"  Spread Betting Accuracy: {metrics['spread_accuracy']:.1%}")
        
        return metrics
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict point differential.
        
        Positive = home team expected to win by X points
        Negative = away team expected to win by X points
        
        Args:
            X: Feature DataFrame
            
        Returns:
            Predicted point differentials
        """
        if self.model is None:
            raise ValueError("Model not trained")
        
        X_features = X[self.feature_names].fillna(0)
        return self.model.predict(X_features)
    
    def recommend_spread_bet(
        self,
        predicted_diff: float,
        home_spread: float,
        home_odds: int = -110,
        away_odds: int = -110,
        min_edge: float = 2.0
    ) -> Dict:
        """
        Recommend a spread bet based on model prediction.
        
        Args:
            predicted_diff: Model's predicted point differential
            home_spread: Bookmaker's home spread (negative = favorite)
            home_odds: Odds for home spread bet
            away_odds: Odds for away spread bet
            min_edge: Minimum edge in points to recommend bet
            
        Returns:
            Recommendation dict
        """
        # Edge = how much model differs from book spread
        edge = predicted_diff - (-home_spread)
        
        if abs(edge) < min_edge:
            return {
                'recommendation': 'PASS',
                'predicted_diff': predicted_diff,
                'book_spread': home_spread,
                'edge': edge,
                'reason': 'Edge below threshold'
            }
        
        if edge > 0:
            # Model predicts home covers more than spread
            return {
                'recommendation': 'HOME',
                'predicted_diff': predicted_diff,
                'book_spread': home_spread,
                'edge': edge,
                'odds': home_odds,
                'reason': f'Model expects home +{edge:.1f} vs spread'
            }
        else:
            return {
                'recommendation': 'AWAY',
                'predicted_diff': predicted_diff,
                'book_spread': home_spread,
                'edge': abs(edge),
                'odds': away_odds,
                'reason': f'Model expects away +{abs(edge):.1f} vs spread'
            }
    
    def save(self, filename: str = None) -> str:
        """Save the trained model."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"spread_{self.model_type}_{timestamp}.joblib"
        
        filepath = self.model_dir / filename
        
        model_data = {
            'model': self.model,
            'model_type': self.model_type,
            'feature_names': self.feature_names,
            'config': self.config,
            'saved_at': datetime.now().isoformat()
        }
        
        joblib.dump(model_data, filepath)
        print(f"Spread model saved: {filepath}")
        return str(filepath)
    
    def load(self, filepath: str):
        """Load a trained model."""
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.model_type = model_data.get('model_type', 'xgboost')
        self.feature_names = model_data['feature_names']
        self.config = model_data.get('config', {})
        
        print(f"Loaded spread model: {filepath}")


def main():
    """Demo the spread predictor."""
    print("Spread Predictor Demo")
    print("=" * 50)
    
    # Create sample data
    np.random.seed(42)
    n_samples = 500
    
    X = pd.DataFrame({
        'home_pts_l5': np.random.normal(110, 10, n_samples),
        'away_pts_l5': np.random.normal(108, 10, n_samples),
        'home_def_rating': np.random.normal(110, 5, n_samples),
        'away_def_rating': np.random.normal(112, 5, n_samples),
        'home_pace': np.random.normal(100, 3, n_samples),
        'GAME_DATE': pd.date_range('2024-01-01', periods=n_samples)
    })
    
    # Simulated point differential (home - away)
    y = (X['home_pts_l5'] - X['away_pts_l5']) + np.random.normal(0, 5, n_samples)
    
    predictor = SpreadPredictor()
    X_train, X_test, y_train, y_test = predictor.prepare_data(X, y)
    
    print(f"\nTraining set: {len(X_train)} games")
    print(f"Test set: {len(X_test)} games")
    
    predictor.train(X_train, y_train)
    
    print()
    predictor.evaluate(X_test, y_test)
    
    # Test recommendation
    print("\nSample Recommendation:")
    pred_diff = predictor.predict(X_test.head(1))[0]
    rec = predictor.recommend_spread_bet(pred_diff, home_spread=-5.5)
    print(f"  Predicted diff: {pred_diff:+.1f}")
    print(f"  Book spread: -5.5")
    print(f"  Recommendation: {rec['recommendation']}")
    print(f"  Edge: {rec['edge']:.1f} points")


if __name__ == "__main__":
    main()
