"""
Totals Prediction Model

Predicts total points for over/under betting.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import joblib
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Tuple


class TotalsPredictor:
    """
    Predicts total points (home + away) for over/under betting.
    """
    
    def __init__(
        self,
        model_type: str = "xgboost",
        model_dir: str = "./models"
    ):
        """
        Initialize totals predictor.
        
        Args:
            model_type: "xgboost" or "gbr"
            model_dir: Directory to save models
        """
        self.model_type = model_type
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.model = None
        self.feature_names = []
        self.config = {
            'test_size': 0.2,
            'xgb_params': {
                'n_estimators': 100,
                'max_depth': 5,
                'learning_rate': 0.1
            }
        }
    
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
            self.model = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=4,
                random_state=42
            )
    
    def prepare_data(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        time_based_split: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Prepare and split data for training."""
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
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict:
        """Train the totals prediction model."""
        self._create_model()
        
        print(f"Training {self.model_type} totals model...")
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
        book_totals: pd.Series = None
    ) -> Dict:
        """Evaluate model on test data."""
        predictions = self.model.predict(X_test)
        
        metrics = {
            'mae': mean_absolute_error(y_test, predictions),
            'rmse': np.sqrt(mean_squared_error(y_test, predictions)),
            'r2': r2_score(y_test, predictions)
        }
        
        # Calculate over/under accuracy if book lines provided
        if book_totals is not None:
            model_over = predictions > book_totals
            actual_over = y_test > book_totals
            
            ou_accuracy = (model_over == actual_over).mean()
            metrics['ou_accuracy'] = ou_accuracy
        
        print(f"Test Results:")
        print(f"  MAE: {metrics['mae']:.2f} points")
        print(f"  RMSE: {metrics['rmse']:.2f} points")
        print(f"  R²: {metrics['r2']:.3f}")
        if 'ou_accuracy' in metrics:
            print(f"  Over/Under Accuracy: {metrics['ou_accuracy']:.1%}")
        
        return metrics
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict total points."""
        if self.model is None:
            raise ValueError("Model not trained")
        
        X_features = X[self.feature_names].fillna(0)
        return self.model.predict(X_features)
    
    def recommend_totals_bet(
        self,
        predicted_total: float,
        book_total: float,
        over_odds: int = -110,
        under_odds: int = -110,
        min_edge: float = 3.0
    ) -> Dict:
        """
        Recommend an over/under bet.
        
        Args:
            predicted_total: Model's predicted total
            book_total: Bookmaker's total line
            over_odds: Odds for over bet
            under_odds: Odds for under bet
            min_edge: Minimum edge in points
            
        Returns:
            Recommendation dict
        """
        edge = predicted_total - book_total
        
        if abs(edge) < min_edge:
            return {
                'recommendation': 'PASS',
                'predicted_total': predicted_total,
                'book_total': book_total,
                'edge': edge,
                'reason': 'Edge below threshold'
            }
        
        if edge > 0:
            return {
                'recommendation': 'OVER',
                'predicted_total': predicted_total,
                'book_total': book_total,
                'edge': edge,
                'odds': over_odds,
                'reason': f'Model predicts +{edge:.1f} points vs line'
            }
        else:
            return {
                'recommendation': 'UNDER',
                'predicted_total': predicted_total,
                'book_total': book_total,
                'edge': abs(edge),
                'odds': under_odds,
                'reason': f'Model predicts -{abs(edge):.1f} points vs line'
            }
    
    def save(self, filename: str = None) -> str:
        """Save the trained model."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"totals_{self.model_type}_{timestamp}.joblib"
        
        filepath = self.model_dir / filename
        
        model_data = {
            'model': self.model,
            'model_type': self.model_type,
            'feature_names': self.feature_names,
            'saved_at': datetime.now().isoformat()
        }
        
        joblib.dump(model_data, filepath)
        print(f"Totals model saved: {filepath}")
        return str(filepath)
    
    def load(self, filepath: str):
        """Load a trained model."""
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.model_type = model_data.get('model_type', 'xgboost')
        self.feature_names = model_data['feature_names']
        
        print(f"Loaded totals model: {filepath}")


def main():
    """Demo the totals predictor."""
    print("Totals Predictor Demo")
    print("=" * 50)
    
    np.random.seed(42)
    n_samples = 500
    
    X = pd.DataFrame({
        'home_pts_l5': np.random.normal(110, 10, n_samples),
        'away_pts_l5': np.random.normal(108, 10, n_samples),
        'home_pace': np.random.normal(100, 3, n_samples),
        'away_pace': np.random.normal(99, 3, n_samples),
        'home_off_rating': np.random.normal(112, 5, n_samples),
        'away_off_rating': np.random.normal(110, 5, n_samples),
        'GAME_DATE': pd.date_range('2024-01-01', periods=n_samples)
    })
    
    # Simulated total points
    y = X['home_pts_l5'] + X['away_pts_l5'] + np.random.normal(0, 8, n_samples)
    
    predictor = TotalsPredictor()
    X_train, X_test, y_train, y_test = predictor.prepare_data(X, y)
    
    print(f"\nTraining set: {len(X_train)} games")
    print(f"Test set: {len(X_test)} games")
    
    predictor.train(X_train, y_train)
    
    print()
    predictor.evaluate(X_test, y_test)
    
    print("\nSample Recommendation:")
    pred = predictor.predict(X_test.head(1))[0]
    rec = predictor.recommend_totals_bet(pred, 218.5)
    print(f"  Predicted: {pred:.1f}")
    print(f"  Book line: 218.5")
    print(f"  Recommendation: {rec['recommendation']}")


if __name__ == "__main__":
    main()
