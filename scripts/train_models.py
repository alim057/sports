"""
Multi-Sport Model Trainer

Trains XGBoost models for NBA, NFL, NCAAF, and MLB using collected training data.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import json
import argparse
import joblib

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report


class MultiSportTrainer:
    """Train prediction models for multiple sports."""
    
    SPORT_FEATURES = {
        'nba': [
            'home_pts_l10', 'away_pts_l10',
            'home_opp_pts_l10', 'away_opp_pts_l10',
            'home_fg_pct_l10', 'away_fg_pct_l10',
            'home_reb_l10', 'away_reb_l10',
            'home_ast_l10', 'away_ast_l10',
            'home_win_l10', 'away_win_l10',
            # Enhanced features
            'home_injury_impact', 'away_injury_impact', 'injury_advantage',
        ],
        'nfl': [
            'home_pts_l5', 'away_pts_l5',
            'home_opp_pts_l5', 'away_opp_pts_l5',
            # Enhanced weather features
            'is_dome', 'temp_f', 'wind_mph', 'is_cold', 'is_windy',
        ],
        'ncaaf': [
            'home_pts_l5', 'away_pts_l5',
            'home_margin_l5', 'away_margin_l5',
            # Enhanced features
            'home_strength', 'away_strength', 'strength_diff',
            'temp_f', 'wind_mph',
        ],
        'mlb': [
            'home_runs_l10', 'away_runs_l10',
            # Enhanced pitching features
            'home_team_era', 'away_team_era', 'era_advantage',
        ]
    }
    
    def __init__(self):
        self.data_dir = project_root / "data"
        self.models_dir = project_root / "models"
        self.models_dir.mkdir(exist_ok=True)
        
    def load_training_data(self, sport: str) -> pd.DataFrame:
        """Load training data for a sport."""
        # Try enhanced data first
        enhanced_path = self.data_dir / f"{sport}_training_enhanced.csv"
        regular_path = self.data_dir / f"{sport}_training_data.csv"
        
        if enhanced_path.exists():
            path = enhanced_path
            print(f"Using ENHANCED data for {sport.upper()}")
        elif regular_path.exists():
            path = regular_path
            print(f"Using regular data for {sport.upper()}")
        else:
            print(f"Error: Training data not found for {sport}")
            return pd.DataFrame()
            
        df = pd.read_csv(path)
        print(f"Loaded {len(df)} records for {sport.upper()}")
        return df
    
    def train_model(self, sport: str) -> dict:
        """Train model for a sport and return metrics."""
        print(f"\n{'='*60}")
        print(f"Training {sport.upper()} model...")
        print(f"{'='*60}")
        
        df = self.load_training_data(sport)
        if df.empty:
            return {'error': 'No training data'}
        
        # Get features for this sport
        features = self.SPORT_FEATURES.get(sport, [])
        
        # Filter to available features
        available_features = [f for f in features if f in df.columns]
        missing_features = [f for f in features if f not in df.columns]
        
        if missing_features:
            print(f"  Warning: Missing features: {missing_features}")
        
        if not available_features:
            print(f"  Error: No features available for training")
            return {'error': 'No features available'}
        
        print(f"  Using features: {available_features}")
        
        # Prepare data
        X = df[available_features].fillna(0)
        y = df['home_won']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        print(f"  Training samples: {len(X_train)}")
        print(f"  Test samples: {len(X_test)}")
        
        # Train XGBoost model
        model = XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss'
        )
        
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Cross-validation
        cv_scores = cross_val_score(model, X, y, cv=5)
        
        print(f"\n  Results:")
        print(f"    Test Accuracy: {accuracy:.4f}")
        print(f"    CV Mean: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
        
        # Feature importance
        importance = {k: float(v) for k, v in zip(available_features, model.feature_importances_)}
        sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        
        print(f"\n  Feature Importance:")
        for feat, imp in sorted_importance[:5]:
            print(f"    {feat}: {imp:.4f}")
        
        # Save model
        model_path = self.models_dir / f"{sport}_moneyline_v2.joblib"
        joblib.dump({
            'model': model,
            'features': available_features,
            'accuracy': accuracy,
            'cv_mean': cv_scores.mean(),
            'trained_at': datetime.now().isoformat()
        }, model_path)
        
        print(f"\n  Saved model to {model_path}")
        
        # Return metrics
        return {
            'sport': sport.upper(),
            'games_trained': len(df),
            'features_used': available_features,
            'test_accuracy': round(accuracy, 4),
            'cv_mean': round(cv_scores.mean(), 4),
            'cv_std': round(cv_scores.std(), 4),
            'feature_importance': dict(sorted_importance),
            'model_path': str(model_path)
        }
    
    def train_all(self) -> dict:
        """Train models for all sports."""
        results = {}
        
        for sport in ['nba', 'nfl', 'ncaaf', 'mlb']:
            try:
                results[sport] = self.train_model(sport)
            except Exception as e:
                print(f"Error training {sport}: {e}")
                results[sport] = {'error': str(e)}
        
        return results
    
    def generate_report(self, results: dict):
        """Generate training report."""
        report_path = self.data_dir / "model_training_report.json"
        
        report = {
            'generated_at': datetime.now().isoformat(),
            'models': results,
            'summary': {
                'total_sports': len(results),
                'successful': sum(1 for r in results.values() if 'error' not in r),
                'avg_accuracy': np.mean([
                    r.get('test_accuracy', 0) 
                    for r in results.values() 
                    if 'test_accuracy' in r
                ])
            }
        }
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n\nSaved training report to {report_path}")
        
        return report


def main():
    parser = argparse.ArgumentParser(description='Train multi-sport prediction models')
    parser.add_argument('--sport', type=str, default='all',
                       choices=['all', 'nba', 'nfl', 'ncaaf', 'mlb'],
                       help='Sport to train model for')
    
    args = parser.parse_args()
    
    trainer = MultiSportTrainer()
    
    print("=" * 60)
    print("Multi-Sport Model Trainer")
    print("=" * 60)
    
    if args.sport == 'all':
        results = trainer.train_all()
    else:
        results = {args.sport: trainer.train_model(args.sport)}
    
    report = trainer.generate_report(results)
    
    # Print summary
    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    
    for sport, metrics in results.items():
        if 'error' in metrics:
            print(f"  {sport.upper()}: ERROR - {metrics['error']}")
        else:
            print(f"  {sport.upper()}: Accuracy={metrics.get('test_accuracy', 0):.1%}, "
                  f"Games={metrics.get('games_trained', 0)}")
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
