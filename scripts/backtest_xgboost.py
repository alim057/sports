"""
Backtest with Trained XGBoost Models

Re-runs the retroactive analysis using the newly trained XGBoost models
instead of the simple heuristic.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import joblib

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from models.backtester import Backtester


def load_models(model_dir: Path):
    """Load trained XGBoost models."""
    models = {}
    
    for name in ['moneyline', 'spread', 'totals']:
        path = model_dir / f"nba_{name}_xgb.joblib"
        if path.exists():
            data = joblib.load(path)
            models[name] = data
            print(f"Loaded {name} model (trained: {data.get('trained_at', 'unknown')})")
    
    return models


def run_xgboost_backtest(training_data_path: Path, models: dict):
    """Run backtest using XGBoost moneyline model predictions."""
    print("\nLoading training data...")
    df = pd.read_csv(training_data_path)
    df['game_date'] = pd.to_datetime(df['game_date'])
    print(f"Loaded {len(df)} games")
    
    # Get model and features
    ml_model = models['moneyline']['model']
    ml_features = models['moneyline']['features']
    
    # Filter to test set (last 20% of data, same as training split)
    test_start = int(len(df) * 0.8)
    test_df = df.iloc[test_start:].copy()
    print(f"Testing on {len(test_df)} games (last 20% of data)")
    
    # Generate predictions
    print("\nGenerating XGBoost predictions...")
    predictions = []
    
    for idx, row in test_df.iterrows():
        # Build feature vector
        X = pd.DataFrame([{col: row[col] for col in ml_features}])
        
        # Get probability
        home_win_prob = ml_model.predict_proba(X)[0][1]
        
        # Determine predicted winner
        predicted_winner = row['home_team'] if home_win_prob > 0.5 else row['away_team']
        actual_winner = row['home_team'] if row['home_won'] == 1 else row['away_team']
        
        # Simulate odds based on probability
        if home_win_prob > 0.5:
            odds = int(-100 * home_win_prob / (1 - home_win_prob))
            odds = max(-250, min(-110, odds))
        else:
            away_prob = 1 - home_win_prob
            odds = int(100 * (1 - away_prob) / away_prob)
            odds = max(100, min(250, odds))
        
        predictions.append({
            'matchup': f"{row['away_team']} @ {row['home_team']}",
            'game_date': row['game_date'],
            'predicted_winner': predicted_winner,
            'actual_winner': actual_winner,
            'probability': home_win_prob if predicted_winner == row['home_team'] else 1 - home_win_prob,
            'odds': odds
        })
    
    print(f"Generated {len(predictions)} predictions")
    
    # Run backtest
    print("\nRunning backtest...")
    backtester = Backtester(initial_bankroll=1000)
    results = backtester.run_backtest(
        predictions,
        kelly_fraction=0.25,
        min_ev=0.05  # 5% minimum EV (higher threshold)
    )
    
    return results, backtester


def main():
    print("=" * 70)
    print("XGBOOST MODEL BACKTEST")
    print("=" * 70)
    
    # Paths
    model_dir = Path(__file__).parent.parent / "models"
    data_path = Path(__file__).parent.parent / "data" / "training_data.csv"
    
    # Load models
    models = load_models(model_dir)
    
    if 'moneyline' not in models:
        print("ERROR: Moneyline model not found!")
        return
    
    # Run backtest
    results, backtester = run_xgboost_backtest(data_path, models)
    
    # Print report
    print("\n" + backtester.generate_report(results))
    
    # Analyze by confidence
    print("\nPerformance by Confidence Level:")
    conf_analysis = backtester.analyze_by_confidence(results)
    print(conf_analysis.to_string())
    
    # Compare to baseline
    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)
    print(f"XGBoost Win Rate: {results['win_rate']:.1%}")
    print(f"Training Accuracy: {models['moneyline'].get('accuracy', 0):.1%}")
    print(f"Bets Placed: {results['bets_placed']}")
    print(f"ROI: {results['roi']:.1%}")


if __name__ == "__main__":
    main()
