"""
Train Spread & Totals Models

Trains point spread and over/under prediction models for NBA.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd
import numpy as np
from datetime import datetime
import joblib


def fetch_nba_games_for_spread():
    """Fetch historical NBA game data for spread training."""
    try:
        from nba_api.stats.endpoints import LeagueGameFinder
        
        print("Fetching NBA game data...")
        games = LeagueGameFinder(
            season_nullable='2023-24',
            league_id_nullable='00',
            season_type_nullable='Regular Season'
        ).get_data_frames()[0]
        
        print(f"Retrieved {len(games)} team-game records")
        return games
    except Exception as e:
        print(f"Error fetching NBA data: {e}")
        return None


def prepare_spread_data(games_df):
    """Prepare features and target for spread prediction."""
    if games_df is None or games_df.empty:
        return None, None
    
    # Filter to home games only (each game appears twice)
    home_games = games_df[games_df['MATCHUP'].str.contains(' vs. ')].copy()
    
    if len(home_games) < 100:
        print(f"Not enough home games: {len(home_games)}")
        return None, None
    
    print(f"Processing {len(home_games)} home games...")
    
    # Create features
    features = pd.DataFrame({
        'pts': home_games['PTS'],
        'fgm': home_games['FGM'],
        'fg3m': home_games['FG3M'],
        'fta': home_games['FTA'],
        'reb': home_games['REB'],
        'ast': home_games['AST'],
        'tov': home_games['TOV'],
    })
    
    # Target: Point differential (positive = home win)
    target = home_games['PLUS_MINUS']
    
    return features, target


def train_spread_model(X, y):
    """Train XGBoost regression model for spread prediction."""
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_error
    import xgboost as xgb
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"Training on {len(X_train)} games, testing on {len(X_test)}...")
    
    model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    train_mae = mean_absolute_error(y_train, train_pred)
    test_mae = mean_absolute_error(y_test, test_pred)
    
    print(f"Train MAE: {train_mae:.2f} points")
    print(f"Test MAE: {test_mae:.2f} points")
    
    return model, X.columns.tolist(), test_mae


def train_totals_model(X, y_pts):
    """Train model for over/under prediction."""
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_error
    import xgboost as xgb
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_pts, test_size=0.2, random_state=42
    )
    
    print(f"\nTraining Totals model on {len(X_train)} games...")
    
    model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    test_pred = model.predict(X_test)
    test_mae = mean_absolute_error(y_test, test_pred)
    
    print(f"Totals Test MAE: {test_mae:.2f} points")
    
    return model, X.columns.tolist(), test_mae


def main():
    print("=" * 60)
    print("SPREAD & TOTALS MODEL TRAINING")
    print("=" * 60)
    
    # Fetch data
    games = fetch_nba_games_for_spread()
    
    if games is None:
        print("Using synthetic data for demo...")
        np.random.seed(42)
        n = 500
        X = pd.DataFrame({
            'pts': np.random.normal(110, 15, n),
            'fgm': np.random.normal(40, 5, n),
            'fg3m': np.random.normal(12, 4, n),
            'fta': np.random.normal(22, 5, n),
            'reb': np.random.normal(44, 6, n),
            'ast': np.random.normal(25, 5, n),
            'tov': np.random.normal(14, 3, n),
        })
        y_spread = np.random.normal(0, 12, n)  # Point differential
        y_totals = X['pts'] * 2 + np.random.normal(0, 10, n)  # Total pts
    else:
        X, y_spread = prepare_spread_data(games)
        if X is None:
            print("Could not prepare data")
            return
        y_totals = X['pts'] * 2  # Approximate total
    
    # Train Spread Model
    print("\n--- SPREAD MODEL ---")
    spread_model, spread_features, spread_mae = train_spread_model(X, y_spread)
    
    # Train Totals Model
    print("\n--- TOTALS MODEL ---")
    totals_model, totals_features, totals_mae = train_totals_model(X, y_totals)
    
    # Save models
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    spread_path = models_dir / "nba_spread_model.joblib"
    joblib.dump({
        'model': spread_model,
        'features': spread_features,
        'mae': spread_mae,
        'trained_at': datetime.now().isoformat()
    }, spread_path)
    print(f"\nSaved: {spread_path}")
    
    totals_path = models_dir / "nba_totals_model.joblib"
    joblib.dump({
        'model': totals_model,
        'features': totals_features,
        'mae': totals_mae,
        'trained_at': datetime.now().isoformat()
    }, totals_path)
    print(f"Saved: {totals_path}")
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print(f"Spread MAE: {spread_mae:.2f} pts | Totals MAE: {totals_mae:.2f} pts")
    print("=" * 60)


if __name__ == "__main__":
    main()
