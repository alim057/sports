"""
XGBoost Model Training Pipeline

Trains XGBoost models for:
1. Moneyline (Win/Loss prediction)
2. Spread (Point differential prediction)
3. Totals (Over/Under prediction)

Uses historical NBA data with rolling features.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import joblib

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, mean_absolute_error

from data.nba_fetcher import NBAFetcher


def fetch_training_data(seasons: list = None) -> pd.DataFrame:
    """Fetch and combine multiple seasons of NBA data."""
    if seasons is None:
        seasons = ["2022-23", "2023-24"]
    
    print(f"Fetching data for seasons: {seasons}")
    fetcher = NBAFetcher()
    
    all_games = []
    for season in seasons:
        print(f"  Fetching {season}...")
        games = fetcher.get_all_games_for_season(season)
        games['SEASON'] = season
        all_games.append(games)
    
    combined = pd.concat(all_games, ignore_index=True)
    print(f"Total game records: {len(combined)}")
    return combined


def build_training_dataset(games: pd.DataFrame) -> pd.DataFrame:
    """Transform game records into training rows with features and targets."""
    print("\nBuilding training dataset...")
    
    # Group by GAME_ID to get matchups
    matchups = []
    game_ids = games['GAME_ID'].unique()
    
    for gid in game_ids:
        game_rows = games[games['GAME_ID'] == gid]
        
        if len(game_rows) != 2:
            continue
        
        home_mask = game_rows['MATCHUP'].str.contains('vs.')
        away_mask = game_rows['MATCHUP'].str.contains('@')
        
        if home_mask.sum() == 0 or away_mask.sum() == 0:
            continue
            
        home_row = game_rows[home_mask].iloc[0]
        away_row = game_rows[away_mask].iloc[0]
        
        matchups.append({
            'game_id': gid,
            'game_date': home_row['GAME_DATE'],
            'season': home_row['SEASON'],
            'home_team': home_row['TEAM_ABBREVIATION'],
            'away_team': away_row['TEAM_ABBREVIATION'],
            'home_pts': home_row['PTS'],
            'away_pts': away_row['PTS'],
            'total_pts': home_row['PTS'] + away_row['PTS'],
            'spread': home_row['PTS'] - away_row['PTS'],  # Positive = home wins
            'home_won': 1 if home_row['WL'] == 'W' else 0,
            # Basic game stats
            'home_fg_pct': home_row.get('FG_PCT', 0.45),
            'away_fg_pct': away_row.get('FG_PCT', 0.45),
            'home_fg3_pct': home_row.get('FG3_PCT', 0.35),
            'away_fg3_pct': away_row.get('FG3_PCT', 0.35),
            'home_reb': home_row.get('REB', 40),
            'away_reb': away_row.get('REB', 40),
            'home_ast': home_row.get('AST', 20),
            'away_ast': away_row.get('AST', 20),
            'home_tov': home_row.get('TOV', 12),
            'away_tov': away_row.get('TOV', 12),
        })
    
    df = pd.DataFrame(matchups)
    df = df.sort_values('game_date').reset_index(drop=True)
    print(f"Created {len(df)} matchup records")
    
    return df


def add_rolling_features(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """Add rolling historical features for each team."""
    print(f"\nAdding rolling {window}-game features...")
    
    teams = set(df['home_team'].unique()) | set(df['away_team'].unique())
    team_history = {t: [] for t in teams}
    
    # First pass: build game history
    for idx, row in df.iterrows():
        home, away = row['home_team'], row['away_team']
        
        team_history[home].append({
            'idx': idx,
            'pts': row['home_pts'],
            'opp_pts': row['away_pts'],
            'won': row['home_won'],
            'fg_pct': row['home_fg_pct'],
            'reb': row['home_reb'],
            'ast': row['home_ast'],
            'tov': row['home_tov']
        })
        team_history[away].append({
            'idx': idx,
            'pts': row['away_pts'],
            'opp_pts': row['home_pts'],
            'won': 1 - row['home_won'],
            'fg_pct': row['away_fg_pct'],
            'reb': row['away_reb'],
            'ast': row['away_ast'],
            'tov': row['away_tov']
        })
    
    # Second pass: calculate rolling features
    features = {
        'home_pts_l10': [], 'away_pts_l10': [],
        'home_opp_pts_l10': [], 'away_opp_pts_l10': [],
        'home_win_l10': [], 'away_win_l10': [],
        'home_fg_pct_l10': [], 'away_fg_pct_l10': [],
        'home_reb_l10': [], 'away_reb_l10': [],
        'home_ast_l10': [], 'away_ast_l10': [],
        'home_tov_l10': [], 'away_tov_l10': [],
    }
    
    for idx, row in df.iterrows():
        home, away = row['home_team'], row['away_team']
        
        home_prior = [g for g in team_history[home] if g['idx'] < idx][-window:]
        away_prior = [g for g in team_history[away] if g['idx'] < idx][-window:]
        
        # Home features
        features['home_pts_l10'].append(np.mean([g['pts'] for g in home_prior]) if home_prior else 105)
        features['home_opp_pts_l10'].append(np.mean([g['opp_pts'] for g in home_prior]) if home_prior else 105)
        features['home_win_l10'].append(np.mean([g['won'] for g in home_prior]) if home_prior else 0.5)
        features['home_fg_pct_l10'].append(np.mean([g['fg_pct'] for g in home_prior]) if home_prior else 0.45)
        features['home_reb_l10'].append(np.mean([g['reb'] for g in home_prior]) if home_prior else 40)
        features['home_ast_l10'].append(np.mean([g['ast'] for g in home_prior]) if home_prior else 22)
        features['home_tov_l10'].append(np.mean([g['tov'] for g in home_prior]) if home_prior else 12)
        
        # Away features
        features['away_pts_l10'].append(np.mean([g['pts'] for g in away_prior]) if away_prior else 105)
        features['away_opp_pts_l10'].append(np.mean([g['opp_pts'] for g in away_prior]) if away_prior else 105)
        features['away_win_l10'].append(np.mean([g['won'] for g in away_prior]) if away_prior else 0.5)
        features['away_fg_pct_l10'].append(np.mean([g['fg_pct'] for g in away_prior]) if away_prior else 0.45)
        features['away_reb_l10'].append(np.mean([g['reb'] for g in away_prior]) if away_prior else 40)
        features['away_ast_l10'].append(np.mean([g['ast'] for g in away_prior]) if away_prior else 22)
        features['away_tov_l10'].append(np.mean([g['tov'] for g in away_prior]) if away_prior else 12)
    
    for col, vals in features.items():
        df[col] = vals
    
    # Derived features (differences)
    df['pts_diff_l10'] = df['home_pts_l10'] - df['away_pts_l10']
    df['def_diff_l10'] = df['away_opp_pts_l10'] - df['home_opp_pts_l10']  # Lower opp pts = better D
    df['win_rate_diff'] = df['home_win_l10'] - df['away_win_l10']
    df['fg_pct_diff'] = df['home_fg_pct_l10'] - df['away_fg_pct_l10']
    df['reb_diff'] = df['home_reb_l10'] - df['away_reb_l10']
    df['ast_diff'] = df['home_ast_l10'] - df['away_ast_l10']
    df['tov_diff'] = df['home_tov_l10'] - df['away_tov_l10']  # Higher is worse
    
    # Totals prediction features
    df['combined_pts_l10'] = df['home_pts_l10'] + df['away_pts_l10']
    df['combined_def_l10'] = df['home_opp_pts_l10'] + df['away_opp_pts_l10']
    
    return df


def train_moneyline_model(df: pd.DataFrame, model_dir: Path):
    """Train XGBoost classifier for win/loss prediction."""
    print("\n" + "=" * 60)
    print("TRAINING MONEYLINE MODEL (Win/Loss)")
    print("=" * 60)
    
    # Filter to games with sufficient history
    train_df = df[df.index >= 100].copy()
    
    feature_cols = [
        'pts_diff_l10', 'def_diff_l10', 'win_rate_diff',
        'fg_pct_diff', 'reb_diff', 'ast_diff', 'tov_diff',
        'home_win_l10', 'away_win_l10'
    ]
    
    X = train_df[feature_cols]
    y = train_df['home_won']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=False  # Time-based split
    )
    
    model = XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nTest Accuracy: {accuracy:.1%}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Away Win', 'Home Win']))
    
    # Feature importance
    print("\nFeature Importance:")
    for feat, imp in sorted(zip(feature_cols, model.feature_importances_), key=lambda x: -x[1]):
        print(f"  {feat}: {imp:.3f}")
    
    # Save model
    model_path = model_dir / "nba_moneyline_xgb.joblib"
    joblib.dump({
        'model': model,
        'features': feature_cols,
        'accuracy': accuracy,
        'trained_at': datetime.now().isoformat()
    }, model_path)
    print(f"\nModel saved to: {model_path}")
    
    return model, feature_cols


def train_spread_model(df: pd.DataFrame, model_dir: Path):
    """Train XGBoost regressor for point spread prediction."""
    print("\n" + "=" * 60)
    print("TRAINING SPREAD MODEL (Point Differential)")
    print("=" * 60)
    
    train_df = df[df.index >= 100].copy()
    
    feature_cols = [
        'pts_diff_l10', 'def_diff_l10', 'win_rate_diff',
        'fg_pct_diff', 'reb_diff', 'ast_diff', 'tov_diff',
        'home_pts_l10', 'away_pts_l10',
        'home_opp_pts_l10', 'away_opp_pts_l10'
    ]
    
    X = train_df[feature_cols]
    y = train_df['spread']  # Home pts - Away pts
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=False
    )
    
    model = XGBRegressor(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    
    print(f"\nTest MAE: {mae:.2f} points")
    print(f"Avg actual spread: {y_test.mean():.2f}")
    print(f"Spread std: {y_test.std():.2f}")
    
    # Win prediction accuracy (did we predict correct side?)
    correct_side = ((y_pred > 0) == (y_test > 0)).mean()
    print(f"Correct side prediction: {correct_side:.1%}")
    
    # Feature importance
    print("\nFeature Importance:")
    for feat, imp in sorted(zip(feature_cols, model.feature_importances_), key=lambda x: -x[1]):
        print(f"  {feat}: {imp:.3f}")
    
    # Save model
    model_path = model_dir / "nba_spread_xgb.joblib"
    joblib.dump({
        'model': model,
        'features': feature_cols,
        'mae': mae,
        'trained_at': datetime.now().isoformat()
    }, model_path)
    print(f"\nModel saved to: {model_path}")
    
    return model, feature_cols


def train_totals_model(df: pd.DataFrame, model_dir: Path):
    """Train XGBoost regressor for total points prediction."""
    print("\n" + "=" * 60)
    print("TRAINING TOTALS MODEL (Over/Under)")
    print("=" * 60)
    
    train_df = df[df.index >= 100].copy()
    
    feature_cols = [
        'combined_pts_l10', 'combined_def_l10',
        'home_pts_l10', 'away_pts_l10',
        'home_opp_pts_l10', 'away_opp_pts_l10',
        'fg_pct_diff', 'home_fg_pct_l10', 'away_fg_pct_l10',
        'ast_diff', 'tov_diff'
    ]
    
    X = train_df[feature_cols]
    y = train_df['total_pts']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=False
    )
    
    model = XGBRegressor(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    
    print(f"\nTest MAE: {mae:.2f} points")
    print(f"Avg actual total: {y_test.mean():.1f}")
    print(f"Total pts std: {y_test.std():.1f}")
    
    # Typical O/U lines are around 220-230, so if MAE < 15, we're doing okay
    # Over/Under accuracy (using a common line of 225)
    test_line = 225
    correct_ou = ((y_pred > test_line) == (y_test > test_line)).mean()
    print(f"O/U accuracy (vs {test_line} line): {correct_ou:.1%}")
    
    # Feature importance
    print("\nFeature Importance:")
    for feat, imp in sorted(zip(feature_cols, model.feature_importances_), key=lambda x: -x[1]):
        print(f"  {feat}: {imp:.3f}")
    
    # Save model
    model_path = model_dir / "nba_totals_xgb.joblib"
    joblib.dump({
        'model': model,
        'features': feature_cols,
        'mae': mae,
        'trained_at': datetime.now().isoformat()
    }, model_path)
    print(f"\nModel saved to: {model_path}")
    
    return model, feature_cols


def main():
    print("=" * 70)
    print("XGBOOST MODEL TRAINING PIPELINE")
    print("=" * 70)
    
    # Setup
    model_dir = Path(__file__).parent.parent / "models"
    model_dir.mkdir(exist_ok=True)
    
    # Fetch data
    games = fetch_training_data(["2022-23", "2023-24"])
    
    # Build dataset
    matchups = build_training_dataset(games)
    matchups = add_rolling_features(matchups)
    
    # Save training data for future reference
    data_path = Path(__file__).parent.parent / "data" / "training_data.csv"
    matchups.to_csv(data_path, index=False)
    print(f"\nTraining data saved to: {data_path}")
    
    # Train models
    train_moneyline_model(matchups, model_dir)
    train_spread_model(matchups, model_dir)
    train_totals_model(matchups, model_dir)
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
