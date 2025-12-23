
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_absolute_error, classification_report
import joblib
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.sport_fetcher import NFLFetcher
from features.nfl_features import NFLFeatureEngine

def prepare_training_data():
    print("Fetching historical NFL data...")
    fetcher = NFLFetcher()
    
    all_games = []
    # Training on 2021, 2022. Testing on 2023.
    # Note: HabitatRing data goes back far. Let's fetch a few recent years.
    seasons = [2018, 2019, 2020, 2021, 2022, 2023]
    
    for season in seasons:
        print(f"  Fetching {season}...")
        games = fetcher._fetch_season_games(str(season))
        if not games.empty:
            games['season'] = season
            all_games.append(games)
    
    if not all_games:
        print("Error: No data fetched.")
        return None
        
    df = pd.concat(all_games, ignore_index=True)
    print(f"Total games fetched: {len(df)}")
    
    # Sort by date/id to ensure chronological order for rolling stats
    # HabitatRing data doesn't have a clean date column always, but 'game_id' usually has year_week
    # Or 'gameday' if available. 
    # Our fallback fetcher might return different columns than nfl_data_py?
    # Let's inspect columns from our verification script output:
    # game_id, home_team, away_team, home_score, away_score
    # We should add a 'gameday' or 'week' if possible. 
    # The habitatring csv has 'gameday'.
    
    if 'gameday' in df.columns:
        df['gameday'] = pd.to_datetime(df['gameday'])
        df = df.sort_values('gameday')
    
    return df

def build_features(df):
    print("Building features...")
    # We need to transform the single-row-per-game into team-level stats
    # For each game, we need the rolling stats of Home and Away teams *prior* to that game.
    
    # Helper to calculate rolling stats
    # We'll stick to simple logic matching our NFLFeatureEngine/AdvancedPredictor logic
    
    team_stats = {} # team -> list of game results
    
    features = []
    targets_ml = [] # 1 if Home Win
    targets_spread = [] # Home Score - Away Score
    targets_total = [] # Home + Away Score
    
    for _, row in df.iterrows():
        home = row['home_team']
        away = row['away_team']
        h_score = row['home_score']
        a_score = row['away_score']
        
        # Skip if scores are missing
        if pd.isna(h_score) or pd.isna(a_score):
            continue
            
        # 1. Build Features for this game based on PAST history
        
        def get_team_rolling(team_id):
            history = team_stats.get(team_id, [])
            if len(history) < 3: return None # Need some history
            
            # History is list of (pts_scored, pts_allowed, win)
            df_hist = pd.DataFrame(history, columns=['pf', 'pa', 'win'])
            
            last_5 = df_hist.tail(5)
            last_10 = df_hist.tail(10)
            
            return {
                'WIN_RATE': df_hist['win'].mean(),
                'PTS_L5': last_5['pf'].mean(),
                'PTS_L10': last_10['pf'].mean(),
                'PA_L5': last_5['pa'].mean(),
                'PA_L10': last_10['pa'].mean()
            }

        h_stats = get_team_rolling(home)
        a_stats = get_team_rolling(away)
        
        if h_stats and a_stats:
            # Create feature dict matching our model input needs
            # Logic: We define the features here. AdvancedPredictor must match.
            feat = {
                'HOME_WIN_RATE': h_stats['WIN_RATE'],
                'AWAY_WIN_RATE': a_stats['WIN_RATE'],
                'HOME_PTS_L5': h_stats['PTS_L5'],
                'AWAY_PTS_L5': a_stats['PTS_L5'],
                'HOME_PA_L5': h_stats['PA_L5'],
                'AWAY_PA_L5': a_stats['PA_L5'],
                'PTS_DIFF_L5': h_stats['PTS_L5'] - a_stats['PTS_L5'],
                'PA_DIFF_L5': h_stats['PA_L5'] - a_stats['PA_L5']
            }
            features.append(feat)
            
            # Targets
            targets_ml.append(1 if h_score > a_score else 0)
            targets_spread.append(h_score - a_score)
            targets_total.append(h_score + a_score)
        
        # 2. Update History AFTER the game
        h_win = 1 if h_score > a_score else 0
        a_win = 1 if a_score > h_score else 0
        
        if home not in team_stats: team_stats[home] = []
        if away not in team_stats: team_stats[away] = []
        
        team_stats[home].append((h_score, a_score, h_win))
        team_stats[away].append((a_score, h_score, a_win))
        
    return pd.DataFrame(features), np.array(targets_ml), np.array(targets_spread), np.array(targets_total)

def train_models():
    # 1. Get Data
    df_raw = prepare_training_data()
    if df_raw is None: return
    
    # 2. Build Dataset
    X, y_ml, y_spread, y_total = build_features(df_raw)
    print(f"Training dataset size: {len(X)}")
    
    if len(X) < 100:
        print("Insufficient data for training.")
        return

    # 3. Split (Time-based split ideally, but random for now is ok for MVP)
    # Actually, simplistic random split is risky for time-series.
    # Let's simple split: Last 20% is test.
    split_idx = int(len(X) * 0.8)
    
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_ml_train, y_ml_test = y_ml[:split_idx], y_ml[split_idx:]
    y_spread_train, y_spread_test = y_spread[:split_idx], y_spread[split_idx:]
    y_total_train, y_total_test = y_total[:split_idx], y_total[split_idx:]
    
    # 4. Train Models
    models_dir = Path(__file__).parent.parent / "models"
    os.makedirs(models_dir, exist_ok=True)
    
    # --- Moneyline Model (Classifier) ---
    print("\nTraining Moneyline Model...")
    ml_model = xgb.XGBClassifier(
        n_estimators=100, 
        max_depth=3, 
        learning_rate=0.05,
        eval_metric='logloss'
    )
    ml_model.fit(X_train, y_ml_train)
    
    preds_ml = ml_model.predict(X_test)
    acc = accuracy_score(y_ml_test, preds_ml)
    print(f"  Moneyline Accuracy: {acc:.1%}")
    print(classification_report(y_ml_test, preds_ml))
    
    # Save
    joblib.dump({
        'model': ml_model,
        'features': list(X.columns),
        'timestamp': pd.Timestamp.now()
    }, models_dir / "nfl_moneyline_xgb.joblib")
    
    
    # --- Spread Model (Regressor) ---
    print("\nTraining Spread Model...")
    spread_model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.05
    )
    spread_model.fit(X_train, y_spread_train)
    
    preds_spread = spread_model.predict(X_test)
    mae_spread = mean_absolute_error(y_spread_test, preds_spread)
    print(f"  Spread MAE: {mae_spread:.2f} pts")
    
    joblib.dump({
        'model': spread_model,
        'features': list(X.columns),
        'timestamp': pd.Timestamp.now()
    }, models_dir / "nfl_spread_xgb.joblib")
    
    
    # --- Totals Model (Regressor) ---
    print("\nTraining Totals Model...")
    total_model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.05
    )
    total_model.fit(X_train, y_total_train)
    
    preds_total = total_model.predict(X_test)
    mae_total = mean_absolute_error(y_total_test, preds_total)
    print(f"  Totals MAE: {mae_total:.2f} pts")
    
    joblib.dump({
        'model': total_model,
        'features': list(X.columns),
        'timestamp': pd.Timestamp.now()
    }, models_dir / "nfl_totals_xgb.joblib")
    
    print("\nDone! Models saved to /models/")

if __name__ == "__main__":
    train_models()
