"""
MLB Game Predictor - Data Fetcher and Model Trainer

Uses MLB-StatsAPI to fetch game data and train prediction model.
"""

import statsapi
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


class MLBFetcher:
    """Fetch MLB game data from MLB-StatsAPI."""
    
    def __init__(self):
        self.team_ids = {}
        self._load_teams()
    
    def _load_teams(self):
        """Load all MLB team IDs."""
        teams = statsapi.get('teams', {'sportId': 1})['teams']
        for team in teams:
            abbr = team.get('abbreviation', team['teamName'][:3].upper())
            self.team_ids[abbr] = team['id']
            self.team_ids[team['id']] = abbr
    
    def get_schedule(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Get games between dates."""
        games = []
        schedule = statsapi.schedule(start_date=start_date, end_date=end_date)
        
        for game in schedule:
            if game['status'] == 'Final':
                games.append({
                    'game_id': game['game_id'],
                    'date': game['game_date'],
                    'home_team': game['home_name'],
                    'away_team': game['away_name'],
                    'home_id': game['home_id'],
                    'away_id': game['away_id'],
                    'home_score': game['home_score'],
                    'away_score': game['away_score'],
                    'home_win': 1 if game['home_score'] > game['away_score'] else 0
                })
        
        return pd.DataFrame(games)
    
    def get_team_stats(self, team_id: int, season: int = 2024) -> Dict:
        """Get team season statistics."""
        try:
            stats = statsapi.get('teams', {'teamId': team_id, 'season': season})
            team = stats['teams'][0] if stats['teams'] else {}
            
            # Get team stats
            team_stats = statsapi.team_stats(team_id, season=season, stats='season')
            
            return {
                'team_id': team_id,
                'wins': team.get('record', {}).get('wins', 0),
                'losses': team.get('record', {}).get('losses', 0),
                'run_diff': 0  # Calculate from games
            }
        except:
            return {'team_id': team_id, 'wins': 0, 'losses': 0, 'run_diff': 0}
    
    def get_standings(self, season: int = 2024) -> pd.DataFrame:
        """Get current standings."""
        standings_data = []
        standings = statsapi.standings_data(leagueId="103,104", season=season)
        
        for div_id, div_data in standings.items():
            for team in div_data['teams']:
                standings_data.append({
                    'team': team['name'],
                    'wins': team['w'],
                    'losses': team['l'],
                    'pct': team['w'] / (team['w'] + team['l']) if (team['w'] + team['l']) > 0 else 0.5,
                    'div': div_data['div_name']
                })
        
        return pd.DataFrame(standings_data)


class MLBTrainer:
    """Train MLB prediction model."""
    
    def __init__(self):
        self.fetcher = MLBFetcher()
        self.model = None
        self.feature_cols = []
    
    def fetch_training_data(self, seasons: List[int] = [2023, 2024]) -> pd.DataFrame:
        """Fetch game data for training."""
        all_games = []
        
        for season in seasons:
            print(f"Fetching {season} season...")
            start = f"{season}-03-28"
            end = f"{season}-10-01"
            
            try:
                games = self.fetcher.get_schedule(start, end)
                games['season'] = season
                all_games.append(games)
                print(f"  Found {len(games)} games")
            except Exception as e:
                print(f"  Error: {e}")
        
        if not all_games:
            return pd.DataFrame()
        
        return pd.concat(all_games, ignore_index=True)
    
    def engineer_features(self, games: pd.DataFrame) -> pd.DataFrame:
        """Create features from game data."""
        # Calculate rolling stats per team
        games = games.sort_values('date')
        
        # Team performance tracking
        team_stats = {}
        
        features = []
        for idx, game in games.iterrows():
            home_id = game['home_id']
            away_id = game['away_id']
            
            # Get or initialize team stats
            if home_id not in team_stats:
                team_stats[home_id] = {'wins': 0, 'losses': 0, 'runs_scored': [], 'runs_allowed': []}
            if away_id not in team_stats:
                team_stats[away_id] = {'wins': 0, 'losses': 0, 'runs_scored': [], 'runs_allowed': []}
            
            home = team_stats[home_id]
            away = team_stats[away_id]
            
            # Calculate features
            home_games = home['wins'] + home['losses']
            away_games = away['wins'] + away['losses']
            
            home_win_pct = home['wins'] / home_games if home_games > 0 else 0.5
            away_win_pct = away['wins'] / away_games if away_games > 0 else 0.5
            
            home_runs_avg = np.mean(home['runs_scored'][-10:]) if home['runs_scored'] else 4.5
            away_runs_avg = np.mean(away['runs_scored'][-10:]) if away['runs_scored'] else 4.5
            
            home_runs_allowed = np.mean(home['runs_allowed'][-10:]) if home['runs_allowed'] else 4.5
            away_runs_allowed = np.mean(away['runs_allowed'][-10:]) if away['runs_allowed'] else 4.5
            
            features.append({
                'game_id': game['game_id'],
                'home_win_pct': home_win_pct,
                'away_win_pct': away_win_pct,
                'home_runs_avg': home_runs_avg,
                'away_runs_avg': away_runs_avg,
                'home_runs_allowed': home_runs_allowed,
                'away_runs_allowed': away_runs_allowed,
                'win_pct_diff': home_win_pct - away_win_pct,
                'run_diff': home_runs_avg - away_runs_avg,
                'home_win': game['home_win']
            })
            
            # Update stats after game
            if game['home_win'] == 1:
                team_stats[home_id]['wins'] += 1
                team_stats[away_id]['losses'] += 1
            else:
                team_stats[home_id]['losses'] += 1
                team_stats[away_id]['wins'] += 1
            
            team_stats[home_id]['runs_scored'].append(game['home_score'])
            team_stats[home_id]['runs_allowed'].append(game['away_score'])
            team_stats[away_id]['runs_scored'].append(game['away_score'])
            team_stats[away_id]['runs_allowed'].append(game['home_score'])
        
        return pd.DataFrame(features)
    
    def train(self, df: pd.DataFrame) -> Dict:
        """Train the model."""
        self.feature_cols = [
            'home_win_pct', 'away_win_pct', 'home_runs_avg', 'away_runs_avg',
            'home_runs_allowed', 'away_runs_allowed', 'win_pct_diff', 'run_diff'
        ]
        
        # Filter rows with enough history
        df = df.dropna()
        
        X = df[self.feature_cols]
        y = df['home_win']
        
        # Time-based split
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Train model
        self.model = XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            random_state=42
        )
        
        self.model.fit(X_train, y_train)
        
        # Evaluate
        train_acc = self.model.score(X_train, y_train)
        test_acc = self.model.score(X_test, y_test)
        
        return {
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'train_size': len(X_train),
            'test_size': len(X_test)
        }
    
    def save(self, path: str = 'models/mlb_model.joblib'):
        """Save model to disk."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            'model': self.model,
            'features': self.feature_cols
        }, path)
        print(f"Model saved to {path}")
    
    def load(self, path: str = 'models/mlb_model.joblib'):
        """Load model from disk."""
        data = joblib.load(path)
        self.model = data['model']
        self.feature_cols = data['features']


def main():
    """Train MLB model."""
    print("=" * 50)
    print("MLB Model Training")
    print("=" * 50)
    
    trainer = MLBTrainer()
    
    # Fetch data
    print("\nFetching game data...")
    games = trainer.fetch_training_data([2023, 2024])
    
    if games.empty:
        print("No games found!")
        return
    
    print(f"\nTotal games: {len(games)}")
    
    # Engineer features
    print("\nEngineering features...")
    features = trainer.engineer_features(games)
    print(f"Feature samples: {len(features)}")
    
    # Train
    print("\nTraining model...")
    results = trainer.train(features)
    
    print(f"\nResults:")
    print(f"  Train Accuracy: {results['train_accuracy']:.1%}")
    print(f"  Test Accuracy: {results['test_accuracy']:.1%}")
    print(f"  Train Size: {results['train_size']}")
    print(f"  Test Size: {results['test_size']}")
    
    # Save
    trainer.save()
    print("\nDone!")


if __name__ == "__main__":
    main()
