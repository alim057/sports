"""
NHL Game Predictor - Data Fetcher and Model Trainer

Uses nhl-api-py to fetch game data and train prediction model.
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import joblib
from pathlib import Path
from xgboost import XGBClassifier


class NHLFetcher:
    """Fetch NHL game data from NHL API."""
    
    BASE_URL = "https://api-web.nhle.com/v1"
    
    def __init__(self):
        self.team_abbrs = {
            'ANA': 'Anaheim Ducks', 'ARI': 'Arizona Coyotes', 'BOS': 'Boston Bruins',
            'BUF': 'Buffalo Sabres', 'CAR': 'Carolina Hurricanes', 'CBJ': 'Columbus Blue Jackets',
            'CGY': 'Calgary Flames', 'CHI': 'Chicago Blackhawks', 'COL': 'Colorado Avalanche',
            'DAL': 'Dallas Stars', 'DET': 'Detroit Red Wings', 'EDM': 'Edmonton Oilers',
            'FLA': 'Florida Panthers', 'LAK': 'Los Angeles Kings', 'MIN': 'Minnesota Wild',
            'MTL': 'Montreal Canadiens', 'NJD': 'New Jersey Devils', 'NSH': 'Nashville Predators',
            'NYI': 'New York Islanders', 'NYR': 'New York Rangers', 'OTT': 'Ottawa Senators',
            'PHI': 'Philadelphia Flyers', 'PIT': 'Pittsburgh Penguins', 'SEA': 'Seattle Kraken',
            'SJS': 'San Jose Sharks', 'STL': 'St. Louis Blues', 'TBL': 'Tampa Bay Lightning',
            'TOR': 'Toronto Maple Leafs', 'VAN': 'Vancouver Canucks', 'VGK': 'Vegas Golden Knights',
            'WPG': 'Winnipeg Jets', 'WSH': 'Washington Capitals'
        }
    
    def get_schedule(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Get games between dates."""
        games = []
        
        # Parse dates
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        current = start
        
        while current <= end:
            date_str = current.strftime("%Y-%m-%d")
            try:
                url = f"{self.BASE_URL}/schedule/{date_str}"
                resp = requests.get(url, timeout=10)
                if resp.status_code == 200:
                    data = resp.json()
                    
                    for day in data.get('gameWeek', []):
                        for game in day.get('games', []):
                            if game.get('gameState') == 'OFF':  # Completed
                                home = game.get('homeTeam', {})
                                away = game.get('awayTeam', {})
                                
                                home_score = home.get('score', 0)
                                away_score = away.get('score', 0)
                                
                                games.append({
                                    'game_id': game.get('id'),
                                    'date': day.get('date'),
                                    'home_team': home.get('abbrev', ''),
                                    'away_team': away.get('abbrev', ''),
                                    'home_score': home_score,
                                    'away_score': away_score,
                                    'home_win': 1 if home_score > away_score else 0
                                })
            except Exception as e:
                pass
            
            current += timedelta(days=7)  # Check weekly to speed up
        
        return pd.DataFrame(games)
    
    def get_standings(self, season: str = "20232024") -> pd.DataFrame:
        """Get current standings."""
        standings_data = []
        
        try:
            url = f"{self.BASE_URL}/standings/{season}"
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                
                for team in data.get('standings', []):
                    standings_data.append({
                        'team': team.get('teamAbbrev', {}).get('default', ''),
                        'wins': team.get('wins', 0),
                        'losses': team.get('losses', 0),
                        'ot_losses': team.get('otLosses', 0),
                        'points': team.get('points', 0),
                        'goals_for': team.get('goalFor', 0),
                        'goals_against': team.get('goalAgainst', 0)
                    })
        except:
            pass
        
        return pd.DataFrame(standings_data)


class NHLTrainer:
    """Train NHL prediction model."""
    
    def __init__(self):
        self.fetcher = NHLFetcher()
        self.model = None
        self.feature_cols = []
    
    def fetch_training_data(self, seasons: List[str] = ["2023-2024"]) -> pd.DataFrame:
        """Fetch game data for training."""
        all_games = []
        
        for season in seasons:
            print(f"Fetching {season} season...")
            year = int(season.split("-")[0])
            start = f"{year}-10-01"
            end = f"{year + 1}-04-15"
            
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
        games = games.sort_values('date')
        
        team_stats = {}
        features = []
        
        for idx, game in games.iterrows():
            home = game['home_team']
            away = game['away_team']
            
            if home not in team_stats:
                team_stats[home] = {'wins': 0, 'losses': 0, 'goals_for': [], 'goals_against': []}
            if away not in team_stats:
                team_stats[away] = {'wins': 0, 'losses': 0, 'goals_for': [], 'goals_against': []}
            
            h = team_stats[home]
            a = team_stats[away]
            
            home_games = h['wins'] + h['losses']
            away_games = a['wins'] + a['losses']
            
            home_win_pct = h['wins'] / home_games if home_games > 0 else 0.5
            away_win_pct = a['wins'] / away_games if away_games > 0 else 0.5
            
            home_goals_avg = np.mean(h['goals_for'][-10:]) if h['goals_for'] else 3.0
            away_goals_avg = np.mean(a['goals_for'][-10:]) if a['goals_for'] else 3.0
            
            home_goals_allowed = np.mean(h['goals_against'][-10:]) if h['goals_against'] else 3.0
            away_goals_allowed = np.mean(a['goals_against'][-10:]) if a['goals_against'] else 3.0
            
            features.append({
                'game_id': game['game_id'],
                'home_win_pct': home_win_pct,
                'away_win_pct': away_win_pct,
                'home_goals_avg': home_goals_avg,
                'away_goals_avg': away_goals_avg,
                'home_goals_allowed': home_goals_allowed,
                'away_goals_allowed': away_goals_allowed,
                'win_pct_diff': home_win_pct - away_win_pct,
                'goal_diff': home_goals_avg - away_goals_avg,
                'home_win': game['home_win']
            })
            
            # Update stats
            if game['home_win'] == 1:
                team_stats[home]['wins'] += 1
                team_stats[away]['losses'] += 1
            else:
                team_stats[home]['losses'] += 1
                team_stats[away]['wins'] += 1
            
            team_stats[home]['goals_for'].append(game['home_score'])
            team_stats[home]['goals_against'].append(game['away_score'])
            team_stats[away]['goals_for'].append(game['away_score'])
            team_stats[away]['goals_against'].append(game['home_score'])
        
        return pd.DataFrame(features)
    
    def train(self, df: pd.DataFrame) -> Dict:
        """Train the model."""
        self.feature_cols = [
            'home_win_pct', 'away_win_pct', 'home_goals_avg', 'away_goals_avg',
            'home_goals_allowed', 'away_goals_allowed', 'win_pct_diff', 'goal_diff'
        ]
        
        df = df.dropna()
        
        X = df[self.feature_cols]
        y = df['home_win']
        
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        self.model = XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            random_state=42
        )
        
        self.model.fit(X_train, y_train)
        
        train_acc = self.model.score(X_train, y_train)
        test_acc = self.model.score(X_test, y_test)
        
        return {
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'train_size': len(X_train),
            'test_size': len(X_test)
        }
    
    def save(self, path: str = 'models/nhl_model.joblib'):
        """Save model to disk."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            'model': self.model,
            'features': self.feature_cols
        }, path)
        print(f"Model saved to {path}")
    
    def load(self, path: str = 'models/nhl_model.joblib'):
        """Load model from disk."""
        data = joblib.load(path)
        self.model = data['model']
        self.feature_cols = data['features']


def main():
    """Train NHL model."""
    print("=" * 50)
    print("NHL Model Training")
    print("=" * 50)
    
    trainer = NHLTrainer()
    
    print("\nFetching game data...")
    games = trainer.fetch_training_data(["2023-2024"])
    
    if games.empty:
        print("No games found!")
        return
    
    print(f"\nTotal games: {len(games)}")
    
    print("\nEngineering features...")
    features = trainer.engineer_features(games)
    print(f"Feature samples: {len(features)}")
    
    print("\nTraining model...")
    results = trainer.train(features)
    
    print(f"\nResults:")
    print(f"  Train Accuracy: {results['train_accuracy']:.1%}")
    print(f"  Test Accuracy: {results['test_accuracy']:.1%}")
    
    trainer.save()
    print("\nDone!")


if __name__ == "__main__":
    main()
