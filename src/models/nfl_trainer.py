"""
NFL Game Predictor - Data Fetcher and Model Trainer

Uses ESPN API to fetch game data and train prediction model.
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import joblib
from pathlib import Path
from xgboost import XGBClassifier


class NFLFetcher:
    """Fetch NFL game data from ESPN API."""
    
    ESPN_API = "https://site.api.espn.com/apis/site/v2/sports/football/nfl"
    
    TEAMS = {
        'ARI': 'Arizona Cardinals', 'ATL': 'Atlanta Falcons', 'BAL': 'Baltimore Ravens',
        'BUF': 'Buffalo Bills', 'CAR': 'Carolina Panthers', 'CHI': 'Chicago Bears',
        'CIN': 'Cincinnati Bengals', 'CLE': 'Cleveland Browns', 'DAL': 'Dallas Cowboys',
        'DEN': 'Denver Broncos', 'DET': 'Detroit Lions', 'GB': 'Green Bay Packers',
        'HOU': 'Houston Texans', 'IND': 'Indianapolis Colts', 'JAX': 'Jacksonville Jaguars',
        'KC': 'Kansas City Chiefs', 'LAC': 'Los Angeles Chargers', 'LAR': 'Los Angeles Rams',
        'LV': 'Las Vegas Raiders', 'MIA': 'Miami Dolphins', 'MIN': 'Minnesota Vikings',
        'NE': 'New England Patriots', 'NO': 'New Orleans Saints', 'NYG': 'New York Giants',
        'NYJ': 'New York Jets', 'PHI': 'Philadelphia Eagles', 'PIT': 'Pittsburgh Steelers',
        'SEA': 'Seattle Seahawks', 'SF': 'San Francisco 49ers', 'TB': 'Tampa Bay Buccaneers',
        'TEN': 'Tennessee Titans', 'WAS': 'Washington Commanders'
    }
    
    def get_schedule(self, season: int, season_type: int = 2) -> pd.DataFrame:
        """
        Get NFL schedule for a season.
        season_type: 1=preseason, 2=regular, 3=postseason
        """
        games = []
        
        for week in range(1, 19):  # 18 weeks regular season
            try:
                url = f"{self.ESPN_API}/scoreboard?seasontype={season_type}&week={week}&dates={season}"
                resp = requests.get(url, timeout=10)
                
                if resp.status_code == 200:
                    data = resp.json()
                    
                    for event in data.get('events', []):
                        competition = event.get('competitions', [{}])[0]
                        
                        if competition.get('status', {}).get('type', {}).get('completed'):
                            competitors = competition.get('competitors', [])
                            
                            if len(competitors) == 2:
                                home = next((c for c in competitors if c.get('homeAway') == 'home'), {})
                                away = next((c for c in competitors if c.get('homeAway') == 'away'), {})
                                
                                home_score = int(home.get('score', 0))
                                away_score = int(away.get('score', 0))
                                
                                games.append({
                                    'game_id': event.get('id'),
                                    'date': event.get('date', '')[:10],
                                    'week': week,
                                    'home_team': home.get('team', {}).get('abbreviation', ''),
                                    'away_team': away.get('team', {}).get('abbreviation', ''),
                                    'home_score': home_score,
                                    'away_score': away_score,
                                    'home_win': 1 if home_score > away_score else 0
                                })
            except Exception as e:
                print(f"Week {week} error: {e}")
        
        return pd.DataFrame(games)
    
    def get_team_stats(self, season: int = 2024) -> pd.DataFrame:
        """Get team statistics."""
        stats = []
        
        try:
            url = f"{self.ESPN_API}/standings?season={season}"
            resp = requests.get(url, timeout=10)
            
            if resp.status_code == 200:
                data = resp.json()
                
                for group in data.get('children', []):
                    for team_entry in group.get('standings', {}).get('entries', []):
                        team = team_entry.get('team', {})
                        team_stats = team_entry.get('stats', [])
                        
                        stat_dict = {s['name']: s['value'] for s in team_stats if 'value' in s}
                        
                        stats.append({
                            'team': team.get('abbreviation', ''),
                            'wins': stat_dict.get('wins', 0),
                            'losses': stat_dict.get('losses', 0),
                            'points_for': stat_dict.get('pointsFor', 0),
                            'points_against': stat_dict.get('pointsAgainst', 0)
                        })
        except:
            pass
        
        return pd.DataFrame(stats)


class NFLTrainer:
    """Train NFL prediction model."""
    
    def __init__(self):
        self.fetcher = NFLFetcher()
        self.model = None
        self.feature_cols = []
    
    def fetch_training_data(self, seasons: List[int] = [2022, 2023, 2024]) -> pd.DataFrame:
        """Fetch game data for training."""
        all_games = []
        
        for season in seasons:
            print(f"Fetching {season} season...")
            try:
                games = self.fetcher.get_schedule(season)
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
        games = games.sort_values(['season', 'week'])
        
        team_stats = {}
        features = []
        
        for idx, game in games.iterrows():
            home = game['home_team']
            away = game['away_team']
            
            if home not in team_stats:
                team_stats[home] = {'wins': 0, 'losses': 0, 'pts_for': [], 'pts_against': []}
            if away not in team_stats:
                team_stats[away] = {'wins': 0, 'losses': 0, 'pts_for': [], 'pts_against': []}
            
            h = team_stats[home]
            a = team_stats[away]
            
            home_games = h['wins'] + h['losses']
            away_games = a['wins'] + a['losses']
            
            home_win_pct = h['wins'] / home_games if home_games > 0 else 0.5
            away_win_pct = a['wins'] / away_games if away_games > 0 else 0.5
            
            home_ppg = np.mean(h['pts_for'][-5:]) if h['pts_for'] else 21.0
            away_ppg = np.mean(a['pts_for'][-5:]) if a['pts_for'] else 21.0
            
            home_ppg_allowed = np.mean(h['pts_against'][-5:]) if h['pts_against'] else 21.0
            away_ppg_allowed = np.mean(a['pts_against'][-5:]) if a['pts_against'] else 21.0
            
            features.append({
                'game_id': game['game_id'],
                'home_win_pct': home_win_pct,
                'away_win_pct': away_win_pct,
                'home_ppg': home_ppg,
                'away_ppg': away_ppg,
                'home_ppg_allowed': home_ppg_allowed,
                'away_ppg_allowed': away_ppg_allowed,
                'win_pct_diff': home_win_pct - away_win_pct,
                'point_diff': home_ppg - away_ppg,
                'home_win': game['home_win']
            })
            
            # Update stats
            if game['home_win'] == 1:
                team_stats[home]['wins'] += 1
                team_stats[away]['losses'] += 1
            else:
                team_stats[home]['losses'] += 1
                team_stats[away]['wins'] += 1
            
            team_stats[home]['pts_for'].append(game['home_score'])
            team_stats[home]['pts_against'].append(game['away_score'])
            team_stats[away]['pts_for'].append(game['away_score'])
            team_stats[away]['pts_against'].append(game['home_score'])
        
        return pd.DataFrame(features)
    
    def train(self, df: pd.DataFrame) -> Dict:
        """Train the model."""
        self.feature_cols = [
            'home_win_pct', 'away_win_pct', 'home_ppg', 'away_ppg',
            'home_ppg_allowed', 'away_ppg_allowed', 'win_pct_diff', 'point_diff'
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
    
    def save(self, path: str = 'models/nfl_model.joblib'):
        """Save model to disk."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            'model': self.model,
            'features': self.feature_cols
        }, path)
        print(f"Model saved to {path}")
    
    def load(self, path: str = 'models/nfl_model.joblib'):
        """Load model from disk."""
        data = joblib.load(path)
        self.model = data['model']
        self.feature_cols = data['features']


def main():
    """Train NFL model."""
    print("=" * 50)
    print("NFL Model Training")
    print("=" * 50)
    
    trainer = NFLTrainer()
    
    print("\nFetching game data...")
    games = trainer.fetch_training_data([2022, 2023, 2024])
    
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
