"""
NCAAF Game Predictor - Data Fetcher and Model Trainer

Uses ESPN API to fetch college football game data.
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List
import joblib
from pathlib import Path
from xgboost import XGBClassifier


class NCAAFFetcher:
    """Fetch NCAAF game data from ESPN API."""
    
    ESPN_API = "https://site.api.espn.com/apis/site/v2/sports/football/college-football"
    
    # Power 5 + Notre Dame team IDs
    TOP_TEAMS = [
        'ALA', 'ARIZ', 'ASU', 'ARK', 'AUB', 'BAY', 'CAL', 'CLEM', 'COLO', 'DUKE',
        'FLA', 'FSU', 'UGA', 'ILL', 'IND', 'IOWA', 'ISU', 'KU', 'KSU', 'UK',
        'LSU', 'LOU', 'MD', 'MIA', 'MICH', 'MSU', 'MINN', 'MISS', 'MSST', 'MIZ',
        'NEB', 'UNC', 'NCST', 'ND', 'OSU', 'OU', 'OKST', 'ORE', 'ORST', 'PSU',
        'PITT', 'PUR', 'RUT', 'USC', 'SC', 'STAN', 'SYR', 'TENN', 'TEX', 'TAMU',
        'TTU', 'TCU', 'UCLA', 'UTAH', 'UVA', 'VT', 'WAKE', 'WASH', 'WSU', 'WVU', 'WIS'
    ]
    
    def get_schedule(self, season: int, weeks: List[int] = None) -> pd.DataFrame:
        """Get NCAAF schedule for a season."""
        games = []
        weeks = weeks or list(range(1, 16))
        
        for week in weeks:
            try:
                url = f"{self.ESPN_API}/scoreboard?seasontype=2&week={week}&dates={season}"
                resp = requests.get(url, timeout=15)
                
                if resp.status_code == 200:
                    data = resp.json()
                    
                    for event in data.get('events', []):
                        competition = event.get('competitions', [{}])[0]
                        
                        if competition.get('status', {}).get('type', {}).get('completed'):
                            competitors = competition.get('competitors', [])
                            
                            if len(competitors) == 2:
                                home = next((c for c in competitors if c.get('homeAway') == 'home'), {})
                                away = next((c for c in competitors if c.get('homeAway') == 'away'), {})
                                
                                home_abbr = home.get('team', {}).get('abbreviation', '')
                                away_abbr = away.get('team', {}).get('abbreviation', '')
                                
                                home_score = int(home.get('score', 0))
                                away_score = int(away.get('score', 0))
                                
                                games.append({
                                    'game_id': event.get('id'),
                                    'date': event.get('date', '')[:10],
                                    'week': week,
                                    'home_team': home_abbr,
                                    'away_team': away_abbr,
                                    'home_name': home.get('team', {}).get('displayName', ''),
                                    'away_name': away.get('team', {}).get('displayName', ''),
                                    'home_score': home_score,
                                    'away_score': away_score,
                                    'home_win': 1 if home_score > away_score else 0
                                })
                                
                print(f"Week {week}: {len([g for g in games if g.get('week') == week])} games")
            except Exception as e:
                print(f"Week {week} error: {e}")
        
        return pd.DataFrame(games)
    
    def get_rankings(self, season: int = 2024, week: int = 1) -> pd.DataFrame:
        """Get AP Top 25 rankings."""
        rankings = []
        
        try:
            url = f"{self.ESPN_API}/rankings?seasontype=2&week={week}&dates={season}"
            resp = requests.get(url, timeout=10)
            
            if resp.status_code == 200:
                data = resp.json()
                
                for ranking in data.get('rankings', []):
                    if 'AP' in ranking.get('name', ''):
                        for rank in ranking.get('ranks', []):
                            team = rank.get('team', {})
                            rankings.append({
                                'rank': rank.get('current'),
                                'team': team.get('abbreviation', ''),
                                'name': team.get('displayName', ''),
                                'record': rank.get('recordSummary', '')
                            })
        except:
            pass
        
        return pd.DataFrame(rankings)


class NCAAFTrainer:
    """Train NCAAF prediction model."""
    
    def __init__(self):
        self.fetcher = NCAAFFetcher()
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
            
            home_ppg = np.mean(h['pts_for'][-5:]) if h['pts_for'] else 28.0
            away_ppg = np.mean(a['pts_for'][-5:]) if a['pts_for'] else 28.0
            
            home_ppg_allowed = np.mean(h['pts_against'][-5:]) if h['pts_against'] else 24.0
            away_ppg_allowed = np.mean(a['pts_against'][-5:]) if a['pts_against'] else 24.0
            
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
    
    def save(self, path: str = 'models/ncaaf_model.joblib'):
        """Save model to disk."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            'model': self.model,
            'features': self.feature_cols
        }, path)
        print(f"Model saved to {path}")


def main():
    """Train NCAAF model."""
    print("=" * 50)
    print("NCAAF Model Training")
    print("=" * 50)
    
    trainer = NCAAFTrainer()
    
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
