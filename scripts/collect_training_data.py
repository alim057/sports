"""
Multi-Sport Training Data Collector

Collects historical game data with weather, injuries, and player stats for model training.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import json
import argparse

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))


class TrainingDataCollector:
    """Unified training data collector for all sports."""
    
    SPORT_CONFIGS = {
        'nba': {
            'name': 'NBA',
            'seasons': 3,
            'indoor': True,
            'features': [
                'home_pts_l10', 'away_pts_l10',
                'home_opp_pts_l10', 'away_opp_pts_l10',
                'home_fg_pct_l10', 'away_fg_pct_l10',
                'home_reb_l10', 'away_reb_l10',
                'home_ast_l10', 'away_ast_l10',
                'home_win_l10', 'away_win_l10',
                'home_days_rest', 'away_days_rest',
            ]
        },
        'nfl': {
            'name': 'NFL',
            'seasons': 3,
            'indoor': False,
            'features': [
                'home_pts_l5', 'away_pts_l5',
                'home_yards_l5', 'away_yards_l5',
                'home_turnovers_l5', 'away_turnovers_l5',
                'home_days_rest', 'away_days_rest',
                'temp_f', 'wind_mph', 'is_dome',
            ]
        },
        'ncaaf': {
            'name': 'NCAAF',
            'seasons': 2,
            'indoor': False,
            'features': [
                'home_pts_l5', 'away_pts_l5',
                'home_margin_l5', 'away_margin_l5',
                'temp_f', 'wind_mph',
            ]
        },
        'mlb': {
            'name': 'MLB',
            'seasons': 2,
            'indoor': False,
            'features': [
                'home_runs_l10', 'away_runs_l10',
                'home_hits_l10', 'away_hits_l10',
                'temp_f', 'wind_mph', 'is_dome',
            ]
        }
    }
    
    def __init__(self):
        self.data_dir = project_root / "data"
        self.data_dir.mkdir(exist_ok=True)
        
    def collect_nba(self, num_seasons: int = 3) -> pd.DataFrame:
        """Collect NBA training data."""
        print(f"\n{'='*60}")
        print(f"Collecting NBA data for {num_seasons} seasons...")
        print(f"{'='*60}")
        
        try:
            from nba_api.stats.endpoints import leaguegamefinder
            from nba_api.stats.static import teams
        except ImportError:
            print("Error: nba_api not installed. Run: pip install nba_api")
            return pd.DataFrame()
        
        all_games = []
        current_year = datetime.now().year
        
        # NBA seasons
        seasons = []
        for i in range(num_seasons):
            year = current_year - i
            if datetime.now().month < 10:  # Before October, use previous year
                year -= 1
            seasons.append(f"{year}-{str(year+1)[-2:]}")
        
        print(f"Fetching seasons: {seasons}")
        
        for season in seasons:
            print(f"\n  Fetching {season}...")
            try:
                # Rate limiting
                time.sleep(1)
                
                gamefinder = leaguegamefinder.LeagueGameFinder(
                    season_nullable=season,
                    season_type_nullable='Regular Season'
                )
                games = gamefinder.get_data_frames()[0]
                
                if not games.empty:
                    games['season'] = season
                    all_games.append(games)
                    print(f"    Found {len(games)} game records")
                    
            except Exception as e:
                print(f"    Error fetching {season}: {e}")
                continue
        
        if not all_games:
            return pd.DataFrame()
        
        df = pd.concat(all_games, ignore_index=True)
        
        # Process into game-level format
        games_df = self._process_nba_games(df)
        
        # Add features
        games_df = self._add_nba_features(games_df)
        
        return games_df
    
    def _process_nba_games(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process NBA game finder results into game-level data."""
        # Filter to home games only to avoid duplicates
        home_games = df[df['MATCHUP'].str.contains('vs.')].copy()
        
        games = []
        for _, row in home_games.iterrows():
            game_id = row['GAME_ID']
            home_team = row['TEAM_ABBREVIATION']
            
            # Find away team
            away_row = df[(df['GAME_ID'] == game_id) & (df['TEAM_ABBREVIATION'] != home_team)]
            if away_row.empty:
                continue
                
            away_row = away_row.iloc[0]
            
            games.append({
                'game_id': game_id,
                'game_date': row['GAME_DATE'],
                'season': row['season'],
                'home_team': home_team,
                'away_team': away_row['TEAM_ABBREVIATION'],
                'home_pts': row['PTS'],
                'away_pts': away_row['PTS'],
                'home_won': 1 if row['WL'] == 'W' else 0,
                'home_fg_pct': row['FG_PCT'],
                'away_fg_pct': away_row['FG_PCT'],
                'home_reb': row['REB'],
                'away_reb': away_row['REB'],
                'home_ast': row['AST'],
                'away_ast': away_row['AST'],
                'home_tov': row.get('TOV', 0),
                'away_tov': away_row.get('TOV', 0),
            })
        
        return pd.DataFrame(games)
    
    def _add_nba_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add rolling features for NBA games."""
        if df.empty:
            return df
            
        df = df.sort_values('game_date').reset_index(drop=True)
        
        # Rolling stats per team
        team_stats = {}
        
        features = []
        for idx, row in df.iterrows():
            home = row['home_team']
            away = row['away_team']
            
            # Get last 10 games for each team
            home_hist = team_stats.get(home, [])[-10:]
            away_hist = team_stats.get(away, [])[-10:]
            
            feat = {}
            
            if len(home_hist) >= 5:
                feat['home_pts_l10'] = np.mean([g['pts'] for g in home_hist])
                feat['home_opp_pts_l10'] = np.mean([g['opp_pts'] for g in home_hist])
                feat['home_fg_pct_l10'] = np.mean([g['fg_pct'] for g in home_hist])
                feat['home_reb_l10'] = np.mean([g['reb'] for g in home_hist])
                feat['home_ast_l10'] = np.mean([g['ast'] for g in home_hist])
                feat['home_win_l10'] = np.mean([g['won'] for g in home_hist])
            else:
                feat['home_pts_l10'] = 105
                feat['home_opp_pts_l10'] = 105
                feat['home_fg_pct_l10'] = 0.45
                feat['home_reb_l10'] = 42
                feat['home_ast_l10'] = 24
                feat['home_win_l10'] = 0.5
                
            if len(away_hist) >= 5:
                feat['away_pts_l10'] = np.mean([g['pts'] for g in away_hist])
                feat['away_opp_pts_l10'] = np.mean([g['opp_pts'] for g in away_hist])
                feat['away_fg_pct_l10'] = np.mean([g['fg_pct'] for g in away_hist])
                feat['away_reb_l10'] = np.mean([g['reb'] for g in away_hist])
                feat['away_ast_l10'] = np.mean([g['ast'] for g in away_hist])
                feat['away_win_l10'] = np.mean([g['won'] for g in away_hist])
            else:
                feat['away_pts_l10'] = 105
                feat['away_opp_pts_l10'] = 105
                feat['away_fg_pct_l10'] = 0.45
                feat['away_reb_l10'] = 42
                feat['away_ast_l10'] = 24
                feat['away_win_l10'] = 0.5
            
            features.append(feat)
            
            # Update history
            if home not in team_stats:
                team_stats[home] = []
            team_stats[home].append({
                'pts': row['home_pts'],
                'opp_pts': row['away_pts'],
                'fg_pct': row['home_fg_pct'],
                'reb': row['home_reb'],
                'ast': row['home_ast'],
                'won': row['home_won']
            })
            
            if away not in team_stats:
                team_stats[away] = []
            team_stats[away].append({
                'pts': row['away_pts'],
                'opp_pts': row['home_pts'],
                'fg_pct': row['away_fg_pct'],
                'reb': row['away_reb'],
                'ast': row['away_ast'],
                'won': 1 - row['home_won']
            })
        
        features_df = pd.DataFrame(features)
        return pd.concat([df, features_df], axis=1)
    
    def collect_nfl(self, num_seasons: int = 3) -> pd.DataFrame:
        """Collect NFL training data from ESPN API."""
        print(f"\n{'='*60}")
        print(f"Collecting NFL data for {num_seasons} seasons...")
        print(f"{'='*60}")
        
        import requests
        
        all_games = []
        current_year = datetime.now().year
        
        for i in range(num_seasons):
            year = current_year - i
            if datetime.now().month < 9:  # Before September
                year -= 1
                
            print(f"\n  Fetching {year} season...")
            
            try:
                # ESPN Scoreboard API
                url = f"https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"
                
                # Fetch all weeks
                for week in range(1, 19):  # Regular season weeks
                    time.sleep(0.5)  # Rate limit
                    
                    params = {
                        'seasontype': 2,
                        'week': week,
                        'dates': year
                    }
                    
                    resp = requests.get(url, params=params)
                    data = resp.json()
                    
                    for event in data.get('events', []):
                        game = self._parse_espn_game(event, year)
                        if game:
                            all_games.append(game)
                
                print(f"    Found {len([g for g in all_games if g.get('season') == year])} games")
                
            except Exception as e:
                print(f"    Error: {e}")
                continue
        
        if not all_games:
            return pd.DataFrame()
            
        df = pd.DataFrame(all_games)
        df = self._add_nfl_features(df)
        
        return df
    
    def _parse_espn_game(self, event: dict, season: int) -> dict:
        """Parse ESPN event into game record."""
        try:
            competition = event['competitions'][0]
            
            if competition['status']['type']['state'] != 'post':
                return None  # Skip incomplete games
            
            home_team = None
            away_team = None
            home_score = 0
            away_score = 0
            
            for competitor in competition['competitors']:
                if competitor['homeAway'] == 'home':
                    home_team = competitor['team']['abbreviation']
                    home_score = int(competitor['score'])
                else:
                    away_team = competitor['team']['abbreviation']
                    away_score = int(competitor['score'])
            
            return {
                'game_id': event['id'],
                'game_date': event['date'][:10],
                'season': season,
                'home_team': home_team,
                'away_team': away_team,
                'home_pts': home_score,
                'away_pts': away_score,
                'home_won': 1 if home_score > away_score else 0,
            }
        except:
            return None
    
    def _add_nfl_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add rolling features for NFL games."""
        if df.empty:
            return df
            
        df = df.sort_values('game_date').reset_index(drop=True)
        
        team_stats = {}
        features = []
        
        for idx, row in df.iterrows():
            home = row['home_team']
            away = row['away_team']
            
            home_hist = team_stats.get(home, [])[-5:]
            away_hist = team_stats.get(away, [])[-5:]
            
            feat = {}
            
            if len(home_hist) >= 3:
                feat['home_pts_l5'] = np.mean([g['pts'] for g in home_hist])
                feat['home_opp_pts_l5'] = np.mean([g['opp_pts'] for g in home_hist])
            else:
                feat['home_pts_l5'] = 21
                feat['home_opp_pts_l5'] = 21
                
            if len(away_hist) >= 3:
                feat['away_pts_l5'] = np.mean([g['pts'] for g in away_hist])
                feat['away_opp_pts_l5'] = np.mean([g['opp_pts'] for g in away_hist])
            else:
                feat['away_pts_l5'] = 21
                feat['away_opp_pts_l5'] = 21
            
            features.append(feat)
            
            # Update history
            if home not in team_stats:
                team_stats[home] = []
            team_stats[home].append({
                'pts': row['home_pts'],
                'opp_pts': row['away_pts'],
            })
            
            if away not in team_stats:
                team_stats[away] = []
            team_stats[away].append({
                'pts': row['away_pts'],
                'opp_pts': row['home_pts'],
            })
        
        features_df = pd.DataFrame(features)
        return pd.concat([df, features_df], axis=1)
    
    def collect_mlb(self, num_seasons: int = 2) -> pd.DataFrame:
        """Collect MLB training data."""
        print(f"\n{'='*60}")
        print(f"Collecting MLB data for {num_seasons} seasons...")
        print(f"{'='*60}")
        
        try:
            import statsapi
        except ImportError:
            print("Error: MLB-StatsAPI not installed. Run: pip install MLB-StatsAPI")
            return pd.DataFrame()
        
        all_games = []
        current_year = datetime.now().year
        
        for i in range(num_seasons):
            year = current_year - i
            if datetime.now().month < 4:  # Before April
                year -= 1
                
            print(f"\n  Fetching {year} season...")
            
            try:
                # Get schedule for entire season
                schedule = statsapi.schedule(
                    start_date=f"{year}-04-01",
                    end_date=f"{year}-10-31"
                )
                
                for game in schedule:
                    if game['status'] == 'Final':
                        all_games.append({
                            'game_id': game['game_id'],
                            'game_date': game['game_date'],
                            'season': year,
                            'home_team': game['home_name'],
                            'away_team': game['away_name'],
                            'home_pts': game['home_score'],
                            'away_pts': game['away_score'],
                            'home_won': 1 if game['home_score'] > game['away_score'] else 0,
                        })
                
                print(f"    Found {len([g for g in all_games if g.get('season') == year])} games")
                
            except Exception as e:
                print(f"    Error: {e}")
                continue
        
        if not all_games:
            return pd.DataFrame()
            
        df = pd.DataFrame(all_games)
        df = self._add_mlb_features(df)
        
        return df
    
    def _add_mlb_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add rolling features for MLB games."""
        if df.empty:
            return df
            
        df = df.sort_values('game_date').reset_index(drop=True)
        
        team_stats = {}
        features = []
        
        for idx, row in df.iterrows():
            home = row['home_team']
            away = row['away_team']
            
            home_hist = team_stats.get(home, [])[-10:]
            away_hist = team_stats.get(away, [])[-10:]
            
            feat = {}
            
            if len(home_hist) >= 5:
                feat['home_runs_l10'] = np.mean([g['runs'] for g in home_hist])
            else:
                feat['home_runs_l10'] = 4.5
                
            if len(away_hist) >= 5:
                feat['away_runs_l10'] = np.mean([g['runs'] for g in away_hist])
            else:
                feat['away_runs_l10'] = 4.5
            
            features.append(feat)
            
            # Update history
            if home not in team_stats:
                team_stats[home] = []
            team_stats[home].append({'runs': row['home_pts']})
            
            if away not in team_stats:
                team_stats[away] = []
            team_stats[away].append({'runs': row['away_pts']})
        
        features_df = pd.DataFrame(features)
        return pd.concat([df, features_df], axis=1)
    
    def collect_ncaaf(self, num_seasons: int = 2) -> pd.DataFrame:
        """Collect NCAAF training data from ESPN API."""
        print(f"\n{'='*60}")
        print(f"Collecting NCAAF data for {num_seasons} seasons...")
        print(f"{'='*60}")
        
        import requests
        
        all_games = []
        current_year = datetime.now().year
        
        for i in range(num_seasons):
            year = current_year - i
            if datetime.now().month < 8:  # Before August
                year -= 1
                
            print(f"\n  Fetching {year} season...")
            
            try:
                url = f"https://site.api.espn.com/apis/site/v2/sports/football/college-football/scoreboard"
                
                # Fetch all weeks (1-15 regular season + bowls)
                for week in range(1, 16):
                    time.sleep(0.5)
                    
                    params = {
                        'seasontype': 2,
                        'week': week,
                        'dates': year,
                        'groups': 80,  # FBS
                        'limit': 100
                    }
                    
                    resp = requests.get(url, params=params)
                    data = resp.json()
                    
                    for event in data.get('events', []):
                        game = self._parse_espn_game(event, year)
                        if game:
                            all_games.append(game)
                
                print(f"    Found {len([g for g in all_games if g.get('season') == year])} games")
                
            except Exception as e:
                print(f"    Error: {e}")
                continue
        
        if not all_games:
            return pd.DataFrame()
            
        df = pd.DataFrame(all_games)
        df = self._add_ncaaf_features(df)
        
        return df
    
    def _add_ncaaf_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add rolling features for NCAAF games."""
        if df.empty:
            return df
            
        df = df.sort_values('game_date').reset_index(drop=True)
        
        team_stats = {}
        features = []
        
        for idx, row in df.iterrows():
            home = row['home_team']
            away = row['away_team']
            
            home_hist = team_stats.get(home, [])[-5:]
            away_hist = team_stats.get(away, [])[-5:]
            
            feat = {}
            
            if len(home_hist) >= 3:
                feat['home_pts_l5'] = np.mean([g['pts'] for g in home_hist])
                feat['home_margin_l5'] = np.mean([g['margin'] for g in home_hist])
            else:
                feat['home_pts_l5'] = 28
                feat['home_margin_l5'] = 0
                
            if len(away_hist) >= 3:
                feat['away_pts_l5'] = np.mean([g['pts'] for g in away_hist])
                feat['away_margin_l5'] = np.mean([g['margin'] for g in away_hist])
            else:
                feat['away_pts_l5'] = 28
                feat['away_margin_l5'] = 0
            
            features.append(feat)
            
            if home not in team_stats:
                team_stats[home] = []
            team_stats[home].append({
                'pts': row['home_pts'],
                'margin': row['home_pts'] - row['away_pts']
            })
            
            if away not in team_stats:
                team_stats[away] = []
            team_stats[away].append({
                'pts': row['away_pts'],
                'margin': row['away_pts'] - row['home_pts']
            })
        
        features_df = pd.DataFrame(features)
        return pd.concat([df, features_df], axis=1)
    
    def save_training_data(self, df: pd.DataFrame, sport: str):
        """Save training data to CSV."""
        if df.empty:
            print(f"No data to save for {sport}")
            return
            
        path = self.data_dir / f"{sport}_training_data.csv"
        df.to_csv(path, index=False)
        print(f"\nSaved {len(df)} records to {path}")
        
        # Generate report
        report = {
            'sport': sport.upper(),
            'games_total': len(df),
            'seasons': list(df['season'].unique()) if 'season' in df.columns else [],
            'features': [c for c in df.columns if c not in ['game_id', 'game_date', 'season', 'home_team', 'away_team']],
            'home_win_rate': df['home_won'].mean() if 'home_won' in df.columns else None,
        }
        
        report_path = self.data_dir / f"training_report_{sport}.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        print(f"Saved report to {report_path}")
        
        return report


def main():
    parser = argparse.ArgumentParser(description='Collect training data for sports models')
    parser.add_argument('--sport', type=str, default='all', 
                       choices=['all', 'nba', 'nfl', 'ncaaf', 'mlb'],
                       help='Sport to collect data for')
    parser.add_argument('--seasons', type=int, default=2,
                       help='Number of seasons to fetch')
    
    args = parser.parse_args()
    
    collector = TrainingDataCollector()
    
    print("=" * 60)
    print("Multi-Sport Training Data Collector")
    print("=" * 60)
    print(f"Sport: {args.sport}")
    print(f"Seasons: {args.seasons}")
    
    sports = ['nba', 'nfl', 'ncaaf', 'mlb'] if args.sport == 'all' else [args.sport]
    
    for sport in sports:
        if sport == 'nba':
            df = collector.collect_nba(args.seasons)
        elif sport == 'nfl':
            df = collector.collect_nfl(args.seasons)
        elif sport == 'ncaaf':
            df = collector.collect_ncaaf(args.seasons)
        elif sport == 'mlb':
            df = collector.collect_mlb(args.seasons)
        else:
            continue
            
        collector.save_training_data(df, sport)
    
    print("\n" + "=" * 60)
    print("Data collection complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
