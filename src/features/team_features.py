"""
Team Feature Engineering

Generates predictive features from raw NBA game data.
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Tuple
from datetime import datetime, timedelta


class TeamFeatureEngine:
    """Generates team-level features for game prediction."""
    
    def __init__(
        self,
        rolling_windows: List[int] = None,
        min_games: int = 5
    ):
        """
        Initialize feature engine.
        
        Args:
            rolling_windows: Windows for rolling averages (default: [5, 10])
            min_games: Minimum games before features are valid
        """
        self.rolling_windows = rolling_windows or [5, 10]
        self.min_games = min_games
    
    def prepare_game_data(self, games_df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare raw game data for feature generation.
        
        Converts per-game records into a format with one row per game
        with both home and away team stats.
        """
        df = games_df.copy()
        
        # Standardize column names
        df.columns = df.columns.str.upper()
        
        # Ensure date is datetime
        if 'GAME_DATE' in df.columns:
            df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
        
        # Sort by date
        df = df.sort_values('GAME_DATE').reset_index(drop=True)
        
        return df
    
    def calculate_rolling_stats(
        self,
        df: pd.DataFrame,
        team_col: str = 'TEAM_ABBREVIATION',
        stats_cols: List[str] = None,
        window: int = 5
    ) -> pd.DataFrame:
        """
        Calculate rolling average statistics for each team.
        
        Args:
            df: Game data with one row per team per game
            team_col: Column with team identifier
            stats_cols: Columns to calculate rolling stats for
            window: Number of games for rolling window
            
        Returns:
            DataFrame with additional rolling stat columns
        """
        if stats_cols is None:
            stats_cols = ['PTS', 'REB', 'AST', 'FG_PCT', 'FG3_PCT', 'FT_PCT']
        
        # Filter to available columns
        stats_cols = [c for c in stats_cols if c in df.columns]
        
        df = df.sort_values(['TEAM_ABBREVIATION', 'GAME_DATE'])
        
        for col in stats_cols:
            # Rolling mean (excluding current game)
            df[f'{col}_L{window}'] = (
                df.groupby(team_col)[col]
                .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
            )
        
        return df
    
    def calculate_home_away_splits(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate home and away win percentages for each team."""
        df = df.copy()
        
        if 'IS_HOME' not in df.columns:
            df['IS_HOME'] = df['MATCHUP'].str.contains(r'vs\.')
        
        # Win indicator
        df['WIN'] = (df['WL'] == 'W').astype(int)
        
        # Rolling home/away win rates
        df = df.sort_values(['TEAM_ABBREVIATION', 'GAME_DATE'])
        
        # Home games only
        home_mask = df['IS_HOME']
        df.loc[home_mask, 'HOME_WIN_RATE'] = (
            df[home_mask]
            .groupby('TEAM_ABBREVIATION')['WIN']
            .transform(lambda x: x.shift(1).expanding().mean())
        )
        
        # Away games only
        away_mask = ~df['IS_HOME']
        df.loc[away_mask, 'AWAY_WIN_RATE'] = (
            df[away_mask]
            .groupby('TEAM_ABBREVIATION')['WIN']
            .transform(lambda x: x.shift(1).expanding().mean())
        )
        
        # Forward fill for games of opposite type
        df['HOME_WIN_RATE'] = df.groupby('TEAM_ABBREVIATION')['HOME_WIN_RATE'].ffill()
        df['AWAY_WIN_RATE'] = df.groupby('TEAM_ABBREVIATION')['AWAY_WIN_RATE'].ffill()
        
        return df
    
    def calculate_rest_days(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate days of rest since last game."""
        df = df.sort_values(['TEAM_ABBREVIATION', 'GAME_DATE'])
        
        df['PREV_GAME_DATE'] = df.groupby('TEAM_ABBREVIATION')['GAME_DATE'].shift(1)
        df['REST_DAYS'] = (df['GAME_DATE'] - df['PREV_GAME_DATE']).dt.days
        
        # Cap at 7 days, fill missing with 3 (average)
        df['REST_DAYS'] = df['REST_DAYS'].clip(0, 7).fillna(3)
        
        # Back-to-back indicator
        df['IS_B2B'] = (df['REST_DAYS'] <= 1).astype(int)
        
        return df
    
    def calculate_streak(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate current win/loss streak."""
        df = df.sort_values(['TEAM_ABBREVIATION', 'GAME_DATE'])
        
        df['WIN'] = (df['WL'] == 'W').astype(int)
        
        def get_streak(group):
            streaks = []
            current_streak = 0
            
            for i, win in enumerate(group['WIN']):
                if i == 0:
                    streaks.append(0)  # No prior game
                else:
                    streaks.append(current_streak)
                
                if win:
                    current_streak = current_streak + 1 if current_streak > 0 else 1
                else:
                    current_streak = current_streak - 1 if current_streak < 0 else -1
            
            group['STREAK'] = streaks
            return group
        
        df = df.groupby('TEAM_ABBREVIATION', group_keys=False).apply(get_streak)
        
        return df
    
    def calculate_offensive_rating(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate simplified offensive/defensive efficiency.
        
        True offensive rating requires possessions data which we approximate.
        """
        df = df.copy()
        
        # Approximate possessions using standard formula
        if all(c in df.columns for c in ['FGA', 'FTA', 'TOV', 'OREB']):
            df['POSS'] = df['FGA'] + 0.44 * df['FTA'] - df['OREB'] + df['TOV']
            df['OFF_RTG'] = 100 * df['PTS'] / df['POSS'].clip(lower=1)
        else:
            # Fallback: just use points
            df['OFF_RTG'] = df['PTS']
        
        return df
    
    def build_matchup_features(
        self,
        home_team_stats: pd.Series,
        away_team_stats: pd.Series
    ) -> dict:
        """
        Build features for a specific matchup.
        
        Args:
            home_team_stats: Stats for home team (most recent)
            away_team_stats: Stats for away team (most recent)
            
        Returns:
            Dictionary of matchup features
        """
        features = {}
        
        # Rolling stat differentials
        for window in self.rolling_windows:
            for stat in ['PTS', 'REB', 'AST']:
                home_stat = home_team_stats.get(f'{stat}_L{window}', 0)
                away_stat = away_team_stats.get(f'{stat}_L{window}', 0)
                features[f'{stat}_DIFF_L{window}'] = home_stat - away_stat
        
        # Win rates
        features['HOME_WIN_RATE'] = home_team_stats.get('HOME_WIN_RATE', 0.5)
        features['AWAY_WIN_RATE'] = away_team_stats.get('AWAY_WIN_RATE', 0.5)
        features['WIN_RATE_DIFF'] = features['HOME_WIN_RATE'] - features['AWAY_WIN_RATE']
        
        # Rest advantage
        features['HOME_REST'] = home_team_stats.get('REST_DAYS', 2)
        features['AWAY_REST'] = away_team_stats.get('REST_DAYS', 2)
        features['REST_ADVANTAGE'] = features['HOME_REST'] - features['AWAY_REST']
        
        # Streaks
        features['HOME_STREAK'] = home_team_stats.get('STREAK', 0)
        features['AWAY_STREAK'] = away_team_stats.get('STREAK', 0)
        features['STREAK_DIFF'] = features['HOME_STREAK'] - features['AWAY_STREAK']
        
        # Back-to-back disadvantage
        features['HOME_B2B'] = home_team_stats.get('IS_B2B', 0)
        features['AWAY_B2B'] = away_team_stats.get('IS_B2B', 0)
        
        return features
    
    def generate_training_data(
        self,
        games_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Generate training data from historical games.
        
        Args:
            games_df: Raw game data from NBA API
            
        Returns:
            X: Feature DataFrame
            y: Target Series (1 if home team won)
        """
        print("Preparing game data...")
        df = self.prepare_game_data(games_df)
        
        print("Calculating rolling stats...")
        for window in self.rolling_windows:
            df = self.calculate_rolling_stats(df, window=window)
        
        print("Calculating home/away splits...")
        df = self.calculate_home_away_splits(df)
        
        print("Calculating rest days...")
        df = self.calculate_rest_days(df)
        
        print("Calculating streaks...")
        df = self.calculate_streak(df)
        
        # Build feature matrix
        print("Building feature matrix...")
        
        # We need to pair home and away teams for each game
        # Group by game_id and create matchup features
        feature_rows = []
        targets = []
        
        # Get unique games
        if 'GAME_ID' in df.columns:
            game_groups = df.groupby('GAME_ID')
            
            for game_id, game in game_groups:
                if len(game) != 2:
                    continue  # Skip incomplete games
                
                home_row = game[game['IS_HOME'] == True]
                away_row = game[game['IS_HOME'] == False]
                
                if len(home_row) == 0 or len(away_row) == 0:
                    continue
                
                home_stats = home_row.iloc[0]
                away_stats = away_row.iloc[0]
                
                # Skip if not enough prior games
                if pd.isna(home_stats.get(f'PTS_L{self.rolling_windows[0]}')):
                    continue
                if pd.isna(away_stats.get(f'PTS_L{self.rolling_windows[0]}')):
                    continue
                
                features = self.build_matchup_features(home_stats, away_stats)
                features['GAME_ID'] = game_id
                features['GAME_DATE'] = home_stats['GAME_DATE']
                feature_rows.append(features)
                
                # Target: did home team win?
                targets.append(1 if home_stats['WL'] == 'W' else 0)
        
        X = pd.DataFrame(feature_rows)
        y = pd.Series(targets, name='HOME_WIN')
        
        print(f"Generated {len(X)} training samples")
        
        return X, y
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature column names used by the model."""
        features = []
        
        # Rolling stat differentials
        for window in self.rolling_windows:
            for stat in ['PTS', 'REB', 'AST']:
                features.append(f'{stat}_DIFF_L{window}')
        
        # Other features
        features.extend([
            'HOME_WIN_RATE', 'AWAY_WIN_RATE', 'WIN_RATE_DIFF',
            'HOME_REST', 'AWAY_REST', 'REST_ADVANTAGE',
            'HOME_STREAK', 'AWAY_STREAK', 'STREAK_DIFF',
            'HOME_B2B', 'AWAY_B2B'
        ])
        
        return features


def main():
    """Demo the feature engine."""
    engine = TeamFeatureEngine(rolling_windows=[5, 10])
    
    print("Team Feature Engine initialized")
    print(f"Rolling windows: {engine.rolling_windows}")
    print(f"Feature columns: {engine.get_feature_names()}")
    
    # Create sample matchup features
    home_stats = pd.Series({
        'PTS_L5': 115.2, 'REB_L5': 44.0, 'AST_L5': 26.5,
        'PTS_L10': 112.8, 'REB_L10': 43.5, 'AST_L10': 25.8,
        'HOME_WIN_RATE': 0.65, 'REST_DAYS': 2, 'STREAK': 3, 'IS_B2B': 0
    })
    
    away_stats = pd.Series({
        'PTS_L5': 108.4, 'REB_L5': 42.0, 'AST_L5': 24.0,
        'PTS_L10': 109.5, 'REB_L10': 41.8, 'AST_L10': 23.5,
        'AWAY_WIN_RATE': 0.45, 'REST_DAYS': 1, 'STREAK': -2, 'IS_B2B': 1
    })
    
    features = engine.build_matchup_features(home_stats, away_stats)
    print("\nSample matchup features:")
    for k, v in features.items():
        print(f"  {k}: {v:.2f}" if isinstance(v, float) else f"  {k}: {v}")


if __name__ == "__main__":
    main()
