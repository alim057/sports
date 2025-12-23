
"""
NFL Feature Engineering

Generates predictive features from raw NFL game data.
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Tuple, Dict
from datetime import datetime

class NFLFeatureEngine:
    """Generates team-level features for NFL game prediction."""
    
    def __init__(
        self,
        rolling_windows: List[int] = None,
        min_games: int = 3
    ):
        """
        Initialize feature engine.
        
        Args:
            rolling_windows: Windows for rolling averages (default: [3, 5])
            min_games: Minimum games before features are valid
        """
        self.rolling_windows = rolling_windows or [3, 5]
        self.min_games = min_games
    
    def prepare_game_data(self, games_df: pd.DataFrame) -> pd.DataFrame:
        """Prepare raw game data for feature generation."""
        df = games_df.copy()
        
        # Standardize column names (map nfl_data_py names to standard)
        # Assuming fetcher returns standard columns, but just in case:
        col_map = {
            'home_team': 'HOME_TEAM',
            'away_team': 'AWAY_TEAM',
            'home_score': 'HOME_SCORE',
            'away_score': 'AWAY_SCORE',
            'week': 'WEEK',
            'season': 'SEASON',
            'game_id': 'GAME_ID',
            'gameday': 'GAME_DATE'
        }
        df = df.rename(columns=col_map)
        
        # Ensure date is datetime
        if 'GAME_DATE' in df.columns:
            df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
            
        return df

    def build_matchup_features(
        self,
        home_team_stats: pd.Series,
        away_team_stats: pd.Series
    ) -> Dict[str, float]:
        """
        Build features for a specific NFL matchup.
        
        Args:
            home_team_stats: Stats for home team
            away_team_stats: Stats for away team
            
        Returns:
            Dictionary of matchup features
        """
        # Features must match training script:
        # 'HOME_WIN_RATE', 'AWAY_WIN_RATE', 
        # 'HOME_PTS_L5', 'AWAY_PTS_L5', 
        # 'HOME_PA_L5', 'AWAY_PA_L5', 
        # 'PTS_DIFF_L5', 'PA_DIFF_L5'
        
        features = {}
        
        # Extract stats with defaults
        h_win = home_team_stats.get('WIN_RATE', 0.5)
        a_win = away_team_stats.get('WIN_RATE', 0.5)
        
        h_pts = home_team_stats.get('PTS_L5', 20.0)
        a_pts = away_team_stats.get('PTS_L5', 20.0)
        
        # We need PA (Points Allowed) which we didn't explicitly calc in get_team_recent_stats yet
        # But we can approximate or defaults.
        # In get_team_recent_stats for NFL we set simple stats.
        # Let's update this to be robust.
        
        h_pa = home_team_stats.get('PA_L5', 20.0)
        a_pa = away_team_stats.get('PA_L5', 20.0)
        
        features['HOME_WIN_RATE'] = h_win
        features['AWAY_WIN_RATE'] = a_win
        features['HOME_PTS_L5'] = h_pts
        features['AWAY_PTS_L5'] = a_pts
        features['HOME_PA_L5'] = h_pa
        features['AWAY_PA_L5'] = a_pa
        features['PTS_DIFF_L5'] = h_pts - a_pts
        features['PA_DIFF_L5'] = h_pa - a_pa
        
        return features

    def get_feature_names(self) -> List[str]:
        """Get list of feature column names used by the model."""
        return [
            'HOME_WIN_RATE', 'AWAY_WIN_RATE', 
            'HOME_PTS_L5', 'AWAY_PTS_L5', 
            'HOME_PA_L5', 'AWAY_PA_L5', 
            'PTS_DIFF_L5', 'PA_DIFF_L5'
        ]
