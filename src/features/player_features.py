"""
Player Feature Engineering

Generates player-level features for enhanced game prediction.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.player_fetcher import PlayerFetcher


class PlayerFeatureEngine:
    """Generates player-based features for game prediction."""
    
    def __init__(self, season: str = "2024-25"):
        """
        Initialize player feature engine.
        
        Args:
            season: NBA season for player stats
        """
        self.season = season
        self.player_fetcher = PlayerFetcher()
        self._team_ratings_cache = {}
    
    def get_team_player_features(
        self,
        team_id: int,
        use_cache: bool = True
    ) -> Dict[str, float]:
        """
        Get player-based features for a team.
        
        Args:
            team_id: NBA team ID
            use_cache: Whether to use cached ratings
            
        Returns:
            Dictionary of player features
        """
        if use_cache and team_id in self._team_ratings_cache:
            return self._team_ratings_cache[team_id]
        
        ratings = self.player_fetcher.get_team_player_ratings(
            team_id, 
            self.season
        )
        
        if use_cache:
            self._team_ratings_cache[team_id] = ratings
        
        return ratings
    
    def build_player_matchup_features(
        self,
        home_team_id: int,
        away_team_id: int
    ) -> Dict[str, float]:
        """
        Build matchup features comparing team player ratings.
        
        Args:
            home_team_id: Home team NBA ID
            away_team_id: Away team NBA ID
            
        Returns:
            Dictionary of player-based matchup features
        """
        home_ratings = self.get_team_player_features(home_team_id)
        away_ratings = self.get_team_player_features(away_team_id)
        
        features = {}
        
        # Raw features for each team
        for key, value in home_ratings.items():
            features[f'home_{key}'] = value
        
        for key, value in away_ratings.items():
            features[f'away_{key}'] = value
        
        # Differential features
        features['player_impact_diff'] = (
            home_ratings['total_player_impact'] - 
            away_ratings['total_player_impact']
        )
        
        features['star_power_diff'] = (
            home_ratings['star_player_pts'] -
            away_ratings['star_player_pts']
        )
        
        features['bench_depth_diff'] = (
            home_ratings['bench_depth_pts'] -
            away_ratings['bench_depth_pts']
        )
        
        features['avg_pts_diff'] = (
            home_ratings['avg_player_pts'] -
            away_ratings['avg_player_pts']
        )
        
        return features
    
    def get_feature_names(self) -> List[str]:
        """Get list of player feature column names."""
        return [
            # Home team player features
            'home_total_player_impact',
            'home_avg_player_pts',
            'home_avg_player_ast',
            'home_avg_player_reb',
            'home_star_player_pts',
            'home_bench_depth_pts',
            # Away team player features
            'away_total_player_impact',
            'away_avg_player_pts',
            'away_avg_player_ast',
            'away_avg_player_reb',
            'away_star_player_pts',
            'away_bench_depth_pts',
            # Differential features
            'player_impact_diff',
            'star_power_diff',
            'bench_depth_diff',
            'avg_pts_diff',
        ]


class CombinedFeatureEngine:
    """Combines team and player features for enhanced predictions."""
    
    def __init__(
        self,
        rolling_windows: List[int] = None,
        season: str = "2024-25",
        include_player_features: bool = True
    ):
        """
        Initialize combined feature engine.
        
        Args:
            rolling_windows: Windows for rolling stats
            season: NBA season
            include_player_features: Whether to include player features
        """
        from features.team_features import TeamFeatureEngine
        
        self.team_engine = TeamFeatureEngine(rolling_windows)
        self.include_player_features = include_player_features
        
        if include_player_features:
            self.player_engine = PlayerFeatureEngine(season)
        else:
            self.player_engine = None
    
    def build_full_matchup_features(
        self,
        home_team_stats: pd.Series,
        away_team_stats: pd.Series,
        home_team_id: Optional[int] = None,
        away_team_id: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Build complete feature set for a matchup.
        
        Args:
            home_team_stats: Team rolling stats for home team
            away_team_stats: Team rolling stats for away team
            home_team_id: NBA team ID for home team (for player features)
            away_team_id: NBA team ID for away team (for player features)
            
        Returns:
            Combined dictionary of all features
        """
        # Get team features
        features = self.team_engine.build_matchup_features(
            home_team_stats,
            away_team_stats
        )
        
        # Add player features if enabled and IDs provided
        if (self.include_player_features and 
            self.player_engine and 
            home_team_id and 
            away_team_id):
            
            player_features = self.player_engine.build_player_matchup_features(
                home_team_id,
                away_team_id
            )
            features.update(player_features)
        
        return features
    
    def get_all_feature_names(self) -> List[str]:
        """Get all feature column names."""
        names = self.team_engine.get_feature_names()
        
        if self.include_player_features and self.player_engine:
            names.extend(self.player_engine.get_feature_names())
        
        return names


def main():
    """Demo player features."""
    from nba_api.stats.static import teams
    
    print("Player Feature Engine Demo")
    print("=" * 50)
    
    engine = PlayerFeatureEngine(season="2024-25")
    
    # Get team IDs
    nba_teams = teams.get_teams()
    lakers = next((t for t in nba_teams if t['abbreviation'] == 'LAL'), None)
    warriors = next((t for t in nba_teams if t['abbreviation'] == 'GSW'), None)
    
    if lakers and warriors:
        print(f"\nBuilding matchup features: Lakers (home) vs Warriors (away)")
        
        features = engine.build_player_matchup_features(
            lakers['id'],
            warriors['id']
        )
        
        print("\nPlayer-based matchup features:")
        for k, v in features.items():
            if 'diff' in k:
                print(f"  {k}: {v:+.1f}")
            else:
                print(f"  {k}: {v:.1f}")


if __name__ == "__main__":
    main()
