"""
Player Data Fetcher

Fetches individual player statistics from NBA API.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, List, Dict
import time

from nba_api.stats.endpoints import (
    playergamelog,
    commonteamroster,
    playerindex,
    leaguedashplayerstats
)
from nba_api.stats.static import players, teams


class PlayerFetcher:
    """Fetches NBA player statistics."""
    
    REQUEST_DELAY = 0.6  # Rate limiting
    
    def __init__(self):
        self._players_cache = None
        self._teams_cache = None
        self._last_request_time = 0
    
    def _rate_limit(self):
        """Ensure we don't overwhelm the NBA API."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.REQUEST_DELAY:
            time.sleep(self.REQUEST_DELAY - elapsed)
        self._last_request_time = time.time()
    
    def get_all_players(self) -> pd.DataFrame:
        """Get all NBA players."""
        if self._players_cache is None:
            all_players = players.get_players()
            self._players_cache = pd.DataFrame(all_players)
        return self._players_cache
    
    def get_player_id(self, player_name: str) -> Optional[int]:
        """Get player ID from name."""
        players_df = self.get_all_players()
        match = players_df[
            players_df['full_name'].str.lower().str.contains(player_name.lower())
        ]
        if len(match) > 0:
            return match.iloc[0]['id']
        return None
    
    def get_team_roster(
        self, 
        team_id: int, 
        season: str = "2024-25"
    ) -> pd.DataFrame:
        """Get current roster for a team."""
        self._rate_limit()
        
        roster = commonteamroster.CommonTeamRoster(
            team_id=team_id,
            season=season
        )
        
        return roster.get_data_frames()[0]
    
    def get_player_game_log(
        self,
        player_id: int,
        season: str = "2024-25"
    ) -> pd.DataFrame:
        """Get game log for a specific player."""
        self._rate_limit()
        
        game_log = playergamelog.PlayerGameLog(
            player_id=player_id,
            season=season,
            season_type_all_star="Regular Season"
        )
        
        df = game_log.get_data_frames()[0]
        df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
        
        return df
    
    def get_league_player_stats(
        self,
        season: str = "2024-25",
        per_mode: str = "PerGame"
    ) -> pd.DataFrame:
        """
        Get league-wide player statistics.
        
        Args:
            season: NBA season
            per_mode: "PerGame", "Totals", "Per36", etc.
            
        Returns:
            DataFrame with all player stats for the season
        """
        self._rate_limit()
        
        stats = leaguedashplayerstats.LeagueDashPlayerStats(
            season=season,
            per_mode_detailed=per_mode,
            season_type_all_star="Regular Season"
        )
        
        return stats.get_data_frames()[0]
    
    def get_top_players_by_team(
        self,
        team_id: int,
        season: str = "2024-25",
        top_n: int = 8
    ) -> pd.DataFrame:
        """
        Get top N players for a team by minutes played.
        
        Args:
            team_id: NBA team ID
            season: Season
            top_n: Number of top players to return
            
        Returns:
            DataFrame with top players and their stats
        """
        # Get league stats and filter by team
        all_stats = self.get_league_player_stats(season)
        team_stats = all_stats[all_stats['TEAM_ID'] == team_id]
        
        # Sort by minutes and get top N
        team_stats = team_stats.sort_values('MIN', ascending=False)
        return team_stats.head(top_n)
    
    def calculate_player_impact_score(
        self,
        player_stats: pd.Series
    ) -> float:
        """
        Calculate a simple player impact score.
        
        This is a simplified version of metrics like PER or RAPTOR.
        """
        # Weighted combination of key stats
        score = (
            player_stats.get('PTS', 0) * 1.0 +
            player_stats.get('REB', 0) * 0.8 +
            player_stats.get('AST', 0) * 1.2 +
            player_stats.get('STL', 0) * 1.5 +
            player_stats.get('BLK', 0) * 1.5 -
            player_stats.get('TOV', 0) * 1.0 +
            player_stats.get('FG_PCT', 0) * 20 +
            player_stats.get('FG3_PCT', 0) * 15
        )
        
        return score
    
    def get_team_player_ratings(
        self,
        team_id: int,
        season: str = "2024-25"
    ) -> Dict[str, float]:
        """
        Get aggregated player ratings for a team.
        
        Returns:
            Dictionary with team player metrics
        """
        try:
            top_players = self.get_top_players_by_team(team_id, season, top_n=8)
            
            if top_players.empty:
                return self._get_default_player_ratings()
            
            # Calculate aggregate stats for top players
            total_impact = sum(
                self.calculate_player_impact_score(row)
                for _, row in top_players.iterrows()
            )
            
            avg_pts = top_players['PTS'].mean()
            avg_ast = top_players['AST'].mean()
            avg_reb = top_players['REB'].mean()
            star_power = top_players.iloc[0]['PTS']  # Best player's scoring
            depth_score = top_players.iloc[3:8]['PTS'].sum()  # Bench scoring
            
            return {
                'total_player_impact': total_impact,
                'avg_player_pts': avg_pts,
                'avg_player_ast': avg_ast,
                'avg_player_reb': avg_reb,
                'star_player_pts': star_power,
                'bench_depth_pts': depth_score,
            }
        except Exception as e:
            print(f"Warning: Could not fetch player ratings: {e}")
            return self._get_default_player_ratings()
    
    def _get_default_player_ratings(self) -> Dict[str, float]:
        """Return default player ratings when data unavailable."""
        return {
            'total_player_impact': 150.0,
            'avg_player_pts': 12.0,
            'avg_player_ast': 3.0,
            'avg_player_reb': 5.0,
            'star_player_pts': 25.0,
            'bench_depth_pts': 30.0,
        }


def main():
    """Test player fetcher."""
    fetcher = PlayerFetcher()
    
    print("Player Fetcher Demo")
    print("=" * 50)
    
    # Get Lakers team ID
    nba_teams = teams.get_teams()
    lakers = next((t for t in nba_teams if t['abbreviation'] == 'LAL'), None)
    
    if lakers:
        print(f"\nFetching Lakers roster...")
        roster = fetcher.get_team_roster(lakers['id'], "2024-25")
        print(f"Found {len(roster)} players on roster")
        print(roster[['PLAYER', 'POSITION', 'AGE']].head())
        
        print(f"\nFetching top Lakers players by minutes...")
        top_players = fetcher.get_top_players_by_team(lakers['id'], "2024-25", top_n=5)
        if not top_players.empty:
            print(top_players[['PLAYER_NAME', 'MIN', 'PTS', 'AST', 'REB']].to_string())
        
        print(f"\nCalculating team player ratings...")
        ratings = fetcher.get_team_player_ratings(lakers['id'], "2024-25")
        for k, v in ratings.items():
            print(f"  {k}: {v:.1f}")


if __name__ == "__main__":
    main()
