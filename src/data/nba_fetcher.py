"""
NBA Data Fetcher

Fetches NBA game stats and schedule data using the nba_api package.
"""

import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, List
import time

from nba_api.stats.endpoints import (
    leaguegamefinder,
    teamgamelog,
    scoreboardv2,
    leaguestandingsv3
)
from nba_api.stats.static import teams


class NBAFetcher:
    """Fetches NBA game and team statistics."""
    
    # Rate limiting to be respectful to NBA.com
    REQUEST_DELAY = 0.6  # seconds between requests
    
    def __init__(self):
        self._teams_cache = None
        self._last_request_time = 0
    
    def _rate_limit(self):
        """Ensure we don't overwhelm the NBA API."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.REQUEST_DELAY:
            time.sleep(self.REQUEST_DELAY - elapsed)
        self._last_request_time = time.time()
    
    def get_all_teams(self) -> pd.DataFrame:
        """Get all NBA teams with their IDs and abbreviations."""
        if self._teams_cache is None:
            nba_teams = teams.get_teams()
            self._teams_cache = pd.DataFrame(nba_teams)
        return self._teams_cache
    
    def get_team_id(self, team_abbr: str) -> Optional[int]:
        """Get team ID from abbreviation (e.g., 'LAL' -> 1610612747)."""
        teams_df = self.get_all_teams()
        match = teams_df[teams_df['abbreviation'] == team_abbr.upper()]
        if len(match) > 0:
            return match.iloc[0]['id']
        return None
    
    def get_team_game_log(
        self, 
        team_id: int, 
        season: str = "2024-25"
    ) -> pd.DataFrame:
        """
        Fetch game log for a specific team.
        
        Args:
            team_id: NBA team ID
            season: Season string (e.g., "2024-25")
            
        Returns:
            DataFrame with game results and stats
        """
        self._rate_limit()
        
        game_log = teamgamelog.TeamGameLog(
            team_id=team_id,
            season=season,
            season_type_all_star="Regular Season"
        )
        
        df = game_log.get_data_frames()[0]
        
        # Parse and clean the data
        df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
        df['IS_HOME'] = df['MATCHUP'].str.contains('vs.')
        df['OPPONENT'] = df['MATCHUP'].str.extract(r'(?:vs\.|@)\s*(\w+)')
        
        return df
    
    def get_all_games_for_season(self, season: str = "2024-25") -> pd.DataFrame:
        """
        Fetch all NBA games for a season.
        
        Args:
            season: Season string (e.g., "2024-25")
            
        Returns:
            DataFrame with all games and basic stats
        """
        self._rate_limit()
        
        game_finder = leaguegamefinder.LeagueGameFinder(
            season_nullable=season,
            league_id_nullable="00",  # NBA
            season_type_nullable="Regular Season"
        )
        
        games_df = game_finder.get_data_frames()[0]
        games_df['GAME_DATE'] = pd.to_datetime(games_df['GAME_DATE'])
        
        return games_df
    
    def get_todays_games(self) -> pd.DataFrame:
        """Get today's scheduled games (Raw API output)."""
        self._rate_limit()
        
        today = datetime.now().strftime('%Y-%m-%d')
        scoreboard = scoreboardv2.ScoreboardV2(game_date=today)
        
        games_df = scoreboard.get_data_frames()[0]  # GameHeader
        return games_df

    def get_schedule(self, date: datetime = None) -> pd.DataFrame:
        """
        Get game schedule for a date with standard columns.
        Returns: DataFrame with ['home_team', 'away_team', 'game_id']
        """
        # Note: nba_api ScoreboardV2 takes a date, default is today.
        
        if date is None:
            date = datetime.now()
            
        date_str = date.strftime('%Y-%m-%d')
        
        self._rate_limit()
        scoreboard = scoreboardv2.ScoreboardV2(game_date=date_str)
        games_df = scoreboard.get_data_frames()[0]
        
        if games_df.empty:
            return pd.DataFrame()
            
        # Map IDs to Abbreviations
        teams_df = self.get_all_teams()
        id_to_abbr = dict(zip(teams_df['id'], teams_df['abbreviation']))
        
        # Create standard dataframe
        result = pd.DataFrame()
        result['game_id'] = games_df['GAME_ID']
        result['home_team'] = games_df['HOME_TEAM_ID'].map(id_to_abbr)
        result['away_team'] = games_df['VISITOR_TEAM_ID'].map(id_to_abbr)
        
        return result

    def get_scores(self, date: datetime = None) -> pd.DataFrame:
        """Get game scores for a specific date."""
        if date is None:
            date = datetime.now()
        date_str = date.strftime('%Y-%m-%d')
        
        self._rate_limit()
        try:
            board = scoreboardv2.ScoreboardV2(game_date=date_str)
            header = board.get_data_frames()[0]
            linescore = board.get_data_frames()[1]
            
            if header.empty:
                return pd.DataFrame()
            
            # Ensure IDs are strings for camparison
            header['GAME_ID'] = header['GAME_ID'].astype(str)
            linescore['GAME_ID'] = linescore['GAME_ID'].astype(str)
            header['HOME_TEAM_ID'] = header['HOME_TEAM_ID'].astype(str)
            header['VISITOR_TEAM_ID'] = header['VISITOR_TEAM_ID'].astype(str)
            linescore['TEAM_ID'] = linescore['TEAM_ID'].astype(str)
            
            # Map IDs to Abbr
            teams_df = self.get_all_teams()
            id_to_abbr = dict(zip(teams_df['id'].astype(str), teams_df['abbreviation']))
            
            scores = []
            for _, game in header.iterrows():
                game_id = game['GAME_ID']
                home_id = game['HOME_TEAM_ID']
                away_id = game['VISITOR_TEAM_ID']
                status_id = game['GAME_STATUS_ID'] # 3 is Final
                
                # Find scores in linescore
                home_line = linescore[(linescore['GAME_ID'] == game_id) & (linescore['TEAM_ID'] == home_id)]
                away_line = linescore[(linescore['GAME_ID'] == game_id) & (linescore['TEAM_ID'] == away_id)]
                
                home_score = home_line['PTS'].iloc[0] if not home_line.empty else None
                away_score = away_line['PTS'].iloc[0] if not away_line.empty else None
                
                if status_id == 3:
                    status = 'Final'
                elif status_id == 2:
                    status = 'Live'
                else:
                    status = 'Scheduled'
                    
                scores.append({
                    'game_id': game_id,
                    'home_team': id_to_abbr.get(home_id),
                    'away_team': id_to_abbr.get(away_id),
                    'home_score': home_score,
                    'away_score': away_score,
                    'status': status
                })
                
            return pd.DataFrame(scores)
        except Exception as e:
            print(f"Error fetching NBA scores: {e}")
            return pd.DataFrame()
    
    def get_standings(self, season: str = "2024-25") -> pd.DataFrame:
        """Get current league standings."""
        self._rate_limit()
        
        standings = leaguestandingsv3.LeagueStandingsV3(
            season=season,
            season_type="Regular Season"
        )
        
        return standings.get_data_frames()[0]
    
    def fetch_historical_data(
        self, 
        seasons: List[str] = None
    ) -> pd.DataFrame:
        """
        Fetch historical game data for model training.
        
        Args:
            seasons: List of seasons to fetch (default: last 3 seasons)
            
        Returns:
            Combined DataFrame of all historical games
        """
        if seasons is None:
            seasons = ["2022-23", "2023-24", "2024-25"]
        
        all_games = []
        
        for season in seasons:
            print(f"Fetching {season} season data...")
            season_games = self.get_all_games_for_season(season)
            season_games['SEASON'] = season
            all_games.append(season_games)
            time.sleep(1)  # Extra delay between seasons
        
        combined = pd.concat(all_games, ignore_index=True)
        combined = combined.sort_values('GAME_DATE').reset_index(drop=True)
        
        return combined


def main():
    """Test the NBA fetcher."""
    fetcher = NBAFetcher()
    
    # Test getting teams
    print("Getting all NBA teams...")
    teams_df = fetcher.get_all_teams()
    print(f"Found {len(teams_df)} teams")
    print(teams_df[['full_name', 'abbreviation']].head())
    
    # Test getting Lakers game log
    print("\nGetting Lakers 2024-25 game log...")
    lal_id = fetcher.get_team_id("LAL")
    games = fetcher.get_team_game_log(lal_id, "2024-25")
    print(f"Found {len(games)} games")
    print(games[['GAME_DATE', 'MATCHUP', 'WL', 'PTS']].head())


if __name__ == "__main__":
    main()
