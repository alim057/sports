"""
Multi-Sport Data Fetcher

Abstract base class and sport-specific implementations for fetching game data.
"""

from abc import ABC, abstractmethod
import pandas as pd
from datetime import datetime
from typing import List, Dict, Optional
import time


class SportFetcher(ABC):
    """Abstract base class for sport-specific data fetchers."""
    
    SPORT_NAME: str = "unknown"
    SPORT_KEY: str = "unknown"
    
    REQUEST_DELAY = 0.6
    
    def __init__(self):
        self._last_request_time = 0
    
    def _rate_limit(self):
        """Rate limiting for API calls."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.REQUEST_DELAY:
            time.sleep(self.REQUEST_DELAY - elapsed)
        self._last_request_time = time.time()
    
    @abstractmethod
    def get_all_teams(self) -> pd.DataFrame:
        """Get all teams in the league."""
        pass
    
    @abstractmethod
    def get_team_games(
        self, 
        team_id: str, 
        season: str = None
    ) -> pd.DataFrame:
        """Get games for a specific team."""
        pass
    
    @abstractmethod
    def get_schedule(
        self, 
        date: datetime = None
    ) -> pd.DataFrame:
        """Get game schedule for a date."""
        pass
    
    @abstractmethod
    def get_standings(self, season: str = None) -> pd.DataFrame:
        """Get current standings."""
        pass
    
    def get_historical_data(
        self,
        seasons: List[str] = None
    ) -> pd.DataFrame:
        """Fetch historical game data for model training."""
        if seasons is None:
            seasons = self.get_default_seasons()
        
        all_games = []
        for season in seasons:
            print(f"Fetching {self.SPORT_NAME} {season} season...")
            games = self._fetch_season_games(season)
            if not games.empty:
                games['SEASON'] = season
                all_games.append(games)
            time.sleep(1)
        
        if not all_games:
            return pd.DataFrame()
        
        return pd.concat(all_games, ignore_index=True)
    
    @abstractmethod
    def _fetch_season_games(self, season: str) -> pd.DataFrame:
        """Fetch all games for a season."""
        pass
    
    @abstractmethod
    def get_default_seasons(self) -> List[str]:
        """Get default seasons to fetch for training."""
        pass


class NFLFetcher(SportFetcher):
    """Fetches NFL game data using nfl_data_py."""
    
    SPORT_NAME = "NFL"
    SPORT_KEY = "americanfootball_nfl"
    
    
    TEAMS_URL = "https://github.com/nflverse/nflverse-data/releases/download/teams/teams_colors_logos.csv"
    SCHEDULE_URL = "http://www.habitatring.com/games.csv"
    
    def __init__(self):
        super().__init__()
        try:
            import nfl_data_py as nfl
            self.nfl = nfl
            self._available = True
        except ImportError:
            print("Warning: nfl_data_py not installed. Using fallback CSV downloader.")
            self._available = False
            self.nfl = None
            
    def _fetch_from_url(self, url: str) -> pd.DataFrame:
        """Helper to fetch CSV directly."""
        try:
            return pd.read_csv(url)
        except Exception as e:
            print(f"Error fetching from {url}: {e}")
            return pd.DataFrame()
    
    def get_all_teams(self) -> pd.DataFrame:
        try:
            if self._available:
                teams = self.nfl.import_team_desc()
            else:
                teams = self._fetch_from_url(self.TEAMS_URL)
                
            if teams.empty: return pd.DataFrame()
            return teams[['team_abbr', 'team_name', 'team_conf', 'team_division']]
        except Exception as e:
            print(f"Error fetching NFL teams: {e}")
            return pd.DataFrame()
    
    def get_team_games(
        self, 
        team_id: str, 
        season: str = None
    ) -> pd.DataFrame:
        season = season or "2024"
        try:
            if self._available:
                schedule = self.nfl.import_schedules([int(season)])
            else:
                schedule = self._fetch_from_url(self.SCHEDULE_URL)
                # Filter by season if using full schedule CSV
                if not schedule.empty and 'season' in schedule.columns:
                    schedule = schedule[schedule['season'] == int(season)]
                    
            if schedule.empty: return pd.DataFrame()
            
            team_games = schedule[
                (schedule['home_team'] == team_id) | 
                (schedule['away_team'] == team_id)
            ]
            return team_games
        except Exception as e:
            print(f"Error fetching NFL games: {e}")
            return pd.DataFrame()
    
    def get_schedule(self, date: datetime = None) -> pd.DataFrame:
        date = date or datetime.now()
        try:
            if self._available:
                schedule = self.nfl.import_schedules([date.year])
            else:
                schedule = self._fetch_from_url(self.SCHEDULE_URL)
                if not schedule.empty and 'season' in schedule.columns:
                    schedule = schedule[schedule['season'] == date.year]

            if not schedule.empty:
                schedule['gameday'] = pd.to_datetime(schedule['gameday'])
                return schedule[schedule['gameday'].dt.date == date.date()]
            return pd.DataFrame()
        except Exception as e:
            print(f"Error fetching NFL schedule: {e}")
            return pd.DataFrame()
    
    def get_standings(self, season: str = None) -> pd.DataFrame:
        # Fallback for standings is complex without the lib, return empty or implement later
        if not self._available:
            return pd.DataFrame()
        
        season = season or "2024"
        try:
            standings = self.nfl.import_seasonal_data([int(season)])
            return standings
        except Exception as e:
            print(f"Error fetching NFL standings: {e}")
            return pd.DataFrame()
    
    def _fetch_season_games(self, season: str) -> pd.DataFrame:
        try:
            if self._available:
                schedule = self.nfl.import_schedules([int(season)])
            else:
                schedule = self._fetch_from_url(self.SCHEDULE_URL)
                if not schedule.empty and 'season' in schedule.columns:
                    schedule = schedule[schedule['season'] == int(season)]

            if schedule.empty: return pd.DataFrame()

            # Filter to completed games
            completed = schedule[schedule['result'].notna()]
            return completed
        except Exception as e:
            print(f"Error fetching NFL season {season}: {e}")
            return pd.DataFrame()
    
    def get_default_seasons(self) -> List[str]:
        return ["2022", "2023", "2024"]


class MLBFetcher(SportFetcher):
    """Fetches MLB game data using MLB-StatsAPI."""
    
    SPORT_NAME = "MLB"
    SPORT_KEY = "baseball_mlb"
    
    def __init__(self):
        super().__init__()
        try:
            import statsapi
            self.api = statsapi
            self._available = True
        except ImportError:
            print("Warning: MLB-StatsAPI not installed. Run: pip install MLB-StatsAPI")
            self._available = False
    
    def get_all_teams(self) -> pd.DataFrame:
        if not self._available:
            return pd.DataFrame()
        
        try:
            teams_data = self.api.get('teams', {'sportId': 1})
            teams = []
            for team in teams_data.get('teams', []):
                teams.append({
                    'team_id': team['id'],
                    'team_abbr': team.get('abbreviation', ''),
                    'team_name': team.get('name', ''),
                    'division': team.get('division', {}).get('name', '')
                })
            return pd.DataFrame(teams)
        except Exception as e:
            print(f"Error fetching MLB teams: {e}")
            return pd.DataFrame()
    
    def get_team_games(
        self, 
        team_id: str, 
        season: str = None
    ) -> pd.DataFrame:
        if not self._available:
            return pd.DataFrame()
        
        season = season or "2024"
        try:
            schedule = self.api.schedule(
                team=team_id,
                start_date=f"{season}-03-01",
                end_date=f"{season}-11-30"
            )
            return pd.DataFrame(schedule)
        except Exception as e:
            print(f"Error fetching MLB games: {e}")
            return pd.DataFrame()
    
    def get_schedule(self, date: datetime = None) -> pd.DataFrame:
        if not self._available:
            return pd.DataFrame()
        
        date = date or datetime.now()
        try:
            date_str = date.strftime("%Y-%m-%d")
            schedule = self.api.schedule(date=date_str)
            return pd.DataFrame(schedule)
        except Exception as e:
            print(f"Error fetching MLB schedule: {e}")
            return pd.DataFrame()
    
    def get_standings(self, season: str = None) -> pd.DataFrame:
        if not self._available:
            return pd.DataFrame()
        
        try:
            standings_data = self.api.standings_data(leagueId="103,104")
            rows = []
            for div_id, div_data in standings_data.items():
                for team in div_data.get('teams', []):
                    rows.append({
                        'division': div_data.get('div_name', ''),
                        'team': team.get('name', ''),
                        'wins': team.get('w', 0),
                        'losses': team.get('l', 0),
                        'pct': team.get('w', 0) / max(1, team.get('w', 0) + team.get('l', 0))
                    })
            return pd.DataFrame(rows)
        except Exception as e:
            print(f"Error fetching MLB standings: {e}")
            return pd.DataFrame()
    
    def _fetch_season_games(self, season: str) -> pd.DataFrame:
        if not self._available:
            return pd.DataFrame()
        
        try:
            schedule = self.api.schedule(
                start_date=f"{season}-03-01",
                end_date=f"{season}-11-30"
            )
            return pd.DataFrame(schedule)
        except Exception as e:
            print(f"Error fetching MLB season {season}: {e}")
            return pd.DataFrame()
    
    def get_default_seasons(self) -> List[str]:
        return ["2022", "2023", "2024"]


class NHLFetcher(SportFetcher):
    """Fetches NHL game data using nhl-api-py."""
    
    SPORT_NAME = "NHL"
    SPORT_KEY = "icehockey_nhl"
    
    def __init__(self):
        super().__init__()
        try:
            from nhlpy import NHLClient
            self.client = NHLClient()
            self._available = True
        except ImportError:
            print("Warning: nhl-api-py not installed. Run: pip install nhl-api-py")
            self._available = False
    
    def get_all_teams(self) -> pd.DataFrame:
        if not self._available:
            return pd.DataFrame()
        
        try:
            teams_data = self.client.teams.all_franchises()
            teams = []
            for team in teams_data:
                teams.append({
                    'team_id': team.get('id'),
                    'team_name': team.get('teamCommonName', ''),
                    'full_name': team.get('fullName', ''),
                })
            return pd.DataFrame(teams)
        except Exception as e:
            print(f"Error fetching NHL teams: {e}")
            return pd.DataFrame()
    
    def get_team_games(
        self, 
        team_id: str, 
        season: str = None
    ) -> pd.DataFrame:
        # NHL API structure different - would need schedule endpoint
        return pd.DataFrame()
    
    def get_schedule(self, date: datetime = None) -> pd.DataFrame:
        if not self._available:
            return pd.DataFrame()
        
        date = date or datetime.now()
        try:
            schedule = self.client.schedule.get_schedule(date=date.strftime("%Y-%m-%d"))
            games = []
            for game in schedule.get('games', []):
                games.append({
                    'game_id': game.get('id'),
                    'home_team': game.get('homeTeam', {}).get('abbrev', ''),
                    'away_team': game.get('awayTeam', {}).get('abbrev', ''),
                    'start_time': game.get('startTimeUTC', ''),
                    'venue': game.get('venue', {}).get('default', '')
                })
            return pd.DataFrame(games)
        except Exception as e:
            print(f"Error fetching NHL schedule: {e}")
            return pd.DataFrame()
    
    def get_standings(self, season: str = None) -> pd.DataFrame:
        if not self._available:
            return pd.DataFrame()
        
        try:
            standings = self.client.standings.get_standings()
            rows = []
            for entry in standings.get('standings', []):
                rows.append({
                    'team': entry.get('teamName', {}).get('default', ''),
                    'team_abbr': entry.get('teamAbbrev', {}).get('default', ''),
                    'wins': entry.get('wins', 0),
                    'losses': entry.get('losses', 0),
                    'points': entry.get('points', 0),
                    'division': entry.get('divisionName', '')
                })
            return pd.DataFrame(rows)
        except Exception as e:
            print(f"Error fetching NHL standings: {e}")
            return pd.DataFrame()
    
    def _fetch_season_games(self, season: str) -> pd.DataFrame:
        # Would need to iterate through dates for the season
        return pd.DataFrame()
    
    def get_default_seasons(self) -> List[str]:
        return ["2022-23", "2023-24", "2024-25"]


class SportFetcherFactory:
    """Factory for creating sport-specific fetchers."""
    
    FETCHERS = {
        'nba': None,  # Use existing NBAFetcher
        'nfl': NFLFetcher,
        'mlb': MLBFetcher,
        'nhl': NHLFetcher
    }
    
    @classmethod
    def get_fetcher(cls, sport: str) -> SportFetcher:
        """Get a fetcher for the specified sport."""
        sport = sport.lower()
        
        if sport == 'nba':
            from data.nba_fetcher import NBAFetcher
            return NBAFetcher()
        
        fetcher_class = cls.FETCHERS.get(sport)
        if fetcher_class is None:
            raise ValueError(f"Unknown sport: {sport}")
        
        return fetcher_class()
    
    @classmethod
    def available_sports(cls) -> List[str]:
        """Get list of available sports."""
        return list(cls.FETCHERS.keys())


def main():
    """Demo multi-sport fetchers."""
    print("Multi-Sport Fetcher Demo")
    print("=" * 50)
    
    # Test NFL
    print("\n[NFL]")
    nfl = NFLFetcher()
    teams = nfl.get_all_teams()
    if not teams.empty:
        print(f"Found {len(teams)} NFL teams")
        print(teams[['team_abbr', 'team_name']].head())
    else:
        print("NFL data not available")
    
    # Test MLB
    print("\n[MLB]")
    mlb = MLBFetcher()
    teams = mlb.get_all_teams()
    if not teams.empty:
        print(f"Found {len(teams)} MLB teams")
        print(teams[['team_abbr', 'team_name']].head())
    else:
        print("MLB data not available")
    
    # Test NHL
    print("\n[NHL]")
    nhl = NHLFetcher()
    standings = nhl.get_standings()
    if not standings.empty:
        print(f"Found {len(standings)} NHL teams in standings")
        print(standings[['team_abbr', 'wins', 'losses', 'points']].head())
    else:
        print("NHL data not available")


if __name__ == "__main__":
    main()
