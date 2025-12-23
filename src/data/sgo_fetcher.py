"""
Sports Game Odds API Fetcher

Unified data fetcher for game data, player props, and betting odds
from the Sports Game Odds (SGO) API.

Free tier limits:
- ~10 requests per minute
- All sports: NBA, NFL, NCAAF, MLB
- 9+ bookmakers: FanDuel, DraftKings, Caesars, BetMGM, etc.
"""

import requests
import pandas as pd
import time
import os
import yaml
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any
from pathlib import Path


class SGOFetcher:
    """
    Sports Game Odds API client.
    
    Fetches game data, odds, and player props from a single unified source.
    """
    
    BASE_URL = "https://api.sportsgameodds.com/v2"
    
    # League IDs for SGO API
    LEAGUES = {
        "nba": "NBA",
        "nfl": "NFL", 
        "ncaaf": "NCAAF",
        "mlb": "MLB",
        "ncaab": "NCAAB",
        "nhl": "NHL"
    }
    
    # Rate limiting: 10 requests/minute = 1 request per 6 seconds minimum
    REQUEST_DELAY = 7  # seconds between requests (safe margin)
    
    def __init__(self, api_key: Optional[str] = None, config_path: Optional[str] = None):
        """
        Initialize the SGO Fetcher.
        
        Args:
            api_key: SGO API key (or load from environment/config)
            config_path: Path to config.yaml
        """
        self.api_key = api_key or os.environ.get("SGO_API_KEY")
        
        if not self.api_key and config_path:
            self.api_key = self._load_api_key_from_config(config_path)
        
        if not self.api_key:
            # Try default config path
            default_config = Path(__file__).parent.parent.parent / "config" / "config.yaml"
            if default_config.exists():
                self.api_key = self._load_api_key_from_config(str(default_config))
        
        self._last_request_time = 0
        self._request_count = 0
    
    def _load_api_key_from_config(self, config_path: str) -> Optional[str]:
        """Load API key from config file."""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config.get('sgo_api', {}).get('api_key')
        except Exception:
            return None
    
    def _rate_limit(self):
        """Enforce rate limiting between API requests."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.REQUEST_DELAY:
            time.sleep(self.REQUEST_DELAY - elapsed)
        self._last_request_time = time.time()
        self._request_count += 1
    
    def _make_request(self, endpoint: str, params: Dict = None) -> Dict:
        """
        Make a request to the SGO API.
        
        Args:
            endpoint: API endpoint (e.g., "/events", "/leagues")
            params: Query parameters
            
        Returns:
            JSON response as dictionary
        """
        if not self.api_key:
            raise ValueError("SGO API key not configured. Set SGO_API_KEY environment variable.")
        
        self._rate_limit()
        
        url = f"{self.BASE_URL}{endpoint}"
        headers = {"x-api-key": self.api_key}
        
        try:
            response = requests.get(url, headers=headers, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            if response.status_code == 429:
                print(f"[SGO] Rate limit hit. Waiting 60 seconds...")
                time.sleep(60)
                return self._make_request(endpoint, params)
            raise Exception(f"SGO API error: {e}")
        except requests.exceptions.RequestException as e:
            raise Exception(f"SGO API request failed: {e}")
    
    # =========================================================================
    # League & Team Data
    # =========================================================================
    
    def get_leagues(self) -> List[Dict]:
        """Get list of available leagues."""
        result = self._make_request("/leagues")
        return result.get("data", [])
    
    def get_league_ids(self) -> List[str]:
        """Get list of league IDs."""
        leagues = self.get_leagues()
        return [league["leagueID"] for league in leagues]
    
    # =========================================================================
    # Event (Game) Data
    # =========================================================================
    
    def fetch_events(
        self, 
        league: str,
        odds_available: bool = True,
        started: Optional[bool] = None,
        limit: int = 100
    ) -> List[Dict]:
        """
        Fetch events (games) for a league.
        
        Args:
            league: League key (nba, nfl, ncaaf, mlb)
            odds_available: Only fetch events with odds available
            started: Filter by started status (None = all)
            limit: Maximum events to return
            
        Returns:
            List of event dictionaries
        """
        league_id = self.LEAGUES.get(league.lower(), league.upper())
        
        params = {
            "leagueID": league_id,
            "limit": limit
        }
        
        if odds_available:
            params["oddsAvailable"] = "true"
        
        if started is not None:
            params["started"] = str(started).lower()
        
        result = self._make_request("/events", params)
        return result.get("data", [])
    
    def get_upcoming_games(self, league: str) -> pd.DataFrame:
        """
        Get upcoming games with odds as a DataFrame.
        
        Args:
            league: League key (nba, nfl, ncaaf, mlb)
            
        Returns:
            DataFrame with game info and odds
        """
        events = self.fetch_events(league, odds_available=True, started=False)
        
        games = []
        for event in events:
            teams = event.get("teams", {})
            status = event.get("status", {})
            odds = event.get("odds", {})
            
            # Extract main market odds
            moneyline_home = None
            moneyline_away = None
            spread_home = None
            spread_line = None
            total_line = None
            
            for odd_id, odd_data in odds.items():
                if odd_data.get("betTypeID") == "ml" and odd_data.get("sideID") == "home":
                    moneyline_home = odd_data.get("bookOdds")
                elif odd_data.get("betTypeID") == "ml" and odd_data.get("sideID") == "away":
                    moneyline_away = odd_data.get("bookOdds")
                elif odd_data.get("betTypeID") == "spread" and odd_data.get("sideID") == "home":
                    spread_home = odd_data.get("bookOdds")
                    spread_line = odd_data.get("bookOverUnder")
                elif odd_data.get("betTypeID") == "ou" and odd_data.get("sideID") == "over" and odd_data.get("statEntityID") == "game":
                    total_line = odd_data.get("bookOverUnder")
            
            games.append({
                "event_id": event.get("eventID"),
                "league": league.upper(),
                "home_team": teams.get("home", {}).get("teamID", "").replace("_NBA", "").replace("_NFL", "").replace("_", " "),
                "away_team": teams.get("away", {}).get("teamID", "").replace("_NBA", "").replace("_NFL", "").replace("_", " "),
                "start_time": status.get("startsAt"),
                "status": status.get("displayLong", "Scheduled"),
                "moneyline_home": moneyline_home,
                "moneyline_away": moneyline_away,
                "spread_home": spread_home,
                "spread_line": spread_line,
                "total_line": total_line,
                "odds_available": status.get("oddsAvailable", False)
            })
        
        return pd.DataFrame(games)
    
    # =========================================================================
    # Odds Data
    # =========================================================================
    
    def get_live_odds(self, league: str) -> pd.DataFrame:
        """
        Get all live odds for a league.
        
        Args:
            league: League key (nba, nfl, ncaaf, mlb)
            
        Returns:
            DataFrame with detailed odds from all bookmakers
        """
        events = self.fetch_events(league, odds_available=True)
        
        odds_data = []
        for event in events:
            event_id = event.get("eventID")
            teams = event.get("teams", {})
            home_team = teams.get("home", {}).get("teamID", "")
            away_team = teams.get("away", {}).get("teamID", "")
            start_time = event.get("status", {}).get("startsAt")
            
            odds = event.get("odds", {})
            
            for odd_id, odd_data in odds.items():
                # Skip player props for this method
                if odd_data.get("playerID"):
                    continue
                
                by_bookmaker = odd_data.get("byBookmaker", {})
                
                for bookmaker, book_data in by_bookmaker.items():
                    odds_data.append({
                        "event_id": event_id,
                        "home_team": home_team,
                        "away_team": away_team,
                        "start_time": start_time,
                        "odd_id": odd_id,
                        "market_name": odd_data.get("marketName"),
                        "stat_id": odd_data.get("statID"),
                        "bet_type": odd_data.get("betTypeID"),
                        "side": odd_data.get("sideID"),
                        "bookmaker": bookmaker,
                        "book_odds": book_data.get("odds"),
                        "line": book_data.get("overUnder"),
                        "fair_odds": odd_data.get("fairOdds"),
                        "fair_line": odd_data.get("fairOverUnder"),
                        "fetched_at": datetime.now(timezone.utc).isoformat()
                    })
        
        return pd.DataFrame(odds_data)
    
    def get_best_odds(self, league: str, market: str = "ml") -> pd.DataFrame:
        """
        Get the best available odds across all bookmakers.
        
        Args:
            league: League key
            market: Market type (ml, spread, ou)
            
        Returns:
            DataFrame with best odds for each game
        """
        odds_df = self.get_live_odds(league)
        
        if odds_df.empty:
            return odds_df
        
        # Filter by market
        odds_df = odds_df[odds_df["bet_type"] == market]
        
        # Convert odds to numeric for comparison
        def parse_odds(odds_str):
            if not odds_str:
                return None
            try:
                return int(odds_str.replace("+", ""))
            except:
                return None
        
        odds_df["odds_numeric"] = odds_df["book_odds"].apply(parse_odds)
        
        # Group by event and side, get best odds
        best_odds = odds_df.loc[odds_df.groupby(["event_id", "side"])["odds_numeric"].idxmax()]
        
        return best_odds
    
    # =========================================================================
    # Player Props
    # =========================================================================
    
    def get_player_props(self, league: str) -> pd.DataFrame:
        """
        Get player prop bets for a league.
        
        Args:
            league: League key (nba, nfl, ncaaf, mlb)
            
        Returns:
            DataFrame with player prop data
        """
        events = self.fetch_events(league, odds_available=True)
        
        props_data = []
        for event in events:
            event_id = event.get("eventID")
            players = event.get("players", {})
            odds = event.get("odds", {})
            start_time = event.get("status", {}).get("startsAt")
            
            for odd_id, odd_data in odds.items():
                player_id = odd_data.get("playerID")
                if not player_id:
                    continue
                
                player_info = players.get(player_id, {})
                
                props_data.append({
                    "event_id": event_id,
                    "start_time": start_time,
                    "player_id": player_id,
                    "player_name": player_info.get("name", ""),
                    "team_id": player_info.get("teamID", ""),
                    "stat_type": odd_data.get("statID"),
                    "period": odd_data.get("periodID"),
                    "bet_type": odd_data.get("betTypeID"),
                    "side": odd_data.get("sideID"),
                    "market_name": odd_data.get("marketName"),
                    "line": odd_data.get("bookOverUnder"),
                    "book_odds": odd_data.get("bookOdds"),
                    "fair_odds": odd_data.get("fairOdds"),
                    "fair_line": odd_data.get("fairOverUnder")
                })
        
        return pd.DataFrame(props_data)
    
    # =========================================================================
    # Utility Methods
    # =========================================================================
    
    def fetch_all_sports(self, sports: List[str] = None) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for multiple sports.
        
        Args:
            sports: List of sport keys (default: all major US sports)
            
        Returns:
            Dictionary mapping sport to DataFrame of games
        """
        if sports is None:
            sports = ["nba", "nfl", "ncaaf", "mlb"]
        
        results = {}
        for sport in sports:
            try:
                print(f"[SGO] Fetching {sport.upper()}...")
                results[sport] = self.get_upcoming_games(sport)
                print(f"[SGO] Found {len(results[sport])} {sport.upper()} games with odds")
            except Exception as e:
                print(f"[SGO] Error fetching {sport}: {e}")
                results[sport] = pd.DataFrame()
        
        return results
    
    @staticmethod
    def american_to_probability(american_odds: str) -> float:
        """
        Convert American odds to implied probability.
        
        Args:
            american_odds: Odds string like "+150" or "-200"
            
        Returns:
            Implied probability (0-1)
        """
        if not american_odds:
            return 0.5
        
        try:
            odds = int(american_odds.replace("+", ""))
            if odds > 0:
                return 100 / (odds + 100)
            else:
                return abs(odds) / (abs(odds) + 100)
        except:
            return 0.5
    
    @staticmethod
    def probability_to_american(probability: float) -> str:
        """
        Convert probability to American odds.
        
        Args:
            probability: Win probability (0-1)
            
        Returns:
            American odds string
        """
        if probability <= 0 or probability >= 1:
            return "+100"
        
        if probability >= 0.5:
            odds = int(-100 * probability / (1 - probability))
            return str(odds)
        else:
            odds = int(100 * (1 - probability) / probability)
            return f"+{odds}"
    
    def get_request_count(self) -> int:
        """Get the number of API requests made in this session."""
        return self._request_count


def main():
    """Test the SGO Fetcher."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Sports Game Odds Fetcher")
    parser.add_argument("--league", type=str, default="nba", help="League to fetch (nba, nfl, ncaaf, mlb)")
    parser.add_argument("--fetch-all", action="store_true", help="Fetch all sports")
    parser.add_argument("--props", action="store_true", help="Fetch player props")
    parser.add_argument("--test", action="store_true", help="Run test mode")
    args = parser.parse_args()
    
    fetcher = SGOFetcher()
    
    if args.test:
        print("=" * 60)
        print("SGO Fetcher Test Mode")
        print("=" * 60)
        
        # Test leagues
        print("\n1. Testing league list...")
        leagues = fetcher.get_leagues()
        print(f"   Available leagues: {[l['leagueID'] for l in leagues]}")
        
        # Test single sport
        print(f"\n2. Testing {args.league.upper()} games...")
        games = fetcher.get_upcoming_games(args.league)
        print(f"   Found {len(games)} games with odds")
        if not games.empty:
            print(games[["home_team", "away_team", "start_time", "moneyline_home", "moneyline_away"]].head())
        
        print(f"\n3. API requests made: {fetcher.get_request_count()}")
        print("=" * 60)
        return
    
    if args.fetch_all:
        results = fetcher.fetch_all_sports()
        for sport, df in results.items():
            print(f"\n{sport.upper()}: {len(df)} games")
            if not df.empty:
                print(df[["home_team", "away_team", "moneyline_home", "moneyline_away"]].head())
        return
    
    if args.props:
        print(f"Fetching player props for {args.league.upper()}...")
        props = fetcher.get_player_props(args.league)
        print(f"Found {len(props)} player props")
        if not props.empty:
            print(props[["player_name", "stat_type", "line", "book_odds"]].head(20))
        return
    
    # Default: fetch games for single sport
    games = fetcher.get_upcoming_games(args.league)
    print(f"Found {len(games)} {args.league.upper()} games with odds")
    if not games.empty:
        print(games)


if __name__ == "__main__":
    main()
