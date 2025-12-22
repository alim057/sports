"""
Betting Odds Fetcher

Fetches live betting odds from The Odds API.
Free tier: 500 requests/month
"""

import requests
import pandas as pd
from datetime import datetime
from typing import Optional, List, Dict
import yaml
from pathlib import Path


class OddsFetcher:
    """Fetches betting odds from The Odds API."""
    
    BASE_URL = "https://api.the-odds-api.com/v4"
    
    # Sport keys for different leagues
    SPORT_KEYS = {
        "nba": "basketball_nba",
        "ncaab": "basketball_ncaab",
        "nfl": "americanfootball_nfl",
        "ncaaf": "americanfootball_ncaaf",
        "mlb": "baseball_mlb",
        "nhl": "icehockey_nhl"
    }
    
    def __init__(self, api_key: Optional[str] = None, config_path: Optional[str] = None):
        """
        Initialize the odds fetcher.
        
        Args:
            api_key: The Odds API key (or load from config)
            config_path: Path to config.yaml
        """
        self.api_key = api_key
        
        if not self.api_key and config_path:
            self._load_api_key_from_config(config_path)
        
        if not self.api_key:
            # Try default config location
            default_config = Path(__file__).parent.parent.parent / "config" / "config.yaml"
            if default_config.exists():
                self._load_api_key_from_config(str(default_config))
        
        self.remaining_requests = None
    
    def _load_api_key_from_config(self, config_path: str):
        """Load API key from config file."""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                self.api_key = config.get('odds_api', {}).get('api_key')
        except Exception as e:
            print(f"Warning: Could not load config from {config_path}: {e}")
    
    def _make_request(self, endpoint: str, params: Dict = None) -> Dict:
        """Make a request to The Odds API."""
        if not self.api_key:
            raise ValueError("API key not set. Get one at https://the-odds-api.com/")
        
        if params is None:
            params = {}
        
        params['apiKey'] = self.api_key
        
        url = f"{self.BASE_URL}{endpoint}"
        response = requests.get(url, params=params)
        
        # Track remaining requests
        self.remaining_requests = response.headers.get('x-requests-remaining')
        
        if response.status_code != 200:
            raise Exception(f"API Error {response.status_code}: {response.text}")
        
        return response.json()
    
    def get_sports(self) -> pd.DataFrame:
        """Get list of available sports."""
        data = self._make_request("/sports")
        return pd.DataFrame(data)
    
    def get_odds(
        self,
        sport: str = "nba",
        regions: str = "us",
        markets: str = "h2h,spreads,totals",
        odds_format: str = "american"
    ) -> pd.DataFrame:
        """
        Get current betting odds for a sport.
        
        Args:
            sport: Sport key (nba, ncaab, nfl, mlb, nhl)
            regions: Bookmaker region (us, us2, uk, eu, au)
            markets: Comma-separated markets (h2h=moneyline, spreads, totals)
            odds_format: american or decimal
            
        Returns:
            DataFrame with odds for upcoming games
        """
        sport_key = self.SPORT_KEYS.get(sport.lower(), sport)
        
        params = {
            'regions': regions,
            'markets': markets,
            'oddsFormat': odds_format
        }
        
        data = self._make_request(f"/sports/{sport_key}/odds", params)
        
        if not data:
            return pd.DataFrame()
        
        # Flatten the nested structure
        rows = []
        for game in data:
            game_id = game['id']
            home_team = game['home_team']
            away_team = game['away_team']
            commence_time = game['commence_time']
            
            for bookmaker in game.get('bookmakers', []):
                book_name = bookmaker['title']
                
                for market in bookmaker.get('markets', []):
                    market_key = market['key']
                    
                    for outcome in market.get('outcomes', []):
                        rows.append({
                            'game_id': game_id,
                            'home_team': home_team,
                            'away_team': away_team,
                            'commence_time': commence_time,
                            'bookmaker': book_name,
                            'market': market_key,
                            'team': outcome.get('name'),
                            'price': outcome.get('price'),
                            'point': outcome.get('point')  # For spreads/totals
                        })
        
        df = pd.DataFrame(rows)
        if not df.empty:
            df['commence_time'] = pd.to_datetime(df['commence_time'])
            df['fetched_at'] = datetime.now()
        
        return df
    
    def get_moneyline_odds(self, sport: str = "nba") -> pd.DataFrame:
        """Get moneyline (head-to-head) odds only."""
        return self.get_odds(sport=sport, markets="h2h")
    
    def get_spread_odds(self, sport: str = "nba") -> pd.DataFrame:
        """Get point spread odds only."""
        return self.get_odds(sport=sport, markets="spreads")
    
    def get_totals_odds(self, sport: str = "nba") -> pd.DataFrame:
        """Get over/under total points odds only."""
        return self.get_odds(sport=sport, markets="totals")
    
    def check_remaining_requests(self) -> Optional[int]:
        """Check remaining API requests for the month."""
        if self.remaining_requests:
            return int(self.remaining_requests)
        return None
    
    @staticmethod
    def american_to_probability(american_odds: int) -> float:
        """
        Convert American odds to implied probability.
        
        Examples:
            -150 -> 0.60 (60% implied win probability)
            +200 -> 0.333 (33.3% implied win probability)
        """
        if american_odds < 0:
            return abs(american_odds) / (abs(american_odds) + 100)
        else:
            return 100 / (american_odds + 100)
    
    @staticmethod
    def probability_to_american(probability: float) -> int:
        """
        Convert probability to American odds.
        
        Examples:
            0.60 -> -150
            0.333 -> +200
        """
        if probability >= 0.5:
            return int(-100 * probability / (1 - probability))
        else:
            return int(100 * (1 - probability) / probability)


def main():
    """Test the odds fetcher."""
    # Note: You need a valid API key for this to work
    fetcher = OddsFetcher()
    
    if not fetcher.api_key or "YOUR_" in fetcher.api_key:
        print("=" * 50)
        print("The Odds API key not configured!")
        print("1. Get a free API key at: https://the-odds-api.com/")
        print("2. Copy config/config.example.yaml to config/config.yaml")
        print("3. Add your API key to config.yaml")
        print("=" * 50)
        
        # Demo odds conversion
        print("\nDemo: Odds conversion functions")
        print(f"-150 American -> {OddsFetcher.american_to_probability(-150):.1%} probability")
        print(f"+200 American -> {OddsFetcher.american_to_probability(200):.1%} probability")
        print(f"60% probability -> {OddsFetcher.probability_to_american(0.60)} American")
        return
    
    print("Fetching NBA odds...")
    odds = fetcher.get_odds("nba")
    
    if not odds.empty:
        print(f"\nFound odds for {odds['game_id'].nunique()} games")
        print(f"Remaining API calls: {fetcher.remaining_requests}")
        print("\nSample moneyline odds:")
        ml_odds = odds[odds['market'] == 'h2h']
        print(ml_odds[['home_team', 'away_team', 'bookmaker', 'team', 'price']].head(10))
    else:
        print("No games found (might be off-season or no games today)")


if __name__ == "__main__":
    main()
