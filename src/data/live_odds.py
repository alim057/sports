"""
Real-Time Odds Integration

Fetches and monitors live betting odds from multiple bookmakers.
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable
import time
import json
from pathlib import Path
import yaml
import threading


class LiveOddsFetcher:
    """Fetches real-time betting odds from multiple sources."""
    
    BASE_URL = "https://api.the-odds-api.com/v4"
    
    # Sport keys for The Odds API
    SPORT_KEYS = {
        'nba': 'basketball_nba',
        'ncaab': 'basketball_ncaab',
        'nfl': 'americanfootball_nfl',
        'ncaaf': 'americanfootball_ncaaf',
        'mlb': 'baseball_mlb',
        'nhl': 'icehockey_nhl',
        'epl': 'soccer_epl',
        'mls': 'soccer_usa_mls',
    }
    
    # Bookmaker filters
    BOOKMAKER_REGIONS = {
        'us': ['fanduel', 'draftkings', 'betmgm', 'caesars', 'pointsbet'],
        'stake': ['stake', 'stakeus'],
        'offshore': ['bovada', 'betonline', 'mybookie'],
    }
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        config_path: Optional[str] = None
    ):
        """
        Initialize live odds fetcher.
        
        Args:
            api_key: The Odds API key
            config_path: Path to config.yaml
        """
        self.api_key = api_key
        self._load_config(config_path)
        self.remaining_requests = None
        self._cache = {}
        self._cache_ttl = 60  # seconds
    
    def _load_config(self, config_path: Optional[str]):
        """Load API key from config."""
        if self.api_key:
            return
        
        paths = [
            config_path,
            "./config/config.yaml",
            Path(__file__).parent.parent.parent / "config" / "config.yaml"
        ]
        
        for path in paths:
            if path and Path(path).exists():
                try:
                    with open(path, 'r') as f:
                        config = yaml.safe_load(f)
                        self.api_key = config.get('odds_api', {}).get('api_key')
                        if self.api_key and "YOUR_" not in self.api_key:
                            return
                except:
                    pass
    
    def _make_request(self, endpoint: str, params: Dict = None) -> Optional[Dict]:
        """Make API request with caching."""
        if not self.api_key:
            print("Warning: The Odds API key not configured")
            return None
        
        if params is None:
            params = {}
        params['apiKey'] = self.api_key
        
        # Check cache
        cache_key = f"{endpoint}:{json.dumps(params, sort_keys=True)}"
        if cache_key in self._cache:
            cached_time, cached_data = self._cache[cache_key]
            if time.time() - cached_time < self._cache_ttl:
                return cached_data
        
        try:
            url = f"{self.BASE_URL}{endpoint}"
            response = requests.get(url, params=params, timeout=10)
            
            self.remaining_requests = response.headers.get('x-requests-remaining')
            
            if response.status_code != 200:
                print(f"API Error {response.status_code}: {response.text}")
                return None
            
            data = response.json()
            self._cache[cache_key] = (time.time(), data)
            return data
            
        except Exception as e:
            print(f"Request error: {e}")
            return None
    
    def get_live_odds(
        self,
        sport: str = 'nba',
        markets: str = 'h2h,spreads,totals',
        regions: str = 'us',
        bookmakers: List[str] = None
    ) -> pd.DataFrame:
        """
        Get current live odds for a sport.
        
        Args:
            sport: Sport key (nba, nfl, mlb, nhl)
            markets: Comma-separated markets (h2h, spreads, totals)
            regions: Bookmaker regions (us, us2, uk, eu, au)
            bookmakers: Optional list of specific bookmakers
            
        Returns:
            DataFrame with current odds
        """
        sport_key = self.SPORT_KEYS.get(sport.lower(), sport)
        
        params = {
            'regions': regions,
            'markets': markets,
            'oddsFormat': 'american'
        }
        
        if bookmakers:
            params['bookmakers'] = ','.join(bookmakers)
        
        data = self._make_request(f"/sports/{sport_key}/odds", params)
        
        if not data:
            return pd.DataFrame()
        
        return self._parse_odds_response(data)
    
    def _parse_odds_response(self, data: List[Dict]) -> pd.DataFrame:
        """Parse odds API response into DataFrame."""
        rows = []
        
        for game in data:
            game_id = game['id']
            home_team = game['home_team']
            away_team = game['away_team']
            commence_time = game['commence_time']
            
            for bookmaker in game.get('bookmakers', []):
                book_name = bookmaker['title']
                last_update = bookmaker.get('last_update')
                
                for market in bookmaker.get('markets', []):
                    market_key = market['key']
                    
                    # Build odds dict for this market
                    odds_dict = {
                        'game_id': game_id,
                        'home_team': home_team,
                        'away_team': away_team,
                        'commence_time': commence_time,
                        'bookmaker': book_name,
                        'market': market_key,
                        'last_update': last_update,
                        'fetched_at': datetime.now().isoformat()
                    }
                    
                    for outcome in market.get('outcomes', []):
                        name = outcome.get('name', '')
                        price = outcome.get('price')
                        point = outcome.get('point')
                        
                        if name == home_team:
                            odds_dict['home_odds'] = price
                            odds_dict['home_point'] = point
                        elif name == away_team:
                            odds_dict['away_odds'] = price
                            odds_dict['away_point'] = point
                        elif name.lower() == 'over':
                            odds_dict['over_odds'] = price
                            odds_dict['total'] = point
                        elif name.lower() == 'under':
                            odds_dict['under_odds'] = price
                    
                    rows.append(odds_dict)
        
        return pd.DataFrame(rows)
    
    def get_best_odds(
        self,
        sport: str = 'nba',
        market: str = 'h2h'
    ) -> pd.DataFrame:
        """
        Get best available odds across all bookmakers.
        
        Args:
            sport: Sport key
            market: Market type (h2h, spreads, totals)
            
        Returns:
            DataFrame with best odds for each game
        """
        odds = self.get_live_odds(sport, markets=market, regions='us,us2')
        
        if odds.empty:
            return pd.DataFrame()
        
        # Group by game and find best odds
        best = []
        for game_id, group in odds.groupby('game_id'):
            home_team = group.iloc[0]['home_team']
            away_team = group.iloc[0]['away_team']
            
            if market == 'h2h':
                # Best home odds (highest positive or least negative)
                best_home = group.loc[group['home_odds'].idxmax()]
                best_away = group.loc[group['away_odds'].idxmax()]
                
                best.append({
                    'game': f"{away_team} @ {home_team}",
                    'home_team': home_team,
                    'away_team': away_team,
                    'best_home_odds': best_home['home_odds'],
                    'best_home_book': best_home['bookmaker'],
                    'best_away_odds': best_away['away_odds'],
                    'best_away_book': best_away['bookmaker'],
                    'commence_time': group.iloc[0]['commence_time']
                })
        
        return pd.DataFrame(best)
    
    def compare_to_stake(
        self,
        sport: str = 'nba'
    ) -> pd.DataFrame:
        """
        Compare Stake odds to market best.
        
        Args:
            sport: Sport key
            
        Returns:
            DataFrame with Stake vs market comparison
        """
        # Get all odds
        all_odds = self.get_live_odds(sport, markets='h2h', regions='us,us2')
        
        if all_odds.empty:
            return pd.DataFrame()
        
        comparisons = []
        
        for game_id, group in all_odds.groupby('game_id'):
            home_team = group.iloc[0]['home_team']
            away_team = group.iloc[0]['away_team']
            
            # Find Stake odds
            stake_odds = group[group['bookmaker'].str.lower().isin(['stake', 'stakeus'])]
            
            if stake_odds.empty:
                continue
            
            stake = stake_odds.iloc[0]
            
            # Find market best
            best_home = group['home_odds'].max()
            best_away = group['away_odds'].max()
            avg_home = group['home_odds'].mean()
            avg_away = group['away_odds'].mean()
            
            comparisons.append({
                'game': f"{away_team} @ {home_team}",
                'home_team': home_team,
                'away_team': away_team,
                'stake_home': stake['home_odds'],
                'stake_away': stake['away_odds'],
                'market_best_home': best_home,
                'market_best_away': best_away,
                'market_avg_home': avg_home,
                'market_avg_away': avg_away,
                'stake_home_vs_best': stake['home_odds'] - best_home,
                'stake_away_vs_best': stake['away_odds'] - best_away,
            })
        
        return pd.DataFrame(comparisons)
    
    def get_multi_sport_odds(
        self,
        sports: List[str] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Get odds for multiple sports at once.
        
        Args:
            sports: List of sport keys (default: nba, nfl, mlb, nhl)
            
        Returns:
            Dictionary mapping sport to odds DataFrame
        """
        if sports is None:
            sports = ['nba', 'nfl', 'mlb', 'nhl']
        
        results = {}
        for sport in sports:
            odds = self.get_live_odds(sport)
            if not odds.empty:
                results[sport] = odds
                print(f"  {sport.upper()}: {odds['game_id'].nunique()} games")
        
        return results
    
    def get_api_usage(self) -> Dict:
        """Get API usage information."""
        return {
            'remaining_requests': self.remaining_requests,
            'api_configured': bool(self.api_key and "YOUR_" not in (self.api_key or ""))
        }


class OddsMonitor:
    """Monitors odds changes in real-time."""
    
    def __init__(
        self,
        fetcher: LiveOddsFetcher,
        sports: List[str] = None,
        interval: int = 60
    ):
        """
        Initialize odds monitor.
        
        Args:
            fetcher: LiveOddsFetcher instance
            sports: Sports to monitor
            interval: Refresh interval in seconds
        """
        self.fetcher = fetcher
        self.sports = sports or ['nba']
        self.interval = interval
        self._running = False
        self._callbacks = []
        self._last_odds = {}
    
    def add_callback(self, callback: Callable):
        """Add callback for odds changes."""
        self._callbacks.append(callback)
    
    def start(self):
        """Start monitoring odds."""
        self._running = True
        self._thread = threading.Thread(target=self._monitor_loop)
        self._thread.daemon = True
        self._thread.start()
        print(f"Odds monitor started for {self.sports}")
    
    def stop(self):
        """Stop monitoring."""
        self._running = False
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        while self._running:
            for sport in self.sports:
                try:
                    current_odds = self.fetcher.get_live_odds(sport)
                    
                    if not current_odds.empty:
                        changes = self._detect_changes(sport, current_odds)
                        
                        if changes:
                            for callback in self._callbacks:
                                callback(sport, changes)
                        
                        self._last_odds[sport] = current_odds
                
                except Exception as e:
                    print(f"Monitor error for {sport}: {e}")
            
            time.sleep(self.interval)
    
    def _detect_changes(
        self,
        sport: str,
        current: pd.DataFrame
    ) -> List[Dict]:
        """Detect significant odds changes."""
        if sport not in self._last_odds:
            return []
        
        previous = self._last_odds[sport]
        changes = []
        
        # Compare odds for same game/book combinations
        for _, row in current.iterrows():
            key = (row['game_id'], row['bookmaker'], row['market'])
            
            prev_match = previous[
                (previous['game_id'] == row['game_id']) &
                (previous['bookmaker'] == row['bookmaker']) &
                (previous['market'] == row['market'])
            ]
            
            if not prev_match.empty:
                prev = prev_match.iloc[0]
                
                home_change = row.get('home_odds', 0) - prev.get('home_odds', 0)
                away_change = row.get('away_odds', 0) - prev.get('away_odds', 0)
                
                if abs(home_change) >= 5 or abs(away_change) >= 5:
                    changes.append({
                        'game': f"{row['away_team']} @ {row['home_team']}",
                        'bookmaker': row['bookmaker'],
                        'home_change': home_change,
                        'away_change': away_change,
                        'new_home': row.get('home_odds'),
                        'new_away': row.get('away_odds')
                    })
        
        return changes


def main():
    """Demo live odds integration."""
    print("Live Odds Integration Demo")
    print("=" * 50)
    
    fetcher = LiveOddsFetcher()
    
    usage = fetcher.get_api_usage()
    if not usage['api_configured']:
        print("\nThe Odds API key not configured!")
        print("Add your key to config/config.yaml:")
        print('  odds_api:')
        print('    api_key: "YOUR_KEY_HERE"')
        print("\nGet a free key at: https://the-odds-api.com/")
        return
    
    print(f"\nAPI Requests Remaining: {usage['remaining_requests']}")
    
    # Get odds for all sports
    print("\nFetching multi-sport odds...")
    odds = fetcher.get_multi_sport_odds(['nba', 'nfl', 'nhl'])
    
    for sport, df in odds.items():
        if not df.empty:
            print(f"\n{sport.upper()} - {df['game_id'].nunique()} games")
            best = fetcher.get_best_odds(sport)
            if not best.empty:
                print(best[['game', 'best_home_odds', 'best_home_book']].head())


if __name__ == "__main__":
    main()
