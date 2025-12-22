"""
Stake.com Odds Scraper

Fetches betting odds directly from Stake's Sports API.
"""

import requests
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional
import json
import time


class StakeScraper:
    """
    Scrapes betting odds directly from Stake's public sports API.
    
    Note: This uses Stake's public API endpoints. Be mindful of rate limits
    and their Terms of Service.
    """
    
    # Stake's Sports API endpoints (public, no auth required for basic data)
    BASE_URL = "https://stake.com/_api/graphql"
    SPORTS_API = "https://sportsbook-api.stake.com/v1"
    
    # GraphQL queries for fetching sports data
    SPORTS_QUERY = """
    query SportsAll {
        sports {
            slug
            name
            position
        }
    }
    """
    
    EVENTS_QUERY = """
    query SportEvents($slug: String!, $limit: Int) {
        sport(slug: $slug) {
            slug
            name
            tournaments {
                name
                events(limit: $limit) {
                    id
                    name
                    startTime
                    status
                    competitors {
                        name
                        position
                    }
                    markets {
                        name
                        outcomes {
                            name
                            odds
                        }
                    }
                }
            }
        }
    }
    """
    
    # Sport mappings
    SPORT_SLUGS = {
        'nba': 'basketball',
        'nfl': 'american-football',
        'mlb': 'baseball',
        'nhl': 'ice-hockey',
        'soccer': 'soccer',
    }
    
    def __init__(self, delay: float = 0.5):
        """
        Initialize Stake scraper.
        
        Args:
            delay: Delay between requests in seconds
        """
        self.delay = delay
        self._last_request = 0
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/json',
            'Origin': 'https://stake.com',
            'Referer': 'https://stake.com/sports',
        })
    
    def _rate_limit(self):
        """Enforce rate limiting."""
        elapsed = time.time() - self._last_request
        if elapsed < self.delay:
            time.sleep(self.delay - elapsed)
        self._last_request = time.time()
    
    def _graphql_request(
        self,
        query: str,
        variables: Dict = None
    ) -> Optional[Dict]:
        """Make GraphQL request to Stake API."""
        self._rate_limit()
        
        try:
            payload = {
                'query': query,
                'variables': variables or {}
            }
            
            response = self.session.post(
                self.BASE_URL,
                json=payload,
                timeout=10
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                print(f"Stake API error: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"Request error: {e}")
            return None
    
    def get_sports_list(self) -> List[Dict]:
        """Get list of all available sports."""
        result = self._graphql_request(self.SPORTS_QUERY)
        
        if result and 'data' in result:
            return result['data'].get('sports', [])
        return []
    
    def get_sport_events(
        self,
        sport: str = 'nba',
        limit: int = 50
    ) -> pd.DataFrame:
        """
        Get upcoming events for a sport.
        
        Args:
            sport: Sport key (nba, nfl, mlb, nhl)
            limit: Max events to return
            
        Returns:
            DataFrame with events and odds
        """
        sport_slug = self.SPORT_SLUGS.get(sport.lower(), sport)
        
        result = self._graphql_request(
            self.EVENTS_QUERY,
            {'slug': sport_slug, 'limit': limit}
        )
        
        if not result or 'data' not in result:
            return pd.DataFrame()
        
        sport_data = result['data'].get('sport', {})
        events = []
        
        for tournament in sport_data.get('tournaments', []):
            tournament_name = tournament.get('name', '')
            
            for event in tournament.get('events', []):
                competitors = event.get('competitors', [])
                home_team = competitors[0].get('name', '') if len(competitors) > 0 else ''
                away_team = competitors[1].get('name', '') if len(competitors) > 1 else ''
                
                for market in event.get('markets', []):
                    market_name = market.get('name', '')
                    
                    event_row = {
                        'event_id': event.get('id'),
                        'tournament': tournament_name,
                        'home_team': home_team,
                        'away_team': away_team,
                        'start_time': event.get('startTime'),
                        'status': event.get('status'),
                        'market': market_name,
                    }
                    
                    for outcome in market.get('outcomes', []):
                        name = outcome.get('name', '')
                        odds = outcome.get('odds')
                        
                        if name == home_team or 'home' in name.lower():
                            event_row['home_odds'] = odds
                        elif name == away_team or 'away' in name.lower():
                            event_row['away_odds'] = odds
                        elif 'over' in name.lower():
                            event_row['over_odds'] = odds
                        elif 'under' in name.lower():
                            event_row['under_odds'] = odds
                    
                    events.append(event_row)
        
        return pd.DataFrame(events)
    
    def get_nba_odds(self) -> pd.DataFrame:
        """Get current NBA betting odds from Stake."""
        return self.get_sport_events('nba')
    
    def get_nfl_odds(self) -> pd.DataFrame:
        """Get current NFL betting odds from Stake."""
        return self.get_sport_events('nfl')
    
    def fetch_alternative_api(
        self,
        sport: str = 'basketball',
        league: str = 'nba'
    ) -> pd.DataFrame:
        """
        Try alternative Stake API endpoint.
        
        This attempts to use the sportsbook-api endpoint if available.
        """
        self._rate_limit()
        
        try:
            # Try the documented sports API
            url = f"{self.SPORTS_API}/sports/{sport}/leagues/{league}/events"
            
            response = self.session.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                return self._parse_sportsbook_response(data)
            else:
                # Fall back to GraphQL
                return self.get_sport_events(sport)
                
        except Exception as e:
            print(f"Alternative API error: {e}")
            return pd.DataFrame()
    
    def _parse_sportsbook_response(self, data: Dict) -> pd.DataFrame:
        """Parse sportsbook API response."""
        if not data:
            return pd.DataFrame()
        
        events = []
        for event in data.get('events', []):
            event_row = {
                'event_id': event.get('id'),
                'home_team': event.get('home', {}).get('name', ''),
                'away_team': event.get('away', {}).get('name', ''),
                'start_time': event.get('startTime'),
                'status': event.get('status'),
            }
            
            # Parse odds/markets
            for market in event.get('markets', []):
                market_type = market.get('type', '')
                
                for selection in market.get('selections', []):
                    name = selection.get('name', '')
                    odds = selection.get('odds', {}).get('decimal')
                    
                    if odds:
                        # Convert decimal to American
                        american = self._decimal_to_american(odds)
                        
                        if 'home' in name.lower():
                            event_row['home_odds'] = american
                        elif 'away' in name.lower():
                            event_row['away_odds'] = american
            
            events.append(event_row)
        
        return pd.DataFrame(events)
    
    def _decimal_to_american(self, decimal_odds: float) -> int:
        """Convert decimal odds to American format."""
        if decimal_odds >= 2.0:
            return int((decimal_odds - 1) * 100)
        else:
            return int(-100 / (decimal_odds - 1))
    
    def compare_with_market(
        self,
        market_odds: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Compare Stake odds with market odds.
        
        Args:
            market_odds: DataFrame with market odds from other sources
            
        Returns:
            Comparison DataFrame
        """
        stake_odds = self.get_nba_odds()
        
        if stake_odds.empty:
            return pd.DataFrame()
        
        # Merge on team names (simplified matching)
        # In practice, would need better team name normalization
        comparisons = []
        
        for _, stake_row in stake_odds.iterrows():
            home = stake_row.get('home_team', '')
            
            # Find matching market game
            market_match = market_odds[
                market_odds['home_team'].str.contains(home[:3], case=False, na=False)
            ]
            
            if not market_match.empty:
                market_row = market_match.iloc[0]
                
                comparisons.append({
                    'game': f"{stake_row.get('away_team', '')} @ {home}",
                    'stake_home': stake_row.get('home_odds'),
                    'stake_away': stake_row.get('away_odds'),
                    'market_home': market_row.get('home_odds'),
                    'market_away': market_row.get('away_odds'),
                    'home_diff': (stake_row.get('home_odds') or 0) - (market_row.get('home_odds') or 0),
                    'away_diff': (stake_row.get('away_odds') or 0) - (market_row.get('away_odds') or 0),
                })
        
        return pd.DataFrame(comparisons)


def main():
    """Demo Stake scraper."""
    print("Stake.com Odds Scraper Demo")
    print("=" * 50)
    
    scraper = StakeScraper()
    
    # Get available sports
    print("\nFetching available sports...")
    sports = scraper.get_sports_list()
    
    if sports:
        print(f"Found {len(sports)} sports:")
        for s in sports[:10]:
            print(f"  - {s.get('name')} ({s.get('slug')})")
    else:
        print("Could not fetch sports list (API may require auth)")
    
    # Try to get NBA events
    print("\nFetching NBA events...")
    nba_odds = scraper.get_nba_odds()
    
    if not nba_odds.empty:
        print(f"Found {len(nba_odds)} NBA betting markets")
        print(nba_odds[['home_team', 'away_team', 'home_odds', 'away_odds']].head())
    else:
        print("No NBA data available (may require authentication)")
    
    print("\n" + "=" * 50)
    print("Note: Some Stake API endpoints may require authentication.")
    print("Use The Odds API for reliable odds data including Stake markets.")


if __name__ == "__main__":
    main()
