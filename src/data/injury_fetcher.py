"""
Injury Data Fetcher

Fetches NBA injury reports from ESPN's hidden API.
"""

import requests
import pandas as pd
from datetime import datetime
from typing import List, Dict, Optional
import time


class InjuryFetcher:
    """Fetches NBA injury data from ESPN."""
    
    # ESPN hidden API endpoints
    ESPN_INJURIES_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries"
    ESPN_TEAMS_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/teams"
    
    # Injury status impact weights (how much does the injury affect the team)
    INJURY_IMPACT = {
        'out': 1.0,           # Fully out
        'doubtful': 0.85,     # Likely out
        'questionable': 0.5,  # 50/50
        'probable': 0.15,     # Likely to play
        'day-to-day': 0.4,    # Uncertain
    }
    
    def __init__(self):
        self._team_cache = {}
        self._last_fetch = None
        self._injuries_cache = None
    
    def _make_request(self, url: str) -> Optional[Dict]:
        """Make HTTP request to ESPN API."""
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error fetching from ESPN: {e}")
            return None
    
    def get_all_team_mappings(self) -> Dict[str, Dict]:
        """Get team ID to abbreviation mappings."""
        if self._team_cache:
            return self._team_cache
        
        data = self._make_request(self.ESPN_TEAMS_URL)
        
        if not data or 'sports' not in data:
            return {}
        
        for sport in data.get('sports', []):
            for league in sport.get('leagues', []):
                for team in league.get('teams', []):
                    team_data = team.get('team', {})
                    team_id = team_data.get('id')
                    abbr = team_data.get('abbreviation', '').upper()
                    name = team_data.get('displayName', '')
                    
                    if team_id and abbr:
                        self._team_cache[abbr] = {
                            'espn_id': team_id,
                            'abbreviation': abbr,
                            'name': name
                        }
        
        return self._team_cache
    
    def get_all_injuries(self, use_cache: bool = True) -> pd.DataFrame:
        """
        Fetch all current NBA injuries.
        
        Args:
            use_cache: Use cached data if available and recent
            
        Returns:
            DataFrame with injury information
        """
        # Return cache if fresh (within 30 minutes)
        if use_cache and self._injuries_cache is not None and self._last_fetch:
            age = (datetime.now() - self._last_fetch).seconds
            if age < 1800:  # 30 minutes
                return self._injuries_cache
        
        data = self._make_request(self.ESPN_INJURIES_URL)
        
        if not data:
            return pd.DataFrame()
        
        injuries = []
        
        for team_data in data.get('items', []):
            team_name = team_data.get('team', {}).get('displayName', 'Unknown')
            team_abbr = team_data.get('team', {}).get('abbreviation', '').upper()
            
            for injury in team_data.get('injuries', []):
                athlete = injury.get('athlete', {})
                
                status = injury.get('status', 'unknown').lower()
                
                injuries.append({
                    'team': team_name,
                    'team_abbr': team_abbr,
                    'player_id': athlete.get('id'),
                    'player_name': athlete.get('displayName', 'Unknown'),
                    'position': athlete.get('position', {}).get('abbreviation', ''),
                    'injury_type': injury.get('type', {}).get('description', 'Unknown'),
                    'injury_detail': injury.get('details', {}).get('detail', ''),
                    'status': status,
                    'impact_score': self.INJURY_IMPACT.get(status, 0.5),
                    'fetched_at': datetime.now()
                })
        
        df = pd.DataFrame(injuries)
        
        # Cache the results
        self._injuries_cache = df
        self._last_fetch = datetime.now()
        
        return df
    
    def get_team_injuries(self, team_abbr: str) -> pd.DataFrame:
        """Get injuries for a specific team."""
        all_injuries = self.get_all_injuries()
        
        if all_injuries.empty:
            return pd.DataFrame()
        
        return all_injuries[
            all_injuries['team_abbr'] == team_abbr.upper()
        ]
    
    def calculate_team_injury_impact(self, team_abbr: str) -> Dict:
        """
        Calculate the overall injury impact for a team.
        
        Returns:
            Dictionary with injury metrics
        """
        team_injuries = self.get_team_injuries(team_abbr)
        
        if team_injuries.empty:
            return {
                'total_injuries': 0,
                'out_count': 0,
                'questionable_count': 0,
                'impact_score': 0.0,
                'injured_players': []
            }
        
        out_count = len(team_injuries[team_injuries['status'] == 'out'])
        questionable_count = len(team_injuries[team_injuries['status'].isin(['questionable', 'doubtful'])])
        
        # Total impact is sum of individual impacts
        total_impact = team_injuries['impact_score'].sum()
        
        injured_players = team_injuries[['player_name', 'position', 'status', 'injury_type']].to_dict('records')
        
        return {
            'total_injuries': len(team_injuries),
            'out_count': out_count,
            'questionable_count': questionable_count,
            'impact_score': total_impact,
            'injured_players': injured_players
        }
    
    def get_matchup_injury_comparison(
        self,
        home_team: str,
        away_team: str
    ) -> Dict:
        """
        Compare injuries between two teams for a matchup.
        
        Args:
            home_team: Home team abbreviation
            away_team: Away team abbreviation
            
        Returns:
            Comparison of injury impacts
        """
        home_impact = self.calculate_team_injury_impact(home_team)
        away_impact = self.calculate_team_injury_impact(away_team)
        
        # Positive = home team has FEWER injuries (advantage)
        injury_advantage = away_impact['impact_score'] - home_impact['impact_score']
        
        return {
            'home_team': home_team,
            'away_team': away_team,
            'home_injuries': home_impact,
            'away_injuries': away_impact,
            'injury_advantage': injury_advantage,
            'home_advantage_pct': self._injury_to_probability_impact(injury_advantage)
        }
    
    def _injury_to_probability_impact(self, injury_diff: float) -> float:
        """
        Convert injury score difference to probability adjustment.
        
        Each injury point roughly translates to ~2% win probability.
        """
        return injury_diff * 0.02  # 2% per injury point
    
    def format_injury_report(self, team_abbr: str) -> str:
        """Get formatted injury report for a team."""
        impact = self.calculate_team_injury_impact(team_abbr)
        
        if impact['total_injuries'] == 0:
            return f"{team_abbr}: No injuries reported"
        
        lines = [f"{team_abbr}: {impact['total_injuries']} injuries (Impact: {impact['impact_score']:.1f})"]
        
        for player in impact['injured_players']:
            status = player['status'].upper()
            lines.append(f"  - {player['player_name']} ({player['position']}): {status} - {player['injury_type']}")
        
        return "\n".join(lines)


def main():
    """Demo injury fetcher."""
    fetcher = InjuryFetcher()
    
    print("NBA Injury Fetcher Demo")
    print("=" * 50)
    
    # Fetch all injuries
    print("\nFetching all NBA injuries...")
    injuries = fetcher.get_all_injuries()
    
    if not injuries.empty:
        print(f"Found {len(injuries)} total injury reports")
        print(f"Teams with injuries: {injuries['team_abbr'].nunique()}")
        
        # Show summary by status
        print("\nInjury Status Summary:")
        status_counts = injuries['status'].value_counts()
        for status, count in status_counts.items():
            print(f"  {status}: {count}")
        
        # Show sample injuries
        print("\nSample Injuries:")
        print(injuries[['team_abbr', 'player_name', 'position', 'status', 'injury_type']].head(10).to_string())
        
        # Demo matchup comparison
        print("\n" + "=" * 50)
        print("Sample Matchup: LAL vs GSW")
        comparison = fetcher.get_matchup_injury_comparison('LAL', 'GSW')
        
        print(f"\nLakers injuries: {comparison['home_injuries']['total_injuries']}")
        print(f"  Out: {comparison['home_injuries']['out_count']}")
        print(f"  Impact: {comparison['home_injuries']['impact_score']:.1f}")
        
        print(f"\nWarriors injuries: {comparison['away_injuries']['total_injuries']}")
        print(f"  Out: {comparison['away_injuries']['out_count']}")
        print(f"  Impact: {comparison['away_injuries']['impact_score']:.1f}")
        
        print(f"\nInjury Advantage: {'LAL' if comparison['injury_advantage'] > 0 else 'GSW'}")
        print(f"Win Probability Adjustment: {comparison['home_advantage_pct']:+.1%}")
    else:
        print("No injury data available")


if __name__ == "__main__":
    main()
