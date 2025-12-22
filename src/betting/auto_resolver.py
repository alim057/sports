"""
Auto Bet Resolution

Automatically fetches final scores and resolves pending bets.
Uses ESPN API (free) to get game results.
"""

import requests
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional


class AutoResolver:
    """Automatically resolve pending bets by fetching final scores."""
    
    ESPN_APIS = {
        'nba': 'https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard',
        'nfl': 'https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard',
        'nhl': 'https://site.api.espn.com/apis/site/v2/sports/hockey/nhl/scoreboard',
        'ncaaf': 'https://site.api.espn.com/apis/site/v2/sports/football/college-football/scoreboard',
        'mlb': 'https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/scoreboard'
    }
    
    def __init__(self, db_path: str = 'data/bets.db'):
        """Initialize with database path."""
        self.db_path = Path(db_path)
    
    def get_final_scores(self, sport: str, date: Optional[datetime] = None) -> List[Dict]:
        """
        Fetch final scores for a date.
        
        Args:
            sport: Sport type (nba, nfl, etc.)
            date: Date to check (default: yesterday)
            
        Returns:
            List of completed games with scores
        """
        if date is None:
            date = datetime.now() - timedelta(days=1)
        
        api_url = self.ESPN_APIS.get(sport.lower())
        if not api_url:
            return []
        
        params = {'dates': date.strftime('%Y%m%d')}
        
        try:
            response = requests.get(api_url, params=params, timeout=10)
            if response.status_code != 200:
                return []
            
            data = response.json()
            games = []
            
            for event in data.get('events', []):
                competition = event.get('competitions', [{}])[0]
                status = competition.get('status', {})
                
                # Only completed games
                if not status.get('type', {}).get('completed'):
                    continue
                
                competitors = competition.get('competitors', [])
                if len(competitors) != 2:
                    continue
                
                home = next((c for c in competitors if c.get('homeAway') == 'home'), {})
                away = next((c for c in competitors if c.get('homeAway') == 'away'), {})
                
                home_score = int(home.get('score', 0))
                away_score = int(away.get('score', 0))
                
                games.append({
                    'game_id': event.get('id'),
                    'date': date.strftime('%Y-%m-%d'),
                    'home_team': home.get('team', {}).get('abbreviation', ''),
                    'away_team': away.get('team', {}).get('abbreviation', ''),
                    'home_score': home_score,
                    'away_score': away_score,
                    'winner': 'home' if home_score > away_score else 'away',
                    'total': home_score + away_score
                })
            
            return games
            
        except Exception as e:
            print(f"Error fetching scores: {e}")
            return []
    
    def get_pending_bets(self) -> List[Dict]:
        """Get all pending bets from database."""
        if not self.db_path.exists():
            return []
        
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT * FROM bets 
                WHERE status = 'pending' OR status IS NULL
            """)
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
        except:
            return []
        finally:
            conn.close()
    
    def resolve_bet(self, bet_id: int, result: str, profit_loss: float):
        """
        Resolve a bet with result.
        
        Args:
            bet_id: Bet ID
            result: 'win', 'loss', or 'push'
            profit_loss: Profit (positive) or loss (negative)
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                UPDATE bets SET
                    status = 'resolved',
                    result = ?,
                    profit_loss = ?,
                    resolved_at = ?
                WHERE id = ?
            """, (result, profit_loss, datetime.now().isoformat(), bet_id))
            conn.commit()
        finally:
            conn.close()
    
    def auto_resolve_all(self, sport: str = None, lookback_days: int = 3) -> Dict:
        """
        Auto-resolve all pending bets.
        
        Args:
            sport: Optional sport filter
            lookback_days: How many days back to check
            
        Returns:
            Summary of resolved bets
        """
        pending = self.get_pending_bets()
        
        if sport:
            pending = [b for b in pending if b.get('sport', '').lower() == sport.lower()]
        
        if not pending:
            return {'resolved': 0, 'message': 'No pending bets'}
        
        # Get scores for recent days
        all_scores = []
        for i in range(lookback_days + 1):
            date = datetime.now() - timedelta(days=i)
            for s in ['nba', 'nfl', 'nhl', 'ncaaf', 'mlb']:
                scores = self.get_final_scores(s, date)
                all_scores.extend(scores)
        
        resolved = 0
        wins = 0
        losses = 0
        total_profit = 0
        
        for bet in pending:
            home = bet.get('home_team', '').upper()
            away = bet.get('away_team', '').upper()
            picked = bet.get('predicted_winner', '').upper()
            stake = bet.get('stake_amount', 100)
            odds = bet.get('odds', -110)
            
            # Find matching game
            game = None
            for g in all_scores:
                if g['home_team'].upper() == home and g['away_team'].upper() == away:
                    game = g
                    break
            
            if not game:
                continue
            
            # Determine if bet won
            actual_winner = game['home_team'] if game['winner'] == 'home' else game['away_team']
            bet_won = actual_winner.upper() == picked
            
            # Calculate profit/loss
            if bet_won:
                if odds > 0:
                    profit = stake * (odds / 100)
                else:
                    profit = stake * (100 / abs(odds))
                result = 'win'
                wins += 1
            else:
                profit = -stake
                result = 'loss'
                losses += 1
            
            total_profit += profit
            
            # Update database
            self.resolve_bet(bet['id'], result, profit)
            resolved += 1
        
        return {
            'resolved': resolved,
            'wins': wins,
            'losses': losses,
            'profit': total_profit,
            'pending_remaining': len(pending) - resolved
        }


class KellyCriterion:
    """Calculate optimal bet size using Kelly Criterion."""
    
    def __init__(self, bankroll: float = 1000, fraction: float = 0.25):
        """
        Initialize Kelly calculator.
        
        Args:
            bankroll: Total bankroll
            fraction: Fractional Kelly (0.25 = quarter Kelly for safety)
        """
        self.bankroll = bankroll
        self.fraction = fraction
    
    def calculate(self, prob: float, odds: int) -> Dict:
        """
        Calculate Kelly bet size.
        
        Args:
            prob: Predicted win probability (0-1)
            odds: American odds
            
        Returns:
            Kelly analysis dict
        """
        # Convert odds to decimal multiplier
        if odds > 0:
            decimal_odds = odds / 100
        else:
            decimal_odds = 100 / abs(odds)
        
        # Kelly formula: f* = (bp - q) / b
        # where b = decimal odds, p = win prob, q = loss prob
        b = decimal_odds
        p = prob
        q = 1 - prob
        
        kelly_fraction = (b * p - q) / b
        
        # Apply fractional Kelly for safety
        adjusted_fraction = kelly_fraction * self.fraction
        
        # Calculate bet amount
        if adjusted_fraction > 0:
            bet_amount = self.bankroll * adjusted_fraction
        else:
            bet_amount = 0
        
        # Calculate expected value
        ev = p * decimal_odds - q
        
        return {
            'probability': prob,
            'odds': odds,
            'kelly_fraction': kelly_fraction,
            'adjusted_fraction': adjusted_fraction,
            'recommended_bet': max(0, min(bet_amount, self.bankroll * 0.1)),  # Cap at 10% of bankroll
            'expected_value': ev,
            'should_bet': kelly_fraction > 0 and ev > 0,
            'confidence_level': 'high' if ev > 0.1 else 'medium' if ev > 0.05 else 'low'
        }
    
    def optimal_bets(self, opportunities: List[Dict]) -> List[Dict]:
        """
        Analyze multiple betting opportunities.
        
        Args:
            opportunities: List of dicts with 'prob' and 'odds'
            
        Returns:
            Sorted list of opportunities by expected value
        """
        results = []
        
        for opp in opportunities:
            analysis = self.calculate(opp['prob'], opp['odds'])
            analysis['matchup'] = opp.get('matchup', 'Unknown')
            analysis['pick'] = opp.get('pick', 'Unknown')
            results.append(analysis)
        
        # Sort by expected value
        results.sort(key=lambda x: x['expected_value'], reverse=True)
        
        return [r for r in results if r['should_bet']]


def main():
    """Test auto-resolution and Kelly criterion."""
    print("=" * 50)
    print("Auto Resolution & Kelly Criterion Test")
    print("=" * 50)
    
    # Test score fetching
    resolver = AutoResolver()
    print("\nFetching yesterday's NBA scores...")
    scores = resolver.get_final_scores('nba')
    for game in scores[:3]:
        print(f"  {game['away_team']} {game['away_score']} @ {game['home_team']} {game['home_score']}")
    
    # Test Kelly criterion
    print("\nKelly Criterion Examples:")
    kelly = KellyCriterion(bankroll=1000, fraction=0.25)
    
    examples = [
        {'prob': 0.55, 'odds': -110, 'matchup': 'LAL vs BOS', 'pick': 'LAL'},
        {'prob': 0.65, 'odds': +150, 'matchup': 'KC vs SF', 'pick': 'SF'},
        {'prob': 0.45, 'odds': +200, 'matchup': 'TOR vs MTL', 'pick': 'TOR'},
    ]
    
    for result in kelly.optimal_bets(examples):
        print(f"\n  {result['matchup']}: Pick {result['pick']}")
        print(f"    Win Prob: {result['probability']:.1%}")
        print(f"    EV: {result['expected_value']:.1%}")
        print(f"    Recommended: ${result['recommended_bet']:.2f}")


if __name__ == '__main__':
    main()
