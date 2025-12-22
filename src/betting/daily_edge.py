"""
Daily Edge Analysis

Finds betting edges using our model vs live market odds.
Now auto-saves recommended bets to tracker for later evaluation.
"""

import warnings
warnings.filterwarnings('ignore')

import sys
from pathlib import Path
from datetime import datetime
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.live_odds import LiveOddsFetcher
from models.advanced_predictor import AdvancedPredictor
from betting.bet_tracker import BetTracker


# API Key
API_KEY = "bd6934ca89728830cd789ca6203dbe8b"

# Team abbreviation lookup
TEAM_ABBR = {
    'Atlanta Hawks': 'ATL', 'Boston Celtics': 'BOS', 'Brooklyn Nets': 'BKN',
    'Charlotte Hornets': 'CHA', 'Chicago Bulls': 'CHI', 'Cleveland Cavaliers': 'CLE',
    'Dallas Mavericks': 'DAL', 'Denver Nuggets': 'DEN', 'Detroit Pistons': 'DET',
    'Golden State Warriors': 'GSW', 'Houston Rockets': 'HOU', 'Indiana Pacers': 'IND',
    'Los Angeles Clippers': 'LAC', 'Los Angeles Lakers': 'LAL', 'Memphis Grizzlies': 'MEM',
    'Miami Heat': 'MIA', 'Milwaukee Bucks': 'MIL', 'Minnesota Timberwolves': 'MIN',
    'New Orleans Pelicans': 'NOP', 'New York Knicks': 'NYK', 'Oklahoma City Thunder': 'OKC',
    'Orlando Magic': 'ORL', 'Philadelphia 76ers': 'PHI', 'Phoenix Suns': 'PHX',
    'Portland Trail Blazers': 'POR', 'Sacramento Kings': 'SAC', 'San Antonio Spurs': 'SAS',
    'Toronto Raptors': 'TOR', 'Utah Jazz': 'UTA', 'Washington Wizards': 'WAS'
}


def get_team_abbr(full_name):
    """Get team abbreviation from full name."""
    return TEAM_ABBR.get(full_name, full_name[:3].upper())


def run_edge_analysis(auto_save: bool = True, stake: float = 50.0):
    """
    Run edge analysis on today's games.
    
    Args:
        auto_save: If True, auto-save recommended bets to tracker
        stake: Default stake amount for saved bets
    """
    print("=" * 70)
    print("EDGE ANALYSIS - NBA Games Today")
    print("=" * 70)
    print()
    
    # Get live odds
    print("Fetching live odds...")
    fetcher = LiveOddsFetcher(api_key=API_KEY)
    odds = fetcher.get_live_odds('nba', markets='h2h')
    
    if odds.empty:
        print("No odds data available")
        return []
    
    # Get h2h markets and average odds across books
    h2h_odds = odds[odds['market'] == 'h2h'].groupby(['home_team', 'away_team']).agg({
        'home_odds': 'mean',
        'away_odds': 'mean'
    }).reset_index()
    
    print(f"Found {len(h2h_odds)} games with odds")
    print()
    
    # Initialize predictor and tracker
    print("Loading prediction model...")
    predictor = AdvancedPredictor()
    tracker = BetTracker() if auto_save else None
    print()
    
    edges = []
    today = datetime.now().strftime("%Y-%m-%d")
    
    for _, row in h2h_odds.iterrows():
        home = row['home_team']
        away = row['away_team']
        home_abbr = get_team_abbr(home)
        away_abbr = get_team_abbr(away)
        home_odds = row['home_odds']
        away_odds = row['away_odds']
        
        try:
            pred = predictor.predict_with_odds(
                home_abbr, away_abbr,
                int(home_odds), int(away_odds)
            )
            
            if 'error' in pred:
                print(f"[{away_abbr} @ {home_abbr}] Error: {pred['error']}")
                continue
            
            ba = pred.get('betting_analysis', {})
            home_ev = ba.get('home_ev', 0)
            away_ev = ba.get('away_ev', 0)
            home_prob = pred['home_win_probability']
            away_prob = pred['away_win_probability']
            
            print(f"[{away_abbr} @ {home_abbr}]")
            print(f"  Model: {home_abbr} {home_prob:.1%} | {away_abbr} {away_prob:.1%}")
            print(f"  Odds: {home_abbr} {int(home_odds):+d} | {away_abbr} {int(away_odds):+d}")
            print(f"  EV: {home_abbr} {home_ev:.1%} | {away_abbr} {away_ev:.1%}")
            
            best_ev = max(home_ev, away_ev)
            if best_ev > 0.02:
                best_team = home_abbr if home_ev > away_ev else away_abbr
                best_prob = home_prob if home_ev > away_ev else away_prob
                best_odds = int(home_odds) if home_ev > away_ev else int(away_odds)
                
                print(f"  >>> EDGE FOUND: {best_team} (+{best_ev:.1%} EV)")
                
                edge_data = {
                    'game': f"{away_abbr} @ {home_abbr}",
                    'home_team': home_abbr,
                    'away_team': away_abbr,
                    'team': best_team,
                    'ev': best_ev,
                    'prob': best_prob,
                    'odds': best_odds
                }
                edges.append(edge_data)
                
                # Auto-save to bet tracker
                if auto_save and tracker:
                    bet_id = tracker.place_bet(
                        home_team=home_abbr,
                        away_team=away_abbr,
                        bet_type='moneyline',
                        selection=best_team,
                        odds=best_odds,
                        stake=stake,
                        model_prob=best_prob,
                        expected_value=best_ev,
                        game_date=today,
                        notes=f"Auto-saved from daily_edge.py"
                    )
                    edge_data['bet_id'] = bet_id
                    
            else:
                print(f"  No significant edge")
            print()
            
        except Exception as e:
            print(f"[{away_abbr} @ {home_abbr}] Error: {str(e)[:50]}")
            print()
    
    # Summary
    print("=" * 70)
    print("RECOMMENDED BETS")
    print("=" * 70)
    
    if edges:
        edges = sorted(edges, key=lambda x: x['ev'], reverse=True)
        
        print(f"\nFound {len(edges)} bets with positive expected value:\n")
        for i, e in enumerate(edges, 1):
            print(f"{i}. {e['game']}")
            print(f"   Bet: {e['team']} ({e['odds']:+d})")
            print(f"   Model Probability: {e['prob']:.1%}")
            print(f"   Expected Value: +{e['ev']:.1%}")
            if 'bet_id' in e:
                print(f"   >>> Saved as Bet #{e['bet_id']}")
            print()
        
        if auto_save:
            print(f"All {len(edges)} bets saved to database.")
            print("To resolve after games: python -c \"from src.betting.bet_tracker import BetTracker; t=BetTracker(); print(t.get_pending_bets())\"")
    else:
        print("\nNo significant edges found in today's games.")
        print("The market odds are fairly priced relative to our model.")
    
    print("=" * 70)
    print("DISCLAIMER: For educational purposes only.")
    print("Past performance does not guarantee future results.")
    print("=" * 70)
    
    return edges


def resolve_todays_bets(scores: dict):
    """
    Resolve today's bets with actual game scores.
    
    Args:
        scores: Dict of {(home_team, away_team): (home_score, away_score)}
        
    Example:
        resolve_todays_bets({
            ('LAL', 'GSW'): (115, 108),
            ('BOS', 'MIA'): (105, 102)
        })
    """
    tracker = BetTracker()
    pending = tracker.get_pending_bets()
    
    resolved = 0
    for _, bet in pending.iterrows():
        key = (bet['home_team'], bet['away_team'])
        if key in scores:
            home_score, away_score = scores[key]
            result, pl = tracker.resolve_by_scores(bet['id'], home_score, away_score)
            print(f"Bet #{bet['id']}: {result.upper()} (${pl:+.2f})")
            resolved += 1
    
    print(f"\nResolved {resolved} bets")
    return resolved


if __name__ == "__main__":
    run_edge_analysis(auto_save=True, stake=50.0)

