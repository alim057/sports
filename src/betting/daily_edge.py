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
    odds = fetcher.get_live_odds('nba', markets='h2h,totals')
    
    if odds.empty:
        print("No odds data available")
        return []
    
    # Process odds by game
    # We need to group by game_id to keep H2H and Totals together
    game_groups = odds.groupby('game_id')
    
    print(f"Found {len(game_groups)} games with odds")
    print()
    
    # Initialize predictor and tracker
    print("Loading prediction model...")
    predictor = AdvancedPredictor()
    tracker = BetTracker() if auto_save else None
    print()
    
    edges = []
    today = datetime.now().strftime("%Y-%m-%d")
    
    for game_id, group in game_groups:
        # Get basic game info
        home = group.iloc[0]['home_team']
        away = group.iloc[0]['away_team']
        home_abbr = get_team_abbr(home)
        away_abbr = get_team_abbr(away)
        
        # Extract H2H odds (avg)
        h2h = group[group['market'] == 'h2h']
        if h2h.empty:
            continue
            
        home_odds = int(h2h['home_odds'].mean())
        away_odds = int(h2h['away_odds'].mean())
        
        # Extract Totals (avg over line)
        totals = group[group['market'] == 'totals']
        total_line = 0
        over_odds = -110
        under_odds = -110
        
        if not totals.empty:
            # simple avg of the most common line or just first
            total_line = totals['total'].mode()[0] if not totals['total'].empty else 225.0
            # Get odds for this line
            line_odds = totals[totals['total'] == total_line]
            if not line_odds.empty:
                over_odds = int(line_odds['over_odds'].mean())
                under_odds = int(line_odds['under_odds'].mean())
        
        print(f"[{away_abbr} @ {home_abbr}]")
        
        try:
            # 1. Moneyline Prediction
            pred = predictor.predict_with_odds(
                home_abbr, away_abbr,
                home_odds, away_odds
            )
            
            if 'error' in pred:
                print(f"  Error: {pred['error']}")
                continue
            
            # Print Moneyline Analysis
            ba = pred.get('betting_analysis', {})
            home_prob = pred['home_win_probability']
            away_prob = pred['away_win_probability']
            
            print(f"  Moneyline: {home_abbr} {home_prob:.1%} ({home_odds:+d}) | {away_abbr} {away_prob:.1%} ({away_odds:+d})")
            
            best_ev = max(ba.get('home_ev', 0), ba.get('away_ev', 0))
            if best_ev > 0.02:
                best_team = ba['best_bet']
                print(f"  >>> ML EDGE: {best_team} (+{best_ev:.1%} EV)")
                
                # Save ML bet
                edges.append({
                    'game': f"{away_abbr} @ {home_abbr}",
                    'bet_type': 'moneyline',
                    'team': best_team,
                    'ev': best_ev,
                    'odds': home_odds if best_team == home_abbr else away_odds,
                    'prob': home_prob if best_team == home_abbr else away_prob
                })
                if auto_save and tracker:
                    tracker.place_bet(
                        home_team=home_abbr, away_team=away_abbr, bet_type='moneyline',
                        selection=best_team, odds=(home_odds if best_team == home_abbr else away_odds),
                        stake=stake, model_prob=(home_prob if best_team == home_abbr else away_prob),
                        expected_value=best_ev, game_date=today, notes="ML Auto-bet"
                    )

            # 2. Totals Prediction
            if total_line > 0:
                t_pred = predictor.predict_total_with_odds(
                    home_abbr, away_abbr, total_line, over_odds, under_odds
                )
                t_ba = t_pred.get('betting_analysis', {})
                over_ev = t_ba.get('over_ev', 0)
                under_ev = t_ba.get('under_ev', 0)
                
                print(f"  Total: {total_line} (Exp: {t_pred['expected_total']:.1f}) | Over {t_pred['over_prob']:.1%} | Under {(1-t_pred['over_prob']):.1%}")
                
                # Report Weather
                if t_pred.get('weather'):
                    w = t_pred['weather']
                    if w.get('is_dome'):
                        print(f"  Weather: Dome (No Impact)")
                    else:
                        print(f"  Weather: {w.get('temp_f')}F, {w.get('wind_mph')}mph, {w.get('condition')}")

                if t_ba.get('note'):
                    print(f"  Note: {t_ba['note']}")
                
                best_total_ev = max(over_ev, under_ev)
                if best_total_ev > 0.02:
                    best_side = t_ba['best_bet'] # "Over X"
                    print(f"  >>> TOT EDGE: {best_side} (+{best_total_ev:.1%} EV)")
                    
                    edges.append({
                        'game': f"{away_abbr} @ {home_abbr}",
                        'bet_type': 'total',
                        'team': best_side,
                        'ev': best_total_ev,
                        'odds': over_odds if "Over" in best_side else under_odds,
                        'prob': t_pred['over_prob'] if "Over" in best_side else (1-t_pred['over_prob'])
                    })
                    if auto_save and tracker:
                        tracker.place_bet(
                            home_team=home_abbr, away_team=away_abbr, bet_type='total',
                            selection=best_side, odds=(over_odds if "Over" in best_side else under_odds),
                            stake=stake, model_prob=(t_pred['over_prob'] if "Over" in best_side else (1-t_pred['over_prob'])),
                            expected_value=best_total_ev, game_date=today, notes=t_ba.get('note', 'Total Auto-bet')
                        )
            
            print()
            
        except Exception as e:
            print(f"  Error: {str(e)[:50]}")
            continue
    
    # Summary
    print("=" * 70)
    print("RECOMMENDED BETS")
    print("=" * 70)
    
    if edges:
        edges = sorted(edges, key=lambda x: x['ev'], reverse=True)
        
        print(f"\nFound {len(edges)} bets with positive expected value:\n")
        for i, e in enumerate(edges, 1):
            print(f"{i}. {e['game']} ({e['bet_type']})")
            print(f"   Bet: {e['team']} ({e['odds']:+d})")
            print(f"   Model Probability: {e['prob']:.1%}")
            print(f"   Expected Value: +{e['ev']:.1%}")
            print()
        
        if auto_save:
            print(f"All {len(edges)} bets saved.")
    else:
        print("\nNo significant edges found.")
    
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

