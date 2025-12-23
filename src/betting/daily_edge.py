
"""
Daily Edge Analysis

Finds betting edges using our model vs live market odds.
Now supports multi-sport analysis (NBA, NFL).
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
    'Toronto Raptors': 'TOR', 'Utah Jazz': 'UTA', 'Washington Wizards': 'WAS',
    
    # NFL Teams
    'Arizona Cardinals': 'ARI', 'Atlanta Falcons': 'ATL', 'Baltimore Ravens': 'BAL',
    'Buffalo Bills': 'BUF', 'Carolina Panthers': 'CAR', 'Chicago Bears': 'CHI',
    'Cincinnati Bengals': 'CIN', 'Cleveland Browns': 'CLE', 'Dallas Cowboys': 'DAL',
    'Denver Broncos': 'DEN', 'Detroit Lions': 'DET', 'Green Bay Packers': 'GB',
    'Houston Texans': 'HOU', 'Indianapolis Colts': 'IND', 'Jacksonville Jaguars': 'JAX',
    'Kansas City Chiefs': 'KC', 'Las Vegas Raiders': 'LV', 'Los Angeles Chargers': 'LAC',
    'Los Angeles Rams': 'LAR', 'Miami Dolphins': 'MIA', 'Minnesota Vikings': 'MIN',
    'New England Patriots': 'NE', 'New Orleans Saints': 'NO', 'New York Giants': 'NYG',
    'New York Jets': 'NYJ', 'Philadelphia Eagles': 'PHI', 'Pittsburgh Steelers': 'PIT',
    'San Francisco 49ers': 'SF', 'Seattle Seahawks': 'SEA', 'Tampa Bay Buccaneers': 'TB',
    'Tennessee Titans': 'TEN', 'Washington Commanders': 'WAS'
}


def get_team_abbr(full_name):
    """Get team abbreviation from full name."""
    return TEAM_ABBR.get(full_name, full_name[:3].upper())


def run_edge_analysis(auto_save: bool = True, stake: float = 50.0):
    """
    Run edge analysis on today's games for multiple sports.
    """
    print("=" * 70)
    print("EDGE ANALYSIS - Multi-Sport")
    print("=" * 70)
    print()
    
    tracker = BetTracker() if auto_save else None
    today = datetime.now().strftime("%Y-%m-%d")
    all_edges = []
    
    sports = ['NBA', 'NFL', 'NCAAF']
    
    for sport in sports:
        print(f"\n--- Processing {sport} ---")
        try:
            # Initialize predictor for this sport
            # Use lower threshold for NFL since model is new/less proven? Or keep same?
            # Keeping same 0.02 (2%) for consistency.
            predictor = AdvancedPredictor(sport=sport, min_ev_threshold=0.02)
            
            # Fetch Schedule
            print(f"  Fetching {sport} schedule...")
            games_df = predictor.sport_fetcher.get_schedule()
            
            if games_df.empty:
                print(f"  No {sport} games found for today.")
                # For NCAAF, we might rely purely on odds if no schedule found via API
                if sport == 'NCAAF':
                     print("  Checking for NCAAF odds directly...")
            else:
                print(f"  Found {len(games_df)} games.")
            
            # Fetch Live Odds for this sport
            if sport == 'NFL':
                sport_key = 'americanfootball_nfl'
            elif sport == 'NCAAF':
                sport_key = 'americanfootball_ncaaf'
            else:
                sport_key = 'basketball_nba'

            odds_fetcher = LiveOddsFetcher(api_key=API_KEY)
            odds = odds_fetcher.get_live_odds(sport_key, markets='h2h,totals')
            
            if odds.empty:
                print("  No odds data available.")
                continue
                
            # Process by Game
            game_groups = odds.groupby('game_id')
            
            for game_id, group in game_groups:
                home = group.iloc[0]['home_team']
                away = group.iloc[0]['away_team']
                home_abbr = get_team_abbr(home)
                away_abbr = get_team_abbr(away)
                
                # Check if this game is in our schedule (optional validation, but good for linking data)
                # For now, we rely on odds mainly.
                
                # Extract H2H odds
                h2h = group[group['market'] == 'h2h']
                if h2h.empty: continue
                
                home_odds = int(h2h['home_odds'].mean())
                away_odds = int(h2h['away_odds'].mean())
                
                print(f"\n[{away_abbr} @ {home_abbr}]")
                
                # Prediction
                pred = predictor.predict_with_odds(home_abbr, away_abbr, home_odds, away_odds)
                
                if 'error' in pred:
                    print(f"  Error: {pred['error']}")
                    continue
                    
                # Analysis Output
                ba = pred.get('betting_analysis', {})
                home_prob = pred['home_win_probability']
                away_prob = pred['away_win_probability']
                model_used = pred.get('model_used', 'unknown')
                
                print(f"  Moneyline: {home_abbr} {home_prob:.1%} ({home_odds:+d}) | {away_abbr} {away_prob:.1%} ({away_odds:+d})")
                
                best_ev = max(ba.get('home_ev', 0), ba.get('away_ev', 0))
                if best_ev > 0.02:
                    best_team = ba['best_bet']
                    print(f"  >>> EDGE: {best_team} (+{best_ev:.1%} EV) [{model_used}]")
                    
                    edge_data = {
                        'sport': sport,
                        'game': f"{away_abbr} @ {home_abbr}",
                        'bet_type': 'moneyline',
                        'team': best_team,
                        'ev': best_ev,
                        'odds': home_odds if best_team == home_abbr else away_odds,
                        'prob': home_prob if best_team == home_abbr else away_prob
                    }
                    all_edges.append(edge_data)
                    
                    if auto_save and tracker:
                        tracker.place_bet(
                            home_team=home_abbr, away_team=away_abbr, bet_type='moneyline',
                            selection=best_team, odds=edge_data['odds'],
                            stake=stake, model_prob=edge_data['prob'],
                            expected_value=best_ev, game_date=today, notes=f"{sport} Auto-bet"
                        )
                else:
                    print(f"  No value (Max EV: {best_ev:.1%})")

        except Exception as e:
            print(f"Error processing {sport}: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    if all_edges:
        all_edges = sorted(all_edges, key=lambda x: x['ev'], reverse=True)
        for i, e in enumerate(all_edges, 1):
            print(f"{i}. [{e['sport']}] {e['game']} -> {e['team']} ({e['prob']:.1%}, EV: {e['ev']:.1%})")
        if auto_save:
            print(f"\nAll {len(all_edges)} bets saved to database.")
    else:
        print("No significant edges found today.")

if __name__ == "__main__":
    run_edge_analysis(auto_save=True, stake=50.0)
