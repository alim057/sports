"""
Dashboard API Server

Serves predictions, live odds, and edge analysis to the web dashboard.
"""

from flask import Flask, jsonify, send_from_directory, request
from flask_cors import CORS
from pathlib import Path
from datetime import datetime
import sys
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from models.advanced_predictor import AdvancedPredictor
from betting.edge_analyzer import EdgeAnalyzer
from data.injury_fetcher import InjuryFetcher
from data.live_odds import LiveOddsFetcher

app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)


@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory('.', path)

def convert_numpy(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy(v) for v in obj]
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


# API Key for The Odds API
API_KEY = "bd6934ca89728830cd789ca6203dbe8b"

# Team abbreviation lookup (all sports)
TEAM_ABBR = {
    # NBA
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
    # NFL
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
    'Tennessee Titans': 'TEN', 'Washington Commanders': 'WAS',
    # NHL
    'Anaheim Ducks': 'ANA', 'Arizona Coyotes': 'ARI', 'Boston Bruins': 'BOS',
    'Buffalo Sabres': 'BUF', 'Calgary Flames': 'CGY', 'Carolina Hurricanes': 'CAR',
    'Chicago Blackhawks': 'CHI', 'Colorado Avalanche': 'COL', 'Columbus Blue Jackets': 'CBJ',
    'Dallas Stars': 'DAL', 'Detroit Red Wings': 'DET', 'Edmonton Oilers': 'EDM',
    'Florida Panthers': 'FLA', 'Los Angeles Kings': 'LAK', 'Minnesota Wild': 'MIN',
    'Montreal Canadiens': 'MTL', 'Nashville Predators': 'NSH', 'New Jersey Devils': 'NJD',
    'New York Islanders': 'NYI', 'New York Rangers': 'NYR', 'Ottawa Senators': 'OTT',
    'Philadelphia Flyers': 'PHI', 'Pittsburgh Penguins': 'PIT', 'San Jose Sharks': 'SJ',
    'Seattle Kraken': 'SEA', 'St. Louis Blues': 'STL', 'Tampa Bay Lightning': 'TBL',
    'Toronto Maple Leafs': 'TOR', 'Vancouver Canucks': 'VAN', 'Vegas Golden Knights': 'VGK',
    'Washington Capitals': 'WSH', 'Winnipeg Jets': 'WPG',
    # Utah Hockey Club (new NHL team)
    'Utah Hockey Club': 'UTA',
}

# Component instances (lazy loading)
_predictor = None
_edge_analyzer = None
_injury_fetcher = None
_odds_fetcher = None

def get_predictor():
    global _predictor
    if _predictor is None:
        _predictor = AdvancedPredictor()
    return _predictor

def get_edge_analyzer():
    global _edge_analyzer
    if _edge_analyzer is None:
        _edge_analyzer = EdgeAnalyzer()
    return _edge_analyzer

def get_injury_fetcher():
    global _injury_fetcher
    if _injury_fetcher is None:
        _injury_fetcher = InjuryFetcher()
    return _injury_fetcher

def get_odds_fetcher():
    global _odds_fetcher
    if _odds_fetcher is None:
        _odds_fetcher = LiveOddsFetcher(api_key=API_KEY)
    return _odds_fetcher

def get_team_abbr(full_name):
    """Get team abbreviation from full name."""
    return TEAM_ABBR.get(full_name, full_name[:3].upper())

def validate_moneyline_odds(home_odds, away_odds):
    """
    Validate that odds look like American moneyline, not spreads.
    Spreads are typically small numbers like -6, +3.5
    Moneyline odds are typically -300 to +900 range (at least abs > 100)
    """
    # Both odds should have absolute value > 100 for moneyline
    if abs(home_odds) < 100 or abs(away_odds) < 100:
        return False
    return True


# ============== Live Data Routes ==============

@app.route('/api/live-odds', methods=['GET'])
def get_live_odds():
    """Get live odds for a sport."""
    try:
        sport = request.args.get('sport', 'nba')
        fetcher = get_odds_fetcher()
        
        odds = fetcher.get_live_odds(sport)
        
        if odds.empty:
            return jsonify({'odds': [], 'message': 'No odds available'})
        
        # Get unique games
        games = odds.groupby('game_id').first().reset_index()
        
        return jsonify({
            'sport': sport,
            'games': len(games),
            'totalRecords': len(odds),
            'lastUpdated': datetime.now().isoformat(),
            'odds': odds.to_dict('records')
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/live-predictions', methods=['GET'])
def get_live_predictions():
    """Get predictions for today's games with live odds."""
    try:
        sport = request.args.get('sport', 'nba')
        fetcher = get_odds_fetcher()
        predictor = get_predictor()
        
        # Get live odds
        odds = fetcher.get_live_odds(sport, markets='h2h')
        
        if odds.empty:
            return jsonify({'predictions': [], 'message': 'No games available'})
        
        # Get h2h odds averaged across books
        h2h = odds[odds['market'] == 'h2h'].groupby(['home_team', 'away_team', 'game_id']).agg({
            'home_odds': 'mean',
            'away_odds': 'mean',
            'commence_time': 'first'
        }).reset_index()
        
        predictions = []
        
        for _, row in h2h.iterrows():
            home = row['home_team']
            away = row['away_team']
            home_abbr = get_team_abbr(home)
            away_abbr = get_team_abbr(away)
            home_odds = int(row['home_odds'])
            away_odds = int(row['away_odds'])
            
            try:
                result = predictor.predict_with_odds(
                    home_abbr, away_abbr, home_odds, away_odds
                )
                
                if 'error' in result:
                    continue
                
                ba = result.get('betting_analysis', {})
                home_ev = ba.get('home_ev', 0)
                away_ev = ba.get('away_ev', 0)
                best_ev = max(home_ev, away_ev)
                
                predictions.append({
                    'gameId': row['game_id'],
                    'homeTeam': home,
                    'awayTeam': away,
                    'homeAbbr': home_abbr,
                    'awayAbbr': away_abbr,
                    'startTime': row['commence_time'],
                    'homeProb': result['home_win_probability'],
                    'awayProb': result['away_win_probability'],
                    'homeOdds': home_odds,
                    'awayOdds': away_odds,
                    'predictedWinner': result['predicted_winner'],
                    'confidence': result['confidence'],
                    'homeEv': home_ev,
                    'awayEv': away_ev,
                    'hasEdge': best_ev > 0.02,
                    'recommendedBet': home_abbr if home_ev > away_ev else away_abbr,
                    'recommendedBetEv': best_ev,
                    'teamAnalysis': result.get('team_analysis', {}),
                    'playerAnalysis': result.get('player_analysis', {})
                })
            except Exception as e:
                print(f"Prediction error for {away_abbr} @ {home_abbr}: {e}")
                continue
        
        # Sort by EV
        predictions.sort(key=lambda x: x['recommendedBetEv'], reverse=True)
        
        return jsonify(convert_numpy({
            'sport': sport,
            'totalGames': len(predictions),
            'gamesWithEdge': sum(1 for p in predictions if p['hasEdge']),
            'lastUpdated': datetime.now().isoformat(),
            'predictions': predictions
        }))
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/edge-analysis', methods=['GET'])
def get_edge_analysis():
    """Get full edge analysis for today's games."""
    try:
        sport = request.args.get('sport', 'nba')
        
        # Sample/demo data when no live games
        SAMPLE_EDGES = {
            'nba': [
                {'game': 'SAS @ WAS', 'team': 'WAS', 'odds': 310, 'modelProbability': 0.42, 'ev': 0.58},
                {'game': 'ORL @ DET', 'team': 'ORL', 'odds': -145, 'modelProbability': 0.65, 'ev': 0.12},
                {'game': 'MIL @ MIN', 'team': 'MIN', 'odds': 125, 'modelProbability': 0.52, 'ev': 0.17},
                {'game': 'LAL @ BOS', 'team': 'BOS', 'odds': -180, 'modelProbability': 0.72, 'ev': 0.08},
            ],
            'nfl': [
                {'game': 'KC @ SF', 'team': 'SF', 'odds': 145, 'modelProbability': 0.48, 'ev': 0.18},
                {'game': 'BUF @ MIA', 'team': 'BUF', 'odds': -125, 'modelProbability': 0.60, 'ev': 0.09},
                {'game': 'DAL @ PHI', 'team': 'PHI', 'odds': -160, 'modelProbability': 0.68, 'ev': 0.07},
            ],
            'nhl': [
                {'game': 'TOR @ MTL', 'team': 'TOR', 'odds': -135, 'modelProbability': 0.62, 'ev': 0.11},
                {'game': 'BOS @ NYR', 'team': 'BOS', 'odds': 110, 'modelProbability': 0.54, 'ev': 0.14},
            ],
            'ncaaf': [
                {'game': 'OSU @ MICH', 'team': 'OSU', 'odds': -175, 'modelProbability': 0.70, 'ev': 0.06},
                {'game': 'UGA @ ALA', 'team': 'UGA', 'odds': 135, 'modelProbability': 0.50, 'ev': 0.17},
            ]
        }
        
        try:
            fetcher = get_odds_fetcher()
            odds = fetcher.get_live_odds(sport, markets='h2h')
        except:
            odds = None
        
        # Use sample data if no live odds
        if odds is None or (hasattr(odds, 'empty') and odds.empty):
            edges = SAMPLE_EDGES.get(sport.lower(), [])
            return jsonify({
                'sport': sport,
                'lastUpdated': datetime.now().isoformat(),
                'summary': {
                    'totalGames': len(edges),
                    'gamesWithEdge': len(edges),
                    'avgEdge': sum(e['ev'] for e in edges) / len(edges) if edges else 0
                },
                'edges': edges,
                'isDemo': True
            })
        
        # Get h2h odds
        h2h = odds[odds['market'] == 'h2h'].groupby(['home_team', 'away_team']).agg({
            'home_odds': 'mean',
            'away_odds': 'mean',
            'commence_time': 'first'
        }).reset_index()
        
        edges = []
        
        for _, row in h2h.iterrows():
            home = row['home_team']
            away = row['away_team']
            home_abbr = get_team_abbr(home)
            away_abbr = get_team_abbr(away)
            home_odds = int(row['home_odds'])
            away_odds = int(row['away_odds'])
            
            # Skip if odds look like spreads (not moneyline)
            if not validate_moneyline_odds(home_odds, away_odds):
                continue
            
            # Simple EV calculation without complex predictor
            def implied_prob(odds):
                if odds < 0:
                    return abs(odds) / (abs(odds) + 100)
                return 100 / (odds + 100)
            
            def calc_ev(model_prob, odds):
                impl = implied_prob(odds)
                if odds > 0:
                    payout = odds / 100
                else:
                    payout = 100 / abs(odds)
                return model_prob * payout - (1 - model_prob)
            
            home_impl = implied_prob(home_odds)
            away_impl = implied_prob(away_odds)
            
            # Simple model: assume slight edge over market + home advantage
            home_model_prob = home_impl + 0.03 + 0.02  # Market + edge + home
            away_model_prob = away_impl + 0.03
            
            # Normalize
            total = home_model_prob + away_model_prob
            home_model_prob /= total
            away_model_prob /= total
            
            home_ev = calc_ev(home_model_prob, home_odds)
            away_ev = calc_ev(away_model_prob, away_odds)
            best_ev = max(home_ev, away_ev)
            
            if best_ev > 0.02:
                best_team = home_abbr if home_ev > away_ev else away_abbr
                best_odds = home_odds if home_ev > away_ev else away_odds
                best_prob = home_model_prob if home_ev > away_ev else away_model_prob
                
                edges.append({
                    'game': f"{away_abbr} @ {home_abbr}",
                    'team': best_team,
                    'odds': best_odds,
                    'modelProbability': best_prob,
                    'ev': best_ev,
                    'startTime': row.get('commence_time', None)
                })
        
        # Sort by EV
        edges.sort(key=lambda x: x['ev'], reverse=True)
        
        return jsonify({
            'sport': sport,
            'lastUpdated': datetime.now().isoformat(),
            'summary': {
                'totalGames': len(h2h),
                'gamesWithEdge': len(edges),
                'avgEdge': sum(e['ev'] for e in edges) / len(edges) if edges else 0
            },
            'edges': edges
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/spread-analysis', methods=['GET'])
def get_spread_analysis():
    """Get spread betting edge analysis for today's games."""
    try:
        sport = request.args.get('sport', 'nba')
        
        try:
            fetcher = get_odds_fetcher()
            odds = fetcher.get_live_odds(sport, markets='spreads')
        except:
            odds = None
        
        if odds is None or (hasattr(odds, 'empty') and odds.empty):
            return jsonify({
                'sport': sport,
                'lastUpdated': datetime.now().isoformat(),
                'summary': {'totalGames': 0, 'gamesWithEdge': 0, 'avgEdge': 0},
                'edges': [],
                'isDemo': True
            })
        
        # Get spread odds
        spreads = odds[odds['market'] == 'spreads'].groupby(['home_team', 'away_team']).agg({
            'home_odds': 'mean',
            'away_odds': 'mean',
            'home_point': 'mean',  # The spread value
            'away_point': 'mean'
        }).reset_index()
        
        edges = []
        
        for _, row in spreads.iterrows():
            home = row['home_team']
            away = row['away_team']
            home_abbr = get_team_abbr(home)
            away_abbr = get_team_abbr(away)
            home_odds = int(row['home_odds'])
            away_odds = int(row['away_odds'])
            home_spread = row.get('home_point', 0)
            away_spread = row.get('away_point', 0)
            
            # Skip if odds aren't valid moneyline format (should be -110 style for spreads)
            if not validate_moneyline_odds(home_odds, away_odds):
                continue
            
            # Dynamic spread model: probability varies based on spread size
            # Larger spreads = harder to cover = lower probability
            def spread_cover_prob(spread, is_home=True):
                spread = abs(float(spread)) if spread else 0
                
                # Base probability that adjusts with spread size
                if spread <= 2.5:
                    base = 0.53  # Small spreads - slight edge
                elif spread <= 5.5:
                    base = 0.51  # Medium spreads
                elif spread <= 9.5:
                    base = 0.49  # Larger spreads - harder to cover
                else:
                    base = 0.47  # Very large spreads (10+) - favorites often don't cover
                
                # Home court/field advantage
                home_boost = 0.02 if is_home else 0
                
                return min(0.60, max(0.40, base + home_boost))
            
            home_cover_prob = spread_cover_prob(home_spread, True)
            away_cover_prob = spread_cover_prob(away_spread, False)
            
            # Calculate EV for each side
            def calc_spread_ev(prob, odds):
                if odds > 0:
                    payout = odds / 100
                else:
                    payout = 100 / abs(odds)
                return prob * payout - (1 - prob)
            
            home_ev = calc_spread_ev(home_cover_prob, home_odds)
            away_ev = calc_spread_ev(away_cover_prob, away_odds)
            best_ev = max(home_ev, away_ev)
            
            if best_ev > 0.02:
                best_team = home_abbr if home_ev > away_ev else away_abbr
                best_odds = home_odds if home_ev > away_ev else away_odds
                best_prob = home_cover_prob if home_ev > away_ev else away_cover_prob
                best_spread = home_spread if home_ev > away_ev else away_spread
                
                edges.append({
                    'game': f"{away_abbr} @ {home_abbr}",
                    'team': best_team,
                    'spread': best_spread,
                    'odds': best_odds,
                    'modelProbability': best_prob,
                    'ev': best_ev,
                    'betType': 'spread'
                })
        
        # Sort by EV
        edges.sort(key=lambda x: x['ev'], reverse=True)
        
        return jsonify({
            'sport': sport,
            'lastUpdated': datetime.now().isoformat(),
            'summary': {
                'totalGames': len(spreads),
                'gamesWithEdge': len(edges),
                'avgEdge': sum(e['ev'] for e in edges) / len(edges) if edges else 0
            },
            'edges': edges
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/totals-analysis', methods=['GET'])
def get_totals_analysis():
    """Get totals (over/under) betting edge analysis for today's games."""
    try:
        sport = request.args.get('sport', 'nba')
        
        try:
            fetcher = get_odds_fetcher()
            odds = fetcher.get_live_odds(sport, markets='totals')
        except:
            odds = None
        
        if odds is None or (hasattr(odds, 'empty') and odds.empty):
            return jsonify({
                'sport': sport,
                'lastUpdated': datetime.now().isoformat(),
                'summary': {'totalGames': 0, 'gamesWithEdge': 0, 'avgEdge': 0},
                'edges': [],
                'isDemo': True
            })
        
        # Get totals odds
        totals = odds[odds['market'] == 'totals'].groupby(['home_team', 'away_team']).agg({
            'over_odds': 'mean',
            'under_odds': 'mean',
            'total_point': 'mean'  # The O/U line
        }).reset_index()
        
        edges = []
        
        for _, row in totals.iterrows():
            home = row['home_team']
            away = row['away_team']
            home_abbr = get_team_abbr(home)
            away_abbr = get_team_abbr(away)
            
            over_odds = row.get('over_odds', -110)
            under_odds = row.get('under_odds', -110)
            total_line = row.get('total_point', 220)
            
            # Handle NaN values
            if pd.isna(over_odds) or pd.isna(under_odds):
                continue
                
            over_odds = int(over_odds)
            under_odds = int(under_odds)
            
            # Skip if odds aren't valid moneyline format
            if not validate_moneyline_odds(over_odds, under_odds):
                continue
            
            # Simple totals model: slight edge toward under for most games
            # (many games go under in low-scoring sports like NHL, MLB)
            base_over_prob = 0.50
            if sport.lower() in ['nhl', 'mlb']:
                base_over_prob = 0.48  # Slight under bias
            elif sport.lower() in ['nba', 'ncaaf']:
                base_over_prob = 0.51  # Slight over bias
            
            over_model_prob = base_over_prob + 0.03  # Our edge
            under_model_prob = 1 - over_model_prob
            
            # Calculate EV for each side
            def calc_totals_ev(prob, odds):
                if odds > 0:
                    payout = odds / 100
                else:
                    payout = 100 / abs(odds)
                return prob * payout - (1 - prob)
            
            over_ev = calc_totals_ev(over_model_prob, over_odds)
            under_ev = calc_totals_ev(under_model_prob, under_odds)
            best_ev = max(over_ev, under_ev)
            
            if best_ev > 0.02:
                pick = 'OVER' if over_ev > under_ev else 'UNDER'
                best_odds = over_odds if over_ev > under_ev else under_odds
                best_prob = over_model_prob if over_ev > under_ev else under_model_prob
                
                edges.append({
                    'game': f"{away_abbr} @ {home_abbr}",
                    'pick': pick,
                    'line': total_line,
                    'odds': best_odds,
                    'modelProbability': best_prob,
                    'ev': best_ev,
                    'betType': 'total'
                })
        
        # Sort by EV
        edges.sort(key=lambda x: x['ev'], reverse=True)
        
        return jsonify({
            'sport': sport,
            'lastUpdated': datetime.now().isoformat(),
            'summary': {
                'totalGames': len(totals),
                'gamesWithEdge': len(edges),
                'avgEdge': sum(e['ev'] for e in edges) / len(edges) if edges else 0
            },
            'edges': edges
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/best-odds', methods=['GET'])
def get_best_odds():
    """Get best available odds across bookmakers."""
    try:
        sport = request.args.get('sport', 'nba')
        fetcher = get_odds_fetcher()
        
        best = fetcher.get_best_odds(sport)
        
        if best.empty:
            return jsonify({'games': []})
        
        return jsonify({
            'sport': sport,
            'games': best.to_dict('records')
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ============== Model & Performance Routes ==============

from betting.bet_tracker import BetTracker

_bet_tracker = None

def get_bet_tracker():
    global _bet_tracker
    if _bet_tracker is None:
        _bet_tracker = BetTracker(db_path="./data/bets.db")
    return _bet_tracker

# ============== Model & Performance Routes ==============

@app.route('/api/performance', methods=['GET'])
def get_performance():
    """Get historical performance stats with $100 starting budget."""
    try:
        STARTING_BANKROLL = 100.0
        
        tracker = get_bet_tracker()
        perf = tracker.get_performance_summary()
        
        # Calculate derived stats
        total_profit = perf.get('total_profit', 0)
        current_bankroll = STARTING_BANKROLL + total_profit
        
        response = {
            'total_bets': perf.get('total_bets', 0),
            'wins': perf.get('wins', 0),
            'losses': perf.get('losses', 0),
            'win_rate': perf.get('win_rate', 0),
            'total_wagered': perf.get('total_staked', 0),
            'total_returned': (perf.get('total_staked', 0) + perf.get('total_profit', 0)),
            'profit': round(total_profit, 2),
            'roi': perf.get('roi', 0),
            'starting_bankroll': STARTING_BANKROLL,
            'current_bankroll': round(current_bankroll, 2),
            'isDemo': False
        }
        
        return jsonify({'performance': response})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/performance-history', methods=['GET'])
def get_performance_history():
    """Get daily P/L history for chart rendering."""
    try:
        tracker = get_bet_tracker()
        history = tracker.get_daily_performance()
        
        if not history:
            return jsonify({'history': [], 'message': 'No resolved bets yet'})
            
        total_profit = history[-1]['cumulative'] if history else 0
        total_bets = sum(h['wins'] + h['losses'] for h in history)
        
        return jsonify({
            'history': history,
            'totalProfit': round(total_profit, 2),
            'totalBets': total_bets,
            'isDemo': False
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/recent-bets', methods=['GET'])
def get_recent_bets():
    """Get recent bets for the Performance tab table."""
    try:
        from flask import request
        status_filter = request.args.get('status', 'all')
        
        # Convert 'all' to None for BetTracker
        tracker_status = None
        if status_filter != 'all':
            tracker_status = status_filter
            
        tracker = get_bet_tracker()
        # Fetch last 50 bets
        bets_df = tracker.get_bet_history(limit=50, status=tracker_status)
        
        if bets_df.empty:
            return jsonify({'bets': [], 'message': 'No bet history found'})
        
        bets = []
        for _, row in bets_df.iterrows():
            result = row.get('result', 'pending').upper()
            profit = row.get('profit_loss', 0)
            
            # Format payout string
            if result == 'WIN':
                payout = f"+{profit:.2f}"
            elif result == 'LOSS':
                payout = f"{profit:.2f}"
            else:
                payout = "-"
                
            bets.append({
                'date': row.get('game_date', ''),
                'sport': row.get('sport', '').upper(),
                'game': f"{row.get('away_team')} @ {row.get('home_team')}",
                'team': row.get('selection', ''),
                'result': result,
                'odds': row.get('odds', 0),
                'payout': payout
            })
            
        return jsonify({'bets': bets})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/injuries', methods=['GET'])
def get_injuries():
    """Get current injury data."""
    try:
        fetcher = get_injury_fetcher()
        injuries = fetcher.get_all_injuries()
        
        if injuries.empty:
            return jsonify({'injuries': []})
        
        return jsonify({
            'injuries': injuries.to_dict('records')
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/injuries/<team>', methods=['GET'])
def get_team_injuries(team):
    """Get injuries for a specific team."""
    try:
        fetcher = get_injury_fetcher()
        impact = fetcher.calculate_team_injury_impact(team.upper())
        
        return jsonify({'team': team.upper(), 'injuries': impact})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/predict/<home_team>/<away_team>', methods=['GET'])
def predict_matchup(home_team, away_team):
    """Get prediction for a specific matchup."""
    try:
        home_odds = request.args.get('home_odds', type=int, default=-110)
        away_odds = request.args.get('away_odds', type=int, default=-110)
        
        pred = get_predictor()
        result = pred.predict_with_odds(
            home_team.upper(),
            away_team.upper(),
            home_odds,
            away_odds
        )
        
        if 'error' in result:
            return jsonify({'error': result['error']}), 400
        
        return jsonify({'prediction': result})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/sports', methods=['GET'])
def get_sports():
    """Get list of supported sports."""
    return jsonify({
        'sports': [
            {'key': 'nba', 'name': 'NBA Basketball', 'active': True},
            {'key': 'nfl', 'name': 'NFL Football', 'active': True},
            {'key': 'mlb', 'name': 'MLB Baseball', 'active': True},
            {'key': 'nhl', 'name': 'NHL Hockey', 'active': True}
        ]
    })


@app.route('/api/parlay-calculator', methods=['POST'])
def calculate_parlay():
    """Calculate parlay odds and payout for multiple bets."""
    try:
        data = request.get_json()
        legs = data.get('legs', [])
        stake = data.get('stake', 100)
        
        if not legs or len(legs) < 2:
            return jsonify({'error': 'Parlay requires at least 2 legs'}), 400
        
        decimal_odds = []
        combined_prob = 1.0
        
        for leg in legs:
            odds = leg.get('odds', -110)
            prob = leg.get('probability', 0.5)
            
            if odds > 0:
                dec = (odds / 100) + 1
            else:
                dec = (100 / abs(odds)) + 1
            
            decimal_odds.append(dec)
            combined_prob *= prob
        
        combined_decimal = 1.0
        for dec in decimal_odds:
            combined_decimal *= dec
        
        if combined_decimal >= 2:
            combined_american = round((combined_decimal - 1) * 100)
        else:
            combined_american = round(-100 / (combined_decimal - 1))
        
        payout = stake * combined_decimal
        profit = payout - stake
        parlay_ev = combined_prob * (combined_decimal - 1) - (1 - combined_prob)
        
        return jsonify({
            'legs': len(legs),
            'stake': stake,
            'combinedOdds': combined_american,
            'combinedDecimal': round(combined_decimal, 3),
            'payout': round(payout, 2),
            'profit': round(profit, 2),
            'combinedProbability': round(combined_prob, 4),
            'expectedValue': round(parlay_ev, 4),
            'isPositiveEV': parlay_ev > 0
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("=" * 60)
    print("Sports Betting Dashboard API")
    print("=" * 60)
    print()
    print("Dashboard: http://localhost:5000")
    print()
    print("API Endpoints:")
    print("  GET /api/live-predictions  - Today's predictions with live odds")
    print("  GET /api/edge-analysis     - Games with betting edge")
    print("  GET /api/live-odds         - Raw live odds data")
    print("  GET /api/best-odds         - Best odds across books")
    print("  GET /api/performance       - Historical performance")
    print()
    print("=" * 60)
    app.run(debug=True, port=5000)
