"""
Sports Betting Predictor - Main Application

A unified CLI for all betting prediction tools.
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))


def cmd_predict(args):
    """Run predictions for today's games."""
    from betting.daily_edge import run_edge_analysis
    
    print("\n" + "=" * 60)
    print("   SPORTS BETTING PREDICTOR")
    print("=" * 60 + "\n")
    
    edges = run_edge_analysis(
        auto_save=not args.no_save,
        stake=args.stake
    )
    return edges


def cmd_resolve(args):
    """Resolve pending bets with game scores."""
    from betting.bet_tracker import BetTracker
    
    tracker = BetTracker()
    pending = tracker.get_pending_bets()
    
    if pending.empty:
        print("No pending bets to resolve.")
        return
    
    print(f"\nPending bets: {len(pending)}")
    print("-" * 50)
    
    for _, bet in pending.iterrows():
        print(f"Bet #{bet['id']}: {bet['away_team']} @ {bet['home_team']} - {bet['selection']} ({bet['odds']:+d})")
    
    print("\nTo resolve, enter scores as: BET_ID HOME_SCORE AWAY_SCORE")
    print("Example: 1 115 108")
    print("Enter 'q' to quit\n")
    
    while True:
        try:
            inp = input("> ").strip()
            if inp.lower() == 'q':
                break
            
            parts = inp.split()
            if len(parts) != 3:
                print("Format: BET_ID HOME_SCORE AWAY_SCORE")
                continue
            
            bet_id, home_score, away_score = int(parts[0]), int(parts[1]), int(parts[2])
            result, pl = tracker.resolve_by_scores(bet_id, home_score, away_score)
            print(f"  -> {result.upper()} (${pl:+.2f})")
            
        except ValueError:
            print("Invalid input. Use numbers only.")
        except Exception as e:
            print(f"Error: {e}")


def cmd_report(args):
    """Show performance report."""
    from betting.bet_tracker import BetTracker
    
    tracker = BetTracker()
    print(tracker.generate_report())


def cmd_analyze(args):
    """Analyze model performance by bet type."""
    from betting.model_analyzer import ModelAnalyzer
    
    analyzer = ModelAnalyzer()
    print(analyzer.generate_analysis_report())


def cmd_train(args):
    """Train or retrain the prediction model."""
    from models.trainer import main as train_main
    
    print("\nStarting model training...")
    train_main()


def cmd_dashboard(args):
    """Start the web dashboard."""
    import subprocess
    
    print("\n" + "=" * 60)
    print("   STARTING WEB DASHBOARD")
    print("=" * 60)
    print("\nDashboard will be available at: http://localhost:5000")
    print("Press Ctrl+C to stop\n")
    
    subprocess.run([
        sys.executable,
        str(Path(__file__).parent / "dashboard" / "server.py")
    ])


def cmd_pending(args):
    """Show pending bets."""
    from betting.bet_tracker import BetTracker
    
    tracker = BetTracker()
    pending = tracker.get_pending_bets()
    
    if pending.empty:
        print("No pending bets.")
        return
    
    print(f"\n{'ID':<5} {'Game':<20} {'Bet':<8} {'Odds':<8} {'Stake':<8} {'EV':<8}")
    print("-" * 60)
    
    for _, bet in pending.iterrows():
        game = f"{bet['away_team']}@{bet['home_team']}"
        print(f"{bet['id']:<5} {game:<20} {bet['selection']:<8} {bet['odds']:+<8} ${bet['stake']:<7.0f} {bet['expected_value']*100:+.1f}%")


def cmd_history(args):
    """Show bet history."""
    from betting.bet_tracker import BetTracker
    
    tracker = BetTracker()
    history = tracker.get_bet_history(limit=args.limit)
    
    if history.empty:
        print("No bet history.")
        return
    
    print(f"\n{'ID':<5} {'Date':<12} {'Game':<15} {'Bet':<6} {'Result':<8} {'P/L':<10}")
    print("-" * 60)
    
    for _, bet in history.iterrows():
        game = f"{bet['away_team']}@{bet['home_team']}"
        result = bet['result'].upper() if bet['result'] else 'PENDING'
        pl = f"${bet['profit_loss']:+.2f}" if bet['profit_loss'] else '-'
        date = str(bet['game_date'])[:10] if bet['game_date'] else '-'
        print(f"{bet['id']:<5} {date:<12} {game:<15} {bet['selection']:<6} {result:<8} {pl:<10}")


def cmd_odds(args):
    """Fetch current live odds."""
    from data.live_odds import LiveOddsFetcher
    
    API_KEY = "bd6934ca89728830cd789ca6203dbe8b"
    fetcher = LiveOddsFetcher(api_key=API_KEY)
    
    sport = args.sport or 'nba'
    print(f"\nFetching {sport.upper()} odds...")
    
    best = fetcher.get_best_odds(sport)
    
    if best.empty:
        print("No odds available.")
        return
    
    print(f"\n{'Game':<40} {'Best Home':<12} {'Book':<15}")
    print("-" * 70)
    
    for _, row in best.iterrows():
        print(f"{row['game']:<40} {row['best_home_odds']:+<12} {row['best_home_book']:<15}")


def cmd_eval(args):
    """Evaluate a specific bet."""
    from models.advanced_predictor import AdvancedPredictor
    
    print("\n" + "=" * 50)
    print("   BET EVALUATOR")
    print("=" * 50)
    
    home = args.home.upper()
    away = args.away.upper()
    
    predictor = AdvancedPredictor()
    
    try:
        result = predictor.predict_with_odds(
            home, away,
            args.home_odds, args.away_odds
        )
        
        if 'error' in result:
            print(f"\nError: {result['error']}")
            return
        
        ba = result.get('betting_analysis', {})
        home_prob = result['home_win_probability']
        away_prob = result['away_win_probability']
        home_ev = ba.get('home_ev', 0)
        away_ev = ba.get('away_ev', 0)
        
        print(f"\n{away} @ {home}")
        print("-" * 40)
        print(f"\nModel Probabilities:")
        print(f"  {home}: {home_prob:.1%}")
        print(f"  {away}: {away_prob:.1%}")
        
        print(f"\nOdds Analysis:")
        print(f"  {home} @ {args.home_odds:+d} -> EV: {home_ev:+.1%}")
        print(f"  {away} @ {args.away_odds:+d} -> EV: {away_ev:+.1%}")
        
        # Recommendation
        best_ev = max(home_ev, away_ev)
        best_team = home if home_ev > away_ev else away
        best_odds = args.home_odds if home_ev > away_ev else args.away_odds
        
        print(f"\n{'='*40}")
        if best_ev > 0.05:
            print(f">>> STRONG BET: {best_team} ({best_odds:+d})")
            print(f"    Expected Value: +{best_ev:.1%}")
        elif best_ev > 0.02:
            print(f">>> SLIGHT EDGE: {best_team} ({best_odds:+d})")
            print(f"    Expected Value: +{best_ev:.1%}")
        elif best_ev > 0:
            print(f">>> MARGINAL: {best_team} has slight edge (+{best_ev:.1%})")
            print("    Consider passing or reducing stake")
        else:
            print(f">>> PASS: No positive EV bet available")
            print(f"    Best option has {best_ev:.1%} EV")
        print("=" * 40)
        
    except Exception as e:
        print(f"\nError: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Sports Betting Predictor - CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python app.py predict           # Find today's betting edges
  python app.py predict --no-save # Find edges without saving
  python app.py pending           # Show pending bets
  python app.py resolve           # Resolve bets with scores
  python app.py report            # Performance report
  python app.py analyze           # Analyze by bet type
  python app.py dashboard         # Start web dashboard
  python app.py odds --sport nfl  # Get NFL odds
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # predict command
    p_predict = subparsers.add_parser('predict', help='Find betting edges for today')
    p_predict.add_argument('--no-save', action='store_true', help='Do not save bets')
    p_predict.add_argument('--stake', type=float, default=50, help='Stake per bet')
    p_predict.set_defaults(func=cmd_predict)
    
    # resolve command
    p_resolve = subparsers.add_parser('resolve', help='Resolve pending bets')
    p_resolve.set_defaults(func=cmd_resolve)
    
    # report command
    p_report = subparsers.add_parser('report', help='Show performance report')
    p_report.set_defaults(func=cmd_report)
    
    # analyze command
    p_analyze = subparsers.add_parser('analyze', help='Analyze model performance')
    p_analyze.set_defaults(func=cmd_analyze)
    
    # train command
    p_train = subparsers.add_parser('train', help='Train prediction model')
    p_train.set_defaults(func=cmd_train)
    
    # dashboard command
    p_dash = subparsers.add_parser('dashboard', help='Start web dashboard')
    p_dash.set_defaults(func=cmd_dashboard)
    
    # pending command
    p_pending = subparsers.add_parser('pending', help='Show pending bets')
    p_pending.set_defaults(func=cmd_pending)
    
    # history command
    p_history = subparsers.add_parser('history', help='Show bet history')
    p_history.add_argument('--limit', type=int, default=20, help='Max bets to show')
    p_history.set_defaults(func=cmd_history)
    
    # odds command
    p_odds = subparsers.add_parser('odds', help='Get live odds')
    p_odds.add_argument('--sport', choices=['nba', 'nfl', 'mlb', 'nhl'], default='nba')
    p_odds.set_defaults(func=cmd_odds)
    
    # eval command
    p_eval = subparsers.add_parser('eval', help='Evaluate a specific bet')
    p_eval.add_argument('home', help='Home team (e.g., LAL)')
    p_eval.add_argument('away', help='Away team (e.g., GSW)')
    p_eval.add_argument('home_odds', type=int, help='Home team odds (e.g., -130)')
    p_eval.add_argument('away_odds', type=int, help='Away team odds (e.g., +110)')
    p_eval.set_defaults(func=cmd_eval)
    
    args = parser.parse_args()
    
    if args.command is None:
        # Default: show menu
        print("\n" + "=" * 50)
        print("   SPORTS BETTING PREDICTOR")
        print("=" * 50)
        print("""
Commands:
  predict    - Find today's betting edges (auto-saves)
  pending    - Show pending bets
  resolve    - Resolve bets with game scores
  report     - Show performance report
  analyze    - Analyze by bet type
  history    - Show bet history
  odds       - Get live odds
  train      - Train/retrain model
  dashboard  - Start web dashboard

Usage: python app.py <command> [options]
       python app.py --help
        """)
    else:
        args.func(args)


if __name__ == "__main__":
    main()
