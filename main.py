"""
Sports Betting Predictor - Main CLI

Command-line interface for data collection, model training, and predictions.
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from data.nba_fetcher import NBAFetcher
from data.odds_fetcher import OddsFetcher
from data.database import Database
from features.team_features import TeamFeatureEngine
from models.trainer import ModelTrainer
from models.predictor import GamePredictor
from betting.evaluator import BettingEvaluator


def update_data(args):
    """Fetch and update NBA game data and odds."""
    print("\n" + "="*60)
    print("UPDATING DATA")
    print("="*60)
    
    db = Database()
    
    # Fetch NBA data
    print("\n[1/2] Fetching NBA game data...")
    nba = NBAFetcher()
    
    seasons = args.seasons.split(",") if args.seasons else ["2024-25"]
    games = nba.fetch_historical_data(seasons)
    
    print(f"Fetched {len(games)} game records")
    
    # Save to database
    db.save_games(games)
    print("Saved games to database")
    
    # Fetch odds (if API key configured)
    print("\n[2/2] Fetching current odds...")
    odds_fetcher = OddsFetcher()
    
    if odds_fetcher.api_key and "YOUR_" not in odds_fetcher.api_key:
        odds = odds_fetcher.get_odds("nba")
        if not odds.empty:
            db.save_odds(odds)
            print(f"Saved odds for {odds['game_id'].nunique()} games")
            print(f"Remaining API calls: {odds_fetcher.remaining_requests}")
        else:
            print("No upcoming games with odds found")
    else:
        print("Odds API key not configured - skipping odds fetch")
        print("Add your key to config/config.yaml to enable")
    
    db.close()
    print("\nData update complete!")


def train_model(args):
    """Train the prediction model."""
    print("\n" + "="*60)
    print("TRAINING MODEL")
    print("="*60)
    
    # Fetch data
    print("\n[1/4] Loading game data...")
    nba = NBAFetcher()
    seasons = args.seasons.split(",") if args.seasons else ["2022-23", "2023-24", "2024-25"]
    games = nba.fetch_historical_data(seasons)
    
    print(f"Loaded {len(games)} game records")
    
    # Generate features
    print("\n[2/4] Engineering features...")
    feature_engine = TeamFeatureEngine()
    X, y = feature_engine.generate_training_data(games)
    
    if len(X) == 0:
        print("ERROR: No training samples generated. Need more game data.")
        return
    
    print(f"Generated {len(X)} training samples with {len(feature_engine.get_feature_names())} features")
    
    # Train model
    print("\n[3/4] Training model...")
    trainer = ModelTrainer(model_type=args.model_type)
    
    X_train, X_test, y_train, y_test = trainer.prepare_data(X, y)
    trainer.train(X_train, y_train, calibrate=True)
    
    # Evaluate
    print("\n[4/4] Evaluating model...")
    metrics = trainer.evaluate(X_test, y_test)
    
    # Save model
    model_path = trainer.save()
    
    print(f"\nModel saved to: {model_path}")
    print("Training complete!")


def predict(args):
    """Generate predictions for upcoming games."""
    print("\n" + "="*60)
    print("GENERATING PREDICTIONS")
    print("="*60)
    
    # Load model
    predictor = GamePredictor()
    
    if predictor.model is None:
        print("ERROR: No trained model found. Run --train first.")
        return
    
    # Get today's games
    print("\nFetching today's games...")
    nba = NBAFetcher()
    
    try:
        todays_games = nba.get_todays_games()
        if todays_games.empty:
            print("No games scheduled for today.")
            return
        
        print(f"Found {len(todays_games)} games scheduled")
    except Exception as e:
        print(f"Error fetching games: {e}")
        print("\nUsing demo predictions...")
        
        # Demo mode with sample data
        demo_games = [
            {'home_team': 'Lakers', 'away_team': 'Warriors', 
             'home_prob': 0.55, 'home_odds': -130, 'away_odds': +110},
            {'home_team': 'Celtics', 'away_team': 'Heat',
             'home_prob': 0.62, 'home_odds': -175, 'away_odds': +145}
        ]
        
        evaluator = BettingEvaluator(
            min_ev_threshold=args.min_ev,
            bankroll=args.bankroll
        )
        
        report = evaluator.generate_daily_report(demo_games)
        print(report)
        return
    
    # For real implementation, would need to:
    # 1. Get recent stats for each team
    # 2. Generate features
    # 3. Run predictions
    # 4. Fetch current odds
    # 5. Evaluate betting value
    
    print("\nFull prediction pipeline requires:")
    print("  1. Historical team stats loaded in database")
    print("  2. Current odds from The Odds API")
    print("  3. Trained model")
    print("\nRun --update-data and --train first.")


def demo(args):
    """Run a demonstration of the system."""
    print("\n" + "="*60)
    print("SPORTS BETTING PREDICTOR - DEMO")
    print("="*60)
    
    # Demo feature engineering
    print("\n[Feature Engineering Demo]")
    feature_engine = TeamFeatureEngine()
    print(f"Features used: {len(feature_engine.get_feature_names())}")
    
    home_stats = {
        'PTS_L5': 115.2, 'PTS_L10': 112.8,
        'REB_L5': 44.0, 'REB_L10': 43.5,
        'AST_L5': 26.5, 'AST_L10': 25.8,
        'HOME_WIN_RATE': 0.65,
        'REST_DAYS': 2, 'STREAK': 3, 'IS_B2B': 0
    }
    
    away_stats = {
        'PTS_L5': 108.4, 'PTS_L10': 109.5,
        'REB_L5': 42.0, 'REB_L10': 41.8,
        'AST_L5': 24.0, 'AST_L10': 23.5,
        'AWAY_WIN_RATE': 0.45,
        'REST_DAYS': 1, 'STREAK': -2, 'IS_B2B': 1
    }
    
    import pandas as pd
    features = feature_engine.build_matchup_features(
        pd.Series(home_stats), pd.Series(away_stats)
    )
    
    print("\nSample matchup (Lakers vs Warriors):")
    print(f"  Points differential (L5): {features['PTS_DIFF_L5']:.1f}")
    print(f"  Rest advantage: {features['REST_ADVANTAGE']} days")
    print(f"  Streak differential: {features['STREAK_DIFF']}")
    
    # Demo betting evaluation
    print("\n[Betting Evaluation Demo]")
    evaluator = BettingEvaluator(bankroll=1000)
    
    game_eval = evaluator.evaluate_game(
        home_team="Lakers",
        away_team="Warriors", 
        home_prob=0.58,
        home_odds=-140,
        away_odds=+120
    )
    
    print(f"\nLakers vs Warriors:")
    print(f"  Model: Lakers 58% to win")
    print(f"  Odds: Lakers -140, Warriors +120")
    print(f"  Lakers EV: {game_eval['home_evaluation']['expected_value']:.1%}")
    print(f"  Warriors EV: {game_eval['away_evaluation']['expected_value']:.1%}")
    print(f"  Best bet: {game_eval['best_bet']['team']}")
    print(f"  Recommended size: ${game_eval['best_bet']['recommended_bet_size']:.2f}")
    
    # Demo report
    print("\n[Daily Report Demo]")
    sample_games = [
        {'home_team': 'Lakers', 'away_team': 'Warriors', 
         'home_prob': 0.58, 'home_odds': -140, 'away_odds': +120},
        {'home_team': 'Celtics', 'away_team': 'Heat',
         'home_prob': 0.65, 'home_odds': -180, 'away_odds': +150},
    ]
    
    print(evaluator.generate_daily_report(sample_games))


def main():
    parser = argparse.ArgumentParser(
        description="NBA Betting Prediction System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --demo                    Run demonstration
  python main.py --update-data             Fetch latest game data
  python main.py --train                   Train prediction model
  python main.py --predict                 Get today's predictions
  python main.py --train --seasons 2023-24,2024-25
        """
    )
    
    # Actions
    parser.add_argument('--demo', action='store_true',
                        help='Run system demonstration')
    parser.add_argument('--update-data', action='store_true',
                        help='Fetch and update game data and odds')
    parser.add_argument('--train', action='store_true',
                        help='Train the prediction model')
    parser.add_argument('--predict', action='store_true',
                        help='Generate predictions for upcoming games')
    
    # Options
    parser.add_argument('--seasons', type=str, default=None,
                        help='Comma-separated seasons (e.g., "2023-24,2024-25")')
    parser.add_argument('--model-type', type=str, default='xgboost',
                        choices=['xgboost', 'logistic'],
                        help='Model type to train')
    parser.add_argument('--min-ev', type=float, default=0.02,
                        help='Minimum EV threshold for bet recommendations')
    parser.add_argument('--bankroll', type=float, default=1000.0,
                        help='Bankroll for bet sizing calculations')
    
    args = parser.parse_args()
    
    # Run requested action
    if args.demo:
        demo(args)
    elif args.update_data:
        update_data(args)
    elif args.train:
        train_model(args)
    elif args.predict:
        predict(args)
    else:
        parser.print_help()
        print("\nRun 'python main.py --demo' to see a demonstration.")


if __name__ == "__main__":
    main()
