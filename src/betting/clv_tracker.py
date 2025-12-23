
import sys
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.data.database import Database, Prediction, Odds, Game
from src.data.odds_fetcher import OddsFetcher

class CLVTracker:
    """
    Tracks Closing Line Value (CLV) for predictions.
    
    CLV = (Decimal Bet Odds / Decimal Closing Odds) - 1
    
    This tracker:
    1. Identifies predictions that haven't been CLV-checked.
    2. Finds the last available odds before game start (Closing Line).
    3. Calculates and stores CLV.
    """
    
    def __init__(self, db_path: str = "./data/betting.db"):
        self.db = Database(db_path)
        self.session = self.db.session
        
    def track_all(self):
        """Run CLV tracking for all untracked, past predictions."""
        print("Starting CLV tracking...")
        
        # Get untracked predictions for games that have started
        now = datetime.now()
        untacked_preds = (
            self.session.query(Prediction)
            .filter(Prediction.game_date < now)
            .filter(Prediction.closing_odds == None)
            .all()
        )
        
        if not untacked_preds:
            print("No new predictions to track.")
            return

        print(f"Found {len(untacked_preds)} predictions to process.")
        
        updated_count = 0
        for pred in untacked_preds:
            if self._process_prediction(pred):
                updated_count += 1
                
        print(f"Update complete. Added CLV for {updated_count} predictions.")
        
    def _process_prediction(self, pred: Prediction) -> bool:
        """Process a single prediction to find its CLV."""
        game_id = pred.game_id
        
        # Find closing odds in our DB
        # We want the latest odds fetch that is <= commence_time
        # But commence_time might be slightly different across bookmakers or updates
        # So we look for odds fetched roughly before the recorded game_date
        
        # Note: Prediction.game_date should align with commence_time
        game_start = pred.game_date
        
        if not game_start:
            # Try to lookup game time if missing
            game = self.session.query(Game).filter_by(game_id=game_id).first()
            if game:
                game_start = game.game_date
            else:
                print(f"Skipping {game_id}: No start time found.")
                return False

        # Tolerance: Allow odds up to 10 mins after 'scheduled' start if nothing else, 
        # but strictly preferring pre-game.
        cutoff = game_start
        
        # Query Odds table
        # Get the latest odds record before cutoff
        # We need to match the team we bet on.
        # Prediction stores "recommended_bet" which might be "LAL -5.5" or just Team Name for Moneyline.
        # For simplicity, if recommended_bet is missing or complex, we'll try to match predicted_winner for Moneyline.
        
        target_team = pred.predicted_winner
        if not target_team:
            return False
            
        # Try to find moneyline odds for this team
        latest_odd = (
            self.session.query(Odds)
            .filter(Odds.game_id == game_id)
            .filter(Odds.market == 'h2h')
            .filter(Odds.team == target_team)
            .filter(Odds.fetched_at <= cutoff)
            .order_by(Odds.fetched_at.desc())
            .first()
        )
        
        if not latest_odd:
            # Fallback: maybe we only have odds from slightly after start?
            # Or maybe no odds at all.
            # print(f"No odds found for {game_id} {target_team}")
            return False
            
        closing_price = latest_odd.price
        
        # We need the 'bet_odds' to calculate CLV.
        # The Prediction table has 'expected_value' but doesn't explicitly store 'bet_odds' in the current schema
        # (It stores probabilities). 
        # Wait, the Handoff said "expected_value" is stored.
        # But we really need the Odds WE bet at.
        # Currently the `save_prediction` uses `home_win_prob`.
        # The `EdgeAnalyzer` tracks `stake_odds` in a separate JSON file, but not in DB `Prediction`.
        # The `Prediction` table has `recommended_bet` which is a string.
        
        # If we can't determine the odds we "bet" at from the DB, we can't calc strict CLV.
        # However, for now, let's assume we bet at the Best Available Odds at the time of prediction.
        # If we don't have that recorded, we might strictly calculate "Closing Probability" vs "Model Probability".
        # But CLV usually implies Odds.
        
        # Let's check if we can infer "Odds at Prediction Time".
        # If we stored `created_at` on Prediction, we can find odds from that time.
        
        prediction_time = pred.created_at
        opening_odd = (
            self.session.query(Odds)
            .filter(Odds.game_id == game_id)
            .filter(Odds.market == 'h2h')
            .filter(Odds.team == target_team)
            .filter(Odds.fetched_at <= prediction_time + timedelta(minutes=30)) # close to creation
            .filter(Odds.fetched_at >= prediction_time - timedelta(minutes=60))
            .order_by(Odds.fetched_at.asc()) # first available around prediction time
            .first()
        )
        
        bet_odds_american = opening_odd.price if opening_odd else None
        
        if bet_odds_american is None:
            # Can't calculate CLV without opening odds
            return False
            
        # Calculate CLV
        # CLV = (Decimal Bet Odds / Decimal Closing Odds) - 1
        
        dec_bet = self._american_to_decimal(bet_odds_american)
        dec_close = self._american_to_decimal(closing_price)
        
        if dec_close == 0:
            return False
            
        clv = (dec_bet / dec_close) - 1
        
        # update record
        pred.closing_odds = closing_price
        pred.clv = clv
        self.session.commit()
        
        print(f"Game {game_id}: Bet {bet_odds_american} ({dec_bet:.2f}) -> Close {closing_price} ({dec_close:.2f}) | CLV: {clv:+.2%}")
        return True

    @staticmethod
    def _american_to_decimal(american: int) -> float:
        if american > 0:
            return 1 + (american / 100)
        else:
            return 1 + (100 / abs(american))
            
if __name__ == "__main__":
    tracker = CLVTracker()
    tracker.track_all()
