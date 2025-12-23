"""
Script to automatically resolve pending bets using fetched game scores.
Iterates through pending bets, fetches scores for the relevant dates/sports, 
and updates the database.
"""

import sys
import os
from datetime import datetime
import pandas as pd
import time

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from betting.bet_tracker import BetTracker
from data.sport_fetcher import SportFetcherFactory

def main():
    print(f"--- Automated Bet Settlement ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')}) ---")
    
    tracker = BetTracker()
    pending = tracker.get_pending_bets()
    
    if pending.empty:
        print("No pending bets to resolve.")
        return

    print(f"Found {len(pending)} pending bets.")
    
    # Group by sport and date to minimize API calls
    grouped = pending.groupby(['sport', 'game_date'])
    
    for (sport_raw, game_date_str), group in grouped:
        sport = sport_raw.lower() if sport_raw else 'nba' # Default to nba if missing
        
        try:
            # Parse date
            # game_date in db is usually YYYY-MM-DD or full timestamp
            # We assume YYYY-MM-DD based on schema
            game_date = pd.to_datetime(game_date_str)
            
            print(f"\nProcessing {sport.upper()} for {game_date.strftime('%Y-%m-%d')} ({len(group)} bets)...")
            
            # Fetch scores
            try:
                # Handle 'nba' separately or via factory if updated
                # Note: nba_fetcher is separate in current codebase structure 
                # but SportFetcherFactory handles 'nba' string by importing it.
                # However, SportFetcherFactory.get_fetcher('nba') returns NBAFetcher
                # which we just updated with get_scores.
                
                fetcher = SportFetcherFactory.get_fetcher(sport)
                scores_df = fetcher.get_scores(date=game_date)
                
                if scores_df.empty:
                    print(f"  No scores found for {sport.upper()} on {game_date.date()}")
                    continue
                    
            except Exception as e:
                print(f"  Error fetching scores for {sport}: {e}")
                continue
            
            # Process each bet in this group
            for _, bet in group.iterrows():
                bet_id = bet['id']
                game_id = bet['game_id']
                home_team = bet['home_team']
                away_team = bet['away_team']
                
                # Find matching game in scores
                match = None
                
                # 1. Try exact game_id match
                if game_id:
                    match = scores_df[scores_df['game_id'] == game_id]
                
                # 2. If no ID match, try team match (fuzzy or exact abbr)
                if match is None or match.empty:
                    # NBA: LAL, GSW. Scores have LAL, GSW.
                    # NFL: ARI, BAL.
                    match = scores_df[
                        (scores_df['home_team'] == home_team) & 
                        (scores_df['away_team'] == away_team)
                    ]
                
                if not match.empty:
                    game_result = match.iloc[0]
                    status = game_result['status']
                    
                    if status == 'Final':
                        home_score = int(game_result['home_score'])
                        away_score = int(game_result['away_score'])
                        
                        print(f"  Resolving Bet #{bet_id} ({bet['bet_type']} {bet['selection']}): {home_team} {home_score} - {away_team} {away_score}")
                        
                        result, pl = tracker.resolve_by_scores(
                            bet_id, 
                            home_score, 
                            away_score
                        )
                    else:
                        print(f"  Game {home_team} vs {away_team} is {status}. Skipping.")
                else:
                    print(f"  Could not find score for {home_team} vs {away_team} (ID: {game_id})")
                    
        except Exception as e:
            print(f"Error processing group {sport} {game_date_str}: {e}")
            continue

    print("\n--- Settlement Complete ---")
    print(tracker.generate_report())

if __name__ == "__main__":
    main()
