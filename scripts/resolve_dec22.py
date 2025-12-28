"""
Manual Bet Resolution for December 22, 2025

Uses hardcoded scores from web search results since nba_api doesn't have future dates.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.betting.bet_tracker import BetTracker


# NBA Scores for December 22, 2025 (from web search)
SCORES_DEC_22 = {
    # Format: (home_team, away_team): (home_score, away_score)
    ('CLE', 'CHA'): (139, 132),  # Cavaliers beat Hornets
    ('CHA', 'CLE'): (132, 139),  # Reverse lookup
    ('BOS', 'IND'): (103, 95),   # Celtics beat Pacers
    ('IND', 'BOS'): (95, 103),
    ('NOP', 'DAL'): (119, 113),  # Pelicans beat Mavericks
    ('DAL', 'NOP'): (113, 119),
    ('DEN', 'UTA'): (135, 112),  # Nuggets beat Jazz
    ('UTA', 'DEN'): (112, 135),
    ('OKC', 'MEM'): (119, 103),  # Thunder beat Grizzlies
    ('MEM', 'OKC'): (103, 119),
    ('GSW', 'ORL'): (120, 115),  # Warriors beat Magic
    ('ORL', 'GSW'): (115, 120),
    ('POR', 'DET'): (102, 110),  # Pistons beat Blazers
    ('DET', 'POR'): (110, 102),
}


def resolve_dec22_bets(dry_run=True):
    """Resolve all pending bets from December 22, 2025."""
    
    tracker = BetTracker()
    pending = tracker.get_pending_bets()
    
    # Filter to Dec 22 bets
    dec22 = pending[pending['game_date'] == '2025-12-22']
    
    print(f"{'='*60}")
    print(f"RESOLVING {len(dec22)} BETS FROM DECEMBER 22, 2025")
    print(f"{'='*60}")
    
    if dry_run:
        print("*** DRY RUN MODE ***\n")
    
    wins = 0
    losses = 0
    pushes = 0
    total_profit = 0
    skipped = 0
    
    for _, bet in dec22.iterrows():
        home = bet['home_team']
        away = bet['away_team']
        selection = bet['selection']
        bet_type = bet['bet_type']
        
        # Look up score
        key = (home, away)
        if key not in SCORES_DEC_22:
            print(f"[?] No score for {away} @ {home} - skipping")
            skipped += 1
            continue
        
        home_score, away_score = SCORES_DEC_22[key]
        
        # Determine result for moneyline
        if bet_type == 'moneyline':
            if selection == home:
                result = 'win' if home_score > away_score else 'loss'
            else:
                result = 'win' if away_score > home_score else 'loss'
        else:
            print(f"[?] Unknown bet type: {bet_type}")
            skipped += 1
            continue
        
        icon = '[W]' if result == 'win' else '[L]'
        print(f"{icon} {away} {away_score} @ {home} {home_score} | Pick: {selection} = {result.upper()}")
        
        if not dry_run:
            actual_result, profit = tracker.resolve_by_scores(bet['id'], home_score, away_score)
            total_profit += profit
            
        if result == 'win':
            wins += 1
        elif result == 'loss':
            losses += 1
        else:
            pushes += 1
    
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Wins:    {wins}")
    print(f"Losses:  {losses}")
    print(f"Pushes:  {pushes}")
    print(f"Skipped: {skipped}")
    if not dry_run:
        print(f"Profit:  ${total_profit:+.2f}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true", help="Actually resolve bets")
    args = parser.parse_args()
    
    resolve_dec22_bets(dry_run=not args.execute)
