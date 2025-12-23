
import pandas as pd
from pathlib import Path

def test_performance_logic():
    print("Testing Performance Logic...")
    csv_path = Path("./data")
    csv_files = list(csv_path.glob("best_bets_*.csv"))
    print(f"Found {len(csv_files)} CSV files: {[f.name for f in csv_files]}")
    
    if not csv_files:
        print("No CSV files found.")
        return

    all_bets = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)
    print(f"Total rows: {len(all_bets)}")
    
    # 1. Check Performance Summary Logic (from edge_analyzer.py/server.py)
    resolved = all_bets[all_bets['result'].isin(['WIN', 'LOSS'])]
    print(f"Resolved rows: {len(resolved)}")
    
    wins = resolved[resolved['result'] == 'WIN']
    losses = resolved[resolved['result'] == 'LOSS']
    print(f"Wins: {len(wins)}")
    print(f"Losses: {len(losses)}")
    
    if len(wins) > 0:
        print("Sample Win Row:")
        print(wins.iloc[0])
    
    # 2. Check Recent Bets Logic (from server.py)
    # Sort by date descending and take recent 20
    recent_bets = all_bets.sort_values('date', ascending=False, kind='mergesort').head(20) # Python sort is stable, pandas might not be default
    # server.py uses default sort (quicksort usually)
    
    print("\nRecent Bets Top 5:")
    print(recent_bets[['game', 'result']].head(5))
    
    # Check if the WIN is in recent_bets
    win_in_recent = recent_bets[recent_bets['result'] == 'WIN']
    print(f"\nWins in Top 20 Recent: {len(win_in_recent)}")

if __name__ == "__main__":
    test_performance_logic()
