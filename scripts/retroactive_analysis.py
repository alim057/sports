"""
Retroactive Analysis Pipeline

Fetches historical NBA data from the 2023-24 season, builds features,
runs prediction simulations, and generates a performance report.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.nba_fetcher import NBAFetcher
from features.team_features import TeamFeatureEngine
from models.backtester import Backtester
from betting.evaluator import BettingEvaluator


def fetch_historical_games(season: str = "2023-24") -> pd.DataFrame:
    """Fetch all games for a season and process into matchups."""
    print(f"\n[1/4] Fetching {season} season data...")
    fetcher = NBAFetcher()
    
    games = fetcher.get_all_games_for_season(season)
    print(f"  Found {len(games)} game records (includes both home/away entries)")
    
    return games


def build_matchup_dataset(games: pd.DataFrame) -> pd.DataFrame:
    """
    Transform raw game data into matchup rows with features.
    Each row represents one game with home/away team stats.
    """
    print("\n[2/4] Building matchup dataset with features...")
    
    # Group by GAME_ID to get matchups
    matchups = []
    game_ids = games['GAME_ID'].unique()
    
    for gid in game_ids:
        game_rows = games[games['GAME_ID'] == gid]
        
        if len(game_rows) != 2:
            continue  # Skip incomplete records
        
        # Identify home and away
        home_row = game_rows[game_rows['MATCHUP'].str.contains('vs.')].iloc[0] if len(game_rows[game_rows['MATCHUP'].str.contains('vs.')]) > 0 else None
        away_row = game_rows[game_rows['MATCHUP'].str.contains('@')].iloc[0] if len(game_rows[game_rows['MATCHUP'].str.contains('@')]) > 0 else None
        
        if home_row is None or away_row is None:
            continue
        
        matchups.append({
            'game_id': gid,
            'game_date': home_row['GAME_DATE'],
            'home_team': home_row['TEAM_ABBREVIATION'],
            'away_team': away_row['TEAM_ABBREVIATION'],
            'home_pts': home_row['PTS'],
            'away_pts': away_row['PTS'],
            'home_won': home_row['WL'] == 'W',
            'actual_winner': home_row['TEAM_ABBREVIATION'] if home_row['WL'] == 'W' else away_row['TEAM_ABBREVIATION'],
            # Basic features
            'home_fg_pct': home_row.get('FG_PCT', 0.45),
            'away_fg_pct': away_row.get('FG_PCT', 0.45),
            'home_reb': home_row.get('REB', 40),
            'away_reb': away_row.get('REB', 40),
            'home_ast': home_row.get('AST', 20),
            'away_ast': away_row.get('AST', 20),
        })
    
    df = pd.DataFrame(matchups)
    df = df.sort_values('game_date').reset_index(drop=True)
    print(f"  Created {len(df)} matchup records")
    
    return df


def add_rolling_features(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """
    Add rolling stats for each team to use as prediction features.
    This simulates what we'd know BEFORE each game.
    """
    print(f"\n  Adding rolling {window}-game features...")
    
    # For each team, calculate their rolling stats before each game
    teams = set(df['home_team'].unique()) | set(df['away_team'].unique())
    
    team_games = {t: [] for t in teams}
    
    # First pass: collect game sequence for each team
    for idx, row in df.iterrows():
        home = row['home_team']
        away = row['away_team']
        
        team_games[home].append({
            'idx': idx,
            'pts': row['home_pts'],
            'won': row['home_won']
        })
        team_games[away].append({
            'idx': idx,
            'pts': row['away_pts'],
            'won': not row['home_won']
        })
    
    # Second pass: calculate rolling features for each row
    home_pts_l5 = []
    away_pts_l5 = []
    home_win_l5 = []
    away_win_l5 = []
    
    for idx, row in df.iterrows():
        home = row['home_team']
        away = row['away_team']
        
        # Get prior games for each team
        home_prior = [g for g in team_games[home] if g['idx'] < idx][-window:]
        away_prior = [g for g in team_games[away] if g['idx'] < idx][-window:]
        
        home_pts_l5.append(np.mean([g['pts'] for g in home_prior]) if home_prior else 105)
        away_pts_l5.append(np.mean([g['pts'] for g in away_prior]) if away_prior else 105)
        home_win_l5.append(np.mean([g['won'] for g in home_prior]) if home_prior else 0.5)
        away_win_l5.append(np.mean([g['won'] for g in away_prior]) if away_prior else 0.5)
    
    df['home_pts_l5'] = home_pts_l5
    df['away_pts_l5'] = away_pts_l5
    df['home_win_l5'] = home_win_l5
    df['away_win_l5'] = away_win_l5
    
    # Feature diffs
    df['pts_diff_l5'] = df['home_pts_l5'] - df['away_pts_l5']
    df['win_rate_diff'] = df['home_win_l5'] - df['away_win_l5']
    
    return df


def simulate_predictions(df: pd.DataFrame) -> list:
    """
    Use a simple heuristic model (similar to our current one) to generate
    predictions for each historical game.
    """
    print("\n[3/4] Simulating predictions on historical data...")
    
    predictions = []
    
    for idx, row in df.iterrows():
        if idx < 50:  # Skip first games where rolling stats aren't stable
            continue
        
        # Simple heuristic probability (same as AdvancedPredictor fallback)
        home_win_prob = 0.5 + (
            row['pts_diff_l5'] * 0.01 +
            row['win_rate_diff'] * 0.3 +
            0.03  # Home court advantage
        )
        home_win_prob = max(0.35, min(0.65, home_win_prob))
        
        # Determine predicted winner
        predicted_winner = row['home_team'] if home_win_prob > 0.5 else row['away_team']
        
        # Simulate odds (in reality we'd use historical odds)
        # For favorite (>50%), assume -130 to -150
        # For underdog, assume +110 to +150
        if home_win_prob > 0.5:
            odds = int(-100 * home_win_prob / (1 - home_win_prob))
            odds = max(-200, min(-110, odds))
        else:
            away_prob = 1 - home_win_prob
            odds = int(100 * (1 - away_prob) / away_prob)
            odds = max(100, min(200, odds))
        
        predictions.append({
            'matchup': f"{row['away_team']} @ {row['home_team']}",
            'game_date': row['game_date'],
            'predicted_winner': predicted_winner,
            'actual_winner': row['actual_winner'],
            'probability': home_win_prob if predicted_winner == row['home_team'] else 1 - home_win_prob,
            'odds': odds
        })
    
    print(f"  Generated {len(predictions)} predictions")
    return predictions


def run_backtest(predictions: list) -> dict:
    """Run the backtester on our predictions."""
    print("\n[4/4] Running backtest simulation...")
    
    backtester = Backtester(initial_bankroll=1000)
    results = backtester.run_backtest(
        predictions,
        kelly_fraction=0.25,
        min_ev=0.03  # 3% minimum EV threshold
    )
    
    return results, backtester


def main():
    print("=" * 70)
    print("RETROACTIVE ANALYSIS - 2023-24 NBA Season")
    print("=" * 70)
    
    # Step 1: Fetch data
    games = fetch_historical_games("2023-24")
    
    # Step 2: Build matchup dataset with features
    matchups = build_matchup_dataset(games)
    matchups = add_rolling_features(matchups)
    
    # Step 3: Simulate predictions
    predictions = simulate_predictions(matchups)
    
    # Step 4: Run backtest
    results, backtester = run_backtest(predictions)
    
    # Print report
    print("\n" + backtester.generate_report(results))
    
    # Analyze by confidence
    print("\nPerformance by Confidence Level:")
    conf_analysis = backtester.analyze_by_confidence(results)
    print(conf_analysis.to_string())
    
    # Save results
    output_path = Path(__file__).parent.parent / "data" / "backtest_results_2023-24.csv"
    if results['results']:
        pd.DataFrame(results['results']).to_csv(output_path, index=False)
        print(f"\nResults saved to: {output_path}")
    
    return results


if __name__ == "__main__":
    main()
