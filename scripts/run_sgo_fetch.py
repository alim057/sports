"""
SGO Fetch Runner

Script to fetch odds from Sports Game Odds API for GitHub Actions.
Reports status for each sport and saves data to database.
"""

import sys
import os
import argparse
from datetime import datetime, timezone
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent  # scripts -> project root
sys.path.insert(0, str(project_root))


def get_active_sports() -> list:
    """
    Determine which sports are currently in season.
    
    Returns:
        List of sport keys that are in season
    """
    now = datetime.now()
    month = now.month
    
    active = []
    
    # NBA: October - June
    if month >= 10 or month <= 6:
        active.append("nba")
    
    # NFL: September - February
    if month >= 9 or month <= 2:
        active.append("nfl")
    
    # NCAAF: August - January
    if month >= 8 or month <= 1:
        active.append("ncaaf")
    
    # MLB: April - October
    if 4 <= month <= 10:
        active.append("mlb")
    
    # NCAA Basketball: November - April
    if month >= 11 or month <= 4:
        active.append("ncaab")
    
    return active


def main():
    parser = argparse.ArgumentParser(description="SGO Fetch Runner")
    parser.add_argument("--sports", nargs="+", help="Sports to fetch (default: all active)")
    parser.add_argument("--force-all", action="store_true", help="Fetch all sports regardless of season")
    parser.add_argument("--save-db", action="store_true", help="Save results to database")
    parser.add_argument("--props", action="store_true", help="Also fetch player props")
    args = parser.parse_args()
    
    # Direct imports to avoid package __init__ that pulls in nba_api
    import importlib.util
    
    # Load sgo_fetcher directly
    sgo_spec = importlib.util.spec_from_file_location("sgo_fetcher", project_root / "src" / "data" / "sgo_fetcher.py")
    sgo_module = importlib.util.module_from_spec(sgo_spec)
    sgo_spec.loader.exec_module(sgo_module)
    SGOFetcher = sgo_module.SGOFetcher
    
    # Load database directly  
    db_spec = importlib.util.spec_from_file_location("database", project_root / "src" / "data" / "database.py")
    db_module = importlib.util.module_from_spec(db_spec)
    db_spec.loader.exec_module(db_module)
    Database = db_module.Database
    
    print("=" * 60)
    print(f"SGO Odds Fetch - {datetime.now(timezone.utc).isoformat()}")
    print("=" * 60)
    
    # Determine sports to fetch
    if args.sports:
        sports = args.sports
    elif args.force_all:
        sports = ["nba", "nfl", "ncaaf", "mlb"]
    else:
        sports = get_active_sports()
    
    print(f"\nActive sports to fetch: {', '.join(s.upper() for s in sports)}")
    
    # Initialize fetcher
    fetcher = SGOFetcher()
    
    # Initialize database if saving
    db = None
    if args.save_db:
        db = Database()
        print(f"Database: Connected")
    
    # Track results for each sport
    results = {}
    errors = {}
    
    for sport in sports:
        print(f"\n{'='*40}")
        print(f"Fetching {sport.upper()}...")
        print(f"{'='*40}")
        
        try:
            # Fetch games with odds
            games_df = fetcher.get_upcoming_games(sport)
            
            if games_df.empty:
                print(f"  [!] {sport.upper()}: No games with odds available")
                results[sport] = {"games": 0, "odds": 0, "props": 0, "status": "NO_DATA"}
            else:
                print(f"  [OK] {sport.upper()}: {len(games_df)} games found")
                
                # Show sample games
                for _, game in games_df.head(3).iterrows():
                    print(f"     - {game['away_team']} @ {game['home_team']}")
                    print(f"       ML: {game['moneyline_away']} / {game['moneyline_home']}")
                
                results[sport] = {
                    "games": len(games_df), 
                    "odds": 0, 
                    "props": 0, 
                    "status": "SUCCESS"
                }
                
                # Save to database
                if db:
                    db.save_sgo_events(games_df)
                    print(f"  [DB] Saved {len(games_df)} events to database")
                
                # Fetch odds details
                odds_df = fetcher.get_live_odds(sport)
                if not odds_df.empty:
                    results[sport]["odds"] = len(odds_df)
                    print(f"  [ODDS] {len(odds_df)} odds entries from bookmakers")
                    
                    if db:
                        db.save_sgo_odds(odds_df)
                
                # Fetch player props if requested
                if args.props:
                    props_df = fetcher.get_player_props(sport)
                    if not props_df.empty:
                        results[sport]["props"] = len(props_df)
                        print(f"  [PROPS] {len(props_df)} player props")
                        
                        if db:
                            db.save_sgo_player_props(props_df)
                    else:
                        print(f"  [!] No player props available for {sport.upper()}")
        
        except Exception as e:
            print(f"  [ERROR] {sport.upper()}: {str(e)}")
            errors[sport] = str(e)
            results[sport] = {"games": 0, "odds": 0, "props": 0, "status": "ERROR"}
    
    # Summary
    print("\n" + "=" * 60)
    print("FETCH SUMMARY")
    print("=" * 60)
    
    total_games = 0
    total_odds = 0
    total_props = 0
    
    for sport, data in results.items():
        status_icon = "[OK]" if data["status"] == "SUCCESS" else "[!]" if data["status"] == "NO_DATA" else "[X]"
        print(f"{status_icon} {sport.upper():8} | Games: {data['games']:3} | Odds: {data['odds']:5} | Props: {data['props']:4}")
        total_games += data["games"]
        total_odds += data["odds"]
        total_props += data["props"]
    
    print("-" * 60)
    print(f"{'TOTAL':12} | Games: {total_games:3} | Odds: {total_odds:5} | Props: {total_props:4}")
    
    if errors:
        print("\n[!] ERRORS:")
        for sport, error in errors.items():
            print(f"   {sport.upper()}: {error}")
    
    print(f"\nAPI requests made: {fetcher.get_request_count()}")
    print(f"Completed at: {datetime.now(timezone.utc).isoformat()}")
    
    # Close database
    if db:
        db.close()
    
    # Exit with error code if any failures
    if errors:
        sys.exit(1)
    
    print("\n[SUCCESS] Fetch completed successfully!")


if __name__ == "__main__":
    main()
