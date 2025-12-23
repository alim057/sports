
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    import nfl_data_py as nfl
    print("[OK] nfl_data_py imported successfully")
except ImportError as e:
    print(f"[FAIL] Failed to import nfl_data_py: {e}")
    # Don't exit, let's see if the fetcher handles it gracefully or we can mock it
    # sys.exit(1) 

from data.sport_fetcher import NFLFetcher

def test_nfl_fetcher():
    print("\nInitializing NFLFetcher...")
    try:
        fetcher = NFLFetcher()
        print("[OK] NFLFetcher initialized")
    except Exception as e:
        print(f"[FAIL] Failed to initialize NFLFetcher: {e}")
        return

    print("\nFetching Teams...")
    try:
        teams = fetcher.get_all_teams()
        if not teams.empty:
            print(f"[OK] Found {len(teams)} teams")
            print(teams[['team_abbr', 'team_name']].head())
        else:
            print("[WARN] No teams found")
    except Exception as e:
        print(f"[FAIL] Failed to fetch teams: {e}")

    print("\nFetching Schedule (2023)...")
    try:
        schedule = fetcher._fetch_season_games('2023')
        if not schedule.empty:
            print(f"[OK] Found {len(schedule)} games for 2023")
            columns = [c for c in ['game_id', 'home_team', 'away_team', 'home_score', 'away_score'] if c in schedule.columns]
            print(schedule[columns].head())
        else:
            print("[WARN] No schedule found for 2023")
    except Exception as e:
        print(f"[FAIL] Failed to fetch schedule: {e}")

if __name__ == "__main__":
    test_nfl_fetcher()
