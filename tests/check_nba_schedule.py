
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.nba_fetcher import NBAFetcher

def check_schedule():
    fetcher = NBAFetcher()
    print("Fetching today's games...")
    try:
        df = fetcher.get_todays_games()
        if df.empty:
            print("No games today.")
        else:
            print("Columns:", df.columns.tolist())
            print("First row:", df.iloc[0].to_dict())
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_schedule()
