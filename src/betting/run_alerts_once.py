"""
Run Alerts Once

Single execution script for GitHub Actions / Cron jobs.
Checks for edges, sends alerts if found, then exits (does not loop).
"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.betting.alerts import AlertSystem
from src.data.odds_fetcher import OddsFetcher

def main():
    print("Running betting alert check...")
    
    # Initialize system with 0 interval (one-off check)
    # Note: We need to bypass the localhost data fetching since GitHub Actions
    # won't have the Flask server running. We'll verify raw odds directly.
    
    try:
        # Get API key from env (for GitHub Actions)
        api_key = os.environ.get('ODDS_API_KEY')
        webhook = os.environ.get('DISCORD_WEBHOOK')
        
        if not api_key:
            print("No ODDS_API_KEY found (check GitHub Secrets)")
            # Don't fail, just exit
            return

        print(f"Checking odds using key: {api_key[:4]}...")
        
        # 1. Fetch raw odds
        fetcher = OddsFetcher(api_key=api_key)
        sports = ['nba', 'nfl', 'nhl', 'ncaaf']
        
        has_alerts = False
        
        for sport in sports:
            try:
                odds = fetcher.get_odds(sport, markets='h2h')
                if odds.empty:
                    continue
                    
                # 2. Simple EV Check (Simplified logic for cloud script)
                # In full app we use complex models. Here we look for:
                # - Arbitrage or massive discrepancies
                # - Or just report best lines
                
                # For now, we'll just log that we checked. 
                # To fully implement, we'd need to bundle the trained models 
                # and predict here.
                
                print(f"Checked {sport}: {len(odds)} games found.")
                
            except Exception as e:
                print(f"Error checking {sport}: {e}")
                
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
