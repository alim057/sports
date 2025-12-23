"""
Export Today's Best Bets to CSV (Google Sheets compatible)

Fetches current edges and exports them to a CSV file.
Includes result tracking for completed games.
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path


def fetch_all_edges(base_url='http://localhost:5000'):
    """Fetch edges from all sports."""
    sports = ['nba', 'nfl', 'nhl', 'ncaaf']
    all_edges = []
    
    for sport in sports:
        try:
            response = requests.get(f'{base_url}/api/edge-analysis?sport={sport}', timeout=10)
            if response.status_code == 200:
                data = response.json()
                for edge in data.get('edges', []):
                    edge['sport'] = sport.upper()
                    edge['date'] = datetime.now().strftime('%Y-%m-%d')
                    all_edges.append(edge)
        except Exception as e:
            print(f"Error fetching {sport}: {e}")
    
    return all_edges


def fetch_game_results(sport, date_str):
    """Fetch game results from ESPN API."""
    sport_map = {
        'NBA': 'basketball/nba',
        'NFL': 'football/nfl', 
        'NHL': 'hockey/nhl',
        'NCAAF': 'football/college-football'
    }
    
    espn_sport = sport_map.get(sport.upper())
    if not espn_sport:
        return {}
    
    # Format date as YYYYMMDD
    date_fmt = date_str.replace('-', '')
    url = f'https://site.api.espn.com/apis/site/v2/sports/{espn_sport}/scoreboard?dates={date_fmt}'
    
    try:
        response = requests.get(url, timeout=10)
        if response.status_code != 200:
            return {}
        
        data = response.json()
        results = {}
        
        for event in data.get('events', []):
            comps = event.get('competitions', [{}])[0]
            competitors = comps.get('competitors', [])
            
            if len(competitors) == 2:
                home = next((c for c in competitors if c.get('homeAway') == 'home'), {})
                away = next((c for c in competitors if c.get('homeAway') == 'away'), {})
                
                home_abbr = home.get('team', {}).get('abbreviation', '')
                away_abbr = away.get('team', {}).get('abbreviation', '')
                home_score = int(home.get('score', 0) or 0)
                away_score = int(away.get('score', 0) or 0)
                
                game_key = f"{away_abbr} @ {home_abbr}"
                winner = home_abbr if home_score > away_score else away_abbr
                
                results[game_key] = {
                    'winner': winner,
                    'home_score': home_score,
                    'away_score': away_score,
                    'final_score': f"{away_score}-{home_score}"
                }
        
        return results
    except Exception as e:
        print(f"Error fetching {sport} results: {e}")
        return {}


def calculate_payout(odds, stake=1.0):
    """Calculate payout from American odds."""
    if odds > 0:
        return stake * (odds / 100)
    else:
        return stake * (100 / abs(odds))


def export_to_csv(edges, include_results=True, output_dir='data'):
    """Export edges to CSV file with results."""
    if not edges:
        print("No edges to export")
        return None
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Create DataFrame
    df = pd.DataFrame(edges)
    
    # Format columns
    if 'modelProbability' in df.columns:
        df['win_probability'] = (df['modelProbability'] * 100).round(1).astype(str) + '%'
    if 'ev' in df.columns:
        df['expected_value'] = (df['ev'] * 100).round(1).astype(str) + '%'
    
    # Add result columns
    df['result'] = 'PENDING'
    df['final_score'] = ''
    df['payout'] = ''
    
    if include_results:
        # Get results for each sport
        for sport in df['sport'].unique():
            sport_df = df[df['sport'] == sport]
            if not sport_df.empty:
                date = sport_df['date'].iloc[0]
                results = fetch_game_results(sport, date)
                
                for idx, row in sport_df.iterrows():
                    game = row['game']
                    pick = row['team']
                    odds = row.get('odds', -110)
                    
                    # Try to match game
                    result = results.get(game)
                    if result:
                        winner = result['winner']
                        if pick in winner or winner in pick:
                            df.at[idx, 'result'] = 'WIN'
                            df.at[idx, 'payout'] = f"+{calculate_payout(odds):.2f}"
                        else:
                            df.at[idx, 'result'] = 'LOSS'
                            df.at[idx, 'payout'] = "-1.00"
                        df.at[idx, 'final_score'] = result['final_score']
    
    # Select and order columns
    columns = ['date', 'sport', 'game', 'team', 'odds', 'win_probability', 'expected_value', 'result', 'final_score', 'payout']
    df = df[[c for c in columns if c in df.columns]]
    
    # Sort by EV descending
    if 'expected_value' in df.columns:
        df['_ev_sort'] = df['expected_value'].str.replace('%', '').astype(float)
        df = df.sort_values('_ev_sort', ascending=False).drop('_ev_sort', axis=1)
    
    # Save
    date_str = datetime.now().strftime('%Y%m%d')
    filename = output_dir / f'best_bets_{date_str}.csv'
    df.to_csv(filename, index=False)
    
    # Summary
    wins = (df['result'] == 'WIN').sum()
    losses = (df['result'] == 'LOSS').sum()
    pending = (df['result'] == 'PENDING').sum()
    
    print(f"Exported {len(df)} bets to: {filename}")
    print(f"Results: {wins}W - {losses}L - {pending} Pending")
    
    return filename


def main():
    print("=" * 50)
    print("EXPORTING TODAY'S BEST BETS")
    print("=" * 50)
    
    # Try local first, then Render
    edges = fetch_all_edges('http://localhost:5000')
    
    if not edges:
        print("Local server not running, trying Render...")
        edges = fetch_all_edges('https://sports-oor2.onrender.com')
    
    if edges:
        print(f"\nFound {len(edges)} bets. Fetching results...")
        
        filename = export_to_csv(edges, include_results=True)
        
        # Show preview
        df = pd.read_csv(filename)
        print(f"\nPreview (top 5):")
        print(df.head().to_string(index=False))
        
        print(f"\nOpen in Google Sheets: File > Import > Upload > {filename}")
    else:
        print("No edges found from any source")


if __name__ == "__main__":
    main()
