"""
Export Today's Best Bets to CSV (Google Sheets compatible)

Fetches current edges and exports them to a CSV file.
"""

import requests
import pandas as pd
from datetime import datetime
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


def export_to_csv(edges, output_dir='data'):
    """Export edges to CSV file."""
    if not edges:
        print("No edges to export")
        return None
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Create DataFrame
    df = pd.DataFrame(edges)
    
    # Rename and format columns for readability
    if 'modelProbability' in df.columns:
        df['win_probability'] = (df['modelProbability'] * 100).round(1).astype(str) + '%'
    if 'ev' in df.columns:
        df['expected_value'] = (df['ev'] * 100).round(1).astype(str) + '%'
    
    # Select and order columns
    columns = ['date', 'sport', 'game', 'team', 'odds', 'win_probability', 'expected_value']
    df = df[[c for c in columns if c in df.columns]]
    
    # Sort by EV descending
    if 'expected_value' in df.columns:
        df['_ev_sort'] = df['expected_value'].str.replace('%', '').astype(float)
        df = df.sort_values('_ev_sort', ascending=False).drop('_ev_sort', axis=1)
    
    # Save
    date_str = datetime.now().strftime('%Y%m%d')
    filename = output_dir / f'best_bets_{date_str}.csv'
    df.to_csv(filename, index=False)
    
    print(f"Exported {len(df)} bets to: {filename}")
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
        print(f"\nFound {len(edges)} bets:")
        for e in edges[:5]:
            print(f"  {e['sport']} | {e['game']} | {e['team']} | EV: {e['ev']*100:.1f}%")
        if len(edges) > 5:
            print(f"  ... and {len(edges)-5} more")
        
        filename = export_to_csv(edges)
        print(f"\nOpen in Google Sheets: File > Import > Upload > {filename}")
    else:
        print("No edges found from any source")


if __name__ == "__main__":
    main()
