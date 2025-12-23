"""Analyze yesterday's bets against actual results."""
import requests
from datetime import datetime, timedelta

def get_nba_scores(date_str):
    """Fetch NBA scores from ESPN for a specific date."""
    url = f'https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates={date_str}'
    r = requests.get(url)
    data = r.json()
    
    results = []
    for game in data.get('events', []):
        comps = game['competitions'][0]['competitors']
        home = next(c for c in comps if c['homeAway'] == 'home')
        away = next(c for c in comps if c['homeAway'] == 'away')
        
        results.append({
            'game': f"{away['team']['abbreviation']} @ {home['team']['abbreviation']}",
            'home': home['team']['abbreviation'],
            'away': away['team']['abbreviation'],
            'home_score': int(home.get('score', 0)),
            'away_score': int(away.get('score', 0)),
            'winner': home['team']['abbreviation'] if int(home.get('score', 0)) > int(away.get('score', 0)) else away['team']['abbreviation']
        })
    return results

def get_nfl_scores(date_str):
    """Fetch NFL scores from ESPN."""
    url = f'https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard?dates={date_str}'
    r = requests.get(url)
    data = r.json()
    
    results = []
    for game in data.get('events', []):
        comps = game['competitions'][0]['competitors']
        home = next(c for c in comps if c['homeAway'] == 'home')
        away = next(c for c in comps if c['homeAway'] == 'away')
        
        results.append({
            'game': f"{away['team']['abbreviation']} @ {home['team']['abbreviation']}",
            'home': home['team']['abbreviation'],
            'away': away['team']['abbreviation'],
            'home_score': int(home.get('score', 0)),
            'away_score': int(away.get('score', 0)),
            'winner': home['team']['abbreviation'] if int(home.get('score', 0)) > int(away.get('score', 0)) else away['team']['abbreviation']
        })
    return results

# Yesterday's bets we recommended (from Dec 21):
YESTERDAY_BETS = [
    # NBA
    {'sport': 'NBA', 'game': 'SAS @ WAS', 'pick': 'WAS', 'odds': 515, 'ev': 14.6},
    {'sport': 'NBA', 'game': 'MEM @ OKC', 'pick': 'MEM', 'odds': 830, 'ev': 14.1},
    {'sport': 'NBA', 'game': 'HOU @ SAC', 'pick': 'SAC', 'odds': 440, 'ev': 13.1},
    {'sport': 'NBA', 'game': 'DET @ POR', 'pick': 'POR', 'odds': 194, 'ev': 2.1},
    # NFL
    {'sport': 'NFL', 'game': 'DEN @ LAC', 'pick': 'DEN', 'odds': 475, 'ev': 14.3},
]

if __name__ == '__main__':
    print("=" * 60)
    print("YESTERDAY'S BET ANALYSIS (Dec 21, 2024)")
    print("=" * 60)
    
    # Get scores
    nba_scores = get_nba_scores('20241221')
    nfl_scores = get_nfl_scores('20241221')
    
    print(f"\nNBA Games ({len(nba_scores)} games):")
    for g in nba_scores:
        print(f"  {g['game']}: {g['away_score']}-{g['home_score']} (Winner: {g['winner']})")
    
    print(f"\nNFL Games ({len(nfl_scores)} games):")
    for g in nfl_scores:
        print(f"  {g['game']}: {g['away_score']}-{g['home_score']} (Winner: {g['winner']})")
    
    # Check our bets
    print("\n" + "=" * 60)
    print("OUR BETS ANALYSIS:")
    print("=" * 60)
    
    wins = 0
    losses = 0
    profit = 0
    
    for bet in YESTERDAY_BETS:
        scores = nba_scores if bet['sport'] == 'NBA' else nfl_scores
        
        # Find matching game
        result = None
        for g in scores:
            if bet['pick'] in g['game']:
                result = g
                break
        
        if result:
            won = result['winner'] == bet['pick']
            if won:
                wins += 1
                payout = bet['odds'] / 100 if bet['odds'] > 0 else 100 / abs(bet['odds'])
                profit += payout
                status = "[WIN]"
            else:
                losses += 1
                profit -= 1
                status = "[LOSS]"
            
            print(f"{bet['sport']} | {bet['game']} | Pick: {bet['pick']} ({'+' if bet['odds'] > 0 else ''}{bet['odds']}) | {status}")
            print(f"    Actual: {result['away_score']}-{result['home_score']} (Winner: {result['winner']})")
        else:
            print(f"{bet['sport']} | {bet['game']} | Pick: {bet['pick']} | [?] GAME NOT FOUND")
    
    print("\n" + "=" * 60)
    print(f"SUMMARY: {wins}W - {losses}L | ROI: {profit:+.2f} units")
    print("=" * 60)
