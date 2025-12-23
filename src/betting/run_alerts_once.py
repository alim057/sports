"""
Run Alerts Once

Single execution script for GitHub Actions / Cron jobs.
Checks for edges, sends alerts if found, then exits (does not loop).
"""

import sys
import os
import requests
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.odds_fetcher import OddsFetcher


def send_discord_alert(webhook_url: str, title: str, message: str, color: int = 0x00FF00):
    """Send a rich embed to Discord."""
    if not webhook_url:
        print(f"[ALERT] {title}\n{message}")
        return
    
    payload = {
        "embeds": [{
            "title": title,
            "description": message,
            "color": color,
            "timestamp": datetime.utcnow().isoformat(),
            "footer": {"text": "Sports Betting Predictor"}
        }]
    }
    
    try:
        resp = requests.post(webhook_url, json=payload)
        if resp.status_code in [200, 204]:
            print(f"Discord alert sent: {title}")
        else:
            print(f"Discord alert failed: {resp.status_code}")
    except Exception as e:
        print(f"Discord error: {e}")


def validate_moneyline_odds(home_odds, away_odds):
    """Filter out spread values being treated as moneyline."""
    if abs(home_odds) < 100 or abs(away_odds) < 100:
        return False
    return True


def implied_prob(odds):
    """Convert American odds to implied probability."""
    if odds < 0:
        return abs(odds) / (abs(odds) + 100)
    return 100 / (odds + 100)


def calc_ev(model_prob, odds):
    """Calculate expected value."""
    if odds > 0:
        payout = odds / 100
    else:
        payout = 100 / abs(odds)
    return model_prob * payout - (1 - model_prob)


def main():
    print(f"Running betting alert check at {datetime.now()}")
    
    # Get API key and webhook from env (for GitHub Actions)
    api_key = os.environ.get('ODDS_API_KEY')
    webhook = os.environ.get('DISCORD_WEBHOOK')
    min_ev = float(os.environ.get('MIN_EV', '0.05'))  # Default 5% EV threshold
    
    if not api_key:
        print("No ODDS_API_KEY found (check GitHub Secrets)")
        return

    print(f"Checking odds using key: {api_key[:4]}...")
    
    # Fetch raw odds
    fetcher = OddsFetcher(api_key=api_key)
    sports = ['nba', 'nfl', 'nhl', 'ncaaf']
    
    all_edges = []
    
    for sport in sports:
        try:
            odds = fetcher.get_odds(sport, markets='h2h')
            if odds.empty:
                print(f"{sport.upper()}: No games found")
                continue
            
            # Group by game
            h2h = odds[odds['market'] == 'h2h'].groupby(['home_team', 'away_team']).agg({
                'home_odds': 'mean',
                'away_odds': 'mean'
            }).reset_index()
            
            print(f"{sport.upper()}: {len(h2h)} games found")
            
            for _, row in h2h.iterrows():
                home_odds = int(row['home_odds'])
                away_odds = int(row['away_odds'])
                
                # Skip if looks like spread
                if not validate_moneyline_odds(home_odds, away_odds):
                    continue
                
                home_impl = implied_prob(home_odds)
                away_impl = implied_prob(away_odds)
                
                # Simple model: assume slight edge over market + home advantage
                home_model_prob = home_impl + 0.03 + 0.02  # Market + edge + home
                away_model_prob = away_impl + 0.03
                
                # Normalize
                total = home_model_prob + away_model_prob
                home_model_prob /= total
                away_model_prob /= total
                
                home_ev = calc_ev(home_model_prob, home_odds)
                away_ev = calc_ev(away_model_prob, away_odds)
                best_ev = max(home_ev, away_ev)
                
                if best_ev >= min_ev:
                    best_team = row['home_team'][:15] if home_ev > away_ev else row['away_team'][:15]
                    best_odds = home_odds if home_ev > away_ev else away_odds
                    best_prob = home_model_prob if home_ev > away_ev else away_model_prob
                    
                    all_edges.append({
                        'sport': sport.upper(),
                        'game': f"{row['away_team'][:15]} @ {row['home_team'][:15]}",
                        'team': best_team,
                        'odds': best_odds,
                        'probability': best_prob,
                        'ev': best_ev
                    })
                    
        except Exception as e:
            print(f"Error checking {sport}: {e}")
    
    # Sort by EV
    all_edges.sort(key=lambda x: x['ev'], reverse=True)
    
    # Send alerts
    if all_edges:
        print(f"\nFound {len(all_edges)} edges above {min_ev*100:.0f}% EV")
        
        # Send individual alerts for top 5 edges
        for i, edge in enumerate(all_edges[:5]):
            title = f"🔥 {edge['sport']} Value Bet: {edge['team']}"
            message = (
                f"**Game:** {edge['game']}\n"
                f"**Bet:** {edge['team']}\n"
                f"**Odds:** {'+' if edge['odds'] > 0 else ''}{edge['odds']}\n"
                f"**Model Prob:** {edge['probability']*100:.1f}%\n"
                f"**Expected Value:** +{edge['ev']*100:.1f}%"
            )
            
            # Green for high EV, yellow for medium
            color = 0x00FF00 if edge['ev'] > 0.10 else 0xFFFF00
            send_discord_alert(webhook, title, message, color)
        
        if len(all_edges) > 5:
            summary_title = f"📊 {len(all_edges) - 5} More Edges Found"
            summary_msg = "\n".join([
                f"• {e['sport']} {e['team']} +{e['ev']*100:.1f}%"
                for e in all_edges[5:10]
            ])
            if len(all_edges) > 10:
                summary_msg += f"\n...and {len(all_edges) - 10} more"
            send_discord_alert(webhook, summary_title, summary_msg, 0x808080)
    else:
        print("\nNo edges above threshold found")
        # Optionally send "no edges" alert
        # send_discord_alert(webhook, "📊 Market Check Complete", "No +EV bets found above threshold", 0x808080)

    print("\nAlert check complete")


if __name__ == "__main__":
    main()
