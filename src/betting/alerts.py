"""
Automated Betting Alerts

Monitors odds for value bets and sends notifications via:
1. Browser Push (simple local)
2. Desktop Notification
3. Discord Webhook (Free & Easy)
"""

import time
import requests
import json
from datetime import datetime
from threading import Thread
import platform
from typing import Dict

# For desktop notifications
try:
    from plyer import notification
    DESKTOP_NOTIFY = True
except ImportError:
    DESKTOP_NOTIFY = False


class AlertSystem:
    """Monitors API for betting edges and sends alerts."""
    
    def __init__(self, check_interval_minutes: int = 5, min_ev: float = 0.05):
        """
        Initialize alert system.
        
        Args:
            check_interval_minutes: How often to check (default 5 mins)
            min_ev: Minimum EV to trigger alert (default 5%)
        """
        self.interval = check_interval_minutes * 60
        self.min_ev = min_ev
        self.running = False
        self.seen_games = set()  # prevent duplicate alerts
        
        # Load config for Discord
        self.discord_webhook = None
        try:
            with open('config/config.yaml', 'r') as f:
                import yaml
                config = yaml.safe_load(f)
                self.discord_webhook = config.get('notifications', {}).get('discord_webhook')
        except:
            pass

    def start(self):
        """Start monitoring in background thread."""
        self.running = True
        t = Thread(target=self._monitor_loop)
        t.daemon = True
        t.start()
        print(f"[*] Alert system started (Interval: {self.interval/60}m, Min EV: {self.min_ev*100}%)")

    def stop(self):
        """Stop monitoring."""
        self.running = False

    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.running:
            try:
                self._check_all_sports()
            except Exception as e:
                print(f"[!] Error in alert loop: {e}")
            
            # Wait for next interval
            time.sleep(self.interval)

    def _check_all_sports(self):
        """Check all sports for value bets."""
        sports = ['nba', 'nfl', 'nhl', 'ncaaf']
        
        for sport in sports:
            try:
                # Use local API to reuse logic
                response = requests.get(f'http://localhost:5000/api/edge-analysis?sport={sport}')
                if response.status_code != 200:
                    continue
                
                data = response.json()
                edges = data.get('edges', [])
                
                for edge in edges:
                    ev = edge.get('ev', 0)
                    if ev >= self.min_ev:
                        self._trigger_alert(sport, edge)
                        
            except Exception as e:
                # API might be down or restarting
                pass

    def _trigger_alert(self, sport: str, edge: Dict):
        """Send alert for a specific edge."""
        game_id = f"{sport}_{edge['game']}_{edge['ev']:.2f}"
        
        # Avoid duplicate alerts for same game/edge
        if game_id in self.seen_games:
            return
        
        self.seen_games.add(game_id)
        
        title = f"🔥 {sport.upper()} Value Bet: {edge['team']}"
        message = (
            f"Game: {edge['game']}\n"
            f"Edge: +{edge['ev']*100:.1f}% EV\n"
            f"Odds: {self._format_odds(edge['odds'])}\n"
            f"Model Prob: {edge['modelProbability']*100:.1f}%"
        )
        
        print(f"\n[ALERT] {title}")
        print(message)
        
        # 1. Desktop Notification
        if DESKTOP_NOTIFY:
            try:
                notification.notify(
                    title=title,
                    message=message,
                    app_name='Sports Bettor',
                    timeout=10
                )
            except:
                pass
        
        # 2. Discord Webhook (if configured)
        if self.discord_webhook:
            payload = {
                "content": f"**{title}**\n{message}",
                "username": "Betting Bot"
            }
            try:
                requests.post(self.discord_webhook, json=payload)
            except:
                pass

    def _format_odds(self, odds):
        return f"+{odds}" if odds > 0 else str(odds)


def main():
    """Test the alert system."""
    print("Testing Alert System...")
    
    # 1. Test Desktop Notification (Mock)
    if DESKTOP_NOTIFY:
        print("Sending test desktop notification...")
        notification.notify(
            title="Test Alert",
            message="This is a test notification from your Betting Bot.",
            app_name='Sports Bettor',
            timeout=5
        )
    else:
        print("Desktop notifications not supported (install plyer: pip install plyer)")

    # 2. Run monitor once
    system = AlertSystem(min_ev=0.01) # Low threshold for test
    print("\nScanning for value bets now...")
    system._check_all_sports()
    print("Done.")


if __name__ == "__main__":
    from typing import Dict
    main()
