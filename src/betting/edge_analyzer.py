"""
Edge Analyzer

Analyzes our model's edge over bookmaker odds, specifically Stake.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.odds_fetcher import OddsFetcher


class EdgeAnalyzer:
    """Analyzes betting edge over bookmakers."""
    
    # Stake is typically in 'us2' region on The Odds API
    # Also check: 'stake' as a bookmaker name
    STAKE_BOOKMAKER_NAMES = ['stake', 'stakeus', 'stake.com']
    
    def __init__(
        self,
        tracking_file: str = "./data/edge_tracking.json",
        odds_fetcher: Optional[OddsFetcher] = None
    ):
        """
        Initialize edge analyzer.
        
        Args:
            tracking_file: Path to store edge tracking data
            odds_fetcher: OddsFetcher instance (or creates new one)
        """
        self.tracking_file = Path(tracking_file)
        self.tracking_file.parent.mkdir(parents=True, exist_ok=True)
        
        self.odds_fetcher = odds_fetcher or OddsFetcher()
        self._tracking_data = self._load_tracking()
    
    def _load_tracking(self) -> Dict:
        """Load tracking data from file."""
        if self.tracking_file.exists():
            try:
                with open(self.tracking_file, 'r') as f:
                    return json.load(f)
            except:
                pass
        
        return {
            'predictions': [],
            'results': [],
            'summary': {
                'total_bets': 0,
                'wins': 0,
                'losses': 0,
                'total_wagered': 0,
                'total_returned': 0,
                'profit': 0,
                'roi': 0
            }
        }
    
    def _save_tracking(self):
        """Save tracking data to file."""
        with open(self.tracking_file, 'w') as f:
            json.dump(self._tracking_data, f, indent=2, default=str)
    
    def get_stake_odds(
        self,
        sport: str = "nba"
    ) -> pd.DataFrame:
        """
        Get current odds from Stake (via The Odds API).
        
        Args:
            sport: Sport to fetch odds for
            
        Returns:
            DataFrame with Stake odds
        """
        if not self.odds_fetcher.api_key or "YOUR_" in (self.odds_fetcher.api_key or ""):
            print("The Odds API key not configured. Cannot fetch live Stake odds.")
            return pd.DataFrame()
        
        # Stake is typically in 'us2' region
        all_odds = self.odds_fetcher.get_odds(
            sport=sport,
            regions="us,us2",  # Include regions where Stake operates
            markets="h2h,spreads,totals"
        )
        
        if all_odds.empty:
            return pd.DataFrame()
        
        # Filter for Stake bookmaker
        stake_odds = all_odds[
            all_odds['bookmaker'].str.lower().isin(self.STAKE_BOOKMAKER_NAMES)
        ]
        
        # If Stake not found, return best available (for demo)
        if stake_odds.empty:
            print("Stake not found in odds data. Using average market odds.")
            # Return consensus/average odds
            return all_odds
        
        return stake_odds
    
    def analyze_edge(
        self,
        predictions: List[Dict],
        bookmaker: str = "stake"
    ) -> pd.DataFrame:
        """
        Analyze our model's edge over bookmaker odds.
        
        Args:
            predictions: List of our predictions with probabilities
            bookmaker: Bookmaker to compare against
            
        Returns:
            DataFrame with edge analysis
        """
        results = []
        
        for pred in predictions:
            home_team = pred['home_team']
            away_team = pred['away_team']
            home_prob = pred['home_win_probability']
            away_prob = pred.get('away_win_probability', 1 - home_prob)
            
            # Get bookmaker's implied probabilities
            home_odds = pred.get('home_odds')
            away_odds = pred.get('away_odds')
            
            if home_odds and away_odds:
                home_implied = self.odds_fetcher.american_to_probability(home_odds)
                away_implied = self.odds_fetcher.american_to_probability(away_odds)
                
                # Calculate edge
                home_edge = home_prob - home_implied
                away_edge = away_prob - away_implied
                
                # Calculate expected value
                home_ev = self._calculate_ev(home_prob, home_odds)
                away_ev = self._calculate_ev(away_prob, away_odds)
                
                results.append({
                    'game': f"{away_team} @ {home_team}",
                    'home_team': home_team,
                    'away_team': away_team,
                    # Our model
                    'model_home_prob': home_prob,
                    'model_away_prob': away_prob,
                    # Bookmaker (Stake)
                    'stake_home_odds': home_odds,
                    'stake_away_odds': away_odds,
                    'stake_home_implied': home_implied,
                    'stake_away_implied': away_implied,
                    # Edge analysis
                    'home_edge': home_edge,
                    'away_edge': away_edge,
                    'home_ev': home_ev,
                    'away_ev': away_ev,
                    # Recommendation
                    'best_bet': home_team if home_ev > away_ev else away_team,
                    'best_ev': max(home_ev, away_ev),
                    'has_edge': max(home_ev, away_ev) > 0
                })
        
        return pd.DataFrame(results)
    
    def _calculate_ev(self, prob: float, american_odds: int) -> float:
        """Calculate expected value as percentage of stake."""
        if american_odds < 0:
            profit = 100 / abs(american_odds)
        else:
            profit = american_odds / 100
        
        return (prob * profit) - ((1 - prob) * 1)
    
    def track_prediction(
        self,
        game_id: str,
        home_team: str,
        away_team: str,
        predicted_winner: str,
        model_probability: float,
        stake_odds: int,
        bet_amount: float
    ):
        """
        Track a prediction for later verification.
        
        Args:
            game_id: Unique game identifier
            home_team: Home team
            away_team: Away team
            predicted_winner: Our predicted winner
            model_probability: Our win probability
            stake_odds: Stake's odds for our predicted winner
            bet_amount: Amount bet
        """
        self._tracking_data['predictions'].append({
            'game_id': game_id,
            'game': f"{away_team} @ {home_team}",
            'predicted_winner': predicted_winner,
            'model_probability': model_probability,
            'stake_odds': stake_odds,
            'bet_amount': bet_amount,
            'predicted_at': datetime.now().isoformat(),
            'status': 'pending'
        })
        
        self._save_tracking()
    
    def record_result(
        self,
        game_id: str,
        actual_winner: str
    ):
        """
        Record the actual result of a game.
        
        Args:
            game_id: Game identifier
            actual_winner: Actual winner of the game
        """
        for pred in self._tracking_data['predictions']:
            if pred['game_id'] == game_id and pred['status'] == 'pending':
                won = pred['predicted_winner'] == actual_winner
                
                if won:
                    # Calculate payout
                    odds = pred['stake_odds']
                    if odds < 0:
                        payout = pred['bet_amount'] * (100 / abs(odds))
                    else:
                        payout = pred['bet_amount'] * (odds / 100)
                    
                    self._tracking_data['summary']['wins'] += 1
                    self._tracking_data['summary']['total_returned'] += pred['bet_amount'] + payout
                else:
                    payout = 0
                    self._tracking_data['summary']['losses'] += 1
                
                pred['status'] = 'won' if won else 'lost'
                pred['actual_winner'] = actual_winner
                pred['payout'] = payout
                pred['resolved_at'] = datetime.now().isoformat()
                
                # Update summary
                self._tracking_data['summary']['total_bets'] += 1
                self._tracking_data['summary']['total_wagered'] += pred['bet_amount']
                
                profit = self._tracking_data['summary']['total_returned'] - self._tracking_data['summary']['total_wagered']
                self._tracking_data['summary']['profit'] = profit
                
                if self._tracking_data['summary']['total_wagered'] > 0:
                    roi = profit / self._tracking_data['summary']['total_wagered']
                    self._tracking_data['summary']['roi'] = roi
                
                self._save_tracking()
                return
    
    def get_performance_summary(self) -> Dict:
        """Get overall performance summary."""
        summary = self._tracking_data['summary'].copy()
        
        # If no tracking data, try to load from CSV exports
        if summary['total_bets'] == 0:
            try:
                csv_path = Path("./data")
                csv_files = list(csv_path.glob("best_bets_*.csv"))
                if csv_files:
                    import pandas as pd
                    all_bets = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)
                    # Filter to resolved bets only
                    resolved = all_bets[all_bets['result'].isin(['WIN', 'LOSS'])]
                    if len(resolved) > 0:
                        wins = len(resolved[resolved['result'] == 'WIN'])
                        losses = len(resolved[resolved['result'] == 'LOSS'])
                        # Parse payout column (format: +7.13 or -1.00)
                        payouts = resolved['payout'].apply(
                            lambda x: float(str(x).replace('+', '')) if pd.notna(x) and str(x).strip() else 0
                        )
                        total_profit = payouts.sum()
                        summary = {
                            'total_bets': len(resolved),
                            'wins': wins,
                            'losses': losses,
                            'total_wagered': len(resolved),  # Assume $1 per bet
                            'total_returned': len(resolved) + total_profit,
                            'profit': total_profit,
                            'roi': total_profit / len(resolved) if len(resolved) > 0 else 0
                        }
            except Exception as e:
                print(f"Warning: Could not load CSV performance data: {e}")
        
        if summary.get('total_bets', 0) > 0:
            summary['win_rate'] = summary['wins'] / summary['total_bets']
        else:
            summary['win_rate'] = 0
        
        return summary
    
    def get_edge_report(
        self,
        predictions: List[Dict]
    ) -> str:
        """
        Generate a formatted edge analysis report.
        
        Args:
            predictions: List of predictions with odds
            
        Returns:
            Formatted report string
        """
        analysis = self.analyze_edge(predictions)
        
        if analysis.empty:
            return "No edge analysis available - need predictions with odds"
        
        lines = []
        lines.append("=" * 70)
        lines.append("EDGE ANALYSIS REPORT - Model vs Stake")
        lines.append("=" * 70)
        lines.append("")
        
        # Summary stats
        games_with_edge = len(analysis[analysis['has_edge']])
        avg_edge = analysis[analysis['has_edge']]['best_ev'].mean() if games_with_edge > 0 else 0
        
        lines.append(f"Total Games Analyzed: {len(analysis)}")
        lines.append(f"Games with Positive EV: {games_with_edge}")
        lines.append(f"Average Edge (on +EV bets): {avg_edge:.1%}")
        lines.append("")
        
        # Individual games
        for _, row in analysis.iterrows():
            lines.append(f"[GAME] {row['game']}")
            lines.append(f"  Model: {row['home_team']} {row['model_home_prob']:.1%} | {row['away_team']} {row['model_away_prob']:.1%}")
            lines.append(f"  Stake: {row['home_team']} {row['stake_home_implied']:.1%} | {row['away_team']} {row['stake_away_implied']:.1%}")
            lines.append(f"  Edge:  {row['home_team']} {row['home_edge']:+.1%} | {row['away_team']} {row['away_edge']:+.1%}")
            
            if row['has_edge']:
                lines.append(f"  >>> EDGE FOUND: {row['best_bet']} (EV: {row['best_ev']:+.1%})")
            else:
                lines.append(f"  No edge identified")
            lines.append("")
        
        # Historical performance
        perf = self.get_performance_summary()
        if perf['total_bets'] > 0:
            lines.append("=" * 70)
            lines.append("HISTORICAL PERFORMANCE")
            lines.append(f"  Total Bets: {perf['total_bets']}")
            lines.append(f"  Win Rate: {perf['win_rate']:.1%} ({perf['wins']}-{perf['losses']})")
            lines.append(f"  Total Wagered: ${perf['total_wagered']:.2f}")
            lines.append(f"  Total Returned: ${perf['total_returned']:.2f}")
            lines.append(f"  Profit/Loss: ${perf['profit']:+.2f}")
            lines.append(f"  ROI: {perf['roi']:+.1%}")
        
        lines.append("=" * 70)
        
        return "\n".join(lines)


def main():
    """Demo edge analyzer."""
    analyzer = EdgeAnalyzer()
    
    print("Edge Analyzer Demo")
    print("=" * 50)
    
    # Sample predictions with Stake-like odds
    sample_predictions = [
        {
            'home_team': 'LAL',
            'away_team': 'GSW',
            'home_win_probability': 0.58,
            'home_odds': -140,
            'away_odds': +120
        },
        {
            'home_team': 'BOS',
            'away_team': 'MIA',
            'home_win_probability': 0.65,
            'home_odds': -180,
            'away_odds': +150
        },
        {
            'home_team': 'DEN',
            'away_team': 'PHX',
            'home_win_probability': 0.52,
            'home_odds': +100,
            'away_odds': -120
        }
    ]
    
    print(analyzer.get_edge_report(sample_predictions))


if __name__ == "__main__":
    main()
