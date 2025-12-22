"""
Backtesting System

Run historical simulations to validate model performance.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))


class Backtester:
    """Backtest predictions against historical data."""
    
    def __init__(self, initial_bankroll: float = 1000):
        """Initialize backtester."""
        self.initial_bankroll = initial_bankroll
        self.results = []
    
    def run_backtest(
        self,
        predictions: List[Dict],
        kelly_fraction: float = 0.25,
        min_ev: float = 0.02
    ) -> Dict:
        """
        Run backtest simulation.
        
        Args:
            predictions: List of dicts with:
                - predicted_winner: Team predicted to win
                - actual_winner: Actual winner
                - probability: Model's win probability
                - odds: American odds
            kelly_fraction: Fractional Kelly (0.25 = quarter Kelly)
            min_ev: Minimum EV to place bet
            
        Returns:
            Backtest results
        """
        bankroll = self.initial_bankroll
        self.results = []
        
        bets_placed = 0
        wins = 0
        losses = 0
        
        for pred in predictions:
            prob = pred['probability']
            odds = pred['odds']
            predicted = pred['predicted_winner']
            actual = pred['actual_winner']
            
            # Calculate EV
            if odds > 0:
                decimal_odds = odds / 100
            else:
                decimal_odds = 100 / abs(odds)
            
            ev = prob * decimal_odds - (1 - prob)
            
            # Only bet if EV > threshold
            if ev < min_ev:
                continue
            
            # Kelly bet sizing
            b = decimal_odds
            kelly = (b * prob - (1 - prob)) / b
            bet_size = bankroll * kelly * kelly_fraction
            bet_size = max(0, min(bet_size, bankroll * 0.1))  # Cap at 10%
            
            if bet_size <= 0:
                continue
            
            bets_placed += 1
            
            # Determine outcome
            won = predicted == actual
            
            if won:
                profit = bet_size * decimal_odds
                wins += 1
            else:
                profit = -bet_size
                losses += 1
            
            bankroll += profit
            
            self.results.append({
                'bet_num': bets_placed,
                'matchup': pred.get('matchup', 'Unknown'),
                'pick': predicted,
                'probability': prob,
                'ev': ev,
                'bet_size': bet_size,
                'won': won,
                'profit': profit,
                'bankroll': bankroll
            })
        
        # Calculate summary stats
        if bets_placed > 0:
            win_rate = wins / bets_placed
            total_profit = bankroll - self.initial_bankroll
            roi = total_profit / self.initial_bankroll
            
            # Calculate Sharpe-like ratio
            profits = [r['profit'] for r in self.results]
            avg_profit = np.mean(profits)
            std_profit = np.std(profits) if len(profits) > 1 else 1
            sharpe = avg_profit / std_profit if std_profit > 0 else 0
        else:
            win_rate = 0
            total_profit = 0
            roi = 0
            sharpe = 0
        
        return {
            'initial_bankroll': self.initial_bankroll,
            'final_bankroll': bankroll,
            'total_profit': total_profit,
            'roi': roi,
            'bets_placed': bets_placed,
            'wins': wins,
            'losses': losses,
            'win_rate': win_rate,
            'sharpe_ratio': sharpe,
            'max_drawdown': self._calculate_drawdown(),
            'results': self.results
        }
    
    def _calculate_drawdown(self) -> float:
        """Calculate maximum drawdown."""
        if not self.results:
            return 0
        
        bankrolls = [r['bankroll'] for r in self.results]
        peak = self.initial_bankroll
        max_dd = 0
        
        for br in bankrolls:
            if br > peak:
                peak = br
            dd = (peak - br) / peak
            max_dd = max(max_dd, dd)
        
        return max_dd
    
    def generate_report(self, results: Dict) -> str:
        """Generate backtest report."""
        lines = [
            "=" * 60,
            "BACKTEST RESULTS",
            "=" * 60,
            "",
            f"Initial Bankroll:  ${results['initial_bankroll']:,.2f}",
            f"Final Bankroll:    ${results['final_bankroll']:,.2f}",
            f"Total Profit:      ${results['total_profit']:,.2f}",
            f"ROI:               {results['roi']:.1%}",
            "",
            f"Bets Placed:       {results['bets_placed']}",
            f"Wins:              {results['wins']}",
            f"Losses:            {results['losses']}",
            f"Win Rate:          {results['win_rate']:.1%}",
            "",
            f"Sharpe Ratio:      {results['sharpe_ratio']:.2f}",
            f"Max Drawdown:      {results['max_drawdown']:.1%}",
            "=" * 60
        ]
        return "\n".join(lines)
    
    def analyze_by_confidence(self, results: Dict) -> pd.DataFrame:
        """Analyze performance by confidence level."""
        if not results['results']:
            return pd.DataFrame()
        
        df = pd.DataFrame(results['results'])
        
        # Bin by probability
        df['confidence_bin'] = pd.cut(
            df['probability'],
            bins=[0, 0.55, 0.60, 0.65, 0.70, 1.0],
            labels=['50-55%', '55-60%', '60-65%', '65-70%', '70%+']
        )
        
        summary = df.groupby('confidence_bin').agg({
            'bet_num': 'count',
            'won': 'sum',
            'profit': 'sum'
        }).rename(columns={'bet_num': 'bets', 'won': 'wins'})
        
        summary['win_rate'] = summary['wins'] / summary['bets']
        summary['avg_profit'] = summary['profit'] / summary['bets']
        
        return summary


def main():
    """Demo backtest with simulated data."""
    print("=" * 60)
    print("Backtest Simulation Demo")
    print("=" * 60)
    
    # Generate sample predictions
    np.random.seed(42)
    n_games = 200
    
    predictions = []
    for i in range(n_games):
        # Simulate model with ~55% edge
        true_prob = 0.5 + np.random.uniform(-0.2, 0.2)
        model_prob = true_prob + np.random.uniform(-0.05, 0.05)
        model_prob = max(0.3, min(0.7, model_prob))
        
        # Random odds
        odds = np.random.choice([-150, -130, -110, +100, +120, +150])
        
        # Determine actual winner based on true probability
        actual = 'Home' if np.random.random() < true_prob else 'Away'
        predicted = 'Home' if model_prob > 0.5 else 'Away'
        
        predictions.append({
            'matchup': f'Game {i+1}',
            'predicted_winner': predicted,
            'actual_winner': actual,
            'probability': model_prob if predicted == 'Home' else 1 - model_prob,
            'odds': odds
        })
    
    # Run backtest
    backtester = Backtester(initial_bankroll=1000)
    results = backtester.run_backtest(predictions, kelly_fraction=0.25, min_ev=0.02)
    
    print(backtester.generate_report(results))
    
    # Analyze by confidence
    print("\nPerformance by Confidence Level:")
    print(backtester.analyze_by_confidence(results))


if __name__ == '__main__':
    main()
