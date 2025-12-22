"""
Model Analyzer

Compares performance across different bet types and models.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import json

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from betting.bet_tracker import BetTracker


class ModelAnalyzer:
    """
    Analyzes and compares model performance across bet types.
    
    Helps identify which bet types work best with the current model.
    """
    
    def __init__(self, tracker: BetTracker = None):
        """
        Initialize model analyzer.
        
        Args:
            tracker: BetTracker instance
        """
        self.tracker = tracker or BetTracker()
    
    def compare_bet_types(self) -> pd.DataFrame:
        """
        Compare performance across moneyline, spread, and totals.
        
        Returns:
            DataFrame with comparison metrics
        """
        return self.tracker.get_performance_by_bet_type()
    
    def get_best_bet_type(self) -> Dict:
        """
        Identify the best performing bet type.
        
        Returns:
            Dictionary with best bet type and metrics
        """
        by_type = self.compare_bet_types()
        
        if len(by_type) == 0:
            return {'best_type': None, 'no_data': True}
        
        # Best by ROI
        best_roi = by_type.loc[by_type['roi'].idxmax()]
        
        # Best by win rate (with minimum bets)
        enough_bets = by_type[by_type['total_bets'] >= 10]
        if len(enough_bets) > 0:
            best_winrate = enough_bets.loc[enough_bets['win_rate'].idxmax()]
        else:
            best_winrate = None
        
        return {
            'best_by_roi': {
                'bet_type': best_roi['bet_type'],
                'roi': best_roi['roi'],
                'total_bets': best_roi['total_bets']
            },
            'best_by_winrate': {
                'bet_type': best_winrate['bet_type'] if best_winrate is not None else None,
                'win_rate': best_winrate['win_rate'] if best_winrate is not None else None
            }
        }
    
    def analyze_ev_accuracy(self) -> Dict:
        """
        Analyze how accurate EV predictions are.
        
        Compares predicted EV to actual ROI.
        """
        return self.tracker.get_ev_analysis()
    
    def get_confidence_breakdown(self) -> pd.DataFrame:
        """
        Analyze performance by model confidence level.
        
        High confidence bets should have better win rates.
        """
        history = self.tracker.get_bet_history(limit=1000)
        
        if len(history) == 0:
            return pd.DataFrame()
        
        # Bucket by model probability
        history['confidence_bucket'] = pd.cut(
            history['model_prob'],
            bins=[0, 0.5, 0.55, 0.6, 0.65, 0.7, 1.0],
            labels=['<50%', '50-55%', '55-60%', '60-65%', '65-70%', '>70%']
        )
        
        return history[history['result'] != 'pending'].groupby('confidence_bucket').agg({
            'id': 'count',
            'result': lambda x: (x == 'win').mean(),
            'profit_loss': 'sum',
            'stake': 'sum'
        }).rename(columns={
            'id': 'total_bets',
            'result': 'win_rate'
        })
    
    def backtest_thresholds(
        self,
        history: pd.DataFrame = None,
        ev_thresholds: List[float] = None,
        prob_thresholds: List[float] = None
    ) -> pd.DataFrame:
        """
        Backtest different EV and probability thresholds.
        
        Helps find optimal bet selection criteria.
        """
        if history is None:
            history = self.tracker.get_bet_history(limit=1000)
        
        if len(history) == 0:
            return pd.DataFrame()
        
        history = history[history['result'] != 'pending']
        
        if ev_thresholds is None:
            ev_thresholds = [0, 0.02, 0.05, 0.1, 0.2]
        if prob_thresholds is None:
            prob_thresholds = [0.5, 0.55, 0.6, 0.65]
        
        results = []
        
        for ev_thresh in ev_thresholds:
            for prob_thresh in prob_thresholds:
                subset = history[
                    (history['expected_value'] >= ev_thresh) &
                    (history['model_prob'] >= prob_thresh)
                ]
                
                if len(subset) == 0:
                    continue
                
                wins = (subset['result'] == 'win').sum()
                total = len(subset)
                profit = subset['profit_loss'].sum()
                staked = subset['stake'].sum()
                
                results.append({
                    'ev_threshold': f'{ev_thresh:.0%}',
                    'prob_threshold': f'{prob_thresh:.0%}',
                    'total_bets': total,
                    'win_rate': wins / total if total > 0 else 0,
                    'total_profit': profit,
                    'roi': profit / staked if staked > 0 else 0
                })
        
        return pd.DataFrame(results)
    
    def get_time_analysis(self) -> pd.DataFrame:
        """
        Analyze performance over time (weekly/monthly).
        
        Identifies if model performance is improving or degrading.
        """
        history = self.tracker.get_bet_history(limit=1000)
        
        if len(history) == 0:
            return pd.DataFrame()
        
        history = history[history['result'] != 'pending']
        history['week'] = pd.to_datetime(history['game_date']).dt.isocalendar().week
        
        return history.groupby('week').agg({
            'id': 'count',
            'result': lambda x: (x == 'win').mean(),
            'profit_loss': 'sum',
            'stake': 'sum'
        }).rename(columns={
            'id': 'total_bets',
            'result': 'win_rate'
        })
    
    def generate_analysis_report(self) -> str:
        """Generate comprehensive model analysis report."""
        report = []
        report.append("=" * 70)
        report.append("MODEL PERFORMANCE ANALYSIS")
        report.append("=" * 70)
        report.append("")
        
        # Bet type comparison
        report.append("BET TYPE COMPARISON")
        report.append("-" * 40)
        by_type = self.compare_bet_types()
        if len(by_type) > 0:
            for _, row in by_type.iterrows():
                report.append(
                    f"  {row['bet_type']:12} | "
                    f"Bets: {row['total_bets']:3} | "
                    f"Win: {row['win_rate']:5.1f}% | "
                    f"ROI: {row['roi']:+6.1f}%"
                )
        else:
            report.append("  No data available")
        report.append("")
        
        # Best bet type
        best = self.get_best_bet_type()
        if best.get('best_by_roi'):
            report.append("RECOMMENDATION")
            report.append("-" * 40)
            report.append(
                f"  Best by ROI: {best['best_by_roi']['bet_type']} "
                f"({best['best_by_roi']['roi']:+.1f}%)"
            )
            if best['best_by_winrate']['bet_type']:
                report.append(
                    f"  Best by Win Rate: {best['best_by_winrate']['bet_type']} "
                    f"({best['best_by_winrate']['win_rate']:.1f}%)"
                )
        report.append("")
        
        # EV accuracy
        ev = self.analyze_ev_accuracy()
        if ev.get('by_ev_bucket'):
            report.append("EV PREDICTION ACCURACY")
            report.append("-" * 40)
            for bucket in ev['by_ev_bucket']:
                report.append(
                    f"  {bucket['ev_bucket']:12} | "
                    f"Predicted: {bucket['avg_predicted_ev']:5.1f}% | "
                    f"Actual ROI: {bucket['actual_roi']:+5.1f}%"
                )
            if ev['ev_correlation'] is not None:
                report.append(f"  Correlation: {ev['ev_correlation']:.3f}")
        report.append("")
        
        report.append("=" * 70)
        
        return "\n".join(report)


def main():
    """Demo the model analyzer."""
    print("Model Analyzer Demo")
    print("=" * 50)
    
    # Create test data
    tracker = BetTracker(db_path="./data/test_bets.db")
    
    # Place variety of bets
    import random
    bet_types = ['moneyline', 'spread', 'total']
    teams = ['LAL', 'GSW', 'BOS', 'MIA', 'DEN', 'PHX']
    
    for i in range(30):
        bet_type = random.choice(bet_types)
        home = random.choice(teams)
        away = random.choice([t for t in teams if t != home])
        
        bet_id = tracker.place_bet(
            home_team=home,
            away_team=away,
            bet_type=bet_type,
            selection=home if bet_type != 'total' else random.choice(['over', 'under']),
            line=None if bet_type == 'moneyline' else random.uniform(-7, 7) if bet_type == 'spread' else random.uniform(210, 230),
            odds=random.choice([-110, -115, -120, 100, 105, 110]),
            stake=50,
            model_prob=random.uniform(0.5, 0.7),
            expected_value=random.uniform(0, 0.2)
        )
        
        # Resolve randomly
        result = random.choices(['win', 'loss'], weights=[0.55, 0.45])[0]
        tracker.resolve_bet(bet_id, result, home_score=random.randint(95, 125), away_score=random.randint(95, 125))
    
    # Analyze
    analyzer = ModelAnalyzer(tracker)
    print(analyzer.generate_analysis_report())


if __name__ == "__main__":
    main()
