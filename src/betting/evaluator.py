"""
Betting Evaluator

Evaluates betting opportunities and generates recommendations.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime


class BettingEvaluator:
    """Evaluates betting opportunities and calculates expected value."""
    
    def __init__(
        self,
        min_ev_threshold: float = 0.02,
        max_kelly_fraction: float = 0.25,
        bankroll: float = 1000.0
    ):
        """
        Initialize evaluator.
        
        Args:
            min_ev_threshold: Minimum EV to recommend a bet (default 2%)
            max_kelly_fraction: Fraction of Kelly criterion to use (default 25%)
            bankroll: Current bankroll for bet sizing
        """
        self.min_ev_threshold = min_ev_threshold
        self.max_kelly_fraction = max_kelly_fraction
        self.bankroll = bankroll
    
    # ==================== Odds Conversion ====================
    
    @staticmethod
    def american_to_decimal(american: int) -> float:
        """Convert American odds to decimal odds."""
        if american > 0:
            return (american / 100) + 1
        else:
            return (100 / abs(american)) + 1
    
    @staticmethod
    def american_to_probability(american: int) -> float:
        """Convert American odds to implied probability."""
        if american < 0:
            return abs(american) / (abs(american) + 100)
        else:
            return 100 / (american + 100)
    
    @staticmethod
    def probability_to_american(prob: float) -> int:
        """Convert probability to American odds."""
        if prob >= 0.5:
            return int(-100 * prob / (1 - prob))
        else:
            return int(100 * (1 - prob) / prob)
    
    @staticmethod
    def decimal_to_american(decimal: float) -> int:
        """Convert decimal odds to American odds."""
        if decimal >= 2.0:
            return int((decimal - 1) * 100)
        else:
            return int(-100 / (decimal - 1))
    
    # ==================== Expected Value ====================
    
    def calculate_ev(
        self, 
        model_prob: float, 
        american_odds: int
    ) -> float:
        """
        Calculate expected value of a bet.
        
        EV = (probability * profit) - ((1 - probability) * stake)
        
        Returns EV as a percentage of stake.
        """
        decimal_odds = self.american_to_decimal(american_odds)
        
        # For a $1 stake
        profit = decimal_odds - 1  # Net profit if win
        stake = 1
        
        ev = (model_prob * profit) - ((1 - model_prob) * stake)
        return ev
    
    def calculate_edge(
        self,
        model_prob: float,
        american_odds: int
    ) -> float:
        """
        Calculate edge over the market.
        
        Edge = model probability - implied probability
        """
        implied_prob = self.american_to_probability(american_odds)
        return model_prob - implied_prob
    
    # ==================== Bet Sizing ====================
    
    def kelly_criterion(
        self,
        model_prob: float,
        american_odds: int
    ) -> float:
        """
        Calculate Kelly criterion bet size.
        
        Kelly = (bp - q) / b
        where:
            b = decimal odds - 1 (net odds)
            p = probability of winning
            q = probability of losing (1 - p)
        
        Returns fraction of bankroll to bet.
        """
        decimal_odds = self.american_to_decimal(american_odds)
        b = decimal_odds - 1
        p = model_prob
        q = 1 - p
        
        if b <= 0:
            return 0.0
        
        kelly = (b * p - q) / b
        
        # Never bet negative (no edge)
        kelly = max(0, kelly)
        
        # Apply fractional Kelly for risk management
        kelly = kelly * self.max_kelly_fraction
        
        # Cap at reasonable max
        kelly = min(kelly, 0.10)  # Max 10% of bankroll
        
        return kelly
    
    def calculate_bet_size(
        self,
        model_prob: float,
        american_odds: int
    ) -> float:
        """Calculate recommended bet size in dollars."""
        kelly = self.kelly_criterion(model_prob, american_odds)
        return self.bankroll * kelly
    
    # ==================== Bet Evaluation ====================
    
    def evaluate_bet(
        self,
        team: str,
        model_prob: float,
        american_odds: int,
        bet_type: str = "moneyline"
    ) -> Dict:
        """
        Evaluate a single betting opportunity.
        
        Args:
            team: Team name
            model_prob: Model's predicted win probability
            american_odds: Current betting odds (American format)
            bet_type: Type of bet (moneyline, spread, total)
            
        Returns:
            Dictionary with evaluation details
        """
        ev = self.calculate_ev(model_prob, american_odds)
        edge = self.calculate_edge(model_prob, american_odds)
        implied_prob = self.american_to_probability(american_odds)
        kelly = self.kelly_criterion(model_prob, american_odds)
        bet_size = self.calculate_bet_size(model_prob, american_odds)
        
        # Determine recommendation
        if ev >= self.min_ev_threshold:
            recommendation = "BET"
            strength = "STRONG" if ev >= 0.05 else "MODERATE"
        else:
            recommendation = "PASS"
            strength = None
        
        return {
            'team': team,
            'bet_type': bet_type,
            'american_odds': american_odds,
            'implied_probability': implied_prob,
            'model_probability': model_prob,
            'edge': edge,
            'expected_value': ev,
            'kelly_fraction': kelly,
            'recommended_bet_size': bet_size,
            'recommendation': recommendation,
            'strength': strength
        }
    
    def evaluate_game(
        self,
        home_team: str,
        away_team: str,
        home_prob: float,
        home_odds: int,
        away_odds: int
    ) -> Dict:
        """
        Evaluate betting opportunities for entire game.
        
        Args:
            home_team: Home team name
            away_team: Away team name
            home_prob: Model's home win probability
            home_odds: American odds for home team
            away_odds: American odds for away team
            
        Returns:
            Dictionary with evaluations for both sides
        """
        away_prob = 1 - home_prob
        
        home_eval = self.evaluate_bet(
            home_team, home_prob, home_odds, "moneyline"
        )
        away_eval = self.evaluate_bet(
            away_team, away_prob, away_odds, "moneyline"
        )
        
        # Determine best bet
        if home_eval['expected_value'] > away_eval['expected_value']:
            best_bet = home_eval
        else:
            best_bet = away_eval
        
        return {
            'home_team': home_team,
            'away_team': away_team,
            'home_evaluation': home_eval,
            'away_evaluation': away_eval,
            'best_bet': best_bet,
            'has_value_bet': best_bet['recommendation'] == 'BET'
        }
    
    def evaluate_spread(
        self,
        team: str,
        model_prob: float,
        spread: float,
        odds: int = -110
    ) -> Dict:
        """
        Evaluate a spread bet.
        
        Args:
            team: Team taking the spread
            model_prob: Model's probability of covering
            spread: Point spread (e.g., -5.5, +3.5)
            odds: American odds (typically -110)
            
        Returns:
            Evaluation dictionary
        """
        result = self.evaluate_bet(
            f"{team} {spread:+.1f}",
            model_prob,
            odds,
            bet_type="spread"
        )
        result['spread'] = spread
        return result
    
    def evaluate_total(
        self,
        total: float,
        over_prob: float,
        over_odds: int = -110,
        under_odds: int = -110
    ) -> Dict:
        """
        Evaluate an over/under total bet.
        
        Args:
            total: Total points line
            over_prob: Model's probability of over hitting
            over_odds: American odds for over
            under_odds: American odds for under
            
        Returns:
            Dictionary with over and under evaluations
        """
        under_prob = 1 - over_prob
        
        over_eval = self.evaluate_bet(
            f"Over {total}", over_prob, over_odds, "total"
        )
        under_eval = self.evaluate_bet(
            f"Under {total}", under_prob, under_odds, "total"
        )
        
        if over_eval['expected_value'] > under_eval['expected_value']:
            best_bet = over_eval
        else:
            best_bet = under_eval
        
        return {
            'total': total,
            'over_evaluation': over_eval,
            'under_evaluation': under_eval,
            'best_bet': best_bet,
            'has_value_bet': best_bet['recommendation'] == 'BET'
        }
    
    # ==================== Batch Processing ====================
    
    def find_value_bets(
        self,
        games: List[Dict]
    ) -> pd.DataFrame:
        """
        Find all value bets from a list of games.
        
        Args:
            games: List of dicts with game info, probabilities, and odds
            
        Returns:
            DataFrame of recommended bets sorted by EV
        """
        value_bets = []
        
        for game in games:
            eval_result = self.evaluate_game(
                home_team=game['home_team'],
                away_team=game['away_team'],
                home_prob=game['home_prob'],
                home_odds=game['home_odds'],
                away_odds=game['away_odds']
            )
            
            if eval_result['has_value_bet']:
                best = eval_result['best_bet']
                value_bets.append({
                    'game': f"{game['away_team']} @ {game['home_team']}",
                    'game_date': game.get('game_date', ''),
                    'bet': best['team'],
                    'bet_type': best['bet_type'],
                    'odds': best['american_odds'],
                    'model_prob': best['model_probability'],
                    'implied_prob': best['implied_probability'],
                    'edge': best['edge'],
                    'ev': best['expected_value'],
                    'recommended_size': best['recommended_bet_size'],
                    'strength': best['strength']
                })
        
        if not value_bets:
            return pd.DataFrame()
        
        df = pd.DataFrame(value_bets)
        df = df.sort_values('ev', ascending=False)
        
        return df
    
    def generate_daily_report(
        self,
        games: List[Dict]
    ) -> str:
        """
        Generate a formatted daily betting report.
        
        Args:
            games: List of games with predictions and odds
            
        Returns:
            Formatted string report
        """
        value_bets = self.find_value_bets(games)
        
        report = []
        report.append("=" * 60)
        report.append(f"DAILY BETTING REPORT - {datetime.now().strftime('%Y-%m-%d')}")
        report.append("=" * 60)
        report.append("")
        
        if value_bets.empty:
            report.append("No value bets identified today.")
            report.append("")
            report.append("All games analyzed have insufficient edge over market odds.")
        else:
            report.append(f"Found {len(value_bets)} value bet(s):\n")
            
            for _, bet in value_bets.iterrows():
                report.append(f"  [{bet['strength']}] {bet['bet']}")
                report.append(f"     Game: {bet['game']}")
                report.append(f"     Odds: {bet['odds']:+d}")
                report.append(f"     Model Prob: {bet['model_prob']:.1%} vs Implied: {bet['implied_prob']:.1%}")
                report.append(f"     Edge: {bet['edge']:.1%}, EV: {bet['ev']:.1%}")
                report.append(f"     Recommended Size: ${bet['recommended_size']:.2f}")
                report.append("")
        
        report.append("=" * 60)
        report.append("DISCLAIMER: Past performance does not guarantee future results.")
        report.append("Bet responsibly. Never bet more than you can afford to lose.")
        report.append("=" * 60)
        
        return "\n".join(report)


def main():
    """Demo the betting evaluator."""
    evaluator = BettingEvaluator(
        min_ev_threshold=0.02,
        max_kelly_fraction=0.25,
        bankroll=1000.0
    )
    
    print("Betting Evaluator Demo")
    print("=" * 50)
    
    # Demo single bet evaluation
    print("\nSingle Bet Evaluation:")
    print("-" * 30)
    
    result = evaluator.evaluate_bet(
        team="Los Angeles Lakers",
        model_prob=0.58,  # Model says 58% chance
        american_odds=-150  # Market says ~60% implied
    )
    
    for k, v in result.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")
    
    # Demo game evaluation
    print("\n\nFull Game Evaluation:")
    print("-" * 30)
    
    game_eval = evaluator.evaluate_game(
        home_team="Lakers",
        away_team="Warriors",
        home_prob=0.55,
        home_odds=-130,
        away_odds=+110
    )
    
    print(f"Home ({game_eval['home_team']}):")
    print(f"  EV: {game_eval['home_evaluation']['expected_value']:.2%}")
    print(f"  Edge: {game_eval['home_evaluation']['edge']:.2%}")
    
    print(f"\nAway ({game_eval['away_team']}):")
    print(f"  EV: {game_eval['away_evaluation']['expected_value']:.2%}")
    print(f"  Edge: {game_eval['away_evaluation']['edge']:.2%}")
    
    print(f"\nBest Bet: {game_eval['best_bet']['team']}")
    print(f"Recommendation: {game_eval['best_bet']['recommendation']}")
    
    # Demo daily report
    print("\n" + "=" * 60)
    print("SAMPLE DAILY REPORT")
    print("=" * 60 + "\n")
    
    sample_games = [
        {'home_team': 'Lakers', 'away_team': 'Warriors', 
         'home_prob': 0.58, 'home_odds': -140, 'away_odds': +120},
        {'home_team': 'Celtics', 'away_team': 'Heat',
         'home_prob': 0.65, 'home_odds': -180, 'away_odds': +150},
        {'home_team': 'Suns', 'away_team': 'Nuggets',
         'home_prob': 0.45, 'home_odds': +100, 'away_odds': -120},
    ]
    
    report = evaluator.generate_daily_report(sample_games)
    print(report)


if __name__ == "__main__":
    main()
