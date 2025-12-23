"""
Advanced Prediction Pipeline

Combines team and player features for enhanced predictions.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.nba_fetcher import NBAFetcher
from data.player_fetcher import PlayerFetcher
from data.odds_fetcher import OddsFetcher
from data.weather_fetcher import WeatherFetcher
from data.injury_fetcher import InjuryFetcher
from features.team_features import TeamFeatureEngine
from features.player_features import PlayerFeatureEngine
from models.predictor import GamePredictor
from betting.evaluator import BettingEvaluator
from nba_api.stats.static import teams as nba_teams_static


class AdvancedPredictor:
    """Advanced prediction system combining team and player analysis."""
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        season: str = "2024-25",
        min_ev_threshold: float = 0.02,
        bankroll: float = 1000.0
    ):
        """
        Initialize advanced predictor.
        
        Args:
            model_path: Path to trained model
            season: Current NBA season
            min_ev_threshold: Minimum EV for bet recommendations
            bankroll: Bankroll for bet sizing
        """
        self.season = season
        
        # Data fetchers
        self.nba_fetcher = NBAFetcher()
        self.player_fetcher = PlayerFetcher()
        self.odds_fetcher = OddsFetcher()
        self.weather_fetcher = WeatherFetcher()
        self.injury_fetcher = InjuryFetcher()
        
        # Feature engines
        self.team_features = TeamFeatureEngine()
        self.player_features = PlayerFeatureEngine(season)
        
        # Model and evaluator
        self.predictor = GamePredictor(model_path=model_path)
        self.evaluator = BettingEvaluator(
            min_ev_threshold=min_ev_threshold,
            bankroll=bankroll
        )
        
        # Team lookup cache
        self._teams = {t['abbreviation']: t for t in nba_teams_static.get_teams()}
        self._team_stats_cache = {}
    
    def get_team_id(self, team_abbr: str) -> Optional[int]:
        """Get team ID from abbreviation."""
        team = self._teams.get(team_abbr.upper())
        return team['id'] if team else None
    
    def get_team_recent_stats(
        self,
        team_abbr: str,
        n_games: int = 10
    ) -> pd.Series:
        """
        Get recent performance stats for a team.
        
        Args:
            team_abbr: Team abbreviation (e.g., 'LAL')
            n_games: Number of recent games to consider
            
        Returns:
            Series with calculated features
        """
        if team_abbr in self._team_stats_cache:
            return self._team_stats_cache[team_abbr]
        
        team_id = self.get_team_id(team_abbr)
        if not team_id:
            return pd.Series()
        
        try:
            # Get game log
            game_log = self.nba_fetcher.get_team_game_log(team_id, self.season)
            
            if game_log.empty or len(game_log) < 5:
                return pd.Series()
            
            # Calculate rolling stats
            recent = game_log.head(n_games)
            
            stats = {
                'PTS_L5': game_log['PTS'].head(5).mean(),
                'PTS_L10': game_log['PTS'].head(10).mean(),
                'REB_L5': game_log['REB'].head(5).mean(),
                'REB_L10': game_log['REB'].head(10).mean(),
                'AST_L5': game_log['AST'].head(5).mean(),
                'AST_L10': game_log['AST'].head(10).mean(),
                'HOME_WIN_RATE': (game_log[game_log['IS_HOME']]['WL'] == 'W').mean() if len(game_log[game_log['IS_HOME']]) > 0 else 0.5,
                'AWAY_WIN_RATE': (game_log[~game_log['IS_HOME']]['WL'] == 'W').mean() if len(game_log[~game_log['IS_HOME']]) > 0 else 0.5,
                'REST_DAYS': 2,  # Default, would need schedule data
                'STREAK': self._calculate_streak(game_log),
                'IS_B2B': 0
            }
            
            result = pd.Series(stats)
            self._team_stats_cache[team_abbr] = result
            return result
            
        except Exception as e:
            print(f"Warning: Could not fetch stats for {team_abbr}: {e}")
            return pd.Series()
    
    def _calculate_streak(self, game_log: pd.DataFrame) -> int:
        """Calculate current win/loss streak."""
        if game_log.empty:
            return 0
        
        streak = 0
        first_result = game_log.iloc[0]['WL']
        
        for _, game in game_log.iterrows():
            if game['WL'] == first_result:
                streak += 1 if first_result == 'W' else -1
            else:
                break
        
        return streak
    
    def predict_game(
        self,
        home_team: str,
        away_team: str,
        include_player_features: bool = True
    ) -> Dict:
        """
        Make a prediction for a single game.
        """
        # Get team stats
        home_stats = self.get_team_recent_stats(home_team)
        away_stats = self.get_team_recent_stats(away_team)
        
        if home_stats.empty or away_stats.empty:
            return {'error': 'Could not fetch team stats'}
        
        # Build team features
        team_feats = self.team_features.build_matchup_features(
            home_stats, away_stats
        )
        
        # Get player analysis if enabled
        player_analysis = {}
        if include_player_features:
            home_id = self.get_team_id(home_team)
            away_id = self.get_team_id(away_team)
            
            if home_id and away_id:
                try:
                    player_feats = self.player_features.build_player_matchup_features(
                        home_id, away_id
                    )
                    player_analysis = {
                        'star_power_advantage': player_feats.get('star_power_diff', 0),
                        'bench_depth_advantage': player_feats.get('bench_depth_diff', 0),
                        'overall_talent_advantage': player_feats.get('player_impact_diff', 0)
                    }
                except Exception as e:
                    print(f"Warning: Player feature error: {e}")
        
        # Get Injury Analysis
        injury_analysis = {}
        try:
            injury_data = self.injury_fetcher.get_matchup_injury_comparison(home_team, away_team)
            injury_analysis = {
                'home_impact': injury_data['home_injuries']['impact_score'],
                'away_impact': injury_data['away_injuries']['impact_score'],
                'net_impact_prob': injury_data['home_advantage_pct'] # e.g. -0.05 if home has more injuries
            }
        except Exception as e:
            print(f"Warning: Injury fetch error: {e}")
            injury_analysis = {'net_impact_prob': 0}

        # Make prediction (using trained model if available)
        home_win_prob = None
        if self.predictor.model is not None and self.predictor.feature_names is not None:
            try:
                X = pd.DataFrame([team_feats])
                # Align features with model's expected feature names
                available_features = set(X.columns) & set(self.predictor.feature_names)
                if len(available_features) >= len(self.predictor.feature_names) * 0.5:
                    # Reindex to match expected features, fill missing with 0
                    X_aligned = X.reindex(columns=self.predictor.feature_names, fill_value=0)
                    probs = self.predictor.model.predict_proba(X_aligned)[0]
                    home_win_prob = probs[1]
                else:
                    print(f"Warning: Only {len(available_features)}/{len(self.predictor.feature_names)} features available")
            except Exception as e:
                print(f"Model prediction failed, using heuristic: {e}")
        
        # Fallback: simple heuristic based on features
        if home_win_prob is None:
            home_win_prob = 0.5 + (
                team_feats.get('PTS_DIFF_L5', 0) * 0.01 +
                team_feats.get('WIN_RATE_DIFF', 0) * 0.3 +
                team_feats.get('REST_ADVANTAGE', 0) * 0.02 +
                0.03  # Home court advantage
            )
            home_win_prob = max(0.1, min(0.9, home_win_prob))
        
        # Apply Injury Adjustment
        # If home_advantage_pct is -0.05, we subtract 5% from home win prob
        injury_adj = injury_analysis.get('net_impact_prob', 0)
        
        # Dampen the adjustment if it's too extreme? 
        # The fetcher already scales it (1 point = 2%). 
        # Let's cap it at +/- 20% to be safe.
        injury_adj = max(-0.20, min(0.20, injury_adj))
        
        home_win_prob_adjusted = home_win_prob + injury_adj
        home_win_prob_adjusted = max(0.05, min(0.95, home_win_prob_adjusted))
        
        result = {
            'home_team': home_team,
            'away_team': away_team,
            'home_win_probability': home_win_prob_adjusted,
            'away_win_probability': 1 - home_win_prob_adjusted,
            'predicted_winner': home_team if home_win_prob_adjusted > 0.5 else away_team,
            'confidence': max(home_win_prob_adjusted, 1 - home_win_prob_adjusted),
            'team_analysis': {
                'home_pts_l5': home_stats.get('PTS_L5', 0),
                'away_pts_l5': away_stats.get('PTS_L5', 0),
                'home_streak': home_stats.get('STREAK', 0),
                'away_streak': away_stats.get('STREAK', 0),
            },
            'player_analysis': player_analysis,
            'injury_analysis': injury_analysis,
            'base_prob': home_win_prob # For debugging/transparency
        }
        
        return result
    
    
    def predict_total(
        self,
        home_team: str,
        away_team: str,
        total_line: float
    ) -> Dict:
        """
        Predict total score probability.
        
        Args:
            home_team: Home team abbreviation
            away_team: Away team abbreviation
            total_line: Bookmaker's total line
            
        Returns:
            Dict with probabilities and weather info
        """
        # Get team stats
        home_stats = self.get_team_recent_stats(home_team)
        away_stats = self.get_team_recent_stats(away_team)
        
        # Default stats if missing
        h_ppg = home_stats.get('PTS_L10', 112.0) if not home_stats.empty else 112.0
        a_ppg = away_stats.get('PTS_L10', 112.0) if not away_stats.empty else 112.0
        
        # Simple heuristic for expected total
        # (Home PPG + Away PPG) / 2 isn't quite right, it's (Home Offense + Away Defense ...)
        # For simplicity: Sum of average scores
        expected_total = (h_ppg + a_ppg) / 2 + (h_ppg + a_ppg) / 2 # Wait, that's just sum
        # Actually expected score = (Home PPG + Away Allowed) / 2...
        # Let's use simple sum of recent scoring averages for now as a baseline
        # (This is a very rough heuristic, Phase 2 ML will improve this)
        expected_total = h_ppg + a_ppg 
        
        # Calculate prob based on difference from line
        # Assuming std dev of ~12 points for NBA totals
        diff = expected_total - total_line
        z_score = diff / 12.0
        
        # Sigmoid to convert to probability
        over_prob = 1 / (1 + np.exp(-z_score))
        
        # Fetch weather (only relevant for outdoor sports like NFL, but code is generic)
        # For NBA this will return "Dome" or None usually
        # But we need full team name for weather fetcher
        # Current fetcher expects full names like "Buffalo Bills"
        # We have abbreviations.
        # We need a lookup.
        
        weather = {}
        # Try to find full name
        home_full = self._teams.get(home_team, {}).get('full_name')
        if home_full:
            weather = self.weather_fetcher.get_weather(home_full, datetime.now()) # Using now for simplicity
        
        return {
            'home_team': home_team,
            'away_team': away_team,
            'total_line': total_line,
            'expected_total': expected_total,
            'over_prob': over_prob,
            'weather': weather
        }

    def predict_total_with_odds(
        self,
        home_team: str,
        away_team: str,
        total_line: float,
        over_odds: int,
        under_odds: int
    ) -> Dict:
        """
        Predict total and evaluate betting value.
        """
        pred = self.predict_total(home_team, away_team, total_line)
        
        # Evaluate
        eval_result = self.evaluator.evaluate_total(
            total=total_line,
            over_prob=pred['over_prob'],
            over_odds=over_odds,
            under_odds=under_odds,
            weather=pred['weather']
        )
        
        pred['betting_analysis'] = {
            'over_ev': eval_result['over_evaluation']['expected_value'],
            'under_ev': eval_result['under_evaluation']['expected_value'],
            'best_bet': eval_result['best_bet']['team'], # "Over X" or "Under X"
            'recommendation': eval_result['best_bet']['recommendation'],
            'recommended_size': eval_result['best_bet']['recommended_bet_size'],
            'note': eval_result['best_bet'].get('note')
        }
        
        return pred

    def predict_with_odds(
        self,
        home_team: str,
        away_team: str,
        home_odds: int,
        away_odds: int
    ) -> Dict:
        """
        Make prediction with betting analysis.
        
        Args:
            home_team: Home team abbreviation
            away_team: Away team abbreviation
            home_odds: American odds for home team
            away_odds: American odds for away team
            
        Returns:
            Prediction with EV analysis
        """
        prediction = self.predict_game(home_team, away_team)
        
        if 'error' in prediction:
            return prediction
        
        # Evaluate betting value
        game_eval = self.evaluator.evaluate_game(
            home_team=home_team,
            away_team=away_team,
            home_prob=prediction['home_win_probability'],
            home_odds=home_odds,
            away_odds=away_odds
        )
        
        prediction['betting_analysis'] = {
            'home_ev': game_eval['home_evaluation']['expected_value'],
            'away_ev': game_eval['away_evaluation']['expected_value'],
            'home_edge': game_eval['home_evaluation']['edge'],
            'away_edge': game_eval['away_evaluation']['edge'],
            'best_bet': game_eval['best_bet']['team'],
            'recommendation': game_eval['best_bet']['recommendation'],
            'recommended_size': game_eval['best_bet']['recommended_bet_size']
        }
        
        return prediction
    
    def generate_predictions_report(
        self,
        games: List[Dict]
    ) -> str:
        """
        Generate a full predictions report.
        
        Args:
            games: List of games with home_team, away_team, and odds
            
        Returns:
            Formatted report string
        """
        lines = []
        lines.append("=" * 70)
        lines.append(f"NBA PREDICTIONS REPORT - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        lines.append("=" * 70)
        lines.append("")
        
        for game in games:
            pred = self.predict_with_odds(
                home_team=game['home_team'],
                away_team=game['away_team'],
                home_odds=game.get('home_odds', -110),
                away_odds=game.get('away_odds', -110)
            )
            
            if 'error' in pred:
                lines.append(f"[ERROR] {game['away_team']} @ {game['home_team']}: {pred['error']}")
                continue
            
            lines.append(f"[GAME] {game['away_team']} @ {game['home_team']}")
            lines.append("-" * 40)
            lines.append(f"  Prediction: {pred['predicted_winner']} wins ({pred['confidence']:.1%} confidence)")
            lines.append(f"  Win Probabilities: {game['home_team']} {pred['home_win_probability']:.1%} | {game['away_team']} {pred['away_win_probability']:.1%}")
            
            # Team analysis
            ta = pred['team_analysis']
            lines.append(f"  Recent Scoring: {game['home_team']} {ta['home_pts_l5']:.1f} PPG | {game['away_team']} {ta['away_pts_l5']:.1f} PPG")
            lines.append(f"  Current Streak: {game['home_team']} {int(ta['home_streak']):+d} | {game['away_team']} {int(ta['away_streak']):+d}")
            
            # Player analysis if available
            if pred['player_analysis']:
                pa = pred['player_analysis']
                lines.append(f"  Star Power Advantage: {pa['star_power_advantage']:+.1f} pts")
                lines.append(f"  Bench Depth Advantage: {pa['bench_depth_advantage']:+.1f} pts")
            
            # Betting analysis
            if 'betting_analysis' in pred:
                ba = pred['betting_analysis']
                lines.append(f"  Betting Value: {game['home_team']} EV {ba['home_ev']:.1%} | {game['away_team']} EV {ba['away_ev']:.1%}")
                
                if ba['recommendation'] == 'BET':
                    lines.append(f"  >>> RECOMMENDED BET: {ba['best_bet']} (${ba['recommended_size']:.2f})")
                else:
                    lines.append(f"  No value bet identified")
            
            lines.append("")
        
        lines.append("=" * 70)
        lines.append("DISCLAIMER: For educational purposes only. Past performance does not")
        lines.append("guarantee future results. Bet responsibly.")
        lines.append("=" * 70)
        
        return "\n".join(lines)


def main():
    """Demo the advanced predictor."""
    print("Initializing Advanced Predictor...")
    predictor = AdvancedPredictor(season="2024-25")
    
    # Sample games to predict
    games = [
        {'home_team': 'LAL', 'away_team': 'GSW', 'home_odds': -130, 'away_odds': +110},
        {'home_team': 'BOS', 'away_team': 'MIA', 'home_odds': -175, 'away_odds': +145},
        {'home_team': 'DEN', 'away_team': 'PHX', 'home_odds': -150, 'away_odds': +125},
    ]
    
    print("\n" + predictor.generate_predictions_report(games))


if __name__ == "__main__":
    main()
