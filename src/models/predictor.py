"""
Predictor Module

Loads trained models and generates predictions for upcoming games.
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from features.team_features import TeamFeatureEngine


class GamePredictor:
    """Makes predictions for upcoming NBA games."""
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        model_dir: str = "./models"
    ):
        """
        Initialize predictor.
        
        Args:
            model_path: Path to specific model file, or None to load latest
            model_dir: Directory containing model files
        """
        self.model_dir = Path(model_dir)
        self.model = None
        self.feature_names = None
        self.model_type = None
        self.feature_engine = TeamFeatureEngine()
        
        if model_path:
            self.load_model(model_path)
        else:
            self._load_latest_model()
    
    def _load_latest_model(self):
        """Load the most recently saved model."""
        model_files = list(self.model_dir.glob("*.joblib"))
        
        if not model_files:
            print("No saved models found. Train a model first.")
            return
        
        # Get most recent by modification time
        latest = max(model_files, key=lambda p: p.stat().st_mtime)
        self.load_model(str(latest))
    
    def load_model(self, model_path: str):
        """Load a specific model file."""
        model_data = joblib.load(model_path)
        
        self.model = model_data['model']
        self.feature_names = model_data['feature_names']
        self.model_type = model_data.get('model_type', 'unknown')
        
        print(f"Loaded model: {Path(model_path).name}")
        print(f"Model type: {self.model_type}")
        print(f"Features: {len(self.feature_names)}")
    
    def predict_game(
        self,
        home_team_stats: pd.Series,
        away_team_stats: pd.Series,
        home_team: str = "Home",
        away_team: str = "Away"
    ) -> Dict:
        """
        Predict outcome for a single game.
        
        Args:
            home_team_stats: Recent stats for home team
            away_team_stats: Recent stats for away team
            home_team: Home team name
            away_team: Away team name
            
        Returns:
            Dictionary with predictions
        """
        if self.model is None:
            raise ValueError("No model loaded. Load a model first.")
        
        # Build features
        features = self.feature_engine.build_matchup_features(
            home_team_stats, 
            away_team_stats
        )
        
        # Create DataFrame with correct feature order
        X = pd.DataFrame([features])
        X = X[self.feature_names].fillna(0)
        
        # Get probabilities
        probs = self.model.predict_proba(X)[0]
        away_win_prob, home_win_prob = probs[0], probs[1]
        
        predicted_winner = home_team if home_win_prob > 0.5 else away_team
        confidence = max(home_win_prob, away_win_prob)
        
        return {
            'home_team': home_team,
            'away_team': away_team,
            'home_win_probability': home_win_prob,
            'away_win_probability': away_win_prob,
            'predicted_winner': predicted_winner,
            'confidence': confidence,
            'features_used': features
        }
    
    def predict_games(
        self,
        games: List[Dict],
        team_stats: Dict[str, pd.Series]
    ) -> pd.DataFrame:
        """
        Predict outcomes for multiple games.
        
        Args:
            games: List of game dicts with 'home_team' and 'away_team'
            team_stats: Dict mapping team names to their recent stats
            
        Returns:
            DataFrame with predictions for all games
        """
        predictions = []
        
        for game in games:
            home_team = game['home_team']
            away_team = game['away_team']
            
            home_stats = team_stats.get(home_team)
            away_stats = team_stats.get(away_team)
            
            if home_stats is None or away_stats is None:
                print(f"Warning: Missing stats for {home_team} vs {away_team}")
                continue
            
            pred = self.predict_game(
                home_stats, away_stats,
                home_team, away_team
            )
            
            pred['game_id'] = game.get('game_id', '')
            pred['game_date'] = game.get('game_date', '')
            predictions.append(pred)
        
        return pd.DataFrame(predictions)
    
    def get_prediction_with_odds(
        self,
        home_team: str,
        away_team: str,
        home_team_stats: pd.Series,
        away_team_stats: pd.Series,
        home_odds: int,
        away_odds: int
    ) -> Dict:
        """
        Get prediction with odds comparison and EV calculation.
        
        Args:
            home_team: Home team name
            away_team: Away team name  
            home_team_stats: Home team recent stats
            away_team_stats: Away team recent stats
            home_odds: American odds for home team
            away_odds: American odds for away team
            
        Returns:
            Dictionary with predictions, EVs, and recommendations
        """
        # Get base prediction
        pred = self.predict_game(
            home_team_stats, 
            away_team_stats,
            home_team,
            away_team
        )
        
        # Calculate implied probabilities from odds
        home_implied = self._american_to_prob(home_odds)
        away_implied = self._american_to_prob(away_odds)
        
        # Calculate edge (model prob - implied prob)
        home_edge = pred['home_win_probability'] - home_implied
        away_edge = pred['away_win_probability'] - away_implied
        
        # Calculate expected value
        home_ev = self._calculate_ev(
            pred['home_win_probability'], 
            home_odds
        )
        away_ev = self._calculate_ev(
            pred['away_win_probability'],
            away_odds
        )
        
        pred.update({
            'home_odds': home_odds,
            'away_odds': away_odds,
            'home_implied_prob': home_implied,
            'away_implied_prob': away_implied,
            'home_edge': home_edge,
            'away_edge': away_edge,
            'home_ev': home_ev,
            'away_ev': away_ev,
            'best_bet': 'home' if home_ev > away_ev else 'away',
            'best_ev': max(home_ev, away_ev)
        })
        
        return pred
    
    def _american_to_prob(self, odds: int) -> float:
        """Convert American odds to probability."""
        if odds < 0:
            return abs(odds) / (abs(odds) + 100)
        else:
            return 100 / (odds + 100)
    
    def _calculate_ev(self, prob: float, odds: int) -> float:
        """
        Calculate expected value of a bet.
        
        EV = (prob * profit) - ((1 - prob) * stake)
        For $100 stake with American odds.
        """
        if odds < 0:
            profit = 100 / abs(odds) * 100  # Profit on $100 bet
        else:
            profit = odds  # Profit on $100 bet
        
        stake = 100
        ev = (prob * profit) - ((1 - prob) * stake)
        
        return ev / stake  # Return as percentage of stake


def main():
    """Demo the predictor."""
    print("GamePredictor Demo")
    print("=" * 50)
    
    predictor = GamePredictor(model_dir="./models")
    
    if predictor.model is None:
        print("\nNo trained model found. Creating demo...")
        
        # Create demo stats
        home_stats = pd.Series({
            'PTS_L5': 115.2, 'PTS_L10': 112.8,
            'REB_L5': 44.0, 'REB_L10': 43.5,
            'AST_L5': 26.5, 'AST_L10': 25.8,
            'HOME_WIN_RATE': 0.65,
            'REST_DAYS': 2, 'STREAK': 3, 'IS_B2B': 0
        })
        
        away_stats = pd.Series({
            'PTS_L5': 108.4, 'PTS_L10': 109.5,
            'REB_L5': 42.0, 'REB_L10': 41.8,
            'AST_L5': 24.0, 'AST_L10': 23.5,
            'AWAY_WIN_RATE': 0.45,
            'REST_DAYS': 1, 'STREAK': -2, 'IS_B2B': 1
        })
        
        print("\nSample matchup features:")
        features = predictor.feature_engine.build_matchup_features(
            home_stats, away_stats
        )
        for k, v in list(features.items())[:5]:
            print(f"  {k}: {v:.2f}" if isinstance(v, float) else f"  {k}: {v}")
        print("  ...")
        
        return
    
    print("\nPrediction system ready!")


if __name__ == "__main__":
    main()
