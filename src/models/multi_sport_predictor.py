"""
Multi-Sport Predictor

Loads sport-specific models and makes predictions for any supported sport.
"""

import joblib
import numpy as np
from pathlib import Path
from typing import Dict, Optional, List
from datetime import datetime


class MultiSportPredictor:
    """Predictor that loads and uses sport-specific models."""
    
    SPORT_MODELS = {
        'nba': 'xgboost_*.joblib',
        'nfl': 'nfl_model.joblib',
        'nhl': 'nhl_model.joblib',
        'ncaaf': 'ncaaf_model.joblib',
        'mlb': 'mlb_model.joblib'
    }
    
    SPORT_FEATURES = {
        'nba': ['home_win_pct', 'away_win_pct', 'pts_diff', 'home_streak', 'away_streak'],
        'nfl': ['home_win_pct', 'away_win_pct', 'home_ppg', 'away_ppg', 
                'home_ppg_allowed', 'away_ppg_allowed', 'win_pct_diff', 'point_diff'],
        'nhl': ['home_win_pct', 'away_win_pct', 'home_goals_avg', 'away_goals_avg',
                'home_goals_allowed', 'away_goals_allowed', 'win_pct_diff', 'goal_diff'],
        'ncaaf': ['home_win_pct', 'away_win_pct', 'home_ppg', 'away_ppg',
                  'home_ppg_allowed', 'away_ppg_allowed', 'win_pct_diff', 'point_diff'],
        'mlb': ['home_win_pct', 'away_win_pct', 'home_runs_avg', 'away_runs_avg',
                'home_runs_allowed', 'away_runs_allowed', 'win_pct_diff', 'run_diff']
    }
    
    def __init__(self, models_dir: str = 'models'):
        """Initialize with models directory."""
        self.models_dir = Path(models_dir)
        self.loaded_models = {}
        self._load_all_models()
    
    def _load_all_models(self):
        """Load all available sport models."""
        for sport, pattern in self.SPORT_MODELS.items():
            try:
                if '*' in pattern:
                    # Find matching files (for NBA which has timestamp)
                    matches = list(self.models_dir.glob(pattern))
                    if matches:
                        model_path = sorted(matches)[-1]  # Most recent
                    else:
                        continue
                else:
                    model_path = self.models_dir / pattern
                    
                if model_path.exists():
                    data = joblib.load(model_path)
                    if isinstance(data, dict):
                        self.loaded_models[sport] = {
                            'model': data.get('model'),
                            'features': data.get('features', data.get('feature_names', []))
                        }
                    else:
                        # Raw model without metadata
                        self.loaded_models[sport] = {
                            'model': data,
                            'features': self.SPORT_FEATURES.get(sport, [])
                        }
                    print(f"Loaded {sport.upper()} model from {model_path.name}")
            except Exception as e:
                print(f"Could not load {sport} model: {e}")
    
    def get_available_sports(self) -> List[str]:
        """Get list of sports with loaded models."""
        return list(self.loaded_models.keys())
    
    def predict(
        self,
        sport: str,
        home_team: str,
        away_team: str,
        home_stats: Optional[Dict] = None,
        away_stats: Optional[Dict] = None
    ) -> Dict:
        """
        Make prediction for a game.
        
        Args:
            sport: Sport type (nba, nfl, nhl, ncaaf, mlb)
            home_team: Home team abbreviation
            away_team: Away team abbreviation
            home_stats: Optional dict with team stats
            away_stats: Optional dict with team stats
            
        Returns:
            Prediction dictionary
        """
        sport = sport.lower()
        
        if sport not in self.loaded_models:
            return {
                'error': f'No model loaded for {sport}',
                'available_sports': self.get_available_sports()
            }
        
        model_data = self.loaded_models[sport]
        model = model_data['model']
        features = model_data['features']
        
        # Build feature vector from stats
        if home_stats and away_stats:
            feature_vector = self._build_features(sport, home_stats, away_stats, features)
        else:
            # Use defaults if no stats provided
            feature_vector = self._get_default_features(sport, features)
        
        try:
            # Reshape for prediction
            X = np.array(feature_vector).reshape(1, -1)
            
            # Get probability
            proba = model.predict_proba(X)[0]
            home_win_prob = proba[1] if len(proba) > 1 else proba[0]
            
            return {
                'sport': sport.upper(),
                'home_team': home_team,
                'away_team': away_team,
                'home_win_probability': float(home_win_prob),
                'away_win_probability': float(1 - home_win_prob),
                'predicted_winner': home_team if home_win_prob > 0.5 else away_team,
                'confidence': float(max(home_win_prob, 1 - home_win_prob)),
                'model_used': sport
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def _build_features(
        self,
        sport: str,
        home_stats: Dict,
        away_stats: Dict,
        features: List[str]
    ) -> List[float]:
        """Build feature vector from team stats."""
        values = []
        
        for feat in features:
            if 'home' in feat.lower():
                key = feat.replace('home_', '')
                values.append(home_stats.get(key, home_stats.get(feat, 0.5)))
            elif 'away' in feat.lower():
                key = feat.replace('away_', '')
                values.append(away_stats.get(key, away_stats.get(feat, 0.5)))
            elif 'diff' in feat.lower():
                # Calculate difference
                if 'win_pct' in feat:
                    values.append(home_stats.get('win_pct', 0.5) - away_stats.get('win_pct', 0.5))
                elif 'point' in feat or 'pts' in feat:
                    values.append(home_stats.get('ppg', 0) - away_stats.get('ppg', 0))
                elif 'goal' in feat:
                    values.append(home_stats.get('goals_avg', 0) - away_stats.get('goals_avg', 0))
                elif 'run' in feat:
                    values.append(home_stats.get('runs_avg', 0) - away_stats.get('runs_avg', 0))
                else:
                    values.append(0)
            else:
                values.append(0.5)
        
        return values
    
    def _get_default_features(self, sport: str, features: List[str]) -> List[float]:
        """Get default feature values (for when no stats provided)."""
        defaults = {
            'nba': [0.5, 0.5, 0, 0, 0],
            'nfl': [0.5, 0.5, 21, 21, 21, 21, 0, 0],
            'nhl': [0.5, 0.5, 3, 3, 3, 3, 0, 0],
            'ncaaf': [0.5, 0.5, 28, 28, 24, 24, 0, 0],
            'mlb': [0.5, 0.5, 4.5, 4.5, 4.5, 4.5, 0, 0]
        }
        return defaults.get(sport, [0.5] * len(features))
    
    def calculate_ev(
        self,
        home_prob: float,
        home_odds: int,
        away_odds: int
    ) -> Dict:
        """Calculate expected value for both sides."""
        def odds_to_implied(odds: int) -> float:
            if odds > 0:
                return 100 / (odds + 100)
            else:
                return abs(odds) / (abs(odds) + 100)
        
        def calculate_expected_value(prob: float, odds: int) -> float:
            implied = odds_to_implied(odds)
            if odds > 0:
                payout = odds / 100
            else:
                payout = 100 / abs(odds)
            return prob * payout - (1 - prob)
        
        home_ev = calculate_expected_value(home_prob, home_odds)
        away_ev = calculate_expected_value(1 - home_prob, away_odds)
        
        return {
            'home_ev': home_ev,
            'away_ev': away_ev,
            'home_implied': odds_to_implied(home_odds),
            'away_implied': odds_to_implied(away_odds),
            'best_bet': 'home' if home_ev > away_ev else 'away',
            'has_edge': max(home_ev, away_ev) > 0
        }


def main():
    """Test multi-sport predictor."""
    predictor = MultiSportPredictor()
    
    print("\nAvailable sports:", predictor.get_available_sports())
    
    # Test each sport
    tests = [
        ('nba', 'LAL', 'BOS'),
        ('nfl', 'KC', 'SF'),
        ('nhl', 'TOR', 'MTL'),
        ('ncaaf', 'OSU', 'MICH'),
    ]
    
    for sport, home, away in tests:
        result = predictor.predict(sport, home, away)
        if 'error' not in result:
            print(f"\n{sport.upper()}: {away} @ {home}")
            print(f"  Winner: {result['predicted_winner']} ({result['confidence']:.1%})")
        else:
            print(f"\n{sport.upper()}: {result['error']}")


if __name__ == '__main__':
    main()
