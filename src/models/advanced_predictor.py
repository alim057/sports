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
import joblib

sys.path.insert(0, str(Path(__file__).parent.parent))


from data.nba_fetcher import NBAFetcher
from data.sport_fetcher import NFLFetcher
from data.player_fetcher import PlayerFetcher
from data.odds_fetcher import OddsFetcher
from data.weather_fetcher import WeatherFetcher
from data.injury_fetcher import InjuryFetcher
from features.team_features import TeamFeatureEngine
from features.nfl_features import NFLFeatureEngine
from features.player_features import PlayerFeatureEngine
from models.predictor import GamePredictor
from betting.evaluator import BettingEvaluator
from data.travel_data import TravelCalculator
from nba_api.stats.static import teams as nba_teams_static


class AdvancedPredictor:
    """Advanced prediction system combining team and player analysis for multiple sports."""
    
    # XGBoost feature names (must match training)
    NBA_MONEYLINE_FEATURES = [
        'pts_diff_l10', 'def_diff_l10', 'win_rate_diff',
        'fg_pct_diff', 'reb_diff', 'ast_diff', 'tov_diff',
        'home_win_l10', 'away_win_l10'
    ]
    
    # NFL Feature names
    NFL_MONEYLINE_FEATURES = [
        'HOME_WIN_RATE', 'AWAY_WIN_RATE', 
        'HOME_PTS_L5', 'AWAY_PTS_L5', 
        'HOME_PA_L5', 'AWAY_PA_L5', 
        'PTS_DIFF_L5', 'PA_DIFF_L5'
    ]
    
    def __init__(
        self,
        sport: str = "NBA",
        model_path: Optional[str] = None,
        season: str = "2024-25",
        min_ev_threshold: float = 0.02,
        bankroll: float = 1000.0
    ):
        """
        Initialize advanced predictor.
        
        Args:
            sport: "NBA" or "NFL"
            model_path: Path to trained model
            season: Current season (e.g. 2024-25 for NBA, 2024 for NFL)
            min_ev_threshold: Minimum EV for bet recommendations
            bankroll: Bankroll for bet sizing
        """
        self.sport = sport.upper()
        self.season = season
        
        # Common Data fetchers
        self.odds_fetcher = OddsFetcher()
        self.weather_fetcher = WeatherFetcher()
        
        # Sport-specific components
        if self.sport == "NBA":
            self.sport_fetcher = NBAFetcher()
            self.player_fetcher = PlayerFetcher()
            self.injury_fetcher = InjuryFetcher()
            self.feature_engine = TeamFeatureEngine()
            self.player_features = PlayerFeatureEngine(season)
            self._teams = {t['abbreviation']: t for t in nba_teams_static.get_teams()}
        elif self.sport == "NFL":
            self.sport_fetcher = NFLFetcher()
            self.player_fetcher = None # NFL Player fetcher Todo
            self.injury_fetcher = None # NFL Injury fetcher Todo
            self.feature_engine = NFLFeatureEngine()
            self.player_features = None
            
            # Load NFL teams cache
            teams_df = self.sport_fetcher.get_all_teams()
            if not teams_df.empty:
                self._teams = teams_df.set_index('team_abbr').to_dict('index')
            else:
                self._teams = {}
        else:
            raise ValueError(f"Unsupported sport: {sport}")

        # Legacy model and evaluator
        self.predictor = GamePredictor(model_path=model_path)
        self.evaluator = BettingEvaluator(
            min_ev_threshold=min_ev_threshold,
            bankroll=bankroll
        )
        
        # Load XGBoost models
        self.xgb_models = self._load_xgboost_models()
        
        # Team lookup cache
        self._team_stats_cache = {}
    
    def _load_xgboost_models(self) -> Dict:
        """Load trained XGBoost models if available."""
        models = {}
        model_dir = Path(__file__).parent.parent.parent / "models"
        
        sport_prefix = self.sport.lower()
        
        for name in ['moneyline', 'spread', 'totals']:
            path = model_dir / f"{sport_prefix}_{name}_xgb.joblib"
            if path.exists():
                try:
                    data = joblib.load(path)
                    models[name] = data
                    print(f"Loaded XGBoost {name} model for {self.sport}")
                except Exception as e:
                    print(f"Warning: Could not load {name} model: {e}")
        
        if not models:
            print(f"No XGBoost models found for {self.sport}, using heuristic fallback")
        
        return models
    
    def _build_xgb_features(self, home_stats: pd.Series, away_stats: pd.Series) -> Optional[Dict]:
        """
        Build feature dictionary for XGBoost model from team stats.
        
        Maps the team stats (from get_team_recent_stats) to the feature names
        expected by the trained XGBoost models.
        """
        try:
            # Delegate to feature engine to get raw named features
            # This ensures consistency between training processing and prediction
            features = self.feature_engine.build_matchup_features(home_stats, away_stats)
            
            # For NBA, we might need to do some extra mapping if the feature engine output
            # doesn't match the XGB model input exactly (legacy issue).
            # But ideally feature_engine.build_matchup_features returns what we need.
            
            if self.sport == "NBA":
               # Map standard FeatureEngine outputs to Legacy NBA Model inputs
               # Model expects: ['pts_diff_l10', 'def_diff_l10', 'win_rate_diff', 'fg_pct_diff', 'reb_diff', 'ast_diff', 'tov_diff', 'home_win_l10', 'away_win_l10']
               # FeatureEngine provides: ['PTS_DIFF_L10', 'HOME_WIN_RATE', 'AWAY_WIN_RATE', 'WIN_RATE_DIFF'...]
               
               mapped_features = {
                   'pts_diff_l10': features.get('PTS_DIFF_L10', 0),
                   # We don't have def_diff directly in new engine, approximate with PTS allowed diff?
                   # Legacy model had 'def_diff'. Let's use PA_DIFF if available or inverse of pts diff
                   'def_diff_l10': -features.get('PTS_DIFF_L10', 0), # Crude approx if PA not avail
                   'win_rate_diff': features.get('WIN_RATE_DIFF', 0),
                   'fg_pct_diff': 0, # Not in new engine yet
                   'reb_diff': features.get('REB_DIFF_L10', 0),
                   'ast_diff': features.get('AST_DIFF_L10', 0),
                   'tov_diff': 0,
                   'home_win_l10': features.get('HOME_WIN_RATE', 0.5),
                   'away_win_l10': features.get('AWAY_WIN_RATE', 0.5)
               }
               return mapped_features 
            
            # For NFL, the feature engine returns exactly what we trained on.
            return features
            
        except Exception as e:
            print(f"Feature building failed: {e}")
            return None
    
    def get_team_id(self, team_abbr: str) -> Optional[int]:
        """Get team ID from abbreviation."""
        if self.sport == "NFL":
            return team_abbr # NFL libraries use abbr as ID usually
            
        team = self._teams.get(team_abbr.upper())
        return team['id'] if team else None
    
    
    def get_team_recent_stats(
        self,
        team_abbr: str,
        n_games: int = 10,
        target_date: Optional[str] = None
    ) -> pd.Series:
        """
        Get recent performance stats for a team.
        """
        if team_abbr in self._team_stats_cache and target_date is None:
            return self._team_stats_cache[team_abbr]
        
        team_id = self.get_team_id(team_abbr)
        # For NFL team_id is the abbr
        if self.sport == 'NFL': 
            team_id = team_abbr

        if not team_id:
            return pd.Series()
        
        try:
            if self.sport == "NBA":
                # Get game log
                game_log = self.sport_fetcher.get_team_game_log(team_id, self.season)
                
                if game_log.empty or len(game_log) < 5:
                    return pd.Series()
                
                # Ensure GAME_DATE is datetime
                game_log['GAME_DATE'] = pd.to_datetime(game_log['GAME_DATE'])
                
                # Calculate rolling stats
                recent = game_log.head(n_games)
                
                # --- Advanced Features: B2B, Rest, Travel ---
                if target_date:
                    game_date = pd.to_datetime(target_date)
                else:
                    game_date = pd.Timestamp.now()
                    
                last_game = game_log.iloc[0]
                last_game_date = last_game['GAME_DATE']
                
                # Days since last game
                days_rest = (game_date - last_game_date).days
                
                # B2B check
                is_b2b = 1 if days_rest == 1 else 0
                
                # Rest days (0 means B2B, 1 means 1 day off, etc.)
                rest_days = max(0, days_rest - 1)
                
                # Travel Distance
                last_game_is_home = last_game['IS_HOME']
                last_opponent = last_game['MATCHUP'].split(' ')[-1]
                
                current_location_abbr = team_abbr if last_game_is_home else last_opponent
                
                stats = {
                    'PTS_L5': game_log['PTS'].head(5).mean(),
                    'PTS_L10': game_log['PTS'].head(10).mean(),
                    'REB_L5': game_log['REB'].head(5).mean(),
                    'REB_L10': game_log['REB'].head(10).mean(),
                    'AST_L5': game_log['AST'].head(5).mean(),
                    'AST_L10': game_log['AST'].head(10).mean(),
                    'HOME_WIN_RATE': (game_log[game_log['IS_HOME']]['WL'] == 'W').mean() if len(game_log[game_log['IS_HOME']]) > 0 else 0.5,
                    'AWAY_WIN_RATE': (game_log[~game_log['IS_HOME']]['WL'] == 'W').mean() if len(game_log[~game_log['IS_HOME']]) > 0 else 0.5,
                    
                    # New Features
                    'REST_DAYS': rest_days,
                    'IS_B2B': is_b2b,
                    'LAST_LOCATION': current_location_abbr, # Returns team abbr of location
                    'STREAK': self._calculate_streak(game_log)
                }
            
            elif self.sport == "NFL":
                # NFL Logic
                # get_team_games returns a df of games for that team
                games_df = self.sport_fetcher.get_team_games(team_id, self.season)
                
                if games_df.empty:
                    return pd.Series()
                
                wins = 0
                total = 0
                
                scores = [] # My scores
                allowed = [] # Opponent scores
                
                for _, game in games_df.iterrows():
                    is_home = (game['home_team'] == team_abbr)
                    my_score = game['home_score'] if is_home else game['away_score']
                    opp_score = game['away_score'] if is_home else game['home_score']
                    
                    if pd.notna(my_score) and pd.notna(opp_score):
                        if my_score > opp_score: wins += 1
                        total += 1
                        scores.append(my_score)
                        allowed.append(opp_score)
                        
                # Stats calculation
                stats = {
                    'WIN_RATE': wins / total if total > 0 else 0.5,
                    'PTS_L5': np.mean(scores[-5:]) if scores else 20.0,
                    'PA_L5': np.mean(allowed[-5:]) if allowed else 20.0, # Added Points Allowed
                    'LAST_LOCATION': team_abbr # simplistic
                }
            
            result = pd.Series(stats)
            if target_date is None:
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
        # Get rolling stats for both teams
        try:
            # Pass target_date to get correct B2B/Rest calculation
            today = datetime.now().strftime('%Y-%m-%d')
            home_stats = self.get_team_recent_stats(home_team, n_games=10, target_date=today)
            away_stats = self.get_team_recent_stats(away_team, n_games=10, target_date=today)
            
            if home_stats.empty or away_stats.empty:
                return {'error': 'Could not fetch team stats'}

            # Calculate Travel Distance for this specific matchup
            # Home team travels from their last location to their Home stadium
            home_last_loc = home_stats.get('LAST_LOCATION', home_team)
            home_travel_dist = TravelCalculator.calculate_distance(home_last_loc, home_team)
            
            # Away team travels from their last location to Home team's stadium
            away_last_loc = away_stats.get('LAST_LOCATION', away_team)
            away_travel_dist = TravelCalculator.calculate_distance(away_last_loc, home_team)
            
            # Add travel to stats
            home_stats['TRAVEL_DIST'] = home_travel_dist
            away_stats['TRAVEL_DIST'] = away_travel_dist
            
            
            # Combine into features dictionary
            # feature_engine is either TeamFeatureEngine or NFLFeatureEngine based on init
            team_feats = self.feature_engine.build_matchup_features(home_stats, away_stats)
            
            # Add new situational features to dictionary for transparency
            team_feats['HOME_REST_DAYS'] = home_stats.get('REST_DAYS', 2)
            team_feats['AWAY_REST_DAYS'] = away_stats.get('REST_DAYS', 2)
            team_feats['HOME_IS_B2B'] = home_stats.get('IS_B2B', 0)
            team_feats['AWAY_IS_B2B'] = away_stats.get('IS_B2B', 0)
            team_feats['HOME_TRAVEL'] = home_travel_dist
            team_feats['AWAY_TRAVEL'] = away_travel_dist
            
            # Calculate situational advantage
            # Positive = Advantage for Home Team
            rest_diff = team_feats['HOME_REST_DAYS'] - team_feats['AWAY_REST_DAYS']
            b2b_advantage = team_feats['AWAY_IS_B2B'] - team_feats['HOME_IS_B2B'] # 1 if Away is B2B, -1 if Home is B2B
            travel_advantage = (away_travel_dist - home_travel_dist) / 1000.0 # 1 point per 1000 miles diff
            
            team_feats['REST_ADVANTAGE'] = rest_diff + b2b_advantage + travel_advantage

            # Get player analysis if enabled and supported
            player_analysis = {}
            if include_player_features and self.player_fetcher:
                try:
                    # Get active roster (could filter by injuries here too)
                    home_roster = self.sport_fetcher.get_team_roster(self.get_team_id(home_team))
                    away_roster = self.sport_fetcher.get_team_roster(self.get_team_id(away_team))
                    
                    # Fetch recent stats for top 8 players
                    # ... implementation details ...
                    pass
                except Exception as e:
                    print(f"Warning: Player feature error: {e}")
            
            # Get Injury Analysis
            injury_analysis = {}
            if self.injury_fetcher:
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
            else:
                 injury_analysis = {'net_impact_prob': 0}

            # Make prediction using XGBoost model if available
            home_win_prob = None
            model_used = 'heuristic'
            
            if 'moneyline' in self.xgb_models:
                try:
                    # Build XGBoost feature vector from team stats
                    # Note: Currently XGBoost model was trained WITHOUT new features (Travel/Rest)
                    # So we only feed it what it knows, and apply heuristic adjustments AFTER.
                    xgb_features = self._build_xgb_features(home_stats, away_stats)
                    
                    if xgb_features is not None:
                        model_data = self.xgb_models['moneyline']
                        model = model_data['model']
                        
                        # Get feature names from model or use defaults based on sport
                        if self.sport == "NBA":
                            feature_names = model_data.get('features', self.NBA_MONEYLINE_FEATURES)
                        else:
                            feature_names = model_data.get('features', self.NFL_MONEYLINE_FEATURES)
                        
                        # Create DataFrame with correct feature order
                        X = pd.DataFrame([xgb_features])
                        # Filter/Order columns only if X has them
                        available_cols = [c for c in feature_names if c in X.columns]
                        if len(available_cols) == len(feature_names):
                             X = X[feature_names]
                        
                        home_win_prob = model.predict_proba(X)[0][1]
                        model_used = 'xgboost'
                except Exception as e:
                    print(f"XGBoost prediction failed: {e}")
            
            # Fallback: simple heuristic based on features
            if home_win_prob is None:
                home_win_prob = 0.5 + (
                    team_feats.get('PTS_DIFF_L5', 0) * 0.01 +
                    team_feats.get('WIN_RATE_DIFF', 0) * 0.3 +
                    team_feats.get('REST_ADVANTAGE', 0) * 0.02 +
                    0.03  # Home court advantage
                )
                home_win_prob = max(0.1, min(0.9, home_win_prob))
                
            # If using XGBoost, apply Situational Modifiers (since model doesn't know them yet)
            if model_used == 'xgboost':
                # 2% shift for significant rest advantage
                situation_modifier = team_feats.get('REST_ADVANTAGE', 0) * 0.02 
                home_win_prob += situation_modifier
                home_win_prob = max(0.05, min(0.95, home_win_prob))
            
            # Apply Injury Adjustment
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
                'model_used': model_used,
                'team_analysis': {
                    'home_pts_l5': home_stats.get('PTS_L5', 0),
                    'away_pts_l5': away_stats.get('PTS_L5', 0),
                    'home_streak': home_stats.get('STREAK', 0),
                    'away_streak': away_stats.get('STREAK', 0),
                },
                'player_analysis': player_analysis,
                'injury_analysis': injury_analysis,
                'situational_analysis': {
                    'rest_advantage': team_feats.get('REST_ADVANTAGE', 0),
                    'home_travel': home_travel_dist,
                    'away_travel': away_travel_dist
                },
                'base_prob': home_win_prob # For debugging/transparency
            }
            
            return result
            
        except Exception as e:
            print(f"Prediction error for {home_team} vs {away_team}: {e}")
            return {'error': str(e)}
    
    
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
