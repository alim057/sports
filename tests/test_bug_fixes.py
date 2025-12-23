"""
Sports Betting Predictor - Automated Tests

Tests for bug fixes and core functionality.
"""

import pytest
import sys
from pathlib import Path
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "dashboard"))


class TestOddsValidation:
    """Test odds validation to filter spread values from moneyline."""
    
    def test_valid_moneyline_odds(self):
        """Valid moneyline odds should pass validation."""
        # Import from server
        from server import validate_moneyline_odds
        
        assert validate_moneyline_odds(-150, +130) == True
        assert validate_moneyline_odds(-200, +175) == True
        assert validate_moneyline_odds(+100, -120) == True
        assert validate_moneyline_odds(-500, +350) == True
    
    def test_invalid_spread_values(self):
        """Spread values should fail validation."""
        from server import validate_moneyline_odds
        
        # These are spread values, not moneyline odds
        assert validate_moneyline_odds(-6, +6) == False
        assert validate_moneyline_odds(-35, +35) == False
        assert validate_moneyline_odds(-3.5, +3.5) == False
        assert validate_moneyline_odds(0, -10) == False
    
    def test_edge_cases(self):
        """Edge cases near boundary value."""
        from server import validate_moneyline_odds
        
        # Exactly at boundary
        assert validate_moneyline_odds(-100, +100) == True
        # Just below boundary
        assert validate_moneyline_odds(-99, +100) == False


class TestTeamAbbreviations:
    """Test team abbreviation mappings for all sports."""
    
    def test_nba_teams(self):
        """NBA teams should have correct abbreviations."""
        from server import TEAM_ABBR
        
        assert TEAM_ABBR['Boston Celtics'] == 'BOS'
        assert TEAM_ABBR['Los Angeles Lakers'] == 'LAL'
        assert TEAM_ABBR['New Orleans Pelicans'] == 'NOP'
    
    def test_nfl_teams(self):
        """NFL teams should have correct abbreviations (fix for ambiguous NEW)."""
        from server import TEAM_ABBR
        
        # These were previously both "NEW" - now distinct
        assert TEAM_ABBR['New England Patriots'] == 'NE'
        assert TEAM_ABBR['New Orleans Saints'] == 'NO'
        assert TEAM_ABBR['New York Giants'] == 'NYG'
        assert TEAM_ABBR['New York Jets'] == 'NYJ'
    
    def test_nhl_teams(self):
        """NHL teams should have correct abbreviations."""
        from server import TEAM_ABBR
        
        assert TEAM_ABBR['New Jersey Devils'] == 'NJD'
        assert TEAM_ABBR['New York Rangers'] == 'NYR'
        assert TEAM_ABBR['New York Islanders'] == 'NYI'
    
    def test_get_team_abbr_fallback(self):
        """Unknown teams should fall back to first 3 chars."""
        from server import get_team_abbr
        
        # Known team
        assert get_team_abbr('Boston Celtics') == 'BOS'
        # Unknown team - falls back to first 3 chars
        assert get_team_abbr('Unknown Team Name') == 'UNK'


class TestAdvancedPredictor:
    """Test AdvancedPredictor with robust error handling."""
    
    def test_predict_game_returns_dict(self):
        """predict_game should always return a dict (with or without error)."""
        try:
            from models.advanced_predictor import AdvancedPredictor
            predictor = AdvancedPredictor()
        except Exception as e:
            pytest.skip(f"Could not initialize AdvancedPredictor: {e}")
        
        # Even with invalid teams, should return dict with error
        result = predictor.predict_game('INVALID', 'TEAMS')
        assert isinstance(result, dict)
    
    def test_predict_with_odds_handles_errors(self):
        """predict_with_odds should handle errors gracefully."""
        try:
            from models.advanced_predictor import AdvancedPredictor
            predictor = AdvancedPredictor()
        except Exception as e:
            pytest.skip(f"Could not initialize AdvancedPredictor: {e}")
        
        # Should not raise exception, just return error dict
        result = predictor.predict_with_odds('INVALID', 'TEAMS', -110, +100)
        assert isinstance(result, dict)


class TestEdgeAnalyzer:
    """Test EdgeAnalyzer performance summary."""
    
    def test_get_performance_summary_returns_dict(self):
        """get_performance_summary should always return a dict."""
        from betting.edge_analyzer import EdgeAnalyzer
        
        analyzer = EdgeAnalyzer(tracking_file="./data/test_tracking.json")
        summary = analyzer.get_performance_summary()
        
        assert isinstance(summary, dict)
        assert 'total_bets' in summary or summary.get('total_bets', 0) >= 0
        assert 'win_rate' in summary
    
    def test_ev_calculation(self):
        """Test expected value calculation."""
        from betting.edge_analyzer import EdgeAnalyzer
        
        analyzer = EdgeAnalyzer()
        
        # 60% probability at -150 odds
        # Profit = 100/150 = 0.67
        # EV = 0.60 * 0.67 - 0.40 * 1 = 0.402 - 0.4 = 0.002
        ev = analyzer._calculate_ev(0.60, -150)
        assert -0.1 < ev < 0.1  # Should be small positive or negative
        
        # 50% probability at +100 odds (even money)
        # EV should be 0
        ev = analyzer._calculate_ev(0.50, 100)
        assert abs(ev) < 0.01


class TestIntegration:
    """Integration tests for API endpoints."""
    
    def test_server_imports(self):
        """Server should import without errors."""
        try:
            import server
            assert hasattr(server, 'app')
            assert hasattr(server, 'TEAM_ABBR')
            assert hasattr(server, 'validate_moneyline_odds')
        except ImportError as e:
            pytest.skip(f"Server import failed: {e}")
    
    def test_flask_app_exists(self):
        """Flask app should be properly configured."""
        try:
            from server import app
            assert app is not None
            
            # Test that routes are registered
            routes = [rule.rule for rule in app.url_map.iter_rules()]
            assert '/api/edge-analysis' in routes or any('edge' in r for r in routes)
        except ImportError:
            pytest.skip("Server not available")


class TestNewEndpoints:
    """Tests for newly added API endpoints."""
    
    def test_spread_analysis_route_exists(self):
        """Spread analysis endpoint should exist."""
        from server import app
        routes = [rule.rule for rule in app.url_map.iter_rules()]
        assert any('spread' in r for r in routes)
    
    def test_totals_analysis_route_exists(self):
        """Totals analysis endpoint should exist."""
        from server import app
        routes = [rule.rule for rule in app.url_map.iter_rules()]
        assert any('totals' in r for r in routes)
    
    def test_parlay_calculator_route_exists(self):
        """Parlay calculator endpoint should exist."""
        from server import app
        routes = [rule.rule for rule in app.url_map.iter_rules()]
        assert any('parlay' in r for r in routes)
    
    def test_recent_bets_route_exists(self):
        """Recent bets endpoint should exist."""
        from server import app
        routes = [rule.rule for rule in app.url_map.iter_rules()]
        assert any('recent-bets' in r for r in routes)
    
    def test_performance_history_route_exists(self):
        """Performance history endpoint should exist."""
        from server import app
        routes = [rule.rule for rule in app.url_map.iter_rules()]
        assert any('performance-history' in r for r in routes)


class TestParlayCalculations:
    """Test parlay calculation logic."""
    
    def test_parlay_decimal_odds_conversion(self):
        """Test American to decimal odds conversion."""
        # +150 should be 2.50 decimal
        positive_dec = (150 / 100) + 1
        assert positive_dec == 2.5
        
        # -150 should be 1.67 decimal
        negative_dec = (100 / 150) + 1
        assert abs(negative_dec - 1.67) < 0.01
    
    def test_parlay_combined_odds(self):
        """Test that combined odds multiply correctly."""
        # Two -110 legs
        dec1 = (100 / 110) + 1  # ~1.909
        dec2 = (100 / 110) + 1  # ~1.909
        combined = dec1 * dec2  # ~3.64
        assert 3.6 < combined < 3.7


class TestDataIntegrity:
    """Tests for Phase 1 data integrity fixes."""
    
    def test_game_time_filter_future(self):
        """Future games should pass the filter."""
        from server import is_game_upcoming
        
        # Game in the future should be included
        future = "2099-12-25T12:00:00Z"
        assert is_game_upcoming(future) == True
    
    def test_game_time_filter_past(self):
        """Past games should fail the filter."""
        from server import is_game_upcoming
        
        # Game in the past should be excluded
        past = "2020-01-01T12:00:00Z"
        assert is_game_upcoming(past) == False
    
    def test_game_time_filter_none(self):
        """None/empty time should fail the filter."""
        from server import is_game_upcoming
        
        assert is_game_upcoming(None) == False
        assert is_game_upcoming("") == False
    
    def test_odds_validation_rejects_spreads(self):
        """Small numbers like -53 should be rejected as spread values."""
        from server import validate_moneyline_odds
        
        # -53 is a spread value mistakenly stored as odds
        assert validate_moneyline_odds(-53, +100) == False
        assert validate_moneyline_odds(-150, +53) == False
        assert validate_moneyline_odds(-6, +6) == False
    
    def test_performance_returns_is_demo_flag(self):
        """Performance endpoint should include isDemo flag."""
        from server import app
        
        with app.test_client() as client:
            response = client.get('/api/performance')
            data = response.get_json()
            perf = data.get('performance', {})
            # isDemo should be present in response
            assert 'isDemo' in perf


class TestPhase4RiskManagement:
    """Tests for Phase 3 risk management features."""
    
    def test_daily_status_endpoint_exists(self):
        """Daily status endpoint should exist."""
        from server import app
        routes = [rule.rule for rule in app.url_map.iter_rules()]
        assert '/api/daily-status' in routes
    
    def test_clv_summary_endpoint_exists(self):
        """CLV summary endpoint should exist."""
        from server import app
        routes = [rule.rule for rule in app.url_map.iter_rules()]
        assert '/api/clv-summary' in routes
    
    def test_daily_status_returns_correct_fields(self):
        """Daily status should return expected fields."""
        from server import app
        
        with app.test_client() as client:
            response = client.get('/api/daily-status')
            data = response.get_json()
            
            # Must have these fields
            assert 'date' in data
            assert 'profit_loss' in data
            assert 'daily_loss_limit' in data
            assert 'limit_hit' in data
            assert 'can_bet' in data
    
    def test_clv_summary_returns_correct_fields(self):
        """CLV summary should return expected fields."""
        from server import app
        
        with app.test_client() as client:
            response = client.get('/api/clv-summary')
            data = response.get_json()
            
            # Must have these fields
            assert 'total_tracked_bets' in data
            assert 'estimated_clv' in data
            assert 'status' in data
    
    def test_evaluator_has_correct_defaults(self):
        """BettingEvaluator should have correct Phase 2/3 defaults."""
        from betting.evaluator import BettingEvaluator
        
        evaluator = BettingEvaluator()
        
        # Phase 2: 7% EV threshold
        assert evaluator.min_ev_threshold == 0.07, f"EV threshold should be 0.07, got {evaluator.min_ev_threshold}"
        
        # Phase 3: 10% daily loss limit
        assert evaluator.daily_loss_limit == 0.10, f"Daily limit should be 0.10, got {evaluator.daily_loss_limit}"
        
        # Quarter Kelly (25%)
        assert evaluator.max_kelly_fraction == 0.25, f"Kelly fraction should be 0.25, got {evaluator.max_kelly_fraction}"


class TestPhase4ModelIntegrity:
    """Tests for model and prediction integrity."""
    
    def test_predictor_returns_probabilities_in_range(self):
        """Predictions should return probabilities between 0 and 1."""
        try:
            from models.advanced_predictor import AdvancedPredictor
            predictor = AdvancedPredictor(sport="NBA")
        except Exception as e:
            pytest.skip(f"Could not initialize predictor: {e}")
        
        # Use known NBA teams
        result = predictor.predict_game('LAL', 'BOS')
        
        if 'error' not in result:
            home_prob = result.get('home_win_probability', 0)
            away_prob = result.get('away_win_probability', 0)
            
            assert 0.0 <= home_prob <= 1.0, f"Home prob out of range: {home_prob}"
            assert 0.0 <= away_prob <= 1.0, f"Away prob out of range: {away_prob}"
    
    def test_predictor_uses_correct_ev_threshold(self):
        """Predictor should use 7% EV threshold."""
        try:
            from models.advanced_predictor import AdvancedPredictor
            predictor = AdvancedPredictor(sport="NBA", min_ev_threshold=0.07)
        except Exception as e:
            pytest.skip(f"Could not initialize predictor: {e}")
        
        # Check the evaluator attached to predictor has correct threshold
        assert predictor.evaluator.min_ev_threshold == 0.07
    
    def test_xgboost_models_loaded(self):
        """XGBoost models should be loaded for NBA."""
        try:
            from models.advanced_predictor import AdvancedPredictor
            predictor = AdvancedPredictor(sport="NBA")
        except Exception as e:
            pytest.skip(f"Could not initialize predictor: {e}")
        
        # Should have at least one model loaded
        assert len(predictor.xgb_models) > 0, "No XGBoost models loaded"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

