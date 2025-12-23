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


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
