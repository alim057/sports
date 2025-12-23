import sys
import os
import pytest
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.append(os.getcwd())

from src.models.advanced_predictor import AdvancedPredictor
from src.betting.evaluator import BettingEvaluator
from src.data.live_odds import LiveOddsFetcher
from src.betting.bet_tracker import BetTracker

def print_header(title):
    print("\n" + "=" * 60)
    print(f" {title}")
    print("=" * 60)

def print_status(check, passed, message=""):
    status = "[PASS]" if passed else "[FAIL]"
    print(f"{status} | {check:<30} | {message}")
    if not passed:
        print(f"   >>> CRITICAL: {message}")
        return False
    return True

def run_verification():
    print_header("SPORTS BETTING SYSTEM - HEALTH CHECK")
    print(f"Timestamp: {datetime.now().isoformat()}")
    
    all_passed = True
    
    # 1. Unit Tests
    print("\n--- 1. Running Test Suite (pytest) ---")
    # Capturing output is tricky with pytest inside script, so we verify return code
    retcode = pytest.main(["tests/test_bug_fixes.py", "-v", "--tb=short"])
    if not print_status("Unit Tests", retcode == 0, f"Pytest exit code: {retcode}"):
        all_passed = False

    # 2. Configuration Validation
    print("\n--- 2. Validating Configurations ---")
    
    try:
        evaluator = BettingEvaluator()
        
        # Check EV Threshold (7%)
        ev_pass = evaluator.min_ev_threshold == 0.07
        if not print_status("Min EV Threshold (7%)", ev_pass, f"Current: {evaluator.min_ev_threshold:.2%}"):
            all_passed = False
            
        # Check Daily Loss Limit (10%)
        loss_pass = evaluator.daily_loss_limit == 0.10
        if not print_status("Daily Loss Limit (10%)", loss_pass, f"Current: {evaluator.daily_loss_limit:.1%}"):
            all_passed = False
            
    except Exception as e:
        print_status("Config Validation", False, str(e))
        all_passed = False

    # 3. Model Health Check
    print("\n--- 3. Checking Prediction Models ---")
    try:
        # Check NBA Predictor
        predictor = AdvancedPredictor(sport="NBA", min_ev_threshold=0.07)
        models_loaded = len(predictor.xgb_models) > 0
        if not print_status("XGBoost Models Loaded", models_loaded, f"Models: {list(predictor.xgb_models.keys())}"):
            all_passed = False
            
        # Basic prediction sanity check (mock data)
        # We don't want to call API here, just check method existence
        has_predict = hasattr(predictor, 'predict_with_odds')
        print_status("Predictor Integrity", has_predict, "Method exists")
        
    except Exception as e:
        print_status("Model Health", False, str(e))
        all_passed = False

    # 4. Database & API Check
    print("\n--- 4. Infrastructure Checks ---")
    try:
        # DB Check
        tracker = BetTracker()
        tracker_pass = os.path.exists(tracker.db_path)
        print_status("Database Exists", tracker_pass, f"Path: {tracker.db_path}")
        
        # API Key Presence
        # Note: We avoid making actual API calls to save quota/time, just check key presence
        from src.betting.daily_edge import API_KEY
        key_pass = API_KEY is not None and len(API_KEY) > 10
        print_status("API Key Configured", key_pass, "Key found in daily_edge.py")
        
    except Exception as e:
        print_status("Infrastructure", False, str(e))
        all_passed = False

    # Final Summary
    print_header("VERIFICATION SUMMARY")
    if all_passed:
        print("\n[PASS] SYSTEM IS HEALTHY. READY FOR PAPER TRADING.")
        return 0
    else:
        print("\n[FAIL] SYSTEM HEALTH CHECK FAILED. FIX ISSUES BEFORE TRADING.")
        return 1

if __name__ == "__main__":
    sys.exit(run_verification())
