
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from models.advanced_predictor import AdvancedPredictor

def test_multisport():
    print("Testing Multi-Sport Capabilities...")
    
    # 1. Test NBA (Regression Test)
    print("\n--- Testing NBA Predictor ---")
    try:
        nba_predictor = AdvancedPredictor(sport="NBA")
        print("[OK] Initialized NBA Predictor")
        
        # Test basic prediction
        pred = nba_predictor.predict_game("LAL", "GSW")
        if 'error' not in pred:
            print(f"[OK] NBA Prediction successful: {pred['predicted_winner']} ({pred['confidence']:.1%})")
            print(f"     Model used: {pred.get('model_used', 'unknown')}")
        else:
            print(f"[FAIL] NBA Prediction error: {pred['error']}")
            
    except Exception as e:
        print(f"[FAIL] NBA Predictor crashed: {e}")

    # 2. Test NFL (New Feature)
    print("\n--- Testing NFL Predictor ---")
    try:
        nfl_predictor = AdvancedPredictor(sport="NFL", season="2023") # Use 2023 since we know we have 2023 schedule in fallback
        print("[OK] Initialized NFL Predictor")
        
        # Test basic prediction
        # KC vs DET was opening game of 2023
        pred = nfl_predictor.predict_game("KC", "DET")
        if 'error' not in pred:
            print(f"[OK] NFL Prediction successful: {pred['predicted_winner']} ({pred['confidence']:.1%})")
            print(f"     Model used: {pred.get('model_used', 'unknown')}")
            print(f"     Win Prob: {pred['home_win_probability']:.1%}")
            print(f"     Situational: {pred['situational_analysis']}")
            print(f"     Team Analysis: {pred['team_analysis']}")
        else:
            print(f"[FAIL] NFL Prediction error: {pred['error']}")
            
    except Exception as e:
        print(f"[FAIL] NFL Predictor crashed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_multisport()
