import sys
import os
from pathlib import Path

# Add src to python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from models.advanced_predictor import AdvancedPredictor

def test_prediction_with_injuries():
    print("Initializing AdvancedPredictor...")
    predictor = AdvancedPredictor()
    
    home_team = 'LAL'
    away_team = 'GSW'
    
    print(f"\nPredicting {away_team} @ {home_team}...")
    result = predictor.predict_game(home_team, away_team)
    
    print("\nPrediction Result:")
    print(f"Home Win Prob (Adjusted): {result['home_win_probability']:.1%}")
    print(f"Base Prob (Before Adjustment): {result.get('base_prob', 'N/A')}")
    
    injury_analysis = result.get('injury_analysis', {})
    print("\nInjury Analysis:")
    print(f"Home Impact: {injury_analysis.get('home_impact')}")
    print(f"Away Impact: {injury_analysis.get('away_impact')}")
    print(f"Net Impact (Home Advantage): {injury_analysis.get('net_impact_prob')}")
    
    # Verification
    if 'injury_analysis' in result and result['injury_analysis'].get('net_impact_prob') is not None:
        print("\nSUCCESS: Injury analysis included in prediction.")
        
        # Verify adjustment logic
        base = result.get('base_prob')
        adj = result['home_win_probability']
        impact = result['injury_analysis']['net_impact_prob']
        
        # Cap impact as per logic
        impact = max(-0.20, min(0.20, impact))
        
        expected_adj = base + impact
        # Floating point tolerance
        if abs(adj - expected_adj) < 0.001:
             print("SUCCESS: Win probability correctly adjusted.")
        else:
             print(f"FAILURE: Win probability adjustment mismatch. Expected {expected_adj}, got {adj}")
    else:
        print("\nFAILURE: Injury analysis missing from result.")

if __name__ == "__main__":
    test_prediction_with_injuries()
