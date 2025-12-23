
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from models.advanced_predictor import AdvancedPredictor

def evaluate_bet():
    print("Evaluating DEN @ UTA Total 249.5...")
    predictor = AdvancedPredictor(sport="NBA")
    
    # User Request: Nuggets/Jazz Over 249.5
    # Assuming Jazz is home (UTA) vs Nuggets (DEN) or vice versa.
    # User said "Nuggets/Jazz", usually Away/Home convention or just names.
    # Checking schedule from API would be best, but I'll assume standard lookup.
    # Let's try DEN away @ UTA home.
    
    home = "UTA"
    away = "DEN"
    total_line = 249.5
    over_odds = -110
    under_odds = -110
    
    try:
        pred = predictor.predict_total_with_odds(home, away, total_line, over_odds, under_odds)
        
        print(f"\nPrediction for {away} @ {home}")
        print(f"Line: {total_line}")
        print(f"Model Expected Total: {pred['expected_total']:.1f}")
        print(f"Over Probability: {pred['over_prob']:.1%}")
        print(f"Under Probability: {1-pred['over_prob']:.1%}")
        
        if 'betting_analysis' in pred:
            ba = pred['betting_analysis']
            print(f"Recommendation: {ba['recommendation']}")
            if ba['recommendation'] == 'BET':
                print(f"Edge: {ba['best_bet']} (EV: {max(ba['over_ev'], ba['under_ev']):.1%})")
            else:
                print(f"No Edge (Max EV: {max(ba['over_ev'], ba['under_ev']):.1%})")
                
        if 'weather' in pred:
            print(f"Weather: {pred['weather']}")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    evaluate_bet()
