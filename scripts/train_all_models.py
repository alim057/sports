"""
Train All Sports Models

Unified script to train models for all supported sports.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def train_nba():
    """Train NBA model."""
    print("\n" + "=" * 60)
    print("Training NBA Model...")
    print("=" * 60)
    
    try:
        from models.trainer import main as nba_main
        nba_main()
        return True
    except Exception as e:
        print(f"NBA training error: {e}")
        return False


def train_nfl():
    """Train NFL model."""
    print("\n" + "=" * 60)
    print("Training NFL Model...")
    print("=" * 60)
    
    try:
        from models.nfl_trainer import NFLTrainer
        
        trainer = NFLTrainer()
        games = trainer.fetch_training_data([2022, 2023, 2024])
        
        if games.empty:
            print("No NFL games found!")
            return False
        
        print(f"Total games: {len(games)}")
        features = trainer.engineer_features(games)
        results = trainer.train(features)
        
        print(f"Train Accuracy: {results['train_accuracy']:.1%}")
        print(f"Test Accuracy: {results['test_accuracy']:.1%}")
        
        trainer.save()
        return True
    except Exception as e:
        print(f"NFL training error: {e}")
        return False


def train_mlb():
    """Train MLB model."""
    print("\n" + "=" * 60)
    print("Training MLB Model...")
    print("=" * 60)
    
    try:
        from models.mlb_trainer import MLBTrainer
        
        trainer = MLBTrainer()
        games = trainer.fetch_training_data([2023, 2024])
        
        if games.empty:
            print("No MLB games found!")
            return False
        
        print(f"Total games: {len(games)}")
        features = trainer.engineer_features(games)
        results = trainer.train(features)
        
        print(f"Train Accuracy: {results['train_accuracy']:.1%}")
        print(f"Test Accuracy: {results['test_accuracy']:.1%}")
        
        trainer.save()
        return True
    except Exception as e:
        print(f"MLB training error: {e}")
        return False


def train_nhl():
    """Train NHL model."""
    print("\n" + "=" * 60)
    print("Training NHL Model...")
    print("=" * 60)
    
    try:
        from models.nhl_trainer import NHLTrainer
        
        trainer = NHLTrainer()
        games = trainer.fetch_training_data(["2023-2024"])
        
        if games.empty:
            print("No NHL games found!")
            return False
        
        print(f"Total games: {len(games)}")
        features = trainer.engineer_features(games)
        results = trainer.train(features)
        
        print(f"Train Accuracy: {results['train_accuracy']:.1%}")
        print(f"Test Accuracy: {results['test_accuracy']:.1%}")
        
        trainer.save()
        return True
    except Exception as e:
        print(f"NHL training error: {e}")
        return False


def train_ncaaf():
    """Train NCAAF model."""
    print("\n" + "=" * 60)
    print("Training NCAAF Model...")
    print("=" * 60)
    
    try:
        from models.ncaaf_trainer import NCAAFTrainer
        
        trainer = NCAAFTrainer()
        games = trainer.fetch_training_data([2022, 2023, 2024])
        
        if games.empty:
            print("No NCAAF games found!")
            return False
        
        print(f"Total games: {len(games)}")
        features = trainer.engineer_features(games)
        results = trainer.train(features)
        
        print(f"Train Accuracy: {results['train_accuracy']:.1%}")
        print(f"Test Accuracy: {results['test_accuracy']:.1%}")
        
        trainer.save()
        return True
    except Exception as e:
        print(f"NCAAF training error: {e}")
        return False


def main():
    """Train all models."""
    print("=" * 60)
    print("  MULTI-SPORT MODEL TRAINING")
    print("=" * 60)
    
    results = {}
    
    # Train each sport
    results['NFL'] = train_nfl()
    results['MLB'] = train_mlb()
    results['NHL'] = train_nhl()
    results['NCAAF'] = train_ncaaf()
    
    # Summary
    print("\n" + "=" * 60)
    print("  TRAINING SUMMARY")
    print("=" * 60)
    
    for sport, success in results.items():
        status = "✓ Success" if success else "✗ Failed"
        print(f"  {sport}: {status}")
    
    print("=" * 60)
    
    trained = sum(1 for v in results.values() if v)
    print(f"\nTrained {trained}/{len(results)} models successfully.")


if __name__ == "__main__":
    main()
