# Sports Betting Predictor

NBA (and future sports) betting prediction system using machine learning.

## Features

- **Data Collection**: NBA stats via `nba_api`, betting odds via The Odds API
- **ML Predictions**: Win probability predictions using XGBoost
- **Bet Evaluation**: Expected value calculation and Kelly criterion sizing

## Quick Start

```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Set up configuration
cp config/config.example.yaml config/config.yaml
# Edit config.yaml with your The Odds API key

# Run the system
python main.py --update-data      # Fetch latest data
python main.py --train            # Train model
python main.py --predict          # Get today's predictions
```

## Project Structure

```
sports-betting-predictor/
├── config/          # Configuration files
├── src/
│   ├── data/        # Data fetching & storage
│   ├── features/    # Feature engineering
│   ├── models/      # ML training & inference
│   └── betting/     # Bet evaluation
├── data/            # Local data storage
├── models/          # Saved model artifacts
└── notebooks/       # Exploration
```

## API Keys

- **NBA Stats**: No API key required (`nba_api` is free)
- **The Odds API**: Free tier (500 calls/month) - https://the-odds-api.com/
