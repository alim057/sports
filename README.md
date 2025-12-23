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

## Discord Alerts Setup

Get real-time betting alerts sent to your phone via Discord:

1. **Create a Discord Webhook:**
   - Go to your Discord server → Settings → Integrations → Webhooks
   - Click "New Webhook", name it (e.g., "Betting Bot"), copy the URL

2. **Add GitHub Secrets** (for automated alerts every 15 min):
   - Go to your GitHub repo → Settings → Secrets and variables → Actions
   - Add these secrets:
     - `ODDS_API_KEY`: Your The Odds API key
     - `DISCORD_WEBHOOK`: Your Discord webhook URL

3. **Done!** GitHub Actions will now check for +EV bets every 15 minutes and alert you on Discord.

## Live Dashboard

- **Local:** `python dashboard/server.py` → http://localhost:5000
- **Deployed:** https://sports-oor2.onrender.com/

