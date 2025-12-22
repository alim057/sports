"""
Database Layer

SQLite storage for games, odds, and predictions.
"""

from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
from pathlib import Path
import pandas as pd
from typing import Optional

Base = declarative_base()


class Game(Base):
    """NBA game record."""
    __tablename__ = 'games'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    game_id = Column(String(50), unique=True, nullable=False)
    game_date = Column(DateTime, nullable=False)
    season = Column(String(10), nullable=False)
    
    home_team = Column(String(50), nullable=False)
    away_team = Column(String(50), nullable=False)
    home_team_id = Column(Integer)
    away_team_id = Column(Integer)
    
    # Final scores (null if game not completed)
    home_score = Column(Integer)
    away_score = Column(Integer)
    winner = Column(String(50))
    
    # Team stats (home team perspective)
    home_pts = Column(Integer)
    home_reb = Column(Integer)
    home_ast = Column(Integer)
    home_fg_pct = Column(Float)
    home_fg3_pct = Column(Float)
    home_ft_pct = Column(Float)
    
    away_pts = Column(Integer)
    away_reb = Column(Integer)
    away_ast = Column(Integer)
    away_fg_pct = Column(Float)
    away_fg3_pct = Column(Float)
    away_ft_pct = Column(Float)
    
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)


class Odds(Base):
    """Betting odds snapshot."""
    __tablename__ = 'odds'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    game_id = Column(String(50), nullable=False)
    
    home_team = Column(String(50), nullable=False)
    away_team = Column(String(50), nullable=False)
    commence_time = Column(DateTime)
    
    bookmaker = Column(String(50), nullable=False)
    market = Column(String(20), nullable=False)  # h2h, spreads, totals
    
    team = Column(String(50))  # Team name or Over/Under
    price = Column(Integer)  # American odds
    point = Column(Float)  # Spread or total points
    
    fetched_at = Column(DateTime, default=datetime.now)


class Prediction(Base):
    """Model prediction record."""
    __tablename__ = 'predictions'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    game_id = Column(String(50), nullable=False)
    
    home_team = Column(String(50), nullable=False)
    away_team = Column(String(50), nullable=False)
    game_date = Column(DateTime)
    
    # Model predictions
    home_win_prob = Column(Float, nullable=False)
    away_win_prob = Column(Float, nullable=False)
    predicted_winner = Column(String(50))
    confidence = Column(Float)
    
    # Actual result (filled in after game)
    actual_winner = Column(String(50))
    was_correct = Column(Boolean)
    
    # Betting recommendation
    recommended_bet = Column(String(100))
    expected_value = Column(Float)
    
    model_version = Column(String(50))
    created_at = Column(DateTime, default=datetime.now)


class Database:
    """Database manager for the betting system."""
    
    def __init__(self, db_path: str = "./data/betting.db"):
        """Initialize database connection."""
        # Ensure directory exists
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        
        self.engine = create_engine(f'sqlite:///{db_path}')
        Base.metadata.create_all(self.engine)
        
        Session = sessionmaker(bind=self.engine)
        self.session = Session()
    
    # ==================== Games ====================
    
    def save_games(self, games_df: pd.DataFrame):
        """Save game records from DataFrame."""
        for _, row in games_df.iterrows():
            game_id = row.get('GAME_ID') or row.get('game_id')
            
            # Check if game exists
            existing = self.session.query(Game).filter_by(game_id=game_id).first()
            
            if existing:
                # Update existing game
                for col, val in row.items():
                    if hasattr(existing, col.lower()):
                        setattr(existing, col.lower(), val)
            else:
                # Create new game
                game = Game(
                    game_id=game_id,
                    game_date=row.get('GAME_DATE', row.get('game_date')),
                    season=row.get('SEASON', row.get('season', '2024-25')),
                    home_team=row.get('home_team', ''),
                    away_team=row.get('away_team', ''),
                )
                self.session.add(game)
        
        self.session.commit()
    
    def get_games(
        self, 
        season: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """Retrieve games as DataFrame."""
        query = self.session.query(Game)
        
        if season:
            query = query.filter(Game.season == season)
        if start_date:
            query = query.filter(Game.game_date >= start_date)
        if end_date:
            query = query.filter(Game.game_date <= end_date)
        
        games = query.all()
        
        if not games:
            return pd.DataFrame()
        
        return pd.DataFrame([{
            c.name: getattr(g, c.name) 
            for c in Game.__table__.columns
        } for g in games])
    
    # ==================== Odds ====================
    
    def save_odds(self, odds_df: pd.DataFrame):
        """Save betting odds from DataFrame."""
        records = odds_df.to_dict('records')
        
        for record in records:
            odds = Odds(
                game_id=record.get('game_id'),
                home_team=record.get('home_team'),
                away_team=record.get('away_team'),
                commence_time=record.get('commence_time'),
                bookmaker=record.get('bookmaker'),
                market=record.get('market'),
                team=record.get('team'),
                price=record.get('price'),
                point=record.get('point'),
                fetched_at=record.get('fetched_at', datetime.now())
            )
            self.session.add(odds)
        
        self.session.commit()
    
    def get_latest_odds(self, game_id: str) -> pd.DataFrame:
        """Get the most recent odds for a specific game."""
        odds_records = (
            self.session.query(Odds)
            .filter(Odds.game_id == game_id)
            .order_by(Odds.fetched_at.desc())
            .all()
        )
        
        if not odds_records:
            return pd.DataFrame()
        
        return pd.DataFrame([{
            c.name: getattr(o, c.name) 
            for c in Odds.__table__.columns
        } for o in odds_records])
    
    # ==================== Predictions ====================
    
    def save_prediction(
        self,
        game_id: str,
        home_team: str,
        away_team: str,
        home_win_prob: float,
        game_date: Optional[datetime] = None,
        model_version: str = "v1",
        recommended_bet: str = None,
        expected_value: float = None
    ):
        """Save a model prediction."""
        away_win_prob = 1 - home_win_prob
        predicted_winner = home_team if home_win_prob > 0.5 else away_team
        confidence = max(home_win_prob, away_win_prob)
        
        prediction = Prediction(
            game_id=game_id,
            home_team=home_team,
            away_team=away_team,
            game_date=game_date,
            home_win_prob=home_win_prob,
            away_win_prob=away_win_prob,
            predicted_winner=predicted_winner,
            confidence=confidence,
            recommended_bet=recommended_bet,
            expected_value=expected_value,
            model_version=model_version
        )
        
        self.session.add(prediction)
        self.session.commit()
    
    def update_prediction_result(
        self,
        game_id: str,
        actual_winner: str
    ):
        """Update prediction with actual game result."""
        prediction = (
            self.session.query(Prediction)
            .filter(Prediction.game_id == game_id)
            .first()
        )
        
        if prediction:
            prediction.actual_winner = actual_winner
            prediction.was_correct = (prediction.predicted_winner == actual_winner)
            self.session.commit()
    
    def get_prediction_accuracy(self, model_version: Optional[str] = None) -> dict:
        """Calculate model prediction accuracy."""
        query = self.session.query(Prediction).filter(Prediction.actual_winner.isnot(None))
        
        if model_version:
            query = query.filter(Prediction.model_version == model_version)
        
        predictions = query.all()
        
        if not predictions:
            return {'total': 0, 'correct': 0, 'accuracy': 0.0}
        
        correct = sum(1 for p in predictions if p.was_correct)
        total = len(predictions)
        
        return {
            'total': total,
            'correct': correct,
            'accuracy': correct / total if total > 0 else 0.0
        }
    
    def close(self):
        """Close database session."""
        self.session.close()


def main():
    """Test the database layer."""
    db = Database("./data/test_betting.db")
    
    print("Database initialized successfully!")
    print(f"Tables created: games, odds, predictions")
    
    # Test saving a prediction
    db.save_prediction(
        game_id="test_001",
        home_team="Los Angeles Lakers",
        away_team="Golden State Warriors",
        home_win_prob=0.55,
        game_date=datetime.now(),
        model_version="test"
    )
    
    print("\nSample prediction saved!")
    
    db.close()
    print("Database connection closed.")


if __name__ == "__main__":
    main()
