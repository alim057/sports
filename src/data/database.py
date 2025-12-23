"""
Database Layer

SQLite storage for games, odds, and predictions.
"""

from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean, Text
from sqlalchemy.orm import sessionmaker, declarative_base
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
    
    # CLV Tracking
    closing_odds = Column(Integer)  # Odds at game start
    clv = Column(Float)  # Closing Line Value: (Bet Odds / Closing Odds) - 1
    
    model_version = Column(String(50))
    created_at = Column(DateTime, default=datetime.now)


# ==================== SGO API Models ====================

class SGOEvent(Base):
    """Game event from Sports Game Odds API."""
    __tablename__ = 'sgo_events'
    
    event_id = Column(String(50), primary_key=True)
    league_id = Column(String(20), nullable=False)
    
    home_team_id = Column(String(100))
    away_team_id = Column(String(100))
    home_team_name = Column(String(100))
    away_team_name = Column(String(100))
    
    start_time = Column(DateTime)
    status = Column(String(30))  # Scheduled, Live, Final, Postponed
    
    home_score = Column(Integer)
    away_score = Column(Integer)
    
    odds_available = Column(Boolean, default=False)
    
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)


class SGOOdds(Base):
    """Odds snapshot from Sports Game Odds API for CLV tracking."""
    __tablename__ = 'sgo_odds'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    event_id = Column(String(50), nullable=False)
    
    odd_id = Column(String(150))  # e.g., "points-game-ou-over"
    market_name = Column(String(150))  # e.g., "Over/Under"
    stat_id = Column(String(50))  # e.g., "points"
    bet_type = Column(String(20))  # ml, spread, ou
    side = Column(String(20))  # home, away, over, under
    
    bookmaker = Column(String(50))
    book_odds = Column(String(15))  # American odds: "+150", "-110"
    fair_odds = Column(String(15))
    line = Column(Float)  # Spread or total line
    
    fetched_at = Column(DateTime, default=datetime.now)


class SGOPlayerProp(Base):
    """Player prop bet from Sports Game Odds API."""
    __tablename__ = 'sgo_player_props'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    event_id = Column(String(50), nullable=False)
    
    player_id = Column(String(100))
    player_name = Column(String(100))
    team_id = Column(String(100))
    
    stat_type = Column(String(50))  # points, rebounds, assists, etc.
    period = Column(String(20))  # game, 1h, 1q, etc.
    bet_type = Column(String(20))  # ou (over/under)
    side = Column(String(20))  # over, under
    
    line = Column(Float)
    book_odds = Column(String(15))
    fair_odds = Column(String(15))
    
    fetched_at = Column(DateTime, default=datetime.now)


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
    
    # ==================== SGO API Methods ====================
    
    def save_sgo_events(self, events_df: pd.DataFrame):
        """Save SGO events from DataFrame."""
        # SGOEvent is defined in this same file
        
        for _, row in events_df.iterrows():
            event_id = row.get('event_id')
            if not event_id:
                continue
            
            # Check if event exists
            existing = self.session.query(SGOEvent).filter_by(event_id=event_id).first()
            
            if existing:
                # Update existing event
                existing.status = row.get('status', existing.status)
                existing.home_score = row.get('home_score', existing.home_score)
                existing.away_score = row.get('away_score', existing.away_score)
                existing.odds_available = row.get('odds_available', existing.odds_available)
            else:
                # Parse start_time
                start_time = row.get('start_time')
                if isinstance(start_time, str):
                    try:
                        start_time = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
                    except:
                        start_time = None
                
                event = SGOEvent(
                    event_id=event_id,
                    league_id=row.get('league', ''),
                    home_team_id=row.get('home_team', ''),
                    away_team_id=row.get('away_team', ''),
                    home_team_name=row.get('home_team', ''),
                    away_team_name=row.get('away_team', ''),
                    start_time=start_time,
                    status=row.get('status', 'Scheduled'),
                    odds_available=row.get('odds_available', False)
                )
                self.session.add(event)
        
        self.session.commit()
    
    def save_sgo_odds(self, odds_df: pd.DataFrame):
        """Save SGO odds snapshots for CLV tracking."""
        # SGOOdds is defined in this same file
        
        for _, row in odds_df.iterrows():
            odds = SGOOdds(
                event_id=row.get('event_id'),
                odd_id=row.get('odd_id'),
                market_name=row.get('market_name'),
                stat_id=row.get('stat_id'),
                bet_type=row.get('bet_type'),
                side=row.get('side'),
                bookmaker=row.get('bookmaker'),
                book_odds=row.get('book_odds'),
                fair_odds=row.get('fair_odds'),
                line=row.get('line'),
                fetched_at=datetime.now()
            )
            self.session.add(odds)
        
        self.session.commit()
    
    def save_sgo_player_props(self, props_df: pd.DataFrame):
        """Save SGO player prop data."""
        # SGOPlayerProp is defined in this same file
        
        for _, row in props_df.iterrows():
            prop = SGOPlayerProp(
                event_id=row.get('event_id'),
                player_id=row.get('player_id'),
                player_name=row.get('player_name'),
                team_id=row.get('team_id'),
                stat_type=row.get('stat_type'),
                period=row.get('period'),
                bet_type=row.get('bet_type'),
                side=row.get('side'),
                line=row.get('line'),
                book_odds=row.get('book_odds'),
                fair_odds=row.get('fair_odds'),
                fetched_at=datetime.now()
            )
            self.session.add(prop)
        
        self.session.commit()
    
    def get_sgo_events(self, league: Optional[str] = None, upcoming_only: bool = False) -> pd.DataFrame:
        """Get SGO events as DataFrame."""
        # SGOEvent is defined in this same file
        
        query = self.session.query(SGOEvent)
        
        if league:
            query = query.filter(SGOEvent.league_id == league.upper())
        
        if upcoming_only:
            query = query.filter(SGOEvent.status == 'Scheduled')
        
        events = query.order_by(SGOEvent.start_time).all()
        
        if not events:
            return pd.DataFrame()
        
        return pd.DataFrame([{
            c.name: getattr(e, c.name)
            for c in SGOEvent.__table__.columns
        } for e in events])
    
    def get_sgo_odds_history(self, event_id: str) -> pd.DataFrame:
        """Get odds history for an event (for CLV calculation)."""
        # SGOOdds is defined in this same file
        
        odds_records = (
            self.session.query(SGOOdds)
            .filter(SGOOdds.event_id == event_id)
            .order_by(SGOOdds.fetched_at)
            .all()
        )
        
        if not odds_records:
            return pd.DataFrame()
        
        return pd.DataFrame([{
            c.name: getattr(o, c.name)
            for c in SGOOdds.__table__.columns
        } for o in odds_records])
    
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
