"""
Bet Tracker

Comprehensive bet tracking with persistent storage and analytics.
"""

import sqlite3
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json


class BetTracker:
    """
    Tracks all bets with persistent SQLite storage.
    
    Features:
    - Record bets (moneyline, spread, total)
    - Resolve bets with actual game results
    - Performance analytics by bet type
    - Historical P/L tracking
    """
    
    def __init__(self, db_path: str = "./data/bets.db"):
        """
        Initialize bet tracker.
        
        Args:
            db_path: Path to SQLite database
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()
    
    def _init_database(self):
        """Initialize database tables."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Main bets table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS bets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                game_id TEXT,
                sport TEXT DEFAULT 'nba',
                game_date DATE,
                home_team TEXT,
                away_team TEXT,
                bet_type TEXT,
                selection TEXT,
                line REAL,
                odds INTEGER,
                stake REAL,
                model_prob REAL,
                expected_value REAL,
                placed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                result TEXT DEFAULT 'pending',
                actual_home_score INTEGER,
                actual_away_score INTEGER,
                profit_loss REAL DEFAULT 0,
                resolved_at TIMESTAMP,
                notes TEXT
            )
        """)
        
        # Daily performance summary
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS daily_summary (
                date DATE PRIMARY KEY,
                total_bets INTEGER,
                wins INTEGER,
                losses INTEGER,
                pushes INTEGER,
                total_staked REAL,
                total_returned REAL,
                profit_loss REAL,
                roi REAL,
                moneyline_pl REAL,
                spread_pl REAL,
                total_pl REAL
            )
        """)
        
        # Model performance tracking
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS model_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date DATE,
                model_type TEXT,
                bet_type TEXT,
                accuracy REAL,
                roi REAL,
                total_bets INTEGER,
                edge_pct REAL,
                notes TEXT
            )
        """)
        
        conn.commit()
        conn.close()
    
    def place_bet(
        self,
        home_team: str,
        away_team: str,
        bet_type: str,
        selection: str,
        odds: int,
        stake: float,
        model_prob: float = None,
        expected_value: float = None,
        line: float = None,
        game_id: str = None,
        game_date: str = None,
        sport: str = 'nba',
        notes: str = None
    ) -> int:
        """
        Record a new bet.
        
        Args:
            home_team: Home team abbreviation
            away_team: Away team abbreviation
            bet_type: 'moneyline', 'spread', or 'total'
            selection: Team or 'over'/'under'
            odds: American odds
            stake: Amount wagered
            model_prob: Model's predicted probability
            expected_value: Expected value percentage
            line: Spread or total line (if applicable)
            game_id: Unique game identifier
            game_date: Date of game
            sport: Sport code
            notes: Additional notes
            
        Returns:
            Bet ID
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO bets (
                game_id, sport, game_date, home_team, away_team,
                bet_type, selection, line, odds, stake,
                model_prob, expected_value, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            game_id, sport, game_date, home_team, away_team,
            bet_type, selection, line, odds, stake,
            model_prob, expected_value, notes
        ))
        
        bet_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        print(f"Bet #{bet_id} placed: {bet_type} {selection} @ {odds:+d} (${stake})")
        return bet_id
    
    def resolve_bet(
        self,
        bet_id: int,
        result: str,
        home_score: int = None,
        away_score: int = None
    ) -> float:
        """
        Resolve a bet with the actual result.
        
        Args:
            bet_id: ID of bet to resolve
            result: 'win', 'loss', or 'push'
            home_score: Actual home team score
            away_score: Actual away team score
            
        Returns:
            Profit/loss amount
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get bet details
        cursor.execute("SELECT stake, odds FROM bets WHERE id = ?", (bet_id,))
        row = cursor.fetchone()
        
        if not row:
            conn.close()
            raise ValueError(f"Bet #{bet_id} not found")
        
        stake, odds = row
        
        # Calculate P/L
        if result == 'win':
            if odds > 0:
                profit_loss = stake * (odds / 100)
            else:
                profit_loss = stake * (100 / abs(odds))
        elif result == 'loss':
            profit_loss = -stake
        else:  # push
            profit_loss = 0
        
        # Update bet
        cursor.execute("""
            UPDATE bets SET
                result = ?,
                actual_home_score = ?,
                actual_away_score = ?,
                profit_loss = ?,
                resolved_at = ?
            WHERE id = ?
        """, (result, home_score, away_score, profit_loss, datetime.now(), bet_id))
        
        conn.commit()
        conn.close()
        
        print(f"Bet #{bet_id} resolved: {result.upper()} (${profit_loss:+.2f})")
        return profit_loss
    
    def resolve_by_scores(
        self,
        bet_id: int,
        home_score: int,
        away_score: int
    ) -> Tuple[str, float]:
        """
        Auto-resolve bet based on actual scores.
        
        Args:
            bet_id: ID of bet to resolve
            home_score: Actual home team score
            away_score: Actual away team score
            
        Returns:
            (result, profit_loss)
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT bet_type, selection, line, home_team, away_team
            FROM bets WHERE id = ?
        """, (bet_id,))
        row = cursor.fetchone()
        conn.close()
        
        if not row:
            raise ValueError(f"Bet #{bet_id} not found")
        
        bet_type, selection, line, home_team, away_team = row
        
        # Determine result based on bet type
        if bet_type == 'moneyline':
            if selection == home_team:
                result = 'win' if home_score > away_score else 'loss'
            else:
                result = 'win' if away_score > home_score else 'loss'
                
        elif bet_type == 'spread':
            # Positive line means underdog, negative means favorite
            if selection == home_team:
                covered = (home_score + (line or 0)) > away_score
            else:
                covered = (away_score + (line or 0)) > home_score
            
            # Check for push
            if selection == home_team:
                diff = (home_score + (line or 0)) - away_score
            else:
                diff = (away_score + (line or 0)) - home_score
            
            if diff == 0:
                result = 'push'
            else:
                result = 'win' if covered else 'loss'
                
        elif bet_type == 'total':
            total = home_score + away_score
            if selection.lower() == 'over':
                if total > (line or 0):
                    result = 'win'
                elif total < (line or 0):
                    result = 'loss'
                else:
                    result = 'push'
            else:  # under
                if total < (line or 0):
                    result = 'win'
                elif total > (line or 0):
                    result = 'loss'
                else:
                    result = 'push'
        else:
            raise ValueError(f"Unknown bet type: {bet_type}")
        
        pl = self.resolve_bet(bet_id, result, home_score, away_score)
        return result, pl
    
    def get_pending_bets(self) -> pd.DataFrame:
        """Get all pending (unresolved) bets."""
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query(
            "SELECT * FROM bets WHERE result = 'pending' ORDER BY placed_at DESC",
            conn
        )
        conn.close()
        return df
    
    def get_bet_history(
        self,
        bet_type: str = None,
        sport: str = None,
        start_date: str = None,
        end_date: str = None,
        status: str = None,
        limit: int = 100
    ) -> pd.DataFrame:
        """Get historical bets with optional filters."""
        query = "SELECT * FROM bets WHERE 1=1"
        params = []
        
        if bet_type:
            query += " AND bet_type = ?"
            params.append(bet_type)
        if sport:
            query += " AND sport = ?"
            params.append(sport)
        if start_date:
            query += " AND game_date >= ?"
            params.append(start_date)
        if end_date:
            query += " AND game_date <= ?"
            params.append(end_date)
            
        if status:
            if status == 'pending':
                query += " AND result = 'pending'"
            elif status == 'resolved':
                query += " AND result != 'pending'"
            elif status in ['win', 'loss', 'push']:
                query += " AND result = ?"
                params.append(status)
        
        query += f" ORDER BY placed_at DESC LIMIT {limit}"
        
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        return df
    
    def get_performance_summary(
        self,
        bet_type: str = None
    ) -> Dict:
        """
        Get overall performance summary.
        
        Args:
            bet_type: Filter by bet type (optional)
            
        Returns:
            Dictionary with performance metrics
        """
        conn = sqlite3.connect(self.db_path)
        
        where_clause = "WHERE result != 'pending'"
        if bet_type:
            where_clause += f" AND bet_type = '{bet_type}'"
        
        query = f"""
            SELECT 
                COUNT(*) as total_bets,
                SUM(CASE WHEN result = 'win' THEN 1 ELSE 0 END) as wins,
                SUM(CASE WHEN result = 'loss' THEN 1 ELSE 0 END) as losses,
                SUM(CASE WHEN result = 'push' THEN 1 ELSE 0 END) as pushes,
                SUM(stake) as total_staked,
                SUM(profit_loss) as total_profit,
                AVG(expected_value) as avg_ev,
                AVG(model_prob) as avg_model_prob
            FROM bets
            {where_clause}
        """
        
        cursor = conn.cursor()
        cursor.execute(query)
        row = cursor.fetchone()
        conn.close()
        
        if not row or row[0] == 0:
            return {
                'total_bets': 0,
                'wins': 0,
                'losses': 0,
                'pushes': 0,
                'win_rate': 0,
                'total_staked': 0,
                'total_profit': 0,
                'roi': 0,
                'avg_ev': 0,
                'avg_model_prob': 0
            }
        
        total_bets, wins, losses, pushes, staked, profit, avg_ev, avg_prob = row
        
        return {
            'total_bets': total_bets or 0,
            'wins': wins or 0,
            'losses': losses or 0,
            'pushes': pushes or 0,
            'win_rate': (wins / total_bets) if total_bets > 0 else 0,
            'total_staked': staked or 0,
            'total_profit': profit or 0,
            'roi': (profit / staked) if staked > 0 else 0,
            'avg_ev': avg_ev or 0,
            'avg_model_prob': avg_prob or 0
        }
    
    def get_performance_by_bet_type(self) -> pd.DataFrame:
        """Get performance breakdown by bet type."""
        conn = sqlite3.connect(self.db_path)
        
        query = """
            SELECT 
                bet_type,
                COUNT(*) as total_bets,
                SUM(CASE WHEN result = 'win' THEN 1 ELSE 0 END) as wins,
                SUM(CASE WHEN result = 'loss' THEN 1 ELSE 0 END) as losses,
                ROUND(SUM(CASE WHEN result = 'win' THEN 1.0 ELSE 0 END) / COUNT(*) * 100, 1) as win_rate,
                SUM(stake) as total_staked,
                SUM(profit_loss) as total_profit,
                ROUND(SUM(profit_loss) / SUM(stake) * 100, 2) as roi
            FROM bets
            WHERE result != 'pending'
            GROUP BY bet_type
            ORDER BY roi DESC
        """
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df
    
    def get_performance_by_team(self, limit: int = 10) -> pd.DataFrame:
        """Get performance breakdown by team bet on."""
        conn = sqlite3.connect(self.db_path)
        
        query = f"""
            SELECT 
                selection as team,
                COUNT(*) as total_bets,
                SUM(CASE WHEN result = 'win' THEN 1 ELSE 0 END) as wins,
                ROUND(SUM(CASE WHEN result = 'win' THEN 1.0 ELSE 0 END) / COUNT(*) * 100, 1) as win_rate,
                SUM(profit_loss) as total_profit,
                ROUND(SUM(profit_loss) / SUM(stake) * 100, 2) as roi
            FROM bets
            WHERE result != 'pending'
            GROUP BY selection
            HAVING total_bets >= 3
            ORDER BY roi DESC
            LIMIT {limit}
        """
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df
    
    def get_ev_analysis(self) -> Dict:
        """
        Analyze correlation between predicted EV and actual results.
        
        This helps understand how well the model's EV predictions
        correspond to actual profitability.
        """
        conn = sqlite3.connect(self.db_path)
        
        query = """
            SELECT 
                CASE 
                    WHEN expected_value >= 0.2 THEN 'high_ev'
                    WHEN expected_value >= 0.05 THEN 'medium_ev'
                    ELSE 'low_ev'
                END as ev_bucket,
                COUNT(*) as total_bets,
                SUM(CASE WHEN result = 'win' THEN 1 ELSE 0 END) as wins,
                ROUND(SUM(CASE WHEN result = 'win' THEN 1.0 ELSE 0 END) / COUNT(*) * 100, 1) as win_rate,
                SUM(profit_loss) as total_profit,
                ROUND(AVG(expected_value) * 100, 1) as avg_predicted_ev,
                ROUND(SUM(profit_loss) / SUM(stake) * 100, 2) as actual_roi
            FROM bets
            WHERE result != 'pending' AND expected_value IS NOT NULL
            GROUP BY ev_bucket
            ORDER BY avg_predicted_ev DESC
        """
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        return {
            'by_ev_bucket': df.to_dict('records'),
            'ev_correlation': self._calculate_ev_correlation()
        }
    
    def _calculate_ev_correlation(self) -> float:
        """Calculate correlation between predicted EV and actual results."""
        conn = sqlite3.connect(self.db_path)
        
        query = """
            SELECT expected_value, 
                   CASE WHEN result = 'win' THEN 1 ELSE 0 END as won
            FROM bets
            WHERE result != 'pending' AND expected_value IS NOT NULL
        """
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        if len(df) < 10:
            return None
        
        return df['expected_value'].corr(df['won'])

    def get_daily_performance(self) -> List[Dict]:
        """
        Get daily P/L and cumulative bankroll history.
        
        Returns:
            List of dicts with date, profit, cumulative, wins, losses
        """
        conn = sqlite3.connect(self.db_path)
        
        # Get all resolved bets sorted by date
        query = """
            SELECT 
                DATE(resolved_at) as date,
                SUM(profit_loss) as daily_profit,
                SUM(CASE WHEN result = 'win' THEN 1 ELSE 0 END) as wins,
                SUM(CASE WHEN result = 'loss' THEN 1 ELSE 0 END) as losses
            FROM bets
            WHERE result != 'pending' AND resolved_at IS NOT NULL
            GROUP BY DATE(resolved_at)
            ORDER BY DATE(resolved_at) ASC
        """
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        result = []
        cumulative = 0
        
        for _, row in df.iterrows():
            cumulative += row['daily_profit']
            result.append({
                'date': row['date'],
                'profit': round(row['daily_profit'], 2),
                'cumulative': round(cumulative, 2),
                'wins': int(row['wins']),
                'losses': int(row['losses'])
            })
            
        return result
    
    def generate_report(self) -> str:
        """Generate a comprehensive performance report."""
        summary = self.get_performance_summary()
        by_type = self.get_performance_by_bet_type()
        by_team = self.get_performance_by_team()
        ev_analysis = self.get_ev_analysis()
        
        report = []
        report.append("=" * 70)
        report.append("BET TRACKING PERFORMANCE REPORT")
        report.append("=" * 70)
        report.append("")
        
        # Overall summary
        report.append("OVERALL PERFORMANCE")
        report.append("-" * 40)
        report.append(f"  Total Bets: {summary['total_bets']}")
        report.append(f"  Record: {summary['wins']}-{summary['losses']}-{summary['pushes']}")
        report.append(f"  Win Rate: {summary['win_rate']:.1%}")
        report.append(f"  Total Staked: ${summary['total_staked']:.2f}")
        report.append(f"  Total Profit: ${summary['total_profit']:+.2f}")
        report.append(f"  ROI: {summary['roi']:.1%}")
        report.append("")
        
        # By bet type
        report.append("PERFORMANCE BY BET TYPE")
        report.append("-" * 40)
        if len(by_type) > 0:
            for _, row in by_type.iterrows():
                report.append(f"  {row['bet_type'].upper()}")
                report.append(f"    Bets: {row['total_bets']} | Win Rate: {row['win_rate']}%")
                report.append(f"    Profit: ${row['total_profit']:+.2f} | ROI: {row['roi']}%")
        else:
            report.append("  No resolved bets yet")
        report.append("")
        
        # By team
        report.append("TOP PERFORMING TEAMS")
        report.append("-" * 40)
        if len(by_team) > 0:
            for _, row in by_team.head(5).iterrows():
                report.append(f"  {row['team']}: {row['wins']}/{row['total_bets']} ({row['win_rate']}%) | ROI: {row['roi']}%")
        else:
            report.append("  No data")
        report.append("")
        
        # EV analysis
        report.append("EV PREDICTION ACCURACY")
        report.append("-" * 40)
        if ev_analysis['by_ev_bucket']:
            for bucket in ev_analysis['by_ev_bucket']:
                report.append(f"  {bucket['ev_bucket']}: Predicted {bucket['avg_predicted_ev']}% EV -> Actual {bucket['actual_roi']}% ROI")
            if ev_analysis['ev_correlation'] is not None:
                report.append(f"  Correlation: {ev_analysis['ev_correlation']:.3f}")
        else:
            report.append("  Not enough data")
        
        report.append("")
        report.append("=" * 70)
        
        return "\n".join(report)


def main():
    """Demo the bet tracker."""
    print("Bet Tracker Demo")
    print("=" * 50)
    
    tracker = BetTracker()
    
    # Place some sample bets
    print("\nPlacing sample bets...")
    
    bet1 = tracker.place_bet(
        home_team="LAL",
        away_team="GSW",
        bet_type="moneyline",
        selection="LAL",
        odds=-130,
        stake=50,
        model_prob=0.65,
        expected_value=0.149,
        game_date="2025-12-21"
    )
    
    bet2 = tracker.place_bet(
        home_team="BOS",
        away_team="MIA",
        bet_type="spread",
        selection="MIA",
        line=7.5,
        odds=-110,
        stake=50,
        model_prob=0.55,
        expected_value=0.08,
        game_date="2025-12-21"
    )
    
    bet3 = tracker.place_bet(
        home_team="DEN",
        away_team="PHX",
        bet_type="total",
        selection="over",
        line=228.5,
        odds=-110,
        stake=50,
        model_prob=0.52,
        expected_value=0.03,
        game_date="2025-12-21"
    )
    
    # Resolve bets with scores
    print("\nResolving bets with game scores...")
    tracker.resolve_by_scores(bet1, 115, 108)  # LAL wins
    tracker.resolve_by_scores(bet2, 105, 102)  # MIA loses by 3, covers +7.5
    tracker.resolve_by_scores(bet3, 118, 112)  # Total 230 > 228.5, over wins
    
    # Show report
    print("\n" + tracker.generate_report())


if __name__ == "__main__":
    main()
