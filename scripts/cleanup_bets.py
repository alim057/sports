"""
Database Cleanup Script

Removes invalid pending bets from the database.
"""

import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "data" / "bets.db"


def cleanup_pending_bets(dry_run=True):
    """Delete all pending bets from the database."""
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Count pending bets
    cursor.execute("SELECT COUNT(*) FROM bets WHERE result = 'pending'")
    pending_count = cursor.fetchone()[0]
    
    print(f"{'='*50}")
    print(f"DATABASE CLEANUP")
    print(f"{'='*50}")
    print(f"Pending bets to delete: {pending_count}")
    
    if dry_run:
        print("\n[DRY RUN] No changes made.")
        print("Run with --execute to delete pending bets.")
    else:
        # Delete pending bets
        cursor.execute("DELETE FROM bets WHERE result = 'pending'")
        deleted = cursor.rowcount
        conn.commit()
        print(f"\n[DELETED] {deleted} pending bets removed.")
        
        # Verify
        cursor.execute("SELECT COUNT(*) FROM bets WHERE result = 'pending'")
        remaining = cursor.fetchone()[0]
        print(f"[VERIFY] Remaining pending: {remaining}")
    
    # Show remaining resolved bets summary
    cursor.execute("""
        SELECT result, COUNT(*), ROUND(SUM(profit_loss), 2)
        FROM bets
        WHERE result != 'pending'
        GROUP BY result
    """)
    print(f"\n{'='*50}")
    print("REMAINING RESOLVED BETS")
    print(f"{'='*50}")
    for row in cursor.fetchall():
        print(f"  {row[0].upper()}: {row[1]} bets, ${row[2]:.2f}")
    
    conn.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true", help="Actually delete bets")
    args = parser.parse_args()
    
    cleanup_pending_bets(dry_run=not args.execute)
