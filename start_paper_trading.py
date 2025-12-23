import sys
import os
import subprocess
from datetime import datetime

def log_session(message):
    """Log a message to the paper trading log file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] {message}\n"
    print(log_entry.strip())
    with open("paper_trading_log.txt", "a") as f:
        f.write(log_entry)

def main():
    print("\n" + "=" * 60)
    print(" 🚀 DAILY PAPER TRADING SESSION STARTER")
    print("=" * 60)
    
    log_session("Session started.")

    # 1. Run System Verification
    print("\n[Step 1] Verifying System Health...")
    log_session("Running verification check...")
    
    # Run in subprocess to ensure clean environment
    result = subprocess.run([sys.executable, "verify_system.py"])
    
    if result.returncode != 0:
        msg = "❌ Verification FAILED. Aborting session."
        print(f"\n{msg}")
        log_session(msg)
        sys.exit(1)
    
    log_session("Verification passed.")

    # 2. Run Daily Edge Analysis
    print("\n[Step 2] Finding Today's Best Bets...")
    log_session("Running daily_edge.py...")
    
    # Run daily edge script
    edge_result = subprocess.run([sys.executable, "src/betting/daily_edge.py"])
    
    if edge_result.returncode != 0:
        msg = "❌ Error running prediction analysis."
        print(f"\n{msg}")
        log_session(msg)
        sys.exit(1)

    log_session("Daily analysis complete.")

    # 3. Final Instructions
    print("\n" + "=" * 60)
    print(" ✅ SESSION COMPLETE")
    print("=" * 60)
    print("\nNext Steps:")
    print("1. Start the dashboard to view details:")
    print(f"   {sys.executable} dashboard/server.py")
    print("2. Check Daily Status API:")
    print("   http://localhost:5000/api/daily-status")
    print("\nGood luck with your paper trading!")
    
    log_session("Session ended successfully.")

if __name__ == "__main__":
    main()
