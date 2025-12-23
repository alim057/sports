
import sys
import os
import json
from flask import Flask

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'dashboard'))

# Import the app to test routes
from dashboard.server import app, get_performance, get_performance_history, get_recent_bets

def test_dashboard_api():
    print("Testing Dashboard API Integration...")
    
    with app.app_context():
        # Test Performance
        print("\n1. Testing /api/performance...")
        perf_resp = get_performance()
        perf_data = json.loads(perf_resp.data)
        print(json.dumps(perf_data, indent=2))
        
        if perf_data.get('performance', {}).get('isDemo') is False:
            print("SUCCESS: Performance is using real data.")
        else:
            print("FAILURE: Performance is still using demo data.")

        # Test History
        print("\n2. Testing /api/performance-history...")
        hist_resp = get_performance_history()
        hist_data = json.loads(hist_resp.data)
        # print(json.dumps(hist_data, indent=2)) # verbose
        if hist_data.get('isDemo') is False:
            print(f"SUCCESS: History is real. Found {len(hist_data.get('history', []))} days.")
        else:
             print("FAILURE: History is still demo.")
             
        # Test Recent Bets - All
        print("\n3a. Testing /api/recent-bets (All)...")
        with app.test_request_context('/api/recent-bets'):
            recent_resp = get_recent_bets()
            if isinstance(recent_resp, tuple):
                 recent_resp = recent_resp[0]
            recent_data = json.loads(recent_resp.data)
            bets = recent_data.get('bets', [])
            print(f"Found {len(bets)} total bets.")

        # Test Recent Bets - Pending
        print("\n3b. Testing /api/recent-bets?status=pending...")
        with app.test_request_context('/api/recent-bets?status=pending'):
            pending_resp = get_recent_bets()
            pending_data = json.loads(pending_resp.data)
            pending_bets = pending_data.get('bets', [])
            print(f"Found {len(pending_bets)} pending bets.")
            if len(pending_bets) > 0 and pending_bets[0]['result'] != 'PENDING':
                print("FAILURE: Filtered for pending but got resolved.")

        # Test Recent Bets - Resolved
        print("\n3c. Testing /api/recent-bets?status=resolved...")
        with app.test_request_context('/api/recent-bets?status=resolved'):
            res_resp = get_recent_bets()
            res_data = json.loads(res_resp.data)
            res_bets = res_data.get('bets', [])
            print(f"Found {len(res_bets)} resolved bets.")
            if len(res_bets) > 0 and res_bets[0]['result'] == 'PENDING':
                print("FAILURE: Filtered for resolved but got pending.")

if __name__ == "__main__":
    test_dashboard_api()
