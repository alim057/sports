@echo off
echo ============================================================
echo   SPORTS BETTING - DAILY PAPER TRADING ROUTINE
echo ============================================================
echo.

cd /d "%~dp0"

echo [Step 1] Running System Verification...
call venv\Scripts\python verify_system.py
if errorlevel 1 (
    echo.
    echo [ERROR] Verification failed! Fix issues before trading.
    pause
    exit /b 1
)

echo.
echo [Step 2] Finding Today's Best Bets...
call venv\Scripts\python src\betting\daily_edge.py

echo.
echo ============================================================
echo   ROUTINE COMPLETE
echo ============================================================
echo.
echo Next: Start the dashboard with 'run_dashboard.bat'
echo.
pause
