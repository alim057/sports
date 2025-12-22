@echo off
echo ================================================
echo   Sports Betting Predictor - Web Dashboard
echo ================================================
echo.
echo Starting server...
cd /d "%~dp0"
call venv\Scripts\activate
echo.
echo Dashboard ready at: http://localhost:5000
echo.
echo Press Ctrl+C to stop the server.
echo ================================================
start http://localhost:5000
python dashboard\server.py
