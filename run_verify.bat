@echo off
echo ============================================================
echo   SPORTS BETTING - VERIFICATION ONLY
echo ============================================================
echo.

cd /d "%~dp0"

call venv\Scripts\python verify_system.py

echo.
pause
