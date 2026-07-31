@echo off
cd /d "%~dp0"

if not exist venv\Scripts\python.exe (
    echo Virtual environment not found. Run setup.bat first.
    pause
    exit /b 1
)

echo Re-pulling historical/season-to-date game data (several minutes)...
venv\Scripts\python.exe -m src.data.fetch_historical

echo.
echo Re-running backtest with refreshed data...
venv\Scripts\python.exe -m src.backtest.backtest

echo.
echo Done. Restart the dashboard (or click "Refresh predictions") to pick up the new data.
pause
