@echo off
cd /d "%~dp0"

echo === MLB Prediction Model setup ===
echo.

if not exist venv\Scripts\python.exe (
    echo Creating virtual environment...
    python -m venv venv
)

echo Installing dependencies...
venv\Scripts\python.exe -m pip install --quiet --upgrade pip
venv\Scripts\pip.exe install --quiet -r requirements.txt

echo.
echo Pulling historical game data from Baseball-Reference (2023-present).
echo This takes several minutes and is rate-limited to be polite to the source site.
echo.
venv\Scripts\python.exe -m src.data.fetch_historical

echo.
echo Running an initial backtest so the dashboard has performance numbers to show...
venv\Scripts\python.exe -m src.backtest.backtest

echo.
echo === Setup complete ===
echo Run run_dashboard.bat to launch the dashboard.
pause
