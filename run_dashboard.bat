@echo off
cd /d "%~dp0"

if not exist venv\Scripts\python.exe (
    echo Virtual environment not found. Run setup.bat first.
    pause
    exit /b 1
)

if not exist data\processed\games.csv (
    echo No historical data found. Run setup.bat first.
    pause
    exit /b 1
)

venv\Scripts\python.exe -m streamlit run dashboard\app.py
pause
