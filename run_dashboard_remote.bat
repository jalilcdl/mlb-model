@echo off
REM ---------------------------------------------------------------------------
REM Launches the dashboard so it's reachable from your other devices (e.g. your
REM phone) over a private Tailscale mesh. See REMOTE_ACCESS.md for the full,
REM one-time setup. Keep this window open and this desktop powered on while you
REM want remote access.
REM
REM Optional password: to require a password in addition to Tailscale, set one
REM before launching (uncomment the next line and change the value):
REM set MLB_DASHBOARD_PASSWORD=changeme
REM ---------------------------------------------------------------------------
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

echo.
echo Starting dashboard for remote access on port 8501.
echo On your phone (with Tailscale connected), open:  http://YOUR-DESKTOP-TAILSCALE-IP:8501
echo Find that IP by running:  tailscale ip -4
echo Press Ctrl+C in this window to stop remote access.
echo.

REM --server.address 0.0.0.0 makes it reachable from other devices on the tailnet
REM (not just this machine); --server.headless true stops it opening a local browser.
venv\Scripts\python.exe -m streamlit run dashboard\app.py --server.address 0.0.0.0 --server.port 8501 --server.headless true
pause
