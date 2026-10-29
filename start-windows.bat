@echo off
rem Double-click to install on first run and start the app.
cd /d "%~dp0"

where npm >nul 2>nul
if errorlevel 1 (
  echo Node.js is not installed.
  echo Install the LTS version from https://nodejs.org, then run this file again.
  start "" "https://nodejs.org"
  pause
  exit /b 1
)

if not exist node_modules (
  echo First run: installing packages, 1-3 minutes...
  call npm install
  if errorlevel 1 (
    echo Installing packages failed. See the Troubleshooting section in README.md.
    pause
    exit /b 1
  )
)

echo Starting the app at http://localhost:5178
echo To stop it, close this window or press Ctrl+C.
call npm run dev -- --open
pause
