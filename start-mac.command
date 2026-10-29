#!/bin/bash
# Double-click (or run "bash start-mac.command") to install on first run and start the app.
cd "$(dirname "$0")" || exit 1

if ! command -v npm >/dev/null 2>&1; then
  echo "Node.js is not installed."
  echo "Install the LTS version from https://nodejs.org, then run this file again."
  open "https://nodejs.org" 2>/dev/null
  read -r -p "Press Enter to close..."
  exit 1
fi

if [ ! -d node_modules ]; then
  echo "First run: installing packages (1-3 minutes)..."
  if ! npm install; then
    echo "Installing packages failed. See the Troubleshooting section in README.md."
    read -r -p "Press Enter to close..."
    exit 1
  fi
fi

echo "Starting the app at http://localhost:5178"
echo "To stop it, close this window or press Ctrl+C."
npm run dev -- --open
