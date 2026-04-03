#!/bin/bash
# Start the Velox V10 trading bot in the background
cd "$(dirname "$0")"

if pgrep -f "python3 main.py" > /dev/null; then
    echo "Bot is already running."
    exit 1
fi

# Pre-launch backup
echo "Running pre-launch backup..."
bash scripts/backup.sh || echo "Backup skipped"

echo "Starting Velox V10 Trading Bot..."
nohup python3 main.py "$@" >> bot.log 2>&1 &
echo "Bot started with PID $!"
echo $! > .bot.pid
