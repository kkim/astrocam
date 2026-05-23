#!/bin/bash

# Kill any existing processes
echo "Cleaning up old processes..."
pkill -9 -f "python3 main.py" || true
pkill -9 -f "vite" || true
pkill -9 -f "uvicorn" || true
sleep 2

# Cleanup function
cleanup() {
    echo "Shutting down AstroCam..."
    kill $BACKEND_PID $FRONTEND_PID 2>/dev/null
    exit 0
}

trap cleanup SIGINT SIGTERM

# Start Backend
echo "Starting Backend..."
cd /home/kio/projects/astrocam/backend
source venv/bin/activate
python3 main.py > ../backend.log 2>&1 &
BACKEND_PID=$!

# Start Frontend
echo "Starting Frontend..."
cd /home/kio/projects/astrocam/frontend
npm run dev -- --host 0.0.0.0 > ../frontend.log 2>&1 &
FRONTEND_PID=$!

echo "AstroCam is running (PIDs: $BACKEND_PID, $FRONTEND_PID)"

# Wait for processes to exit
wait $BACKEND_PID $FRONTEND_PID
