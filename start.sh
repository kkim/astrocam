#!/bin/bash

# Kill any existing processes
echo "Cleaning up old processes..."
pkill -9 -f "python3 main.py" || true
pkill -9 -f "vite" || true
pkill -9 -f "uvicorn" || true
sleep 4

# Start Backend
echo "Starting Backend..."
cd /home/kio/projects/astrocam/backend
source venv/bin/activate
# Start in background but keep PID
python3 main.py > ../backend.log 2>&1 &
BACKEND_PID=$!

# Start Frontend
echo "Starting Frontend..."
cd /home/kio/projects/astrocam/frontend
npm run dev > ../frontend.log 2>&1 &
FRONTEND_PID=$!

# Get local IP address
IP_ADDR=$(hostname -I | awk '{print $1}')

echo "-------------------------------------------"
echo "AstroCam is starting!"
echo "Local Access:"
echo "  Backend:  http://localhost:8000"
echo "  Frontend: http://localhost:5173"
echo ""
echo "Network Access (for your phone):"
echo "  Backend:  http://$IP_ADDR:8000"
echo "  Frontend: http://$IP_ADDR:5173"
echo "-------------------------------------------"
echo "Logs are in backend.log and frontend.log"

# Function to kill child processes
cleanup() {
    echo "Shutting down..."
    kill $BACKEND_PID $FRONTEND_PID
    exit
}

trap cleanup SIGINT SIGTERM EXIT
wait
