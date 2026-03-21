# AstroCam for SVBONY SV205

A modern, high-performance web interface for controlling and capturing images from the SVBONY SV205 astronomy camera on Linux.

## Features
- **Live Preview**: Real-time MJPEG stream (1280x720).
- **Camera Controls**: Hardware-level adjustment of exposure, gain, brightness, contrast, etc.
- **Image Capture**: Save high-quality snapshots to the project folder.
- **Modern UI**: Clean, dark-themed interface built with React.

## Quick Start
1. Connect your SV205 camera via USB.
2. Run the startup script:
   ```bash
   ./start.sh
   ```
3. Open your browser to:
   - **Frontend**: [http://localhost:5173](http://localhost:5173)
   - **Backend API**: [http://localhost:8000](http://localhost:8000)

## Project Structure
- `backend/`: FastAPI server and OpenCV camera controller.
- `frontend/`: React + TypeScript frontend.
- `start.sh`: Shell script to run both services in parallel.
- `capture_*.jpg`: Images captured via the interface will appear here.

## Requirements
- Python 3.13+
- Node.js 20+
- `libv4l-dev` (usually pre-installed on Linux)
