# AstroCam Project Context

## Overview
AstroCam is an automated astrophotography rig using a Raspberry Pi, an SV205 camera, and a DC motor-driven equatorial mount.

## Current Technical State
- **Backend:** FastAPI (Python 3).
  - `astro_rig.py`: Factory for switching between `RealAstroRig` and `MockAstroRig`.
  - `real_rig.py`: OpenCV capture, PWM motor control with ramping (GPIO 18), and frame averaging.
  - `logger.py`: In-memory event logger for UI feedback.
- **Frontend:** React + Vite + TypeScript.
  - Dashboard with live stream, health telemetry, motor presets (Sidereal/Drift), and system logs.
- **Infrastructure:**
  - `astrocam.service`: systemd service (Type=simple).
  - `start.sh`: Uses `wait` and a `SIGTERM` trap to keep background processes (Vite/FastAPI) alive and ensure clean shutdown.

## Key Conventions
- **Mock Mode:** System defaults to `MockAstroRig` if camera hardware isn't detected. Mock rig simulates noise and stacking for UI testing.
- **Stacking Logic:** Changing any camera parameter (exposure, gain, etc.) or the `average` (N) count automatically resets the Tier 1 accumulator to prevent "ghosting" artifacts.
- **Paths:** Captures are stored in `/home/kio/projects/astrocam/captures`.
- **API:** Backend runs on port 8000; Frontend on 5173.

## Active Roadmap: Stacking & Panorama
1. **Tier 1 Stacking:** (Implemented) Real-time accumulation using `cv2.accumulateWeighted` in both Real and Mock rigs. Reduced noise in live view.
2. **Tier 2 Stacking:** (Implemented) High-quality frames saved to disk via manual capture or sequence mode.
3. **Panorama:** (Next Up) Sync captures with "Controlled Drift" motor offsets.
