# AstroCam WebUI Documentation

The AstroCam WebUI is a React-based interface designed for real-time telescope control and astrophotography. It communicates with a FastAPI backend on the Raspberry Pi.

## 1. Accessing the Interface

The UI is hosted on port **5173**. 
- **Local:** `http://localhost:5173`
- **Network:** `http://<your-pi-ip>:5173`

## 2. Dashboard Overview

### 2.1 Live Stream
The central viewport shows the live camera feed.
- **Mock Mode:** If no camera is detected, the UI displays a simulated starfield with a moving planet for testing.
- **Luminance:** Shows real-time average brightness of the center of the frame.
- **FPS:** Displays the current frame processing rate.

### 2.2 Sidebar Controls
The sidebar contains three primary control sections:

#### **Mount Control**
Used to drive the equatorial mount via PWM on GPIO 18.
- **Tracking Speed (0.2% steps):** Fine-tune the motor voltage.
- **Sidereal (85.0%):** Standard celestial tracking rate.
- **Drift +/-:** Adjusts speed in 0.2% increments to induce or correct for celestial drift (useful for panoramas).
- **MOCK Badge:** Indicates the motor is running in simulation mode.

#### **Sequence Capture**
Automates the saving of Tier-1 (noise-reduced) stacks.
- **Frames:** Total number of images to capture.
- **Interval (s):** Time delay between each saved frame.
- **Progress Bar:** Appears during an active sequence to show completion status.
- **Storage:** Images are saved to `/home/kio/projects/astrocam/captures/` on the Pi.

#### **Camera Settings**
Standard V4L2 controls:
- **Exposure:** Toggle between **Auto** and **Manual**. In Manual mode, an Exposure slider appears for precise control.
- **Gain/Brightness/Contrast/Saturation:** Adjust sensor parameters.
- **Average (N frames):** Real-time stacking. Higher values significantly reduce noise but increase motion blur.
- **Resolution:** Switch between 8MP, 1080p, 720p, and VGA.

## 3. Advanced Features

### Real-time Stacking (Tier 1)
By adjusting the **Average** slider, the system uses `cv2.accumulateWeighted` to blend incoming frames. This is essential for bringing out faint detail in dark-sky objects while monitoring the live feed.

### Mobile Responsiveness
The UI includes a collapsible sidebar for use on smartphones at the telescope. Use the **Menu (hamburger)** icon in the top right to toggle controls.
