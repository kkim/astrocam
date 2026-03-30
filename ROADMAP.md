# AstroCam Evolution Roadmap

This document outlines the high-level plan for transforming AstroCam into an automated astrophotography and panorama system.

## 1. Hardware Control: Equatorial Mount Drive
The system will interface with a voltage controller on the Raspberry Pi to manage telescope tracking.

### Goals:
- **Sidereal Tracking:** Maintain constant voltage to match celestial movement.
- **Controlled Drift:** Adjust voltage (speed) to be slightly faster or slower than sidereal rate.
- **Panorama Support:** Use the drift caused by speed offsets and polar misalignment to "scan" the sky for wide-angle panoramas.

### Implementation:
- **Backend:** Create a `motor_control.py` module to handle GPIO/PWM signals.
- **Frontend:** Add a "Mount Control" section with a fine-tuned speed slider and presets (Sidereal, Drift+, Drift-).

## 2. Advanced Hierarchical Stacking
Transition from simple frame averaging to a multi-tier stacking engine for high dynamic range (HDR) and deep-space imaging.

### Tier 1: Real-time Accumulation (Short-term)
- Accumulate ~100 frames in memory (using `cv2.accumulateWeighted` or similar).
- **Purpose:** Reduce sensor thermal noise and read noise for a clean live view.

### Tier 2: Sequence Capture (Medium-term)
- Automatically save Tier 1 stacks to disk as high-quality files (TIFF/FITS).
- **Purpose:** Build a library of "clean" sub-exposures over a long observation session.

### Tier 3: Master Stack (Long-term)
- Combine multiple Tier 2 images into a final high-bit-depth master image.
- **Purpose:** Achieve maximum dynamic range and detail for final processing.

## 3. Panorama Strategy
- Use the **Controlled Drift** (from Section 1) to naturally move the telescope across a target area.
- Synchronize Tier 2 captures so that each saved image represents a "tile" in the panorama.
- Ensure sufficient overlap between tiles for post-session stitching.

---
*Last updated: March 21, 2026*
