import os
import glob
from typing import Tuple, Optional
import cv2
from logger import event_logger

def find_camera_device(preferred_id: Optional[int] = None) -> Tuple[Optional[int], Optional[str]]:
    """
    Auto-discovers the best available video capture device (V4L2).
    
    Prioritizes dedicated astronomy cameras (SVBONY / SV205), followed by
    other USB cameras, before falling back to any valid video device.
    
    Returns:
        (device_index, device_name) or (None, None) if no camera is available.
    """
    # If a specific numeric device index was requested, test it first
    if preferred_id is not None and isinstance(preferred_id, int):
        cap = cv2.VideoCapture(preferred_id, cv2.CAP_V4L2)
        if cap.isOpened():
            cap.release()
            dev_name = _get_device_name_by_index(preferred_id) or f"Camera {preferred_id}"
            return preferred_id, dev_name

    sysfs_devices = sorted(
        glob.glob("/sys/class/video4linux/video*"),
        key=lambda p: int(os.path.basename(p).replace("video", "")) if os.path.basename(p).replace("video", "").isdigit() else 999
    )

    if not sysfs_devices:
        event_logger.log("Camera Discovery: No /sys/class/video4linux devices found.")
        return None, None

    candidates = []
    for dev_path in sysfs_devices:
        basename = os.path.basename(dev_path)
        idx_str = basename.replace("video", "")
        if not idx_str.isdigit():
            continue
        idx = int(idx_str)

        name = "Unknown"
        name_file = os.path.join(dev_path, "name")
        if os.path.exists(name_file):
            try:
                with open(name_file, "r") as f:
                    name = f.read().strip()
            except Exception:
                pass

        # Calculate priority:
        # 0: SVBONY / SV205
        # 1: General USB Camera
        # 2: Other V4L2 device
        name_lower = name.lower()
        if "svbony" in name_lower or "sv205" in name_lower:
            priority = 0
        elif "camera" in name_lower or "usb" in name_lower or "webcam" in name_lower:
            priority = 1
        elif "pispbe" in name_lower or "hevc" in name_lower or "bcm2835" in name_lower:
            priority = 3 # Hardware decoder/ISP endpoints (usually not direct image sensor)
        else:
            priority = 2

        candidates.append((priority, idx, name))

    # Sort by priority ascending, then by index ascending
    candidates.sort(key=lambda c: (c[0], c[1]))

    for priority, idx, name in candidates:
        if priority >= 3:
            # Skip known hardware decoders / ISP nodes
            continue
        try:
            cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
            if cap.isOpened():
                cap.release()
                event_logger.log(f"Camera Discovery: Selected /dev/video{idx} ('{name}')")
                return idx, name
        except Exception as e:
            event_logger.log(f"Camera Discovery: Failed opening /dev/video{idx}: {e}")

    # Fallback: probe indices 0 through 3 directly
    for probe_idx in range(4):
        try:
            cap = cv2.VideoCapture(probe_idx, cv2.CAP_V4L2)
            if cap.isOpened():
                cap.release()
                name = _get_device_name_by_index(probe_idx) or f"Camera {probe_idx}"
                event_logger.log(f"Camera Discovery: Fallback found /dev/video{probe_idx} ('{name}')")
                return probe_idx, name
        except Exception:
            pass

    event_logger.log("Camera Discovery: No functional camera hardware detected.")
    return None, None

def _get_device_name_by_index(idx: int) -> Optional[str]:
    name_file = f"/sys/class/video4linux/video{idx}/name"
    if os.path.exists(name_file):
        try:
            with open(name_file, "r") as f:
                return f.read().strip()
        except Exception:
            pass
    return None
