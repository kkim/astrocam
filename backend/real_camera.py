import cv2
import threading
import time
import numpy as np
from typing import Optional, Dict, Any
from logger import event_logger
from camera_discovery import find_camera_device

class RealCamera:
    """
    Manages physical USB camera hardware (e.g. SVBONY SV205) using V4L2.
    Includes automatic device discovery, stream recovery, and reconnect capabilities.
    """
    def __init__(self, camera_id: Optional[int] = None):
        self.camera_id = camera_id
        self.device_name = "Unknown"
        self.cap: Optional[cv2.VideoCapture] = None
        self.width, self.height = 1920, 1080
        self.format = 'MJPG'
        self.raw_frame: Optional[np.ndarray] = None
        self.lock = threading.Lock()
        self.is_running = True
        self._last_error_log_time = 0
        
        self.params: Dict[str, Any] = {
            "brightness": 128, "contrast": 32, "saturation": 64,
            "gain": 0, "exposure": 156, "sharpness": 2, "auto_exposure": 0
        }
        
        self._init_camera()
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()

    def _init_camera(self) -> bool:
        if self.cap:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None

        dev_idx, dev_name = find_camera_device(self.camera_id)
        if dev_idx is None:
            now = time.time()
            if now - self._last_error_log_time > 10:
                event_logger.log("Camera Error: Could not detect or open camera hardware")
                self._last_error_log_time = now
            with self.lock:
                self.raw_frame = None
            return False

        self.camera_id = dev_idx
        self.device_name = dev_name or f"Device {dev_idx}"

        self.cap = cv2.VideoCapture(self.camera_id, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            now = time.time()
            if now - self._last_error_log_time > 10:
                event_logger.log(f"Camera Error: Could not open /dev/video{self.camera_id} ('{self.device_name}')")
                self._last_error_log_time = now
            with self.lock:
                self.raw_frame = None
            return False

        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self.format))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # Re-apply stored parameters
        self._apply_all_params()

        event_logger.log(f"Camera Connected: /dev/video{self.camera_id} ('{self.device_name}') at {self.width}x{self.height}")
        return True

    def _apply_all_params(self):
        if not self.cap or not self.cap.isOpened():
            return
        mapping = {
            "brightness": cv2.CAP_PROP_BRIGHTNESS,
            "contrast": cv2.CAP_PROP_CONTRAST,
            "saturation": cv2.CAP_PROP_SATURATION,
            "gain": cv2.CAP_PROP_GAIN,
            "exposure": cv2.CAP_PROP_EXPOSURE,
            "sharpness": cv2.CAP_PROP_SHARPNESS,
            "auto_exposure": cv2.CAP_PROP_AUTO_EXPOSURE,
        }
        for prop, value in self.params.items():
            if prop in mapping:
                val = (3 if value > 0 else 1) if prop == "auto_exposure" else value
                try:
                    self.cap.set(mapping[prop], val)
                except Exception:
                    pass

    def reconnect(self) -> bool:
        """Explicitly re-scans devices and reconnects to camera hardware."""
        event_logger.log("Camera: Manual reconnect requested...")
        # Reset camera_id to allow dynamic discovery if device path moved
        self.camera_id = None
        success = self._init_camera()
        return success

    def _capture_loop(self):
        consecutive_failures = 0
        while self.is_running:
            if not self.cap or not self.cap.isOpened():
                time.sleep(1.0)
                self._init_camera()
                continue

            if self.cap.grab():
                ret, frame = self.cap.retrieve()
                if ret and frame is not None:
                    consecutive_failures = 0
                    with self.lock:
                        self.raw_frame = frame
                else:
                    consecutive_failures += 1
            else:
                consecutive_failures += 1
                time.sleep(0.01)

            # If frames fail repeatedly, the USB device likely dropped or reset
            if consecutive_failures > 30:
                event_logger.log("Camera warning: frame stream dropped. Attempting recovery...")
                with self.lock:
                    self.raw_frame = None
                if self.cap:
                    try:
                        self.cap.release()
                    except Exception:
                        pass
                    self.cap = None
                consecutive_failures = 0
                time.sleep(1.5)
                self._init_camera()

    def get_raw_frame(self) -> Optional[np.ndarray]:
        with self.lock:
            if self.raw_frame is None:
                return None
            return self.raw_frame.copy()

    def set_camera_param(self, prop: str, value: float) -> bool:
        mapping = {
            "brightness": cv2.CAP_PROP_BRIGHTNESS,
            "contrast": cv2.CAP_PROP_CONTRAST,
            "saturation": cv2.CAP_PROP_SATURATION,
            "gain": cv2.CAP_PROP_GAIN,
            "exposure": cv2.CAP_PROP_EXPOSURE,
            "sharpness": cv2.CAP_PROP_SHARPNESS,
            "auto_exposure": cv2.CAP_PROP_AUTO_EXPOSURE,
        }
        with self.lock:
            if prop in mapping:
                self.params[prop] = value
                if self.cap and self.cap.isOpened():
                    val = (3 if value > 0 else 1) if prop == "auto_exposure" else value
                    try:
                        self.cap.set(mapping[prop], val)
                    except Exception:
                        pass
        return True

    def get_camera_params(self) -> Dict[str, Any]:
        res = {**self.params}
        if self.cap and self.cap.isOpened():
            for p, prop_id in {
                "brightness": cv2.CAP_PROP_BRIGHTNESS,
                "exposure": cv2.CAP_PROP_EXPOSURE,
                "auto_exposure": cv2.CAP_PROP_AUTO_EXPOSURE,
            }.items():
                try:
                    val = self.cap.get(prop_id)
                    res[p] = (1 if val >= 3 else 0) if p == "auto_exposure" else val
                except Exception:
                    pass
        return res

    def get_camera_status(self) -> Dict[str, Any]:
        is_connected = self.cap is not None and self.cap.isOpened() and self.raw_frame is not None
        return {
            "connected": is_connected,
            "width": self.width,
            "height": self.height,
            "camera_id": self.camera_id,
            "device_name": self.device_name,
            "camera_mode": "real"
        }

    def close(self):
        self.is_running = False
        if self.cap:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None
