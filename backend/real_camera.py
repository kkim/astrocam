import cv2
import threading
import time
import subprocess
import numpy as np
from typing import Optional, Dict, Any, Tuple
from logger import event_logger
from camera_discovery import find_camera_device

class RealCamera:
    """
    Manages physical USB camera hardware (e.g. SVBONY SV205) using V4L2.
    Includes automatic device discovery, stream recovery, and reconnect capabilities.
    """
    def __init__(self, camera_id: Optional[int] = None, format: str = 'YUYV'):
        self.camera_id = camera_id
        self.device_name = "Unknown"
        self.cap: Optional[cv2.VideoCapture] = None
        self.width, self.height = 1920, 1080
        self.format = format.upper()
        self.raw_frame: Optional[np.ndarray] = None
        self.frame_id: int = 0
        self.lock = threading.Lock()
        self.is_running = True
        self._last_error_log_time = 0
        
        self.params: Dict[str, Any] = {
            "brightness": 0, "contrast": 50, "saturation": 0,
            "gain": 64, "exposure": 1000, "sharpness": 50, "auto_exposure": 0,
            "auto_wb": 0, "gamma": 100
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

        # Attempt preferred format (default: uncompressed YUYV)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self.format))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # Allow brief driver settling and retry grab check before falling back
        grabbed = False
        for _ in range(3):
            if self.cap.grab():
                grabbed = True
                break
            time.sleep(0.05)

        if not grabbed:
            if self.format != 'MJPG':
                event_logger.log(f"Camera warning: Format {self.format} failed to capture, falling back to MJPG...")
                self.format = 'MJPG'
                self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self.format))
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        # Re-apply stored parameters (exposure, gain, auto_wb=0, etc.)
        self._apply_all_params()

        event_logger.log(f"Camera Connected: /dev/video{self.camera_id} ('{self.device_name}') at {self.width}x{self.height} [{self.format}]")
        return True

    def _set_v4l2_ctrl(self, ctrl_name: str, value: int):
        if self.camera_id is not None:
            try:
                subprocess.run(
                    ["v4l2-ctl", "-d", f"/dev/video{self.camera_id}", f"--set-ctrl={ctrl_name}={value}"],
                    capture_output=True,
                    timeout=1.0
                )
            except Exception:
                pass

    def _apply_all_params(self):
        if not self.cap or not self.cap.isOpened():
            return

        # Always enable dynamic framerate so manual exposures (>33ms) are not clipped to 30 FPS
        self._set_v4l2_ctrl("exposure_dynamic_framerate", 1)

        auto_exp = self.params.get("auto_exposure", 0)
        if auto_exp == 0:
            # SV205 Cold-Boot Wakeup / ISP Prime:
            # On startup, the SV205 ISP's internal exposure integration engine is asleep.
            # Briefly pulsing auto_exposure (mode 3) initializes the sensor's counters and sensitivity,
            # then locking to manual (mode 1) ensures manual exposures immediately take effect.
            self._set_v4l2_ctrl("auto_exposure", 3)
            time.sleep(0.2)
            self._set_v4l2_ctrl("auto_exposure", 1)
            try:
                self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
            except Exception:
                pass

            # Apply target manual exposure
            if "exposure" in self.params:
                exp_val = int(self.params["exposure"])
                self._set_v4l2_ctrl("exposure_time_absolute", exp_val)
                try:
                    self.cap.set(cv2.CAP_PROP_EXPOSURE, exp_val)
                except Exception:
                    pass
        else:
            self._set_v4l2_ctrl("auto_exposure", 3)
            try:
                self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)
            except Exception:
                pass

        # 3. Apply other controls
        ctrl_map = {
            "brightness": (cv2.CAP_PROP_BRIGHTNESS, "brightness"),
            "contrast": (cv2.CAP_PROP_CONTRAST, "contrast"),
            "saturation": (cv2.CAP_PROP_SATURATION, "saturation"),
            "gain": (cv2.CAP_PROP_GAIN, "gain"),
            "sharpness": (cv2.CAP_PROP_SHARPNESS, "sharpness"),
            "auto_wb": (cv2.CAP_PROP_AUTO_WB, "white_balance_automatic"),
            "white_balance_automatic": (cv2.CAP_PROP_AUTO_WB, "white_balance_automatic"),
            "gamma": (cv2.CAP_PROP_GAMMA, "gamma"),
        }
        for prop, (cv_prop, v4l_name) in ctrl_map.items():
            if prop in self.params:
                value = self.params[prop]
                if prop in ("auto_wb", "white_balance_automatic"):
                    val = 1 if value > 0 else 0
                elif prop == "gamma":
                    val = int(np.clip(value, 100, 500))
                elif prop == "brightness":
                    val = int(np.clip(value, -128, 128))
                else:
                    val = int(value)
                try:
                    self.cap.set(cv_prop, val)
                except Exception:
                    pass
                self._set_v4l2_ctrl(v4l_name, val)

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
                        self.frame_id += 1
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

    def get_fresh_frame(self, last_frame_id: int = -1) -> Tuple[Optional[np.ndarray], int]:
        """Returns the frame ONLY if it is newer than last_frame_id, along with current frame_id."""
        with self.lock:
            if self.raw_frame is None or self.frame_id == last_frame_id:
                return None, self.frame_id
            return self.raw_frame.copy(), self.frame_id

    def set_camera_param(self, prop: str, value: float) -> bool:
        with self.lock:
            if prop == "auto_exposure":
                self.params["auto_exposure"] = 1 if value > 0 else 0
                auto_val = 3 if value > 0 else 1
                self._set_v4l2_ctrl("auto_exposure", auto_val)
                if self.cap and self.cap.isOpened():
                    try:
                        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, auto_val)
                    except Exception:
                        pass
                # If reverting to manual mode, immediately re-apply target manual exposure
                if value == 0 and "exposure" in self.params:
                    exp_val = int(self.params["exposure"])
                    self._set_v4l2_ctrl("exposure_time_absolute", exp_val)
                    if self.cap and self.cap.isOpened():
                        try:
                            self.cap.set(cv2.CAP_PROP_EXPOSURE, exp_val)
                        except Exception:
                            pass
                return True

            if prop == "exposure":
                # Setting manual exposure automatically enforces auto_exposure=0 (Manual Mode)
                self.params["auto_exposure"] = 0
                self.params["exposure"] = float(value)
                self._set_v4l2_ctrl("exposure_dynamic_framerate", 1)
                self._set_v4l2_ctrl("auto_exposure", 1)
                if self.cap and self.cap.isOpened():
                    try:
                        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
                    except Exception:
                        pass
                exp_val = int(value)
                self._set_v4l2_ctrl("exposure_time_absolute", exp_val)
                if self.cap and self.cap.isOpened():
                    try:
                        self.cap.set(cv2.CAP_PROP_EXPOSURE, exp_val)
                    except Exception:
                        pass
                return True

            ctrl_map = {
                "brightness": (cv2.CAP_PROP_BRIGHTNESS, "brightness"),
                "contrast": (cv2.CAP_PROP_CONTRAST, "contrast"),
                "saturation": (cv2.CAP_PROP_SATURATION, "saturation"),
                "gain": (cv2.CAP_PROP_GAIN, "gain"),
                "sharpness": (cv2.CAP_PROP_SHARPNESS, "sharpness"),
                "auto_wb": (cv2.CAP_PROP_AUTO_WB, "white_balance_automatic"),
                "white_balance_automatic": (cv2.CAP_PROP_AUTO_WB, "white_balance_automatic"),
                "gamma": (cv2.CAP_PROP_GAMMA, "gamma"),
            }
            if prop in ctrl_map:
                if prop == "gamma":
                    value = float(np.clip(value, 100, 500))
                elif prop == "brightness":
                    value = float(np.clip(value, -128, 128))
                self.params[prop] = value
                cv_prop, v4l_name = ctrl_map[prop]
                val = 1 if prop in ("auto_wb", "white_balance_automatic") and value > 0 else int(value)
                if self.cap and self.cap.isOpened():
                    try:
                        self.cap.set(cv_prop, val)
                    except Exception:
                        pass
                self._set_v4l2_ctrl(v4l_name, val)
                return True
        return False

    def set_format(self, fmt: str) -> bool:
        fmt = fmt.upper()
        if fmt not in ("YUYV", "MJPG"):
            event_logger.log(f"Camera error: Unsupported format {fmt}. Must be YUYV or MJPG.")
            return False
        if fmt == self.format:
            return True
        with self.lock:
            old_fmt = self.format
            self.format = fmt
            event_logger.log(f"Camera: Switching format from {old_fmt} to {fmt}...")
            success = self._init_camera()
            if not success:
                event_logger.log(f"Camera: Failed to initialize in {fmt}, reverting to {old_fmt}...")
                self.format = old_fmt
                self._init_camera()
                return False
            return True

    def get_camera_params(self) -> Dict[str, Any]:
        res = {**self.params, "format": self.format}
        if self.cap and self.cap.isOpened():
            for p, prop_id in {
                "brightness": cv2.CAP_PROP_BRIGHTNESS,
                "exposure": cv2.CAP_PROP_EXPOSURE,
                "auto_exposure": cv2.CAP_PROP_AUTO_EXPOSURE,
                "auto_wb": cv2.CAP_PROP_AUTO_WB,
                "gamma": cv2.CAP_PROP_GAMMA,
            }.items():
                try:
                    val = self.cap.get(prop_id)
                    if p == "auto_exposure" and val > 0:
                        res[p] = 1 if val >= 3 else 0
                    elif p == "auto_wb" and val >= 0:
                        res[p] = 1 if val > 0 else 0
                    elif val > 0:
                        res[p] = val
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
            "camera_mode": "real",
            "format": self.format
        }

    def close(self):
        self.is_running = False
        if hasattr(self, "capture_thread") and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=1.0)
        if self.cap:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None
