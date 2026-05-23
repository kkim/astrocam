import cv2
import threading
import time
import numpy as np
import os
import json
from datetime import datetime
from base_rig import BaseAstroRig
from logger import event_logger

try:
    from gpiozero import PWMOutputDevice
    HAS_GPIO = True
except (ImportError, RuntimeError):
    HAS_GPIO = False

class RealAstroRig(BaseAstroRig):
    def __init__(self, camera_id=0, motor_pin=18):
        self.camera_id = camera_id
        self.motor_pin_number = motor_pin
        
        # Camera State
        self.cap = None
        self.width, self.height = 1920, 1080
        self.format = 'MJPG'
        self.n_avg = 1
        self.acc_frame = None
        self.latest_frame = None
        self.lock = threading.Lock()
        self.is_running = True
        self.fps = 0.0
        self._frame_count = 0
        self._last_fps_calc_time = time.time()
        
        self.params = {
            "brightness": 128, "contrast": 32, "saturation": 64,
            "gain": 0, "exposure": 156, "sharpness": 2, "auto_exposure": 0
        }
        
        # Motor State
        self.pin = None
        self.current_duty = 0.0
        self.target_duty = 0.0
        self.ramp_thread = None
        self.stop_ramping = threading.Event()
        
        # Sequence State
        self.sequence_info = {
            "active": False, "count": 0, "total": 0, "interval": 0,
            "last_capture_time": 0, "directory": "/home/kio/projects/astrocam/captures"
        }
        self.sequence_stop_event = threading.Event()
        
        if not os.path.exists(self.sequence_info["directory"]):
            os.makedirs(self.sequence_info["directory"])

        self._last_error_log_time = 0
        self._init_camera()
        self._init_motor()
        
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()

    def _init_camera(self):
        if self.cap: self.cap.release()
        self.cap = cv2.VideoCapture(self.camera_id, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            # Throttle error logging to once every 10 seconds
            if time.time() - self._last_error_log_time > 10:
                event_logger.log("Error: Could not open camera hardware")
                self._last_error_log_time = time.time()
            with self.lock:
                self.latest_frame = None
            return False
        
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self.format))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return True

    def _init_motor(self):
        if HAS_GPIO:
            try:
                self.pin = PWMOutputDevice(self.motor_pin_number)
                self.pin.value = 0
            except Exception as e:
                event_logger.log(f"Error: Motor init failed: {e}")

    def _capture_loop(self):
        while self.is_running:
            if not self.cap or not self.cap.isOpened():
                time.sleep(1); self._init_camera(); continue
            if self.cap.grab():
                ret, frame = self.cap.retrieve()
                if ret and frame is not None:
                    self._process_frame(frame)
            else:
                time.sleep(0.01)

    def _process_frame(self, frame):
        with self.lock:
            if self.n_avg <= 1:
                self.latest_frame = frame
                self.acc_frame = None
            else:
                alpha = 1.0 / self.n_avg
                if self.acc_frame is None or self.acc_frame.shape != frame.shape:
                    self.acc_frame = frame.astype('float32')
                else:
                    cv2.accumulateWeighted(frame, self.acc_frame, alpha)
                self.latest_frame = cv2.convertScaleAbs(self.acc_frame)
            
            self._frame_count += 1
            now = time.time()
            if now - self._last_fps_calc_time >= 2.0:
                self.fps = self._frame_count / (now - self._last_fps_calc_time)
                self._frame_count = 0
                self._last_fps_calc_time = now

    def get_frame(self):
        with self.lock:
            if self.latest_frame is None: return None
            display_w = 1280
            display_h = int(display_w * (self.height / self.width))
            img = cv2.resize(self.latest_frame, (display_w, display_h))
            ret, jpeg = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 80])
            return jpeg.tobytes() if ret else None

    def set_camera_param(self, prop, value):
        mapping = {
            "brightness": cv2.CAP_PROP_BRIGHTNESS, "contrast": cv2.CAP_PROP_CONTRAST,
            "saturation": cv2.CAP_PROP_SATURATION, "gain": cv2.CAP_PROP_GAIN,
            "exposure": cv2.CAP_PROP_EXPOSURE, "sharpness": cv2.CAP_PROP_SHARPNESS,
            "auto_exposure": cv2.CAP_PROP_AUTO_EXPOSURE
        }
        with self.lock:
            if prop == "average":
                new_n = int(value)
                if new_n != self.n_avg:
                    self.n_avg = new_n
                    self.acc_frame = None # Reset stack
            elif prop in mapping:
                self.params[prop] = value
                if self.cap and self.cap.isOpened():
                    val = (3 if value > 0 else 1) if prop == "auto_exposure" else value
                    self.cap.set(mapping[prop], val)
                    # Reset stack on camera param changes to avoid ghosts
                    self.acc_frame = None
        return True

    def capture_frame(self):
        with self.lock:
            if self.latest_frame is None: return {"success": False, "error": "No frame"}
            frame = self.latest_frame.copy()
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        filename = f"cap_{timestamp}.jpg"
        filepath = os.path.join(self.sequence_info["directory"], filename)
        if cv2.imwrite(filepath, frame):
            event_logger.log(f"Manual capture saved: {filename}")
            return {"success": True, "filename": filename}
        return {"success": False, "error": "Failed to write file"}

    def get_camera_params(self):
        res = {"average": self.n_avg, **self.params}
        if self.cap and self.cap.isOpened():
            for p, prop_id in {"brightness": cv2.CAP_PROP_BRIGHTNESS, "exposure": cv2.CAP_PROP_EXPOSURE, "auto_exposure": cv2.CAP_PROP_AUTO_EXPOSURE}.items():
                try: 
                    val = self.cap.get(prop_id)
                    res[p] = (1 if val >= 3 else 0) if p == "auto_exposure" else val
                except: pass
        return res

    def get_camera_status(self):
        return {"connected": self.cap is not None and self.cap.isOpened(), "fps": round(self.fps, 1), "width": self.width, "height": self.height}

    def start_sequence(self, count, interval):
        if self.sequence_info["active"]: return False
        self.sequence_info.update({"active": True, "total": count, "count": 0, "interval": interval})
        self.sequence_stop_event.clear()
        threading.Thread(target=self._sequence_loop, daemon=True).start()
        return True

    def _sequence_loop(self):
        while self.sequence_info["count"] < self.sequence_info["total"] and not self.sequence_stop_event.is_set():
            if time.time() - self.sequence_info["last_capture_time"] >= self.sequence_info["interval"]:
                with self.lock: frame = self.latest_frame.copy() if self.latest_frame is not None else None
                if frame is not None:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                    filepath = os.path.join(self.sequence_info["directory"], f"seq_{timestamp}.jpg")
                    if cv2.imwrite(filepath, frame):
                        self.sequence_info["count"] += 1
                        self.sequence_info["last_capture_time"] = time.time()
                        event_logger.log(f"Sequence: {self.sequence_info['count']}/{self.sequence_info['total']}")
            time.sleep(0.1)
        self.sequence_info["active"] = False

    def get_sequence_status(self):
        return self.sequence_info

    def set_motor_speed(self, speed, ramp_time=0.5):
        self.target_duty = speed
        if self.ramp_thread and self.ramp_thread.is_alive():
            self.stop_ramping.set(); self.ramp_thread.join()
        self.stop_ramping.clear()
        self.ramp_thread = threading.Thread(target=self._ramp_logic, args=(ramp_time,), daemon=True)
        self.ramp_thread.start()
        return True

    def _ramp_logic(self, ramp_time):
        start, end = self.current_duty, self.target_duty
        steps = 20
        for i in range(1, steps + 1):
            if self.stop_ramping.is_set(): break
            self.current_duty = start + (end - start) * (i / steps)
            if self.pin: self.pin.value = self.current_duty / 100.0
            time.sleep(ramp_time / steps)
        self.current_duty = end

    def get_motor_status(self):
        return {"duty_cycle": round(self.current_duty, 2), "voltage": round(3.3 * (self.current_duty / 100.0), 2), "mock_mode": not HAS_GPIO}

    def close(self):
        self.is_running = False
        self.stop_ramping.set(); self.sequence_stop_event.set()
        if self.cap: self.cap.release()
        if self.pin: self.pin.value = 0; self.pin.close()
