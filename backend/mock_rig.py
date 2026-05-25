import cv2
import threading
import time
import numpy as np
import os
from datetime import datetime
from base_rig import BaseAstroRig
from logger import event_logger

class MockAstroRig(BaseAstroRig):
    def __init__(self):
        self.width = 1920
        self.height = 1080
        self.full_width = int(self.width * 1.5)
        self.full_height = int(self.height * 1.5)
        self.n_avg = 1
        self.is_running = True
        self.lock = threading.Lock()
        
        # Camera parameters
        self.params = {
            "brightness": 128, "contrast": 32, "saturation": 64,
            "gain": 0, "exposure": 156, "sharpness": 2, "auto_exposure": 0
        }
        
        # Motor state
        self.current_duty = 0.0
        self.target_duty = 0.0
        self.ramp_thread = None
        self.stop_ramping = threading.Event()
        
        # Sequence state
        self.sequence_info = {
            "active": False, "count": 0, "total": 0, "interval": 0,
            "last_capture_time": 0, "directory": "/home/kio/projects/astrocam/captures"
        }
        self.sequence_stop_event = threading.Event()
        
        # Ensure captures directory exists
        if not os.path.exists(self.sequence_info["directory"]):
            os.makedirs(self.sequence_info["directory"])

        # Generate static starfield
        self.static_starfield = self._generate_static_starfield()
        self.start_time = time.time()

        # Main simulation thread
        self.latest_frame = None
        self.acc_frame = None
        self.thread = threading.Thread(target=self._sim_loop, daemon=True)
        self.thread.start()

    def _generate_static_starfield(self):
        # Dark background
        img = np.zeros((self.full_height, self.full_width, 3), dtype=np.uint8)
        
        # Seed for consistent starfield
        rng = np.random.default_rng(42)
        
        # Add thousands of stars
        for _ in range(2000):
            x = rng.integers(0, self.full_width)
            y = rng.integers(0, self.full_height)
            size = rng.choice([0, 1, 2], p=[0.8, 0.15, 0.05])
            brightness = int(rng.integers(150, 255))
            cv2.circle(img, (x, y), int(size), (brightness, brightness, brightness), -1, cv2.LINE_AA)
            
        # Add a "nebula" hint
        cv2.GaussianBlur(img, (15, 15), 0, dst=img)
        
        return img

    def _sim_loop(self):
        while self.is_running:
            frame = self._generate_mock_frame()
            self._process_frame(frame)
            time.sleep(0.04) # 25 FPS

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

    def _generate_mock_frame(self):
        # Period: 4 minutes (240 seconds)
        period = 240.0
        elapsed = time.time() - self.start_time
        theta = (elapsed % period) / period * 2 * np.pi
        
        # Path: Circular/Elliptical crop offsets
        # Max horizontal slack is 0.5 * width. Center of slack is 0.25 * width.
        # Starting at top center means theta=0 -> x=0.25W, y=0
        cx = int(self.width * 0.25 + self.width * 0.25 * np.sin(theta))
        cy = int(self.height * 0.25 - self.height * 0.25 * np.cos(theta))
        
        # Crop from static starfield
        crop = self.static_starfield[cy:cy+self.height, cx:cx+self.width].copy()
        
        # Add dynamic sensor noise
        noise = np.random.normal(12, 5, (self.height, self.width, 3)).astype(np.uint8)
        frame = cv2.addWeighted(crop, 1.0, noise, 0.5, 0)

        cv2.putText(frame, f"MOCK DRIFT - {time.strftime('%H:%M:%S')}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1)
        return frame

    def get_frame(self):
        with self.lock:
            if self.latest_frame is None: 
                return None
            display_w = 1280
            display_h = int(display_w * (self.height / self.width))
            img = cv2.resize(self.latest_frame, (display_w, display_h))
            ret, jpeg = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ret:
                return jpeg.tobytes()
            return None

    def set_camera_param(self, prop, value):
        with self.lock:
            if prop == "average":
                new_n = int(value)
                if new_n != self.n_avg:
                    self.n_avg = new_n
                    self.acc_frame = None # Reset stack
            else:
                self.params[prop] = value
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
        return {**self.params, "average": self.n_avg}

    def get_camera_status(self):
        return {"connected": True, "mean_brightness": 12.75, "fps": 30.0, "width": self.width, "height": self.height}

    def start_sequence(self, count, interval):
        if self.sequence_info["active"]: return False
        self.sequence_info.update({"active": True, "total": count, "count": 0, "interval": interval})
        self.sequence_stop_event.clear()
        threading.Thread(target=self._sequence_loop, daemon=True).start()
        event_logger.log(f"Mock Sequence: Starting {count} frames")
        return True

    def _sequence_loop(self):
        while self.sequence_info["count"] < self.sequence_info["total"] and not self.sequence_stop_event.is_set():
            if time.time() - self.sequence_info["last_capture_time"] >= self.sequence_info["interval"]:
                with self.lock: frame = self.latest_frame.copy()
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                filename = f"mock_seq_{timestamp}.jpg"
                filepath = os.path.join(self.sequence_info["directory"], filename)
                if cv2.imwrite(filepath, frame):
                    self.sequence_info["count"] += 1
                    self.sequence_info["last_capture_time"] = time.time()
                    event_logger.log(f"Mock Sequence: {self.sequence_info['count']}/{self.sequence_info['total']}")
            time.sleep(0.1)
        self.sequence_info["active"] = False

    def get_sequence_status(self):
        return self.sequence_info

    def set_motor_speed(self, speed, ramp_time=0.5):
        self.target_duty = speed
        if self.ramp_thread and self.ramp_thread.is_alive():
            self.stop_ramping.set()
            self.ramp_thread.join()
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
            time.sleep(ramp_time / steps)
        self.current_duty = end

    def get_motor_status(self):
        return {"duty_cycle": round(self.current_duty, 2), "target_duty": self.target_duty, "voltage": round(3.3 * (self.current_duty / 100.0), 2), "mock_mode": True}

    def close(self):
        self.is_running = False
        self.stop_ramping.set()
        self.sequence_stop_event.set()
