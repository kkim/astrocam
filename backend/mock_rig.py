import cv2
import threading
import time
import numpy as np
import os
from base_rig import BaseAstroRig
from logger import event_logger

class MockAstroRig(BaseAstroRig):
    def __init__(self):
        self.width = 1920
        self.height = 1080
        self.full_width = int(self.width * 2.0)
        self.full_height = int(self.height * 1.5)
        self.is_running = True
        self.lock = threading.Lock()
        
        # Camera parameters
        self.params = {
            "brightness": 128, "contrast": 32, "saturation": 64,
            "gain": 0, "exposure": 156, "sharpness": 2, "auto_exposure": 0
        }
        
        # Motor state
        self.current_duty = 80.0
        self.target_duty = 80.0
        self.ramp_thread = None
        self.stop_ramping = threading.Event()
        
        # Drift position
        self.pos_x = self.width * 0.25
        self.pos_y = self.height * 0.25
        self.last_update_time = time.time()

        # Load persisted simulation parameters from config.json if available
        import json
        config_data = {}
        if os.path.exists("config.json"):
            try:
                with open("config.json", "r") as f:
                    config_data = json.load(f)
            except Exception:
                pass

        # Simulated Tracking drift parameters (diurnal rot angle & raw drift)
        self.camera_angle = np.deg2rad(config_data.get("camera_angle", 45.0))
        self.sim_drift_speed = float(config_data.get("sim_drift_speed", 60.0))
        if self.sim_drift_speed < 30.0:
            self.sim_drift_speed = 60.0 # Upgrade legacy slow settings
        self.sim_drift_angle = self.camera_angle  # Perfectly aligned to RA axis (perfect polar alignment)

        # Generate static starfield
        self.static_starfield = self._generate_static_starfield()
        self.start_time = time.time()

        # Main simulation thread
        self.raw_frame = None
        self.thread = threading.Thread(target=self._sim_loop, daemon=True)
        self.thread.start()

    def _generate_static_starfield(self):
        img = np.zeros((self.full_height, self.full_width, 3), dtype=np.uint8)
        rng = np.random.default_rng(42)
        for _ in range(5000):
            x = rng.integers(0, self.full_width)
            y = rng.integers(0, self.full_height)
            size = rng.choice([0, 1, 2, 4], p=[0.9, 0.09, 0.009, 0.001])
            brightness = int(rng.integers(150, 255))
            cv2.circle(img, (x, y), int(size), (brightness, brightness, brightness), -1)
        return img

    def _sim_loop(self):
        while self.is_running:
            now = time.time()
            dt = now - self.last_update_time
            
            # Diurnal drift vector
            v_diurnal_x = self.sim_drift_speed * np.cos(self.sim_drift_angle)
            v_diurnal_y = self.sim_drift_speed * np.sin(self.sim_drift_angle)
            
            # Mount compensation vector: at 85% duty cycle, mount speed matches sim_drift_speed
            # and moves in the direction opposite to the camera_angle to cancel out drift.
            v_mount_mag = - (self.current_duty / 85.0) * self.sim_drift_speed
            v_mount_x = v_mount_mag * np.cos(self.camera_angle)
            v_mount_y = v_mount_mag * np.sin(self.camera_angle)
            
            v_x = v_diurnal_x + v_mount_x
            v_y = v_diurnal_y + v_mount_y
            
            self.pos_x = (self.pos_x + v_x * dt) % (self.full_width - self.width)
            self.pos_y = (self.pos_y + v_y * dt) % (self.full_height - self.height)
            
            self.last_update_time = now
            
            frame = self._generate_mock_frame()
            with self.lock:
                self.raw_frame = frame
            time.sleep(0.04) # 25 FPS

    def _generate_mock_frame(self):
        cx = int(self.pos_x)
        cy = int(self.pos_y)
        crop = self.static_starfield[cy:cy+self.height, cx:cx+self.width].copy()
        
        # Add dynamic sensor noise
        noise = np.random.normal(5, 2, (self.height, self.width, 3)).astype(np.uint8)
        frame = cv2.add(crop, noise)

        cv2.putText(frame, f"MOCK DRIFT - {time.strftime('%H:%M:%S')}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(frame, f"Speed: {self.current_duty:.1f}%", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1)
        return frame

    def get_raw_frame(self):
        with self.lock:
            if self.raw_frame is None: return None
            return self.raw_frame.copy()

    def set_camera_param(self, prop, value):
        with self.lock:
            self.params[prop] = value
        return True

    def get_camera_params(self):
        return self.params

    def get_camera_status(self):
        return {"connected": True, "width": self.width, "height": self.height}

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
        return {"duty_cycle": round(self.current_duty, 2), "voltage": round(3.3 * (self.current_duty / 100.0), 2), "mock_mode": True}

    def close(self):
        self.is_running = False
        self.stop_ramping.set()

    def set_camera_angle(self, angle_deg):
        with self.lock:
            self.camera_angle = np.deg2rad(angle_deg % 360.0)
            self.sim_drift_angle = self.camera_angle  # Perfectly aligned to RA axis
            self._save_to_config("camera_angle", float(angle_deg % 360.0))
            self._save_to_config("sim_drift_angle", float(angle_deg % 360.0))
        return True

    def set_sim_drift(self, speed, angle_deg):
        with self.lock:
            self.sim_drift_speed = float(speed)
            # Drift angle is locked to the mount's RA axis (camera angle) under perfect polar alignment
            self.sim_drift_angle = self.camera_angle
            self._save_to_config("sim_drift_speed", float(speed))
            self._save_to_config("sim_drift_angle", float(np.rad2deg(self.camera_angle)))
        return True

    def _save_to_config(self, key, value):
        import json
        config_data = {}
        if os.path.exists("config.json"):
            try:
                with open("config.json", "r") as f:
                    config_data = json.load(f)
            except Exception:
                pass
        config_data[key] = value
        try:
            with open("config.json", "w") as f:
                json.dump(config_data, f)
        except Exception:
            pass
