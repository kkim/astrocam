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

    def _create_star_profile(self, radius: int = 5, peak: float = 220.0, sigma: float = 2.0) -> np.ndarray:
        y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
        dist_sq = x*x + y*y
        mask = dist_sq <= radius*radius
        profile = np.exp(-dist_sq / (2.0 * sigma**2)) * mask
        return (profile * peak).astype(np.float32)

    def _generate_static_starfield(self):
        canvas = np.zeros((self.full_height, self.full_width, 3), dtype=np.float32)
        radius = 5

        bright_star = (int(self.full_width * 0.45), int(self.full_height * 0.45), 220.0)
        dim_stars = [
            (bright_star[0] + 250, bright_star[1] - 170, 10.0),
            (bright_star[0] - 320, bright_star[1] + 180, 9.5),
        ]
        very_dim_stars = [
            (bright_star[0] - 140, bright_star[1] - 210, 2.8),
            (bright_star[0] + 370, bright_star[1] + 120, 2.0),
            (bright_star[0] + 160, bright_star[1] + 250, 1.6),
            (bright_star[0] - 410, bright_star[1] - 150, 2.2),
        ]

        all_stars = [bright_star] + dim_stars + very_dim_stars
        for sx, sy, peak in all_stars:
            prof = self._create_star_profile(radius=radius, peak=peak)
            y1 = max(0, sy - radius)
            y2 = min(self.full_height, sy + radius + 1)
            x1 = max(0, sx - radius)
            x2 = min(self.full_width, sx + radius + 1)
            py1 = y1 - (sy - radius)
            py2 = py1 + (y2 - y1)
            px1 = x1 - (sx - radius)
            px2 = px1 + (x2 - x1)
            canvas[y1:y2, x1:x2, :] += prof[py1:py2, px1:px2, None]

        return canvas

    def _sim_loop(self):
        while self.is_running:
            now = time.time()
            dt = now - self.last_update_time
            
            # Diurnal drift vector
            v_diurnal_x = self.sim_drift_speed * np.cos(self.sim_drift_angle)
            v_diurnal_y = self.sim_drift_speed * np.sin(self.sim_drift_angle)
            
            # Mount compensation vector: at 85% duty cycle, mount speed matches sim_drift_speed
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
        
        exp_val = float(self.params.get("exposure", 156))
        gain_val = float(self.params.get("gain", 0))
        exp_factor = max(exp_val / 156.0, 0.01)
        gain_factor = 1.0 + (gain_val / 64.0)
        flux_factor = exp_factor * gain_factor

        star_signal = crop * flux_factor

        brightness_val = float(self.params.get("brightness", 128))
        base_bg = 24.0
        bg_level = base_bg + (brightness_val - 128.0) * 0.8

        noise_std = 2.5 * np.sqrt(gain_factor)
        noise = np.random.normal(0.0, noise_std, (self.height, self.width, 3)).astype(np.float32)

        img = bg_level + star_signal + noise

        contrast_val = float(self.params.get("contrast", 32))
        if contrast_val != 32:
            slope = contrast_val / 32.0
            img = 128.0 + (img - 128.0) * slope

        sharpness_val = float(self.params.get("sharpness", 0))
        if sharpness_val > 0:
            k = sharpness_val * 0.35
            kernel = np.array([[0, -k, 0], [-k, 1.0 + 4 * k, -k], [0, -k, 0]], dtype=np.float32)
            img = cv2.filter2D(img, -1, kernel)

        saturation_val = float(self.params.get("saturation", 64))
        if saturation_val == 0:
            gray = 0.299 * img[:, :, 2] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 0]
            img[:, :, 0] = gray
            img[:, :, 1] = gray
            img[:, :, 2] = gray
        elif saturation_val != 64:
            sat_scale = max(saturation_val / 64.0, 0.0)
            gray = (0.299 * img[:, :, 2] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 0])[:, :, None]
            img = gray + (img - gray) * sat_scale

        frame = np.clip(img, 0.0, 255.0).astype(np.uint8)
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
