import cv2
import threading
import time
import numpy as np
import os
import json
from typing import Optional, Dict, Any, Callable
from logger import event_logger

class MockCamera:
    """
    Simulates starfield acquisition with realistic celestial drift,
    telescope mount compensation, optical noise, and parameter controls.
    """
    def __init__(self, get_duty_callback: Optional[Callable[[], float]] = None):
        self.width = 1920
        self.height = 1080
        self.full_width = int(self.width * 2.0)
        self.full_height = int(self.height * 1.5)
        self.is_running = True
        self.lock = threading.Lock()
        self.get_duty_callback = get_duty_callback

        self.params: Dict[str, Any] = {
            "brightness": 128, "contrast": 32, "saturation": 64,
            "gain": 0, "exposure": 156, "sharpness": 2, "auto_exposure": 0
        }

        # Drift position
        self.pos_x = self.width * 0.25
        self.pos_y = self.height * 0.25
        self.last_update_time = time.time()

        # Load persisted simulation parameters
        config_data = {}
        if os.path.exists("config.json"):
            try:
                with open("config.json", "r") as f:
                    config_data = json.load(f)
            except Exception:
                pass

        self.camera_angle = np.deg2rad(config_data.get("camera_angle", 45.0))
        self.sim_drift_speed = float(config_data.get("sim_drift_speed", 60.0))
        if self.sim_drift_speed < 30.0:
            self.sim_drift_speed = 60.0
        self.sim_drift_angle = self.camera_angle

        self.static_starfield = self._generate_static_starfield()
        self.raw_frame: Optional[np.ndarray] = None

        self.thread = threading.Thread(target=self._sim_loop, daemon=True)
        self.thread.start()

    def _generate_static_starfield(self) -> np.ndarray:
        img = np.zeros((self.full_height, self.full_width, 3), dtype=np.uint8)
        rng = np.random.default_rng(42)
        for _ in range(5000):
            x = rng.integers(0, self.full_width)
            y = rng.integers(0, self.full_height)
            size = rng.choice([0, 1, 2, 4], p=[0.9, 0.09, 0.009, 0.001])
            brightness = int(rng.integers(150, 255))
            cv2.circle(img, (x, y), int(size), (brightness, brightness, brightness), -1)
        return img

    def _get_current_duty(self) -> float:
        if self.get_duty_callback:
            try:
                return float(self.get_duty_callback())
            except Exception:
                pass
        return 80.0

    def _sim_loop(self):
        while self.is_running:
            now = time.time()
            dt = now - self.last_update_time

            duty = self._get_current_duty()

            # Diurnal drift vector
            v_diurnal_x = self.sim_drift_speed * np.cos(self.sim_drift_angle)
            v_diurnal_y = self.sim_drift_speed * np.sin(self.sim_drift_angle)

            # Mount compensation vector: at 85% duty cycle, mount speed matches sim_drift_speed
            v_mount_mag = - (duty / 85.0) * self.sim_drift_speed
            v_mount_x = v_mount_mag * np.cos(self.camera_angle)
            v_mount_y = v_mount_mag * np.sin(self.camera_angle)

            v_x = v_diurnal_x + v_mount_x
            v_y = v_diurnal_y + v_mount_y

            self.pos_x = (self.pos_x + v_x * dt) % (self.full_width - self.width)
            self.pos_y = (self.pos_y + v_y * dt) % (self.full_height - self.height)
            self.last_update_time = now

            frame = self._generate_mock_frame(duty)
            with self.lock:
                self.raw_frame = frame
            time.sleep(0.04) # ~25 FPS

    def _generate_mock_frame(self, duty: float) -> np.ndarray:
        cx = int(self.pos_x)
        cy = int(self.pos_y)
        crop = self.static_starfield[cy : cy + self.height, cx : cx + self.width].copy()

        # Add sensor noise
        noise = np.random.normal(5, 2, (self.height, self.width, 3)).astype(np.uint8)
        frame = cv2.add(crop, noise)

        cv2.putText(frame, f"MOCK DRIFT - {time.strftime('%H:%M:%S')}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(frame, f"Speed: {duty:.1f}%", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1)
        return frame

    def reconnect(self) -> bool:
        """Mock camera reconnect is instantaneous."""
        event_logger.log("Camera: Mock camera reconnected.")
        return True

    def get_raw_frame(self) -> Optional[np.ndarray]:
        with self.lock:
            if self.raw_frame is None:
                return None
            return self.raw_frame.copy()

    def set_camera_param(self, prop: str, value: float) -> bool:
        with self.lock:
            self.params[prop] = value
        return True

    def get_camera_params(self) -> Dict[str, Any]:
        return {**self.params}

    def get_camera_status(self) -> Dict[str, Any]:
        return {
            "connected": True,
            "width": self.width,
            "height": self.height,
            "camera_mode": "mock",
            "device_name": "Simulated Starfield Camera",
            "sim_drift_angle": float(np.rad2deg(self.sim_drift_angle)),
            "sim_drift_speed": self.sim_drift_speed,
            "camera_angle": float(np.rad2deg(self.camera_angle))
        }

    def set_camera_angle(self, angle_degrees: float) -> bool:
        with self.lock:
            self.camera_angle = np.deg2rad(angle_degrees)
            self._save_sim_config()
            event_logger.log(f"Mock camera angle set to {angle_degrees:.1f}°")
        return True

    def set_sim_drift(self, speed_pixels_per_sec: float, angle_degrees: float) -> bool:
        with self.lock:
            self.sim_drift_speed = float(speed_pixels_per_sec)
            self.sim_drift_angle = np.deg2rad(angle_degrees)
            self._save_sim_config()
            event_logger.log(f"Mock drift set to {speed_pixels_per_sec:.1f} px/s at {angle_degrees:.1f}°")
        return True

    def _save_sim_config(self):
        try:
            config_data = {}
            if os.path.exists("config.json"):
                try:
                    with open("config.json", "r") as f:
                        config_data = json.load(f)
                except Exception:
                    pass
            config_data["camera_angle"] = float(np.rad2deg(self.camera_angle))
            config_data["sim_drift_speed"] = float(self.sim_drift_speed)
            config_data["sim_drift_angle"] = float(np.rad2deg(self.sim_drift_angle))
            with open("config.json", "w") as f:
                json.dump(config_data, f)
        except Exception as e:
            event_logger.log(f"Error saving mock camera config: {e}")

    def close(self):
        self.is_running = False
