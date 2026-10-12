import cv2
import threading
import time
import numpy as np
import os
import json
from typing import Optional, Dict, Any, Callable, Tuple
from logger import event_logger

class MockCamera:
    """
    Simulates starfield acquisition with realistic celestial drift,
    telescope mount compensation, optical noise, and realistic SV205 camera parameter controls.
    """
    def __init__(self, get_duty_callback: Optional[Callable[[], float]] = None):
        self.width = 1920
        self.height = 1080
        self.full_width = int(self.width * 2.0)
        self.full_height = int(self.height * 1.5)
        self.is_running = True
        self.frame_id: int = 0
        self.lock = threading.Lock()
        self.get_duty_callback = get_duty_callback

        self.params: Dict[str, Any] = {
            "brightness": 128, "contrast": 32, "saturation": 64,
            "gain": 0, "exposure": 156, "sharpness": 0, "auto_exposure": 0
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

    def _create_star_profile(self, radius: int = 5, peak: float = 220.0, sigma: float = 2.0) -> np.ndarray:
        """Creates a smooth astronomical point spread function (PSF) of given radius."""
        y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
        dist_sq = x*x + y*y
        mask = dist_sq <= radius*radius
        profile = np.exp(-dist_sq / (2.0 * sigma**2)) * mask
        return (profile * peak).astype(np.float32)

    def _generate_static_starfield(self) -> np.ndarray:
        """
        Generates realistic starfield matching SV205 telescopic observations:
        - One bright guide star (radius 5, peak ~220)
        - 1-2 dim stars of brightness ~10 (radius 5)
        - 3-5 very dim stars of brightness ~1-3 (radius 5)
        """
        canvas = np.zeros((self.full_height, self.full_width, 3), dtype=np.float32)
        radius = 5

        # Guide star positioned near center of drift track
        bright_star = (int(self.full_width * 0.45), int(self.full_height * 0.45), 220.0)

        # 2 dim stars (~10 counts above baseline)
        dim_stars = [
            (bright_star[0] + 250, bright_star[1] - 170, 10.0),
            (bright_star[0] - 320, bright_star[1] + 180, 9.5),
        ]

        # 4 very dim stars (1-3 counts above baseline)
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
                self.frame_id += 1

            # Realistic camera exposure integration time
            exp_sec = float(self.params.get("exposure", 156)) * 0.0001
            time.sleep(max(exp_sec, 0.033))

    def _generate_mock_frame(self, duty: float) -> np.ndarray:
        cx = int(self.pos_x)
        cy = int(self.pos_y)
        crop = self.static_starfield[cy : cy + self.height, cx : cx + self.width].copy()

        # 1. Exposure and Gain controls
        exp_val = float(self.params.get("exposure", 156))
        gain_val = float(self.params.get("gain", 0))

        # Base exposure 156 (~15.6ms) is unity flux
        exp_factor = max(exp_val / 156.0, 0.01)
        # Gain: 0 is unity 1x, 255 is ~5x analog/digital amplification
        gain_factor = 1.0 + (gain_val / 64.0)
        flux_factor = exp_factor * gain_factor

        star_signal = crop * flux_factor

        # 2. Brightness (black level offset):
        # 128 is neutral black level (~24 counts background).
        # Setting < 128 pulls black level down (clips blacks/faint stars to 0).
        # Setting > 128 lifts black level.
        brightness_val = float(self.params.get("brightness", 128))
        base_bg = 24.0
        bg_level = base_bg + (brightness_val - 128.0) * 0.8

        # 3. Gaussian sensor noise:
        # Standard deviation ~2.5 counts, amplified by sensor gain
        noise_std = 2.5 * np.sqrt(gain_factor)
        noise = np.random.normal(0.0, noise_std, (self.height, self.width, 3)).astype(np.float32)

        # 4. Synthesize raw sensor frame
        img = bg_level + star_signal + noise

        # 5. Contrast (S-Curve):
        # 32 is neutral linear. > 32 stretches midtones, crushing shadows and blowing highlights.
        contrast_val = float(self.params.get("contrast", 32))
        if contrast_val != 32:
            slope = contrast_val / 32.0
            img = 128.0 + (img - 128.0) * slope

        # 6. Sharpness (Edge Enhancement):
        # 0 is off (recommended). > 0 applies an unsharp edge kernel, creating dark halos and ringing.
        sharpness_val = float(self.params.get("sharpness", 0))
        if sharpness_val > 0:
            k = sharpness_val * 0.35
            kernel = np.array([
                [0,    -k,          0],
                [-k,   1.0 + 4 * k, -k],
                [0,    -k,          0]
            ], dtype=np.float32)
            img = cv2.filter2D(img, -1, kernel)

        # 7. Saturation (Chroma):
        # 64 is neutral color. 0 is monochrome.
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

    def reconnect(self) -> bool:
        """Mock camera reconnect is instantaneous."""
        event_logger.log("Camera: Mock camera reconnected.")
        return True

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
