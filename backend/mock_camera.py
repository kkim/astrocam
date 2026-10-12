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

        self.radius = 5
        self.noise_canvas = np.clip(
            24.0 + np.random.normal(0.0, 2.5, (self.height + 64, self.width + 64, 3)),
            0.0, 255.0
        ).astype(np.uint8)
        self.star_patches = self._setup_star_patches()
        self.raw_frame: Optional[np.ndarray] = None

        self.thread = threading.Thread(target=self._sim_loop, daemon=True)
        self.thread.start()

    def _create_star_patch(self, radius: int = 5, peak: float = 220.0, sigma: float = 2.0) -> np.ndarray:
        """Creates a smooth astronomical point spread function (PSF) of given radius."""
        y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
        dist_sq = x*x + y*y
        mask = dist_sq <= radius*radius
        prof = np.exp(-dist_sq / (2.0 * sigma**2)) * mask * peak
        patch = np.repeat(prof[:, :, None], 3, axis=2)
        return patch.astype(np.float32)

    def _setup_star_patches(self) -> list:
        """
        Precomputes realistic star profiles matching SV205 telescopic observations:
        - One bright guide star (radius 5, peak ~220)
        - 1-2 dim stars of brightness ~10 (radius 5)
        - 3-5 very dim stars of brightness ~1-3 (radius 5)
        """
        radius = self.radius

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
        return [(sx, sy, self._create_star_patch(radius=radius, peak=peak)) for sx, sy, peak in all_stars]

    def _get_current_duty(self) -> float:
        if self.get_duty_callback:
            try:
                return float(self.get_duty_callback())
            except Exception:
                pass
        return 80.0

    def _sim_loop(self):
        target_fps = 36.0
        target_dt = 1.0 / target_fps  # Target period yielding ~33 FPS with system scheduling

        while self.is_running:
            t_start = time.perf_counter()
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

            # Realistic camera exposure integration time:
            # At standard short exposure (156 = ~15.6ms), frame rate runs at full 33 FPS.
            # When exposure exceeds 30.3ms (e.g. 500ms), frame rate naturally slows down.
            exp_sec = float(self.params.get("exposure", 156)) * 0.0001
            target_interval = max(target_dt, exp_sec)

            elapsed = time.perf_counter() - t_start
            sleep_time = max(0.001, target_interval - elapsed)
            time.sleep(sleep_time)

    def _generate_mock_frame(self, duty: float) -> np.ndarray:
        # Dynamic slice from precomputed noise canvas to simulate sensor read noise
        oy = (self.frame_id * 17) % 64
        ox = (self.frame_id * 29) % 64
        frame = self.noise_canvas[oy : oy + self.height, ox : ox + self.width].copy()

        # 1. Exposure and Gain controls
        exp_val = float(self.params.get("exposure", 156))
        gain_val = float(self.params.get("gain", 0))

        exp_factor = max(exp_val / 156.0, 0.01)
        gain_factor = 1.0 + (gain_val / 64.0)
        flux_factor = exp_factor * gain_factor

        cx = int(self.pos_x)
        cy = int(self.pos_y)
        r = self.radius

        for sx, sy, patch in self.star_patches:
            fx = sx - cx
            fy = sy - cy
            if (fx - r) >= 0 and (fx + r + 1) <= self.width and (fy - r) >= 0 and (fy + r + 1) <= self.height:
                scaled = np.clip(patch * flux_factor, 0.0, 255.0).astype(np.uint8)
                frame[fy - r : fy + r + 1, fx - r : fx + r + 1] = cv2.add(
                    frame[fy - r : fy + r + 1, fx - r : fx + r + 1], scaled
                )

        # 2. Brightness (black level offset)
        brightness_val = float(self.params.get("brightness", 128))
        if brightness_val != 128:
            diff = int((brightness_val - 128.0) * 0.8)
            if diff > 0:
                frame = cv2.add(frame, diff)
            elif diff < 0:
                frame = cv2.subtract(frame, -diff)

        # 3. Contrast (S-Curve)
        contrast_val = float(self.params.get("contrast", 32))
        if contrast_val != 32:
            slope = contrast_val / 32.0
            x = np.arange(256, dtype=np.float32)
            lut = np.clip(128.0 + (x - 128.0) * slope, 0.0, 255.0).astype(np.uint8)
            frame = cv2.LUT(frame, lut)

        # 4. Sharpness (Edge Enhancement)
        sharpness_val = float(self.params.get("sharpness", 0))
        if sharpness_val > 0:
            k = sharpness_val * 0.35
            kernel = np.array([
                [0,    -k,          0],
                [-k,   1.0 + 4 * k, -k],
                [0,    -k,          0]
            ], dtype=np.float32)
            frame = cv2.filter2D(frame, -1, kernel)

        # 5. Saturation (Chroma)
        saturation_val = float(self.params.get("saturation", 64))
        if saturation_val == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        elif saturation_val != 64:
            sat_scale = max(saturation_val / 64.0, 0.0)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)[:, :, None].astype(np.float32)
            f_flt = frame.astype(np.float32)
            frame = np.clip(gray + (f_flt - gray) * sat_scale, 0.0, 255.0).astype(np.uint8)

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
