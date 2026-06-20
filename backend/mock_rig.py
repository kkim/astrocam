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
        self.full_width = int(self.width * 2.0)
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
        self.sim_drift_angle = np.deg2rad(config_data.get("sim_drift_angle", 45.0))


        # Tracking State
        self.auto_tracking = False
        self.tracking_status = "inactive"
        self.ref_frame = None
        self.tracking_drift = [0.0, 0.0]
        self.ra_drift = 0.0
        self.dec_drift = 0.0
        self.calib_g = np.array([0.2, 0.0]) # default guess with positive feedback sign
        self.calib_state = "learning"
        self.calib_updates = 0
        self.calib_nudged = False
        self.last_tracking_time = 0.0
        self.last_u = 85.0
        self.last_d = np.array([0.0, 0.0])
        self.last_v = np.array([0.0, 0.0])
        
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
        self.raw_frame = None
        self.acc_frame = None
        self.thread = threading.Thread(target=self._sim_loop, daemon=True)
        self.thread.start()

    def _generate_static_starfield(self):
        # Clean black background
        img = np.zeros((self.full_height, self.full_width, 3), dtype=np.uint8)
        
        # Seed for consistent starfield
        rng = np.random.default_rng(42)
        
        # Add thousands of SHARP stars with varied sizes
        for _ in range(5000):
            x = rng.integers(0, self.full_width)
            y = rng.integers(0, self.full_height)
            
            # Size mapping: 1px->rad 0, 2px->rad 1, 3px->rad 2, 5px->rad 4
            size = rng.choice([0, 1, 2, 4], p=[0.9, 0.09, 0.009, 0.001])
            brightness = int(rng.integers(150, 255))
            
            # Grayscale stars
            cv2.circle(img, (x, y), int(size), (brightness, brightness, brightness), -1)
            
        return img

    def _sim_loop(self):
        while self.is_running:
            # Update simulated position based on drift
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
            self._process_frame(frame)
            time.sleep(0.04) # 25 FPS

    def _process_frame(self, frame):
        with self.lock:
            self.raw_frame = frame
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
        
        # Run tracking update outside the lock
        if self.auto_tracking:
            self._update_tracking(self.latest_frame)

    def _generate_mock_frame(self):
        # Use simulated positions for crop
        cx = int(self.pos_x)
        cy = int(self.pos_y)
        
        # Crop from static starfield
        crop = self.static_starfield[cy:cy+self.height, cx:cx+self.width].copy()
        
        # Add dynamic sensor noise
        noise = np.random.normal(5, 2, (self.height, self.width, 3)).astype(np.uint8)
        frame = cv2.add(crop, noise)

        cv2.putText(frame, f"MOCK DRIFT - {time.strftime('%H:%M:%S')}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(frame, f"Speed: {self.current_duty:.1f}%", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1)
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

    def get_raw_frame(self):
        with self.lock:
            if self.raw_frame is None: return None
            return self.raw_frame.copy()

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

    def set_auto_tracking(self, enable: bool):
        with self.lock:
            if enable == self.auto_tracking:
                return
            self.auto_tracking = enable
            if enable:
                self.ref_frame = None
                self.tracking_status = "calibrating"
                self.calib_g = np.array([0.2, 0.0])
                self.calib_state = "learning"
                self.calib_updates = 0
                self.calib_nudged = False
                self.last_tracking_time = 0.0
                event_logger.log("Auto-tracking enabled.")
            else:
                self.tracking_status = "inactive"
                self.ref_frame = None
                event_logger.log("Auto-tracking disabled.")

    def get_tracking_status(self):
        with self.lock:
            d_mag = np.linalg.norm(self.tracking_drift)
            g_mag = np.linalg.norm(self.calib_g)
            ra_ratio = 0.0
            if d_mag > 0.01 and g_mag > 0.01:
                d_ra = np.dot(self.tracking_drift, self.calib_g) / g_mag
                ra_ratio = float(abs(d_ra) / d_mag)
                ra_ratio = min(max(ra_ratio, 0.0), 1.0)

            # Return raw diurnal drift parameters directly to the UI
            sim_drift_speed = self.sim_drift_speed
            sim_drift_angle = float(np.rad2deg(self.sim_drift_angle) % 360.0)

            return {
                "active": self.auto_tracking,
                "status": self.tracking_status,
                "drift_x": round(self.tracking_drift[0], 2),
                "drift_y": round(self.tracking_drift[1], 2),
                "ra_drift": round(self.ra_drift, 3),
                "dec_drift": round(self.dec_drift, 3),
                "ra_ratio": round(ra_ratio, 3),
                "sim_drift_speed": round(sim_drift_speed, 2),
                "sim_drift_angle": round(sim_drift_angle, 1),
                "sim_camera_angle": round(np.rad2deg(self.camera_angle) % 360.0, 1),
                "calib_angle": round(np.rad2deg(np.arctan2(self.calib_g[1], self.calib_g[0])) % 360.0, 1) if np.linalg.norm(self.calib_g) > 0.01 else 0.0,
                "calib_magnitude": round(np.linalg.norm(self.calib_g), 2),
                "calib_state": self.calib_state
            }

    def _update_tracking(self, frame):
        now = time.time()
        if self.ref_frame is None:
            self.ref_frame = frame.copy()
            self.ref_time = now
            self.last_tracking_time = now
            self.last_u = self.current_duty
            self.last_d = np.array([0.0, 0.0])
            self.last_v = np.array([0.0, 0.0])
            self.tracking_drift = [0.0, 0.0]
            self.ra_drift = 0.0
            self.dec_drift = 0.0
            self.tracking_status = "calibrating"
            self.calib_state = "learning"
            self.calib_updates = 0
            self.calib_nudged = False
            event_logger.log("Calibration started. Tracking reference frame locked.")
            return

        dt = now - self.last_tracking_time
        if dt < 1.5:
            return

        # Compute shift
        from alignment_utils import align_images
        T = align_images(self.ref_frame, frame, translation_only=True)
        dx = float(T[0, 2])
        dy = float(T[1, 2])

        # If alignment failed or returned exact identity, skip update to prevent corruption
        if abs(dx) < 1e-5 and abs(dy) < 1e-5:
            return

        # Position error d_k (current relative to reference)
        d_k = np.array([-dx, -dy])
        self.tracking_drift = [float(d_k[0]), float(d_k[1])]

        # Measured velocity since last update step
        v_k = (d_k - self.last_d) / dt

        # Continuous calibration update using delta duty cycle and delta velocity
        delta_u = self.current_duty - self.last_u
        delta_v = v_k - self.last_v

        if abs(delta_u) > 0.05:
            # Use high learning rate initially for fast convergence, then lower it for stability
            if self.calib_state == "learning":
                alpha = 0.8
            else:
                alpha = 0.1
                
            reg = 0.005
            g_update = alpha * (delta_v - self.calib_g * delta_u) * delta_u / (delta_u**2 + reg)
            self.calib_g += g_update

            # Bound g
            g_mag = np.linalg.norm(self.calib_g)
            if g_mag < 0.05:
                self.calib_g = (self.calib_g / (g_mag + 1e-5)) * 0.05
            elif g_mag > 50.0:
                self.calib_g = (self.calib_g / g_mag) * 50.0

            self.calib_updates += 1
            if self.calib_updates >= 5:
                self.tracking_status = "tracking"
                self.calib_state = "converged"

        # Apply calibration nudge if we are still learning and haven't nudged yet
        if self.calib_state == "learning" and not self.calib_nudged:
            self.calib_nudged = True
            nudge = 1.0
            new_duty = np.clip(self.current_duty + nudge, 0.0, 100.0)
            event_logger.log(f"Calibration nudge applied (+{nudge}% duty cycle) for excitation.")
            self.set_motor_speed(float(new_duty), ramp_time=0.2)
        else:
            # Standard closed-loop correction
            g_dir = self.calib_g
            g_mag_sq = np.dot(g_dir, g_dir)

            if g_mag_sq > 0.01:
                d_ra = np.dot(d_k, g_dir) / np.sqrt(g_mag_sq)
                self.ra_drift = float(np.dot(v_k, g_dir) / np.sqrt(g_mag_sq))

                # Dec component
                d_dec_vec = d_k - (np.dot(d_k, g_dir) / g_mag_sq) * g_dir
                v_dec_vec = v_k - (np.dot(v_k, g_dir) / g_mag_sq) * g_dir
                self.dec_drift = float(np.linalg.norm(v_dec_vec))

                # PD-style guiding step: proportional error correction with velocity damping
                Kp = 0.05
                Kd = 0.20
                u_correction = - Kp * (np.dot(d_k, g_dir) / g_mag_sq) - Kd * (np.dot(v_k, g_dir) / g_mag_sq)
                u_correction = np.clip(u_correction, -0.1, 0.1)

                new_duty = np.clip(self.current_duty + u_correction, 0.0, 100.0)
                self.set_motor_speed(float(new_duty), ramp_time=0.2)
            else:
                self.ra_drift = 0.0
                self.dec_drift = 0.0

        self.last_u = self.current_duty
        self.last_v = v_k
        self.last_d = d_k
        self.last_tracking_time = now

    def close(self):
        self.is_running = False
        self.stop_ramping.set()
        self.sequence_stop_event.set()

    def set_camera_angle(self, angle_deg):
        with self.lock:
            self.camera_angle = np.deg2rad(angle_deg % 360.0)
            self._save_to_config("camera_angle", float(angle_deg % 360.0))
            # Reset calibration state so the tracking loop has to re-learn the new angle!
            self.calib_state = "learning"
            self.calib_updates = 0
            self.calib_nudged = False
            self.tracking_status = "calibrating"
            event_logger.log(f"Mock Camera Rotation Angle set to {angle_deg:.1f}°. Re-calibrating tracking.")
        return True

    def set_sim_drift(self, speed, angle_deg):
        with self.lock:
            self.sim_drift_speed = float(speed)
            self.sim_drift_angle = np.deg2rad(angle_deg % 360.0)
            self._save_to_config("sim_drift_speed", float(speed))
            self._save_to_config("sim_drift_angle", float(angle_deg % 360.0))
            # Reset calibration state as the sky drift speed/direction changed
            self.calib_state = "learning"
            self.calib_updates = 0
            self.calib_nudged = False
            self.tracking_status = "calibrating"
            event_logger.log(f"Mock Sim Drift set to: speed={speed:.1f} px/s, angle={angle_deg:.1f}°. Re-calibrating tracking.")
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
