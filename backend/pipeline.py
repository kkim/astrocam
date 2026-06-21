import cv2
import numpy as np
import time
import threading
import os
from datetime import datetime
from logger import event_logger
from alignment_utils import align_images, detect_stars

class AstroPipeline:
    """
    AstroPipeline manages the image processing and closed-loop motor control pipeline.
    
    It operates as a decoupled controller between raw frame acquisition (the Rigs) and 
    the user interface. Its core responsibilities include:
    1. Tier 1 Stacking: Performs real-time frame accumulation using a running weighted
       average to reduce sensor noise in the live stream.
    2. Sub-pixel Image Alignment: Matches stars between the active frame and a locked
       reference frame using Star Neighborhood Descriptors to detect camera drift.
    3. Auto-Tracking Calibration: Solves the sensor-to-mount coordinate transformation
       and gain vector g (how star velocity responds to motor duty cycle u) using a 
       continuous rolling estimator, running in parallel with the PD controller.
    4. Closed-loop PD Guiding: Projects the measured drift and velocity errors onto the
       estimated calibration vector to calculate proportional and derivative duty cycle
       corrections, driving drift to zero.
    5. Capture & Sequence Management: Orchestrates manual captures and automated sequences
       saved directly to the local captures directory.

    Pseudocode of the Calibration and Guiding Loop:
    -----------------------------------------------
    def _update_tracking(frame):
        # 1. Rate Limiting (Thread-safe check)
        lock(self):
            if elapsed_time() < 1.5s: return
            update_last_tracking_time()
        
        # 2. Image Alignment
        dx, dy = align_images(ref_frame, frame)
        d_k = [-dx, -dy]            # star displacement
        v_k = (d_k - last_d) / dt   # star velocity
        
        lock(self):
            # Append current state to rolling history
            history_u.append(current_duty)
            history_d.append(d_k)
            calib_updates += 1
            
            # Apply initial nudge of +5.0% on step 0
            if calib_updates == 0:
                apply_motor_duty(u0 + 5.0%)
                
            W = 5 # response delay in steps
            if calib_updates >= 2 * W:
                # Compute displacement change over W steps (nudged vs baseline)
                delta_d_curr = d_k - d_mid
                delta_d_prev = d_mid - d_prev
                delta_u = u_curr - u_prev
                
                # Update calibration gain vector if signal is strong
                if abs(delta_u) > 1.0%:
                    self.calib_g = ((delta_d_curr - delta_d_prev) / delta_u) / W
            
            # Active Closed-Loop PD Guiding
            d_ra = dot(d_k, g) / |g|^2
            v_ra = dot(v_k, g) / |g|^2
            u_correction = - Kp * d_ra - Kd * v_ra
            u_correction = clip(u_correction, -1.5%, 1.5%)
            apply_motor_duty(current_duty + u_correction)
                
            last_d = d_k
    """
    
    def __init__(self, rig):
        """
        Initializes the pipeline with a camera/motor rig and starts the background 
        processing thread.
        
        Args:
            rig: An instance of BaseAstroRig (MockAstroRig or RealAstroRig).
        """
        self.rig = rig
        
        # Pipeline processing state
        self.n_avg = 1
        self.acc_frame = None
        self.latest_frame = None
        self.lock = threading.Lock()
        self.is_running = True
        self.fps = 0.0
        self.mean_brightness = 0.0
        self._frame_count = 0
        self._last_fps_calc_time = time.time()
        
        # Tracking & Calibration State
        self.auto_tracking = False
        self.tracking_status = "inactive"
        self.ref_frame = None
        self.pending_relock = False
        self.prev_alignment_frame = None
        self.reference_stars = []
        self.tracking_drift = [0.0, 0.0]
        self.ra_drift = 0.0
        self.dec_drift = 0.0
        self.calib_g = np.array([0.2, 0.0]) # default guess
        self.calib_state = "calibrating"
        self.last_tracking_time = 0.0
        self.last_u = 0.0
        self.last_d = np.array([0.0, 0.0])
        self.last_v = np.array([0.0, 0.0])
        
        # Calibration rolling history variables
        self.calib_u0 = 80.0
        self.calib_updates = 0
        self.history_u = []
        self.history_d = []
        
        # Sequence state
        self.sequence_info = {
            "active": False, "count": 0, "total": 0, "interval": 0,
            "last_capture_time": 0, "directory": "/home/kio/projects/astrocam/captures"
        }
        self.sequence_stop_event = threading.Event()
        
        # Start background processing thread
        self.thread = threading.Thread(target=self._processing_loop, daemon=True)
        self.thread.start()

    def _processing_loop(self):
        """
        Main worker loop running in a background thread. Continuous loops to acquire 
        raw frames from the rig and feed them into the processing pipeline.
        """
        while self.is_running:
            frame = self.rig.get_raw_frame()
            if frame is not None:
                self._process_frame(frame)
                time.sleep(0.01) # ~100 Hz polling maximum
            else:
                time.sleep(0.04) # ~25 Hz sleep if no frame

    def _process_frame(self, frame):
        """
        Applies Tier 1 weighted average stacking to the raw frame to reduce noise, 
        calculates performance telemetry (FPS, brightness), and triggers the 
        closed-loop tracking logic if auto-tracking is enabled.
        
        Args:
            frame: Raw BGR/grayscale numpy array acquired from the camera rig.
        """
        with self.lock:
            # Tier 1 Stacking (Weighted averaging)
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
            
            # FPS Calculation
            self._frame_count += 1
            now = time.time()
            if now - self._last_fps_calc_time >= 2.0:
                self.fps = self._frame_count / (now - self._last_fps_calc_time)
                self._frame_count = 0
                self._last_fps_calc_time = now
                
            # Mean Brightness calculation
            if self.latest_frame is not None:
                self.mean_brightness = float(np.mean(self.latest_frame))
        
        # Run drift measurement and auto-tracking control step
        if self.latest_frame is not None:
            self._update_tracking(self.latest_frame)

    def get_frame(self):
        """
        Downsamples and compresses the latest processed frame into JPEG format for 
        streaming to the Web UI dashboard.
        
        Returns:
            bytes: JPEG-encoded image bytes or None if no frame is available.
        """
        with self.lock:
            if self.latest_frame is None: return None
            h, w = self.latest_frame.shape[:2]
            display_w = 1280
            display_h = int(display_w * (h / w))
            img = cv2.resize(self.latest_frame, (display_w, display_h))
            
            # Draw reference star crosshairs if auto-tracking is active and reference stars are detected
            if self.auto_tracking and self.reference_stars:
                scale_x = display_w / w
                scale_y = display_h / h
                for rx, ry in self.reference_stars:
                    cx = int(rx * scale_x)
                    cy = int(ry * scale_y)
                    # Draw a cyan crosshair and target circle
                    cv2.drawMarker(img, (cx, cy), (0, 255, 255), markerType=cv2.MARKER_CROSS, markerSize=16, thickness=1)
                    cv2.circle(img, (cx, cy), 6, (0, 255, 255), 1)

            ret, jpeg = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 80])
            return jpeg.tobytes() if ret else None

    def set_n_avg(self, n):
        """
        Updates the running average frame count. Changing this value resets the 
        accumulator to prevent ghosting artifacts when settings are changed.
        
        Args:
            n: Integer number of frames to average.
        """
        with self.lock:
            new_n = int(n)
            if new_n != self.n_avg:
                self.n_avg = new_n
                self.acc_frame = None
        return True

    def set_auto_tracking(self, enable: bool):
        """
        Enables or disables the closed-loop auto-guiding loop. Disabling resets all 
        calibrations and sets the motor back to passive mode. Enabling kicks off 
        the calibration sequence.
        
        Args:
            enable: Boolean indicating whether to start or stop auto-tracking.
        """
        with self.lock:
            if enable == self.auto_tracking:
                if enable:
                    self.pending_relock = True
                    event_logger.log("Auto-tracking reference frame relocked to current view.")
                return
            self.auto_tracking = enable
            if enable:
                self.ref_frame = None
                self.pending_relock = False
                self.tracking_status = "calibrating"
                self.calib_g = np.array([0.2, 0.0])
                self.calib_state = "calibrating"
                self.last_tracking_time = 0.0
                self.calib_u0 = self.rig.current_duty
                self.calib_updates = 0
                self.history_u = []
                self.history_d = []
                event_logger.log("Auto-tracking enabled.")
            else:
                self.tracking_status = "inactive"
                self.ref_frame = None
                self.pending_relock = False
                self.reference_stars = []
                event_logger.log("Auto-tracking disabled.")

    def get_tracking_status(self):
        """
        Gathers current autoguider telemetry, including active status, drift errors,
        drift speed, and camera position angle.
        
        Returns:
            dict: Telemetry data serialized for frontend updates.
        """
        with self.lock:
            # drift speed is the magnitude of the latest measured velocity
            v_mag = np.linalg.norm(self.last_v)
            
            # Camera PA: if we have calibrated calib_g, use it. Otherwise, use the angle of measured velocity
            if self.calib_state == "active" and np.linalg.norm(self.calib_g) > 0.01:
                camera_pa = np.rad2deg(np.arctan2(self.calib_g[1], self.calib_g[0])) % 180.0
            else:
                if v_mag > 0.1:
                    camera_pa = np.rad2deg(np.arctan2(self.last_v[1], self.last_v[0])) % 180.0
                else:
                    camera_pa = np.rad2deg(np.arctan2(self.calib_g[1], self.calib_g[0])) % 180.0
            
            # Fetch simulated properties dynamically if using the Mock Rig
            sim_drift_speed = None
            sim_drift_angle = None
            sim_camera_angle = None
            if hasattr(self.rig, "sim_drift_speed"):
                sim_drift_speed = getattr(self.rig, "sim_drift_speed")
                sim_drift_angle = float(np.rad2deg(getattr(self.rig, "sim_drift_angle")) % 360.0)
                sim_camera_angle = float(np.rad2deg(getattr(self.rig, "camera_angle")) % 360.0)

            return {
                "active": self.auto_tracking,
                "status": self.tracking_status,
                "drift_speed_x": round(self.last_v[0], 3),
                "drift_speed_y": round(self.last_v[1], 3),
                "drift_speed": round(v_mag, 3),
                "camera_pa": round(camera_pa, 1),
                "sim_drift_speed": sim_drift_speed,
                "sim_drift_angle": sim_drift_angle,
                "sim_camera_angle": sim_camera_angle,
                "calib_state": self.calib_state
            }

    def _detect_reference_stars(self):
        """
        Detects the top 5 brightest stars in the reference frame
        for visual crosshair overlay on the live stream.
        """
        if self.ref_frame is None:
            self.reference_stars = []
            return
        
        # Convert to grayscale
        if len(self.ref_frame.shape) == 3:
            gray = cv2.cvtColor(self.ref_frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = self.ref_frame
            
        # Detect stars on full resolution frame to get accurate coordinates
        stars = detect_stars(gray, threshold=15, min_dist=15)
        # detect_stars returns stars sorted by brightness, so take the top 5
        self.reference_stars = stars[:5]
        event_logger.log(f"Detected {len(self.reference_stars)} reference stars for guiding crosshairs.")

    def _update_tracking(self, frame):
        """
        Executes the drift measurement step and, if auto_tracking is enabled,
        the closed-loop tracking control step.
        """
        now = time.time()
        
        # Check and update timestamp under lock immediately to prevent concurrent entry
        with self.lock:
            if self.prev_alignment_frame is None:
                self.prev_alignment_frame = frame.copy()
                self.ref_frame = frame.copy()
                self.ref_time = now
                self.last_tracking_time = now
                self.last_u = self.rig.current_duty
                self.last_d = np.array([0.0, 0.0])
                self.last_v = np.array([0.0, 0.0])
                self.tracking_drift = [0.0, 0.0]
                self.calib_updates = 0
                self.history_u = [self.rig.current_duty]
                self.history_d = [np.array([0.0, 0.0])]
                self.calib_u0 = self.rig.current_duty
                event_logger.log("Tracking measurement started.")
                return

            if self.auto_tracking and self.ref_frame is None:
                self.ref_frame = frame.copy()
                self.ref_time = now
                self.last_tracking_time = now
                self.last_u = self.rig.current_duty
                self.last_d = np.array([0.0, 0.0])
                self.last_v = np.array([0.0, 0.0])
                self.tracking_drift = [0.0, 0.0]
                self.calib_updates = 0
                self.history_u = [self.rig.current_duty]
                self.history_d = [np.array([0.0, 0.0])]
                self.calib_u0 = self.rig.current_duty
                self._detect_reference_stars()
                event_logger.log(f"Auto-tracking started (continuous). Tracking reference locked at {self.calib_u0}%.")
                return

            if self.auto_tracking and self.pending_relock:
                # Try to align to keep the history if possible
                T = align_images(self.ref_frame, frame, translation_only=True)
                dx = float(T[0, 2])
                dy = float(T[1, 2])
                if abs(dx) > 1e-5 or abs(dy) > 1e-5:
                    d_k = np.array([-dx, -dy])
                    self.history_d = [d - d_k for d in self.history_d]
                    event_logger.log("Relocked tracking reference. Shifted calibration history.")
                else:
                    self.calib_updates = 0
                    self.history_u = [self.rig.current_duty]
                    self.history_d = [np.array([0.0, 0.0])]
                    event_logger.log("Relocked tracking reference. Calibration history reset.")
                
                self.ref_frame = frame.copy()
                self.ref_time = now
                self.last_tracking_time = now
                self.tracking_drift = [0.0, 0.0]
                self.last_d = np.array([0.0, 0.0])
                self.last_v = np.array([0.0, 0.0])
                self.pending_relock = False
                self._detect_reference_stars()
                return

            dt = now - self.last_tracking_time
            if dt < 1.5:
                return
            self.last_tracking_time = now

        # Sub-pixel translation alignment (runs outside lock to keep UI responsive)
        if self.auto_tracking:
            T = align_images(self.ref_frame, frame, translation_only=True)
        else:
            T = align_images(self.prev_alignment_frame, frame, translation_only=True)
            
        dx = float(T[0, 2])
        dy = float(T[1, 2])

        if abs(dx) < 1e-5 and abs(dy) < 1e-5:
            return

        with self.lock:
            if self.auto_tracking:
                # Position error d_k (current relative to reference)
                d_k = np.array([-dx, -dy])
                self.tracking_drift = [float(d_k[0]), float(d_k[1])]

                # Measured velocity since last update step
                v_k = (d_k - self.last_d) / dt
                self.last_v = v_k
                self.last_d = d_k

                # Append current state to history
                self.history_u.append(self.rig.current_duty)
                self.history_d.append(d_k)
                self.calib_updates += 1

                W = 5 # response delay window
                
                # Apply nudge of +5% at step W to excite the system
                if self.calib_updates == W:
                    new_duty = np.clip(self.rig.current_duty + 5.0, 0.0, 100.0)
                    self.rig.set_motor_speed(float(new_duty), ramp_time=0.2)
                    event_logger.log(f"Calibration nudge of +5.0% applied at step {W} (target {new_duty}%).")

                # Continuous update of calib_g if we have enough history
                if self.calib_updates >= 2 * W:
                    d_curr = self.history_d[-1]
                    d_mid = self.history_d[-W-1]
                    d_prev = self.history_d[-2*W-1]
                    
                    delta_d_curr = d_curr - d_mid
                    delta_d_prev = d_mid - d_prev
                    
                    u_curr = self.history_u[-1]
                    u_prev = self.history_u[-2*W-1]
                    delta_u = u_curr - u_prev
                    
                    # Only update when we have a strong signal (delta_u > 1.0%)
                    if abs(delta_u) > 1.0:
                        self.calib_g = ((delta_d_curr - delta_d_prev) / delta_u) / W
                        
                        # Bound g to prevent extreme values
                        g_mag = np.linalg.norm(self.calib_g)
                        if g_mag < 0.01:
                            self.calib_g = (self.calib_g / (g_mag + 1e-5)) * 0.01
                        elif g_mag > 10.0:
                            self.calib_g = (self.calib_g / g_mag) * 10.0
                            
                        # Update status for UI to indicate calibration has run
                        if self.calib_state != "active":
                            self.calib_state = "active"
                            self.tracking_status = "tracking"
                            event_logger.log(f"Calibration active. Vector: mag={np.linalg.norm(self.calib_g):.3f}, angle={np.rad2deg(np.arctan2(self.calib_g[1], self.calib_g[0])):.1f}°")
                            
                            # Automatic relock to start guiding from zero position error
                            self.ref_frame = frame.copy()
                            self.history_d = [d - d_k for d in self.history_d]
                            d_k = np.array([0.0, 0.0])
                            v_k = np.array([0.0, 0.0])
                            self.tracking_drift = [0.0, 0.0]
                            self.last_d = np.array([0.0, 0.0])
                            self._detect_reference_stars()
                            
                            # Set motor speed back to u0 to cancel out the nudge we applied
                            self.rig.set_motor_speed(float(self.calib_u0), ramp_time=0.2)

                # Continuous PD guiding using the current calib_g estimation
                g_dir = self.calib_g
                g_mag_sq = np.dot(g_dir, g_dir)

                if g_mag_sq > 1e-5:
                    self.ra_drift = float(np.dot(v_k, g_dir) / np.sqrt(g_mag_sq))

                    # Dec component
                    d_dec_vec = d_k - (np.dot(d_k, g_dir) / g_mag_sq) * g_dir
                    v_dec_vec = v_k - (np.dot(v_k, g_dir) / g_mag_sq) * g_dir
                    self.dec_drift = float(np.linalg.norm(v_dec_vec))

                    # PD-style guiding step: proportional error correction with velocity damping
                    Kp = 0.08
                    Kd = 0.15
                    u_correction = - Kp * (np.dot(d_k, g_dir) / g_mag_sq) - Kd * (np.dot(v_k, g_dir) / g_mag_sq)
                    u_correction = np.clip(u_correction, -1.5, 1.5) # limit max change per step to prevent oscillation

                    new_duty = np.clip(self.rig.current_duty + u_correction, 0.0, 100.0)
                    self.rig.set_motor_speed(float(new_duty), ramp_time=0.2)
                else:
                    self.ra_drift = 0.0
                    self.dec_drift = 0.0
            else:
                # Passive mode: just measure instantaneous velocity relative to previous frame
                d_k = np.array([-dx, -dy])
                self.tracking_drift[0] += float(d_k[0])
                self.tracking_drift[1] += float(d_k[1])
                v_k = d_k / dt
                self.last_v = v_k
                self.last_d = d_k

            self.prev_alignment_frame = frame.copy()
            self.last_u = self.rig.current_duty
            self.last_v = v_k
            self.last_d = d_k

    def capture_frame(self):
        """
        Captures the latest processed image frame from the pipeline and saves it 
        to disk in the captures folder.
        
        Returns:
            dict: JSON success response containing the filename, or an error details dictionary.
        """
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

    def start_sequence(self, count, interval):
        """
        Starts an automated capture sequence taking photos at set intervals.
        
        Args:
            count: Total number of frames to capture.
            interval: Float time in seconds between captures.
            
        Returns:
            bool: True if sequence successfully started, False if a sequence is already active.
        """
        if self.sequence_info["active"]: return False
        self.sequence_info.update({"active": True, "total": count, "count": 0, "interval": interval})
        self.sequence_stop_event.clear()
        threading.Thread(target=self._sequence_loop, daemon=True).start()
        return True

    def _sequence_loop(self):
        """
        Background sequence thread loop that triggers frame captures at the requested intervals.
        """
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
        """
        Queries status and progress of the active sequence.
        
        Returns:
            dict: Sequence state properties.
        """
        return self.sequence_info

    def close(self):
        """
        Closes background pipeline threads and stops active sequences.
        """
        self.is_running = False
        self.sequence_stop_event.set()


def update_duty_cycle(history_v, history_u, g):
    """
    Computes the next duty cycle u_{t+1} to drive the tracking error (u*g - v) to 0.
    
    Args:
        history_v (list of np.ndarray): History of observed velocity vectors (px/s).
        history_u (list of float): History of applied duty cycles (%).
        g (np.ndarray): Calibration gain vector (px/s per % duty cycle).
        
    Returns:
        float: The next duty cycle u_{t+1}.
    """
    g_mag_sq = np.dot(g, g)
    if g_mag_sq < 1e-8:
        return history_u[-1] if history_u else 80.0
        
    # Calculate the ideal duty cycle at each step in the history
    u_ideals = []
    for u_k, v_k in zip(history_u, history_v):
        v_proj = np.dot(v_k, g) / g_mag_sq
        u_ideal = u_k - v_proj
        u_ideals.append(u_ideal)
        
    return float(np.mean(u_ideals))
