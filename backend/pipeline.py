import cv2
import numpy as np
import time
import threading
import os
import json
import collections
from datetime import datetime
from typing import Optional, List, Tuple, Dict, Any
from logger import event_logger
from alignment_utils import align_images, detect_stars
from recorder import StreamRecorder

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
        self.accumulation_count = 1
        self.acc_buffer: Optional[np.ndarray] = None
        self.acc_count = 0
        self.latest_frame = None
        self.lock = threading.Lock()
        self.is_running = True
        self.fps = 0.0
        self.mean_brightness = 0.0
        self.histogram = [0] * 256
        self.enable_histogram = True
        self.enable_corrupted_filter = True
        self.enable_luminance_check = True
        self.hist_roi = [0, 0, 1920, 1080]
        self.hist_stride = 4
        self._frame_count = 0
        self._last_fps_calc_time = time.time()

        # Restore saved pipeline and camera settings from config.json
        cfg_path = os.path.join(os.path.dirname(__file__), "config.json")
        if os.path.exists(cfg_path):
            try:
                with open(cfg_path, "r") as f:
                    cfg_json = json.load(f)
                    p_cfg = cfg_json.get("pipeline_settings", {})
                    if "histogram" in p_cfg:
                        self.enable_histogram = bool(p_cfg["histogram"])
                    if "corrupted_filter" in p_cfg:
                        self.enable_corrupted_filter = bool(p_cfg["corrupted_filter"])
                    if "luminance_check" in p_cfg:
                        self.enable_luminance_check = bool(p_cfg["luminance_check"])
                    if "roi" in p_cfg and len(p_cfg["roi"]) == 4:
                        self.hist_roi = [int(v) for v in p_cfg["roi"]]
                    if "stride" in p_cfg:
                        self.hist_stride = max(1, min(64, int(p_cfg["stride"])))
                    c_cfg = cfg_json.get("camera_settings", {})
                    if "accumulation_count" in c_cfg:
                        self.accumulation_count = max(1, int(c_cfg["accumulation_count"]))
            except Exception:
                pass

        # Star Peak Tracking & Corrupted Frame Filter
        self.peak_history = collections.deque(maxlen=10)
        self.corrupted_frames_dropped = 0
        self._consecutive_peak_drops = 0
        self._last_corrupt_log_time = 0.0
        
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
        self.inlier_ratio = 0.0
        
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
        
        # Dedicated stream recorder
        self.recorder = StreamRecorder()

        # Black Calibration & Defect Mask State
        self.is_calibrating = False
        self.calib_target_frames = 15
        self._calib_frames: List[np.ndarray] = []
        self.defect_mask: Optional[np.ndarray] = None
        self.calibration_meta: Optional[Dict[str, Any]] = None
        self._load_existing_calibration()

        # Setting-Indexed Dark Baseline Calibration State
        self.baseline_subtract: bool = False
        self.baseline_pedestal: float = 10.0
        self.active_baseline: Optional[np.ndarray] = None
        self.active_baseline_key: str = ""
        self.is_capturing_baseline: bool = False
        self.baseline_capture_target: int = 10
        self.baseline_capture_key: str = ""
        self.baseline_capture_frames: List[np.ndarray] = []

        # Start background processing thread
        self.thread = threading.Thread(target=self._processing_loop, daemon=True)
        self.thread.start()

    def _processing_loop(self):
        """
        Main worker loop running in a background thread. Continuous loops to acquire 
        fresh raw frames from the rig and feed them into the processing pipeline.
        """
        last_frame_id = -1
        while self.is_running:
            if hasattr(self.rig, "get_fresh_frame"):
                frame, frame_id = self.rig.get_fresh_frame(last_frame_id)
            else:
                frame = self.rig.get_raw_frame()
                frame_id = last_frame_id + 1

            if frame is not None and frame_id != last_frame_id:
                last_frame_id = frame_id
                self._process_frame(frame)
            else:
                time.sleep(0.002) # Waiting for fresh exposure from sensor

    def _process_frame(self, frame):
        """
        Applies block sum accumulation to raw frames, scales the current accumulation
        divided by current frame count into latest_frame for the live view canvas,
        calculates telemetry (FPS, brightness), and forwards completed stacks
        to the recorder when recording is active.
        
        Args:
            frame: Raw BGR/grayscale numpy array acquired from the camera rig.
        """
        completed_stack = None
        completed_count = 0

        with self.lock:
            # Handle baseline capture if active
            if self.is_capturing_baseline:
                self.baseline_capture_frames.append(frame.copy())
                if len(self.baseline_capture_frames) >= self.baseline_capture_target:
                    self._finalize_baseline_capture()

            # Handle Black Calibration collection if active
            if self.is_calibrating:
                self._calib_frames.append(frame.copy())
                if len(self._calib_frames) >= self.calib_target_frames:
                    self._finalize_black_calibration()

            # Baseline subtraction with fixed positive pedestal (e.g. +10.0)
            if self.baseline_subtract and self.active_baseline is not None and self.active_baseline.shape == frame.shape:
                calibrated_f32 = frame.astype(np.float32) - self.active_baseline + self.baseline_pedestal
                proc_frame = np.clip(calibrated_f32, 0.0, 255.0).astype(np.uint8)
            else:
                proc_frame = frame

            # 1. Star peak detection & corrupted frame filter (conditional for speed):
            if self.enable_luminance_check or self.enable_corrupted_filter:
                gray = cv2.cvtColor(proc_frame, cv2.COLOR_BGR2GRAY) if proc_frame.ndim == 3 else proc_frame
                blurred = cv2.blur(gray, (3, 3))
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(blurred)
                if self.enable_luminance_check:
                    self.mean_brightness = float(max_val)

                # Corrupted frame drop filter
                if self.enable_corrupted_filter and max_val >= 15:
                    if len(self.peak_history) >= 3:
                        med_x = float(np.median([p[0] for p in self.peak_history]))
                        med_y = float(np.median([p[1] for p in self.peak_history]))
                        dist = float(np.hypot(max_loc[0] - med_x, max_loc[1] - med_y))

                        if dist > 30.0:
                            self._consecutive_peak_drops += 1
                            if self._consecutive_peak_drops < 8:
                                self.corrupted_frames_dropped += 1
                                now = time.time()
                                if now - self._last_corrupt_log_time >= 1.0:
                                    event_logger.log(
                                        f"Dropped corrupted frame: star peak jumped {dist:.1f}px to {max_loc} "
                                        f"(median: ({med_x:.0f}, {med_y:.0f}))"
                                    )
                                    self._last_corrupt_log_time = now
                                return  # Discard frame from accumulation, stream, and disk!
                            else:
                                # Slew or new target acquisition: reset rolling history
                                self.peak_history.clear()
                                self._consecutive_peak_drops = 0
                        else:
                            self._consecutive_peak_drops = 0

                    self.peak_history.append(max_loc)

            # Continuous Live Block Accumulation (1x = raw, 10x, 25x, 100x)
            if self.accumulation_count <= 1:
                self.latest_frame = proc_frame
                self.acc_buffer = None
                self.acc_count = 1
            else:
                if self.acc_buffer is None or self.acc_buffer.shape != proc_frame.shape:
                    self.acc_buffer = proc_frame.astype(np.float32)
                    self.acc_count = 1
                else:
                    self.acc_buffer += proc_frame.astype(np.float32)
                    self.acc_count += 1

                # Draw current accumulation divided by current count into the view canvas
                # Saturated float32 -> uint8 scaled by 1.0 / acc_count via single SIMD instruction
                self.latest_frame = cv2.convertScaleAbs(self.acc_buffer, alpha=1.0 / self.acc_count)

                # Check if block reached target accumulation
                if self.acc_count >= self.accumulation_count:
                    completed_count = self.acc_count
                    avg_float = self.acc_buffer / self.acc_count
                    if self.recorder.file_format in ("tif", "tiff"):
                        # If monochrome (e.g. saturation=0 or identical R,G,B), save single channel
                        if avg_float.ndim == 3 and np.array_equal(avg_float[..., 0], avg_float[..., 1]) and np.array_equal(avg_float[..., 0], avg_float[..., 2]):
                            avg_float = avg_float[..., 0]
                        completed_stack = np.clip(np.round(avg_float * 257.0), 0, 65535).astype(np.uint16)
                    else:
                        completed_stack = self.latest_frame.copy()

                    self.acc_buffer = None
                    self.acc_count = 0

            # Live display defect softening (visualization ONLY - acc_buffer and saved TIFFs remain 100% pure raw)
            if self.defect_mask is not None and self.latest_frame is not None:
                fh, fw = self.latest_frame.shape[:2]
                if self.defect_mask.shape == (fh, fw):
                    disp_frame = self.latest_frame.copy()
                    blurred_disp = cv2.medianBlur(disp_frame, 3)
                    disp_frame[self.defect_mask > 0] = blurred_disp[self.defect_mask > 0]
                    self.latest_frame = disp_frame

            # Calculate 256-bin luminance histogram on latest displayed frame or ROI
            if self.enable_histogram and self.latest_frame is not None:
                fh, fw = self.latest_frame.shape[:2]
                rx, ry, rw, rh = self.hist_roi
                rx = max(0, min(rx, fw - 1))
                ry = max(0, min(ry, fh - 1))
                rw = max(1, min(rw, fw - rx))
                rh = max(1, min(rh, fh - ry))
                roi = self.latest_frame[ry:ry+rh, rx:rx+rw]
                stride = max(1, int(self.hist_stride))
                sample = roi[::stride, ::stride] if stride > 1 else roi

                if sample.ndim == 3:
                    sample_gray = cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY)
                else:
                    sample_gray = sample
                hist = cv2.calcHist([sample_gray], [0], None, [256], [0, 256])
                self.histogram = [int(v[0]) for v in hist]

            # FPS Calculation
            self._frame_count += 1
            now = time.time()
            if now - self._last_fps_calc_time >= 2.0:
                self.fps = self._frame_count / (now - self._last_fps_calc_time)
                self._frame_count = 0
                self._last_fps_calc_time = now

        # Handle Stream Recording
        if self.recorder.is_recording:
            if self.accumulation_count <= 1:
                self.recorder.add_frame(frame)
            elif completed_stack is not None:
                self.recorder.add_completed_stack(completed_stack, completed_count)

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

    def get_histogram(self):
        """Returns the current 256-bin luminance histogram."""
        with self.lock:
            return list(self.histogram)

    def get_processing_config(self):
        """Returns current live processing stage toggles, histogram ROI, and subsample stride."""
        with self.lock:
            return {
                "histogram": self.enable_histogram,
                "corrupted_filter": self.enable_corrupted_filter,
                "luminance_check": self.enable_luminance_check,
                "roi": list(self.hist_roi),
                "stride": self.hist_stride
            }

    def set_processing_config(self, config: dict):
        """Updates live processing stage toggles, histogram ROI, and/or subsample stride."""
        with self.lock:
            if "histogram" in config:
                self.enable_histogram = bool(config["histogram"])
            if "corrupted_filter" in config:
                self.enable_corrupted_filter = bool(config["corrupted_filter"])
            if "luminance_check" in config:
                self.enable_luminance_check = bool(config["luminance_check"])
            if "roi" in config and isinstance(config["roi"], (list, tuple)) and len(config["roi"]) == 4:
                self.hist_roi = [int(v) for v in config["roi"]]
            if "stride" in config:
                self.hist_stride = max(1, min(64, int(config["stride"])))
        event_logger.log(f"Pipeline processing config updated: {config}")
        return True

    def set_accumulation_count(self, count: int) -> bool:
        """
        Updates the stacking accumulation count (1x, 10x, 25x, 100x).
        Resets active accumulator to prevent ghosting artifacts.
        """
        with self.lock:
            new_count = max(1, int(count))
            if new_count != self.accumulation_count:
                self.accumulation_count = new_count
                self.acc_buffer = None
                self.acc_count = 0
                event_logger.log(f"Stacking accumulation window set to {new_count}x")
        return True

    def reset_accumulation(self):
        """Resets the live accumulator buffer (e.g. on camera parameter change)."""
        with self.lock:
            self.acc_buffer = None
            self.acc_count = 0

    def _load_existing_calibration(self):
        """Loads persistent defect mask and calibration metadata from disk if available."""
        try:
            calib_dir = os.path.join(self.recorder.base_dir, "calibration")
            mask_path = os.path.join(calib_dir, "defect_mask.npy")
            meta_path = os.path.join(calib_dir, "calibration_meta.json")
            if os.path.exists(mask_path):
                self.defect_mask = np.load(mask_path)
                if os.path.exists(meta_path):
                    with open(meta_path, "r") as f:
                        self.calibration_meta = json.load(f)
                count = int(np.sum(self.defect_mask > 0)) if self.defect_mask is not None else 0
                event_logger.log(f"Loaded existing black calibration: {count} defective pixels.")
        except Exception as e:
            event_logger.log(f"Error loading black calibration: {e}")

    def start_black_calibration(self, target_frames: int = 15) -> Dict[str, Any]:
        """
        Starts collecting incoming dark frames to compute a master dark and defect pixel map.
        
        Args:
            target_frames: Number of dark frames to collect and average (default 15).
        """
        with self.lock:
            self.is_calibrating = True
            self.calib_target_frames = max(3, min(60, int(target_frames)))
            self._calib_frames.clear()
            event_logger.log(f"Black calibration started ({self.calib_target_frames} frames).")
            return {
                "success": True,
                "status": "calibrating",
                "target_frames": self.calib_target_frames,
                "frames_collected": 0
            }

    def _finalize_black_calibration(self):
        """
        Processes collected dark frames into a master dark and hot-pixel/cluster defect map.
        Calculates background pedestal statistics (median and MAD-based sigma).
        Detects both isolated hot pixels and multi-pixel clusters.
        Saves calibration to disk for use in live preview softening and mosaic inverse-weighting.
        """
        if not self._calib_frames:
            self.is_calibrating = False
            return

        frames = np.array(self._calib_frames, dtype=np.float32)
        master_dark = np.mean(frames, axis=0)

        # Convert to grayscale luminance for defect detection
        if master_dark.ndim == 3:
            dark_gray = cv2.cvtColor(master_dark.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
        else:
            dark_gray = master_dark.copy()

        # Robust background estimation
        med = float(np.median(dark_gray))
        mad = float(np.median(np.abs(dark_gray - med)))
        sigma = max(1.0, 1.4826 * mad)

        # Detect hot pixels & clusters: threshold at least 5*sigma above median, with min 6 ADU headroom
        threshold = max(med + 6.0, med + 5.0 * sigma)
        defect_mask = (dark_gray > threshold).astype(np.uint8)

        # Count individual defect pixels
        defect_count = int(np.sum(defect_mask))

        meta = {
            "timestamp": datetime.now().isoformat(),
            "frames_collected": len(self._calib_frames),
            "defect_pixel_count": defect_count,
            "median_bg": round(med, 2),
            "sigma_bg": round(sigma, 2),
            "threshold": round(threshold, 2),
        }

        calib_dir = os.path.join(self.recorder.base_dir, "calibration")
        os.makedirs(calib_dir, exist_ok=True)

        try:
            np.save(os.path.join(calib_dir, "defect_mask.npy"), defect_mask)
            np.save(os.path.join(calib_dir, "master_dark.npy"), master_dark)
            with open(os.path.join(calib_dir, "calibration_meta.json"), "w") as f:
                json.dump(meta, f, indent=2)
        except Exception as e:
            event_logger.log(f"Failed to save black calibration: {e}")

        self.defect_mask = defect_mask
        self.calibration_meta = meta
        self.is_calibrating = False
        self._calib_frames.clear()

        event_logger.log(
            f"Black calibration complete: {defect_count} defective/hot pixels mapped "
            f"(bg={med:.1f}, sigma={sigma:.2f}, threshold={threshold:.1f})."
        )

    def get_calibration_status(self) -> Dict[str, Any]:
        """Returns the current black calibration status and metadata."""
        with self.lock:
            return {
                "is_calibrating": self.is_calibrating,
                "frames_collected": len(self._calib_frames),
                "target_frames": self.calib_target_frames,
                "calibrated": self.defect_mask is not None,
                "meta": self.calibration_meta
            }

    def clear_black_calibration(self) -> Dict[str, Any]:
        """Clears active black calibration and removes calibration files from disk."""
        with self.lock:
            self.defect_mask = None
            self.calibration_meta = None
            self.is_calibrating = False
            self._calib_frames.clear()

            calib_dir = os.path.join(self.recorder.base_dir, "calibration")
            for fname in ["defect_mask.npy", "master_dark.npy", "calibration_meta.json"]:
                fpath = os.path.join(calib_dir, fname)
                if os.path.exists(fpath):
                    try:
                        os.remove(fpath)
                    except Exception:
                        pass

            event_logger.log("Black calibration cleared.")
            return {"success": True, "calibrated": False}

    def start_baseline_capture(self, key: str, frames: int = 10) -> Dict[str, Any]:
        """Starts capturing frames with cap on to compute dark baseline for a specific setting key."""
        with self.lock:
            self.is_capturing_baseline = True
            self.baseline_capture_target = max(3, min(30, int(frames)))
            self.baseline_capture_key = key
            self.baseline_capture_frames.clear()
            event_logger.log(f"Baseline capture started for {key} ({self.baseline_capture_target} frames, cap on).")
            return {
                "success": True,
                "is_capturing": True,
                "key": key,
                "target_frames": self.baseline_capture_target
            }

    def _finalize_baseline_capture(self):
        """Averages collected dark frames, saves the baseline to disk, and enables subtraction."""
        if not self.baseline_capture_frames:
            self.is_capturing_baseline = False
            return
        
        frames = np.array(self.baseline_capture_frames, dtype=np.float32)
        master_baseline = np.mean(frames, axis=0)
        
        key = self.baseline_capture_key or "default"
        baselines_dir = os.path.join(self.recorder.base_dir, "calibration", "baselines")
        os.makedirs(baselines_dir, exist_ok=True)
        file_path = os.path.join(baselines_dir, f"{key}.npy")
        try:
            np.save(file_path, master_baseline)
            event_logger.log(f"Saved baseline for {key} ({len(frames)} frames averaged).")
        except Exception as e:
            event_logger.log(f"Error saving baseline {key}: {e}")
            
        self.active_baseline = master_baseline
        self.active_baseline_key = key
        self.baseline_subtract = True  # Automatically enable subtraction after capturing
        self.is_capturing_baseline = False
        self.baseline_capture_frames.clear()

    def check_baseline_status(self, key: str) -> Dict[str, Any]:
        """Checks if a baseline exists on disk for a given setting key and reports status."""
        baselines_dir = os.path.join(self.recorder.base_dir, "calibration", "baselines")
        file_path = os.path.join(baselines_dir, f"{key}.npy")
        exists = os.path.exists(file_path)
        
        with self.lock:
            if exists and self.active_baseline_key != key:
                try:
                    self.active_baseline = np.load(file_path)
                    self.active_baseline_key = key
                except Exception as e:
                    event_logger.log(f"Error loading baseline {key}: {e}")
            elif not exists and self.active_baseline_key == key:
                self.active_baseline = None
                self.active_baseline_key = ""
                
            return {
                "key": key,
                "exists": exists,
                "subtract": self.baseline_subtract,
                "pedestal": self.baseline_pedestal,
                "is_capturing": self.is_capturing_baseline,
                "active_key": self.active_baseline_key
            }

    def set_baseline_config(self, subtract: Optional[bool] = None, pedestal: Optional[float] = None, key: Optional[str] = None) -> Dict[str, Any]:
        """Updates baseline subtraction toggle, pedestal value, or active setting key."""
        baselines_dir = os.path.join(self.recorder.base_dir, "calibration", "baselines")
        with self.lock:
            if subtract is not None:
                self.baseline_subtract = bool(subtract)
                event_logger.log(f"Baseline subtract set to {self.baseline_subtract}")
            if pedestal is not None:
                self.baseline_pedestal = max(0.0, min(100.0, float(pedestal)))
                event_logger.log(f"Baseline pedestal set to {self.baseline_pedestal}")
            if key is not None and key != self.active_baseline_key:
                file_path = os.path.join(baselines_dir, f"{key}.npy")
                if os.path.exists(file_path):
                    try:
                        self.active_baseline = np.load(file_path)
                        self.active_baseline_key = key
                    except Exception as e:
                        event_logger.log(f"Error loading baseline {key}: {e}")
                else:
                    self.active_baseline = None
                    self.active_baseline_key = key
                    
        return self.check_baseline_status(self.active_baseline_key or key or "")

    def set_n_avg(self, n):
        """Backwards compatibility for legacy average calls."""
        return self.set_accumulation_count(n)

    @property
    def n_avg(self) -> int:
        """Backwards compatibility for legacy average callers."""
        return self.accumulation_count

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
                "inlier_ratio": round(self.inlier_ratio, 3),
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
                search_center = self.reference_stars[0] if self.reference_stars else None
                T = align_images(self.ref_frame, frame, translation_only=True, search_center=search_center)
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
            search_center = self.reference_stars[0] if self.reference_stars else None
            T, metrics = align_images(self.ref_frame, frame, translation_only=True, return_metrics=True, search_center=search_center)
        else:
            T, metrics = align_images(self.prev_alignment_frame, frame, translation_only=True, return_metrics=True)
            
        dx = float(T[0, 2])
        dy = float(T[1, 2])
        inlier_ratio = metrics["inlier_ratio"]

        if abs(dx) < 1e-5 and abs(dy) < 1e-5:
            with self.lock:
                self.inlier_ratio = inlier_ratio
            return

        with self.lock:
            self.inlier_ratio = inlier_ratio
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
            
            fmt = getattr(self.recorder, "file_format", "tif").lower()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            
            if fmt in ("tif", "tiff"):
                filename = f"cap_{timestamp}.tif"
                if self.accumulation_count > 1 and self.acc_buffer is not None and self.acc_count > 0:
                    avg_float = self.acc_buffer / self.acc_count
                else:
                    avg_float = self.latest_frame.astype(np.float32)
                if avg_float.ndim == 3 and np.array_equal(avg_float[..., 0], avg_float[..., 1]) and np.array_equal(avg_float[..., 0], avg_float[..., 2]):
                    avg_float = avg_float[..., 0]
                save_frame = np.clip(np.round(avg_float * 257.0), 0, 65535).astype(np.uint16)
            else:
                filename = f"cap_{timestamp}.jpg"
                save_frame = self.latest_frame.copy()
        
        filepath = os.path.join(self.sequence_info["directory"], filename)
        if cv2.imwrite(filepath, save_frame):
            try:
                from sky_nav import write_nav_sidecar
                write_nav_sidecar(filepath)
            except Exception:
                pass
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
                        try:
                            from sky_nav import write_nav_sidecar
                            write_nav_sidecar(filepath)
                        except Exception:
                            pass
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

    def start_recording(self, accumulation_count=None, accumulation_ms=None, file_format=None):
        """Starts continuous stream recording to a timestamped folder."""
        count = accumulation_count or accumulation_ms or self.accumulation_count
        with self.lock:
            self.accumulation_count = max(1, int(count))
            self.acc_buffer = None
            self.acc_count = 0
        return self.recorder.start_recording(
            accumulation_count=self.accumulation_count,
            file_format=file_format
        )

    def stop_recording(self):
        """Stops active stream recording and flushes pending frames."""
        with self.lock:
            if self.recorder.is_recording and self.accumulation_count > 1 and self.acc_buffer is not None and self.acc_count > 0:
                avg_float = self.acc_buffer / self.acc_count
                if self.recorder.file_format in ("tif", "tiff"):
                    if avg_float.ndim == 3 and np.array_equal(avg_float[..., 0], avg_float[..., 1]) and np.array_equal(avg_float[..., 0], avg_float[..., 2]):
                        avg_float = avg_float[..., 0]
                    stack_img = np.clip(np.round(avg_float * 257.0), 0, 65535).astype(np.uint16)
                else:
                    stack_img = self.latest_frame.copy()
                self.recorder.add_completed_stack(stack_img, self.acc_count)
                self.acc_buffer = None
                self.acc_count = 0
        return self.recorder.stop_recording()

    def get_recording_status(self):
        """Returns live stream recording status and metrics."""
        return self.recorder.get_status()

    def close(self):
        """
        Closes background pipeline threads and stops active sequences and recordings.
        """
        self.is_running = False
        self.sequence_stop_event.set()
        if hasattr(self, 'recorder'):
            self.recorder.stop_recording()


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
