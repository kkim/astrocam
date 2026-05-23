import cv2
import threading
import time
import numpy as np
from logger import event_logger

class SV205Camera:
    def __init__(self, device_id=0):
        self.device_id = device_id
        self.cap = None
        self.lock = threading.Lock()
        self.latest_frame = None
        self.last_frame_time = 0.0
        self.is_running = True
        self.connected = False
        self.mock_mode = False
        self.mean_brightness = 0.0
        self.fail_count = 0
        self.fps = 0.0
        self.n_avg = 1
        self.acc_frame = None
        self._frame_count = 0
        self._last_fps_calc_time = time.time()
        
        # Sequence capture state
        self.sequence_info = {
            "active": False,
            "count": 0,
            "total": 0,
            "interval": 0,
            "last_capture_time": 0,
            "directory": "/home/kio/projects/astrocam/captures"
        }
        
        # Ensure directory exists
        import os
        if not os.path.exists(self.sequence_info["directory"]):
            os.makedirs(self.sequence_info["directory"])
            
        self.sequence_stop_event = threading.Event()
        
        # Default resolution
        self.width = 1920
        self.height = 1080
        self.format = 'MJPG'
        
        # Default parameter values for mock mode / state tracking
        self.params = {
            "brightness": 128,
            "contrast": 32,
            "saturation": 64,
            "gain": 0,
            "exposure": 156,
            "sharpness": 2,
            "auto_exposure": 0
        }
        
        self._init_camera()
        
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()

    def _init_camera(self):
        with self.lock:
            if self.cap:
                self.cap.release()
            
            self.cap = cv2.VideoCapture(self.device_id, cv2.CAP_V4L2)
            if not self.cap.isOpened():
                event_logger.log("Failed to open camera hardware - Entering MOCK MODE")
                self.mock_mode = True
                self.connected = True
                return True
                
            self.mock_mode = False
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self.format))
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            # Disable auto-exposure initially
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
            
            event_logger.log(f"Camera initialized ({self.width}x{self.height} {self.format})")
            return True

    def set_resolution(self, width, height):
        self.width = width
        self.height = height
        return self._init_camera()

    def _capture_loop(self):
        while self.is_running:
            try:
                frame = None
                if self.mock_mode:
                    frame = self._generate_mock_frame()
                    time.sleep(0.033) # ~30 FPS
                else:
                    if not self.cap or not self.cap.isOpened():
                        time.sleep(1)
                        self._init_camera()
                        continue

                    if self.cap.grab():
                        ret, retrieved_frame = self.cap.retrieve()
                        if ret and retrieved_frame is not None:
                            frame = retrieved_frame
                        else:
                            self._handle_fail()
                    else:
                        self._handle_fail()

                if frame is not None:
                    self._process_frame(frame)
                    
            except Exception as e:
                event_logger.log(f"Capture loop error: {e}")
                self._handle_fail()
                time.sleep(0.5)

    def _process_frame(self, frame):
        with self.lock:
            # Averaging / Stacking logic
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

            self.last_frame_time = time.time()
            self.connected = True
            self.fail_count = 0
            
            # FPS Calculation
            self._frame_count += 1
            now = time.time()
            elapsed = now - self._last_fps_calc_time
            if elapsed >= 2.0:
                self.fps = self._frame_count / elapsed
                self._frame_count = 0
                self._last_fps_calc_time = now
            
            # Fast brightness check
            h, w = frame.shape[:2]
            self.mean_brightness = float(frame[h//2, w//2].mean())

    def _generate_mock_frame(self):
        # 1. Create background with random noise: mean 5% (12.75), std 5% (12.75)
        mean = 255 * 0.05
        std = 255 * 0.05
        noise = np.random.normal(mean, std, (self.height, self.width, 3)).astype(np.float32)
        frame = np.clip(noise, 0, 255).astype(np.uint8)

        # Use a fixed seed for stars
        np.random.seed(42)
        
        # 2. 100 stars: 100% brightness, r=0.5px, anti-aliased
        # (r=0.5 is effectively a single pixel or very small circle)
        for _ in range(100):
            x, y = np.random.randint(0, self.width), np.random.randint(0, self.height)
            cv2.circle(frame, (x, y), 0, (255, 255, 255), -1, cv2.LINE_AA)
            
        # 3. 20 stars: 100% brightness, r=1.0px, anti-aliased
        for _ in range(20):
            x, y = np.random.randint(0, self.width), np.random.randint(0, self.height)
            cv2.circle(frame, (x, y), 1, (255, 255, 255), -1, cv2.LINE_AA)
            
        # 4. 10 stars: 100% brightness, r=1.5px, anti-aliased
        for _ in range(10):
            x, y = np.random.randint(0, self.width), np.random.randint(0, self.height)
            cv2.circle(frame, (x, y), 2, (255, 255, 255), -1, cv2.LINE_AA)
            
        # 5. 2 stars: 100% brightness, r=2.0px, anti-aliased
        for _ in range(2):
            x, y = np.random.randint(0, self.width), np.random.randint(0, self.height)
            cv2.circle(frame, (x, y), 3, (255, 255, 255), -1, cv2.LINE_AA)

        np.random.seed(None) # Reset seed
        
        # Minimal text
        cv2.putText(frame, f"MOCK - {time.strftime('%H:%M:%S')}", (20, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1)
        
        return frame

    def _handle_fail(self):
        self.fail_count += 1
        if self.fail_count > 15:
            with self.lock:
                self.connected = False
            self._init_camera()
            self.fail_count = 0
        time.sleep(0.01)

    def get_status(self):
        with self.lock:
            return {
                "connected": self.connected,
                "mean_brightness": self.mean_brightness,
                "fps": round(self.fps, 1),
                "last_frame_time": self.last_frame_time,
                "width": self.width,
                "height": self.height,
                "timestamp": time.time()
            }

    def get_frame(self):
        with self.lock:
            if self.latest_frame is None:
                return None
            
            display_w = 1280 if self.width >= 1280 else self.width
            display_h = int(display_w * (self.height / self.width))
            
            img = cv2.resize(self.latest_frame, (display_w, display_h))
            ret, jpeg = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if not ret:
                return None
            return jpeg.tobytes()

    def set_param(self, prop, value):
        mapping = {
            "brightness": cv2.CAP_PROP_BRIGHTNESS,
            "contrast": cv2.CAP_PROP_CONTRAST,
            "saturation": cv2.CAP_PROP_SATURATION,
            "gain": cv2.CAP_PROP_GAIN,
            "exposure": cv2.CAP_PROP_EXPOSURE,
            "sharpness": cv2.CAP_PROP_SHARPNESS,
            "auto_exposure": cv2.CAP_PROP_AUTO_EXPOSURE,
            "average": "N_AVG"
        }
        if prop in mapping:
            if prop == "average":
                with self.lock:
                    self.n_avg = int(value)
            else:
                self.params[prop] = value
                if not self.mock_mode and self.cap and self.cap.isOpened():
                    if prop == "auto_exposure":
                        val = 3 if value > 0 else 1
                        self.cap.set(mapping[prop], val)
                    else:
                        self.cap.set(mapping[prop], value)
            return True
        return False

    def get_params(self):
        props = ["brightness", "contrast", "saturation", "gain", "exposure", "sharpness", "auto_exposure"]
        mapping = {
            "brightness": cv2.CAP_PROP_BRIGHTNESS,
            "contrast": cv2.CAP_PROP_CONTRAST,
            "saturation": cv2.CAP_PROP_SATURATION,
            "gain": cv2.CAP_PROP_GAIN,
            "exposure": cv2.CAP_PROP_EXPOSURE,
            "sharpness": cv2.CAP_PROP_SHARPNESS,
            "auto_exposure": cv2.CAP_PROP_AUTO_EXPOSURE
        }
        res = {"average": self.n_avg}
        
        # Merge our tracked params (good for mock mode or hardware caching)
        res.update(self.params)
        
        # If hardware is available, try to get actual values
        if not self.mock_mode and self.cap and self.cap.isOpened():
            for p in props:
                try:
                    val = self.cap.get(mapping[p])
                    if p == "auto_exposure":
                        res[p] = 1 if val >= 3 else 0
                    else:
                        res[p] = val
                except:
                    pass
                    
        return res

    def start_sequence(self, count, interval_sec):
        if self.sequence_info["active"]:
            return False
        
        self.sequence_info["active"] = True
        self.sequence_info["total"] = count
        self.sequence_info["count"] = 0
        self.sequence_info["interval"] = interval_sec
        self.sequence_stop_event.clear()
        
        event_logger.log(f"Sequence: Starting {count} frames @ {interval_sec}s")
        thread = threading.Thread(target=self._sequence_loop, daemon=True)
        thread.start()
        return True

    def _sequence_loop(self):
        import os
        from datetime import datetime
        
        while self.sequence_info["count"] < self.sequence_info["total"] and not self.sequence_stop_event.is_set():
            # Wait for next interval
            now = time.time()
            if now - self.sequence_info["last_capture_time"] >= self.sequence_info["interval"]:
                frame = None
                with self.lock:
                    if self.latest_frame is not None:
                        frame = self.latest_frame.copy()
                
                if frame is not None:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                    filename = f"seq_{timestamp}.jpg"
                    filepath = os.path.join(self.sequence_info["directory"], filename)
                    
                    success = cv2.imwrite(filepath, frame)
                    if success:
                        self.sequence_info["count"] += 1
                        self.sequence_info["last_capture_time"] = time.time()
                        event_logger.log(f"Sequence: {self.sequence_info['count']}/{self.sequence_info['total']} - {filename}")
                    else:
                        event_logger.log(f"Error: Failed to write {filepath}")
            
            time.sleep(0.1)
            
        self.sequence_info["active"] = False
        event_logger.log("Sequence: Complete")

    def close(self):
        self.is_running = False
        self.thread.join()
        if self.cap:
            self.cap.release()
