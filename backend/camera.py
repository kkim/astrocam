import cv2
import threading
import time
import numpy as np

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
        
        # Default resolution
        self.width = 1920
        self.height = 1080
        self.format = 'MJPG'
        
        self._init_camera()
        
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()

    def _init_camera(self):
        with self.lock:
            if self.cap:
                self.cap.release()
            
            self.cap = cv2.VideoCapture(self.device_id, cv2.CAP_V4L2)
            if not self.cap.isOpened():
                print("Failed to open camera hardware - Entering MOCK MODE")
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
            
            print(f"Camera initialized ({self.width}x{self.height} {self.format})")
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
                print(f"Capture loop error: {e}")
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
        # Create a dark space-like background
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        # Add dynamic noise that can be averaged out
        noise = np.random.normal(0, 20, (self.height, self.width, 3)).astype(np.int16)
        frame = cv2.add(frame.astype(np.int16), noise)
        frame = np.clip(frame, 0, 255).astype(np.uint8)

        # Add some persistent "stars" (random noise)
        np.random.seed(42) # Keep stars in the same place
        star_noise = np.random.randint(0, 255, (self.height, self.width), dtype=np.uint8)
        mask = star_noise > 252
        frame[mask] = [255, 255, 255]
        np.random.seed(None) # Reset seed
        
        # Add a moving "planet"
        t = time.time()
        cx = int(self.width / 2 + (self.width / 4) * np.cos(t * 0.5))
        cy = int(self.height / 2 + (self.height / 4) * np.sin(t * 0.5))
        cv2.circle(frame, (cx, cy), 50, (100, 150, 255), -1)
        
        # Add text
        cv2.putText(frame, f"MOCK MODE - {time.strftime('%H:%M:%S')}", (50, 100), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
        cv2.putText(frame, f"Res: {self.width}x{self.height}", (50, 180), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
        cv2.putText(frame, f"Avg: {self.n_avg} frames", (50, 260), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
        
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
            elif prop == "auto_exposure":
                val = 3 if value > 0 else 1
                self.cap.set(mapping[prop], val)
            else:
                self.cap.set(mapping[prop], value)
            return True
        return False

    def get_params(self):
        props = ["brightness", "contrast", "saturation", "gain", "exposure", "sharpness"]
        mapping = {
            "brightness": cv2.CAP_PROP_BRIGHTNESS,
            "contrast": cv2.CAP_PROP_CONTRAST,
            "saturation": cv2.CAP_PROP_SATURATION,
            "gain": cv2.CAP_PROP_GAIN,
            "exposure": cv2.CAP_PROP_EXPOSURE,
            "sharpness": cv2.CAP_PROP_SHARPNESS
        }
        res = {"average": self.n_avg}
        if not self.cap or not self.cap.isOpened(): return res
        for p in props:
            try:
                res[p] = self.cap.get(mapping[p])
            except:
                res[p] = 0
        return res

    def close(self):
        self.is_running = False
        self.thread.join()
        if self.cap:
            self.cap.release()
