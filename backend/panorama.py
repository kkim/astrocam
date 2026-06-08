import numpy as np
import cv2
import threading
import time
import os
from datetime import datetime
from logger import event_logger

class FrameAligner:
    def __init__(self):
        self.orb = cv2.ORB_create(nfeatures=1000)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def get_translation(self, frame1, frame2):
        """Estimate (dx, dy) translation between frame1 and frame2 using RANSAC."""
        # Convert to grayscale
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        # Detect and compute
        kp1, des1 = self.orb.detectAndCompute(gray1, None)
        kp2, des2 = self.orb.detectAndCompute(gray2, None)

        if des1 is None or des2 is None or len(kp1) < 10 or len(kp2) < 10:
            return 0.0, 0.0

        # Match using BFMatcher
        matches = self.matcher.match(des1, des2)
        if len(matches) < 10:
            return 0.0, 0.0

        # Extract coordinates
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        # Use RANSAC to find the translation
        # Even if we only want translation, findHomography with RANSAC is great at filtering
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        
        if M is not None:
            # For translation, we can just look at the last column of the matrix
            # Or better, take the median of the INLIER matches
            inliers_mask = mask.ravel().tolist()
            dxs = []
            dys = []
            for i, is_inlier in enumerate(inliers_mask):
                if is_inlier:
                    p1 = src_pts[i][0]
                    p2 = dst_pts[i][0]
                    dxs.append(p2[0] - p1[0])
                    dys.append(p2[1] - p1[1])
            
            if len(dxs) > 0:
                return float(np.median(dxs)), float(np.median(dys))

        return 0.0, 0.0

class PanoramaManager:
    def __init__(self, rig):
        self.rig = rig
        self.is_active = False
        self.total_frames = 0
        self.current_frame = 0
        self.drift_step = 0.0 # manual pixels per frame
        self.auto_align = False
        
        self.sum_buffer = None
        self.weight_buffer = None
        self.offset_x = 0.0
        self.offset_y = 0.0
        
        self.aligner = FrameAligner()
        self.prev_frame = None
        
        self.lock = threading.Lock()
        self.thread = None

    def start(self, frames, drift_step, auto_align=False):
        if self.is_active:
            return False
            
        self.total_frames = frames
        self.drift_step = drift_step
        self.auto_align = auto_align
        self.current_frame = 0
        self.offset_x = 0.0
        self.offset_y = 0.0
        
        self.sum_buffer = None
        self.weight_buffer = None
        self.prev_frame = None
        
        self.is_active = True
        self.thread = threading.Thread(target=self._run_panorama, daemon=True)
        self.thread.start()
        event_logger.log(f"Panorama started: {frames} frames, auto_align={auto_align}")
        return True

    def _run_panorama(self):
        try:
            for i in range(self.total_frames):
                if not self.is_active:
                    break
                
                # 1. Get raw frame from rig
                frame = self.rig.get_raw_frame()
                
                if frame is not None:
                    # 2. Alignment
                    if self.auto_align and self.prev_frame is not None:
                        dx, dy = self.aligner.get_translation(self.prev_frame, frame)
                        # Subtract observed star motion to get camera motion
                        self.offset_x -= dx
                        self.offset_y -= dy
                        if i % 5 == 0: # Log every 5 frames to avoid spamming
                            event_logger.log(f"Auto-Align: dx={dx:.1f}, dy={dy:.1f} (Total: {self.offset_x:.1f}, {self.offset_y:.1f})")
                    elif not self.auto_align:
                        # Manual drift
                        self.offset_x += self.drift_step
                    
                    # 3. Accumulate
                    self._accumulate(frame)
                    self.current_frame += 1
                    
                    # Store current for next alignment
                    self.prev_frame = frame
                else:
                    event_logger.log(f"Panorama Warning: Frame {i} is None")
                
                # In auto-align mode, we don't need a fixed sleep if we want maximum speed,
                # but let's keep it to avoid overwhelming the CPU
                time.sleep(0.5)
                
            self._finalize()
        except Exception as e:
            event_logger.log(f"Panorama Error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.is_active = False

    def _accumulate(self, frame):
        h, w = frame.shape[:2]
        
        # Initialize buffers if needed
        if self.sum_buffer is None:
            # Create a very large buffer to accommodate drift in any direction
            # For 1080p, we allow up to 4x width and 2x height for the panorama
            buf_w = w * 4
            buf_h = h * 2
            self.sum_buffer = np.zeros((buf_h, buf_w, 3), dtype=np.float32)
            self.weight_buffer = np.zeros((buf_h, buf_w), dtype=np.float32)
            
            # Start in the middle-left
            self.base_x = w # 1 frame width margin on the left
            self.base_y = h // 2 # 0.5 frame height margin on top

        # Calculate current position on canvas
        curr_x = int(self.base_x + self.offset_x)
        curr_y = int(self.base_y + self.offset_y)
        
        # Bounds checking
        if curr_x < 0 or curr_y < 0 or curr_x + w > self.sum_buffer.shape[1] or curr_y + h > self.sum_buffer.shape[0]:
            event_logger.log("Panorama Warning: Frame outside buffer bounds. Skipping.")
            return

        # Accumulate into Sum
        self.sum_buffer[curr_y:curr_y+h, curr_x:curr_x+w] += frame.astype(np.float32)
        self.weight_buffer[curr_y:curr_y+h, curr_x:curr_x+w] += 1.0

    def _finalize(self):
        if self.sum_buffer is None:
            event_logger.log("Panorama Error: No frames were accumulated")
            return

        event_logger.log(f"Panorama Finalizing: {self.current_frame} frames accumulated")
        mask = self.weight_buffer > 0
        result = np.zeros_like(self.sum_buffer, dtype=np.float32)
        result[mask] = self.sum_buffer[mask] / self.weight_buffer[mask][:, np.newaxis]
        
        # Convert to uint8
        final_img = np.clip(result, 0, 255).astype(np.uint8)
        
        # Optional: Crop to content
        gray = cv2.cvtColor(final_img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            x, y, w, h = cv2.boundingRect(np.concatenate(contours))
            final_img = final_img[y:y+h, x:x+w]

        # Save result
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"panorama_{timestamp}.jpg"
        save_path = "/home/kio/projects/astrocam/captures"
        if not os.path.exists(save_path):
            os.makedirs(save_path)
            
        cv2.imwrite(os.path.join(save_path, filename), final_img)
        event_logger.log(f"Panorama complete: Saved to {filename}")

    def get_status(self):
        return {
            "active": self.is_active,
            "current": self.current_frame,
            "total": self.total_frames,
            "progress": (self.current_frame / self.total_frames * 100) if self.total_frames > 0 else 0,
            "offset_x": round(self.offset_x, 1),
            "offset_y": round(self.offset_y, 1)
        }

    def stop(self):
        self.is_active = False
        event_logger.log("Panorama cancelled")
