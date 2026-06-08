import numpy as np
import cv2
import threading
import time
import os
from datetime import datetime
from logger import event_logger

class FrameAligner:
    def __init__(self):
        self.orb = cv2.ORB_create(nfeatures=2000)
        # Use FLANN or BFMatcher with KNN for ratio test
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

    def get_translation(self, frame1, frame2):
        """Estimate (dx, dy) translation using RANSAC and Ratio Test."""
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        kp1, des1 = self.orb.detectAndCompute(gray1, None)
        kp2, des2 = self.orb.detectAndCompute(gray2, None)

        if des1 is None or des2 is None or len(kp1) < 10 or len(kp2) < 10:
            return 0.0, 0.0

        # KNN Match for Ratio Test
        matches = self.matcher.knnMatch(des1, des2, k=2)
        
        good_matches = []
        for m_n in matches:
            if len(m_n) == 2:
                m, n = m_n
                if m.distance < 0.7 * n.distance:
                    good_matches.append(m)
        
        if len(good_matches) < 4:
            return 0.0, 0.0

        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # RANSAC for translation
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 3.0)
        
        if M is not None:
            inliers_mask = mask.ravel().tolist()
            dxs = [dst_pts[i][0][0] - src_pts[i][0][0] for i, m in enumerate(inliers_mask) if m]
            dys = [dst_pts[i][0][1] - src_pts[i][0][1] for i, m in enumerate(inliers_mask) if m]
            
            if len(dxs) >= 4:
                dx = float(np.median(dxs))
                dy = float(np.median(dys))
                
                # Sanity Check: If jump is > 50px, it's almost certainly a false match in our rig
                if abs(dx) > 50 or abs(dy) > 50:
                    return 0.0, 0.0
                    
                return dx, dy

        return 0.0, 0.0

class PanoramaManager:
    def __init__(self, rig):
        self.rig = rig
        self.is_active = False
        self.total_frames = 0
        self.current_frame = 0
        self.drift_step = 0.0
        self.auto_align = False
        self.sum_buffer = None
        self.weight_buffer = None
        self.offset_x = 0.0
        self.offset_y = 0.0
        self.prev_frame = None

    def start(self, frames, drift_step=10.0, auto_align=False):
        if self.is_active: return False
        self.total_frames = frames
        self.drift_step = drift_step
        self.auto_align = auto_align
        self.current_frame = 0
        self.offset_x = 0.0
        self.offset_y = 0.0
        self.sum_buffer = None
        self.weight_buffer = None
        self.prev_frame = None
        self.aligner = FrameAligner()
        self.is_active = True
        threading.Thread(target=self._run_panorama, daemon=True).start()
        event_logger.log(f"Panorama started: {frames} frames, auto_align={auto_align}")
        return True

    def _run_panorama(self):
        try:
            for i in range(self.total_frames):
                if not self.is_active: break
                frame = self.rig.get_raw_frame()
                if frame is not None:
                    if self.auto_align and self.prev_frame is not None:
                        dx, dy = self.aligner.get_translation(self.prev_frame, frame)
                        # We only update if we got a valid non-zero match, 
                        # or we can just accept 0 if the star actually didn't move.
                        # But we definitely want to avoid the "big jumps".
                        self.offset_x -= dx
                        self.offset_y -= dy
                        event_logger.log(f"F{i} dx={dx:.1f} TotalX={self.offset_x:.1f}")
                    elif not self.auto_align:
                        self.offset_x += self.drift_step
                    
                    self._accumulate(frame)
                    self.current_frame += 1
                    self.prev_frame = frame
                time.sleep(0.5)
            self._finalize()
        except Exception as e:
            event_logger.log(f"Panorama Error: {e}")
        finally:
            self.is_active = False

    def _accumulate(self, frame):
        h, w = frame.shape[:2]
        if self.sum_buffer is None:
            buf_w, buf_h = w * 6, h * 3
            self.sum_buffer = np.zeros((buf_h, buf_w, 3), dtype=np.float32)
            self.weight_buffer = np.zeros((buf_h, buf_w), dtype=np.float32)
            self.base_x, self.base_y = (buf_w - w) // 2, (buf_h - h) // 2

        curr_x, curr_y = int(self.base_x + self.offset_x), int(self.base_y + self.offset_y)
        if curr_x < 0 or curr_y < 0 or curr_x + w > self.sum_buffer.shape[1] or curr_y + h > self.sum_buffer.shape[0]:
            event_logger.log(f"Out of bounds: x={curr_x}, y={curr_y}")
            return
        self.sum_buffer[curr_y:curr_y+h, curr_x:curr_x+w] += frame.astype(np.float32)
        self.weight_buffer[curr_y:curr_y+h, curr_x:curr_x+w] += 1.0

    def _finalize(self):
        if self.sum_buffer is None: return
        event_logger.log(f"Finalizing {self.current_frame} frames")
        mask = self.weight_buffer > 0
        res = np.zeros_like(self.sum_buffer, dtype=np.float32)
        res[mask] = self.sum_buffer[mask] / self.weight_buffer[mask][:, np.newaxis]
        final = np.clip(res, 0, 255).astype(np.uint8)
        gray = cv2.cvtColor(final, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            x, y, w, h = cv2.boundingRect(np.concatenate(contours))
            final = final[y:y+h, x:x+w]
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"panorama_{ts}.jpg"
        cv2.imwrite(os.path.join("/home/kio/projects/astrocam/captures", filename), final)
        event_logger.log(f"Panorama complete: {filename}")

    def get_status(self):
        return {
            "active": self.is_active, "current": self.current_frame, "total": self.total_frames,
            "progress": (self.current_frame / self.total_frames * 100) if self.total_frames > 0 else 0,
            "offset_x": round(self.offset_x, 1), "offset_y": round(self.offset_y, 1)
        }

    def stop(self):
        self.is_active = False
        event_logger.log("Panorama cancelled")
