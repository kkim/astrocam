import numpy as np
import cv2
import threading
import time
import os
from datetime import datetime
from logger import event_logger

class PanoramaManager:
    def __init__(self, rig):
        self.rig = rig
        self.is_active = False
        self.total_frames = 0
        self.current_frame = 0
        self.drift_step = 0.0 # pixels per frame
        
        self.sum_buffer = None
        self.weight_buffer = None
        self.offset_x = 0.0
        self.offset_y = 0.0
        
        self.lock = threading.Lock()
        self.thread = None

    def start(self, frames, drift_step):
        if self.is_active:
            return False
            
        self.total_frames = frames
        self.drift_step = drift_step
        self.current_frame = 0
        self.offset_x = 0.0
        self.offset_y = 0.0
        
        # We don't know the exact size yet, will initialize on first frame
        self.sum_buffer = None
        self.weight_buffer = None
        
        self.is_active = True
        self.thread = threading.Thread(target=self._run_panorama, daemon=True)
        self.thread.start()
        event_logger.log(f"Panorama started: {frames} frames, step {drift_step}px")
        return True

    def _run_panorama(self):
        try:
            for i in range(self.total_frames):
                if not self.is_active:
                    break
                
                # 1. Get raw frame from rig
                # We need the full resolution frame, not the MJPEG stream bytes
                # Let's assume rig.latest_frame is available via a new method or lock
                frame = self.rig.latest_frame.copy() if self.rig.latest_frame is not None else None
                
                if frame is not None:
                    self._accumulate(frame)
                    self.current_frame += 1
                
                # 2. Wait for drift to occur (simulate or real wait)
                # In real life, we just wait. In mock, we rely on the motor speed.
                # For simplicity, we assume the drift happens between frames.
                time.sleep(0.5) # Allow some time for movement
                
                # Update expected offset
                self.offset_x += self.drift_step
                
            self._finalize()
        except Exception as e:
            event_logger.log(f"Panorama Error: {e}")
        finally:
            self.is_active = False

    def _accumulate(self, frame):
        h, w = frame.shape[:2]
        
        # Initialize buffers if needed
        # Buffers are large enough to hold total drift + 1 frame
        if self.sum_buffer is None:
            max_drift = abs(self.drift_step * self.total_frames)
            buf_w = w + int(max_drift) + 100
            buf_h = h + 100
            self.sum_buffer = np.zeros((buf_h, buf_w, 3), dtype=np.float32)
            self.weight_buffer = np.zeros((buf_h, buf_w), dtype=np.float32)
            # Center vertically, start at left
            self.base_y = 50
            self.base_x = 50 if self.drift_step >= 0 else int(max_drift) + 50

        # Calculate current position on canvas
        curr_x = int(self.base_x + self.offset_x)
        curr_y = int(self.base_y + self.offset_y)
        
        # Accumulate into Sum
        self.sum_buffer[curr_y:curr_y+h, curr_x:curr_x+w] += frame.astype(np.float32)
        
        # Accumulate into Weight (weighted by center-weighted mask to avoid seams)
        # For now, just a simple box weight
        self.weight_buffer[curr_y:curr_y+h, curr_x:curr_x+w] += 1.0

    def _finalize(self):
        # Divide Sum by Weight
        mask = self.weight_buffer > 0
        result = np.zeros_like(self.sum_buffer, dtype=np.float32)
        result[mask] = self.sum_buffer[mask] / self.weight_buffer[mask][:, np.newaxis]
        
        # Convert to uint8
        final_img = np.clip(result, 0, 255).astype(np.uint8)
        
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
            "progress": (self.current_frame / self.total_frames * 100) if self.total_frames > 0 else 0
        }

    def stop(self):
        self.is_active = False
        event_logger.log("Panorama cancelled")
