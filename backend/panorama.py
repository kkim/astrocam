import numpy as np
import cv2
import threading
import time
import os
from datetime import datetime
from logger import event_logger
from alignment_utils import align_images, transform, compose_transforms, accumulate_panorama_frame

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
        self.offset_angle = 0.0
        self.prev_frame = None
        self.T_cumulative = np.eye(2, 3, dtype=np.float32)

    def start(self, frames, drift_step=10.0, auto_align=False):
        if self.is_active: return False
        self.total_frames = frames
        self.drift_step = drift_step
        self.auto_align = auto_align
        self.current_frame = 0
        self.offset_x = 0.0
        self.offset_y = 0.0
        self.offset_angle = 0.0
        self.sum_buffer = None
        self.weight_buffer = None
        self.prev_frame = None
        self.T_cumulative = np.eye(2, 3, dtype=np.float32)
        self.is_active = True
        threading.Thread(target=self._run_panorama, daemon=True).start()
        event_logger.log(f"Panorama started: {frames} frames")
        return True

    def _run_panorama(self):
        try:
            for i in range(self.total_frames):
                if not self.is_active: break
                frame = self.rig.get_raw_frame()
                if frame is not None:
                    # Initialize buffers on the first frame
                    if self.sum_buffer is None:
                        h, w = frame.shape[:2]
                        buf_w, buf_h = w * 6, h * 3
                        self.sum_buffer = np.zeros((buf_h, buf_w, 3), dtype=np.float32)
                        self.weight_buffer = np.zeros((buf_h, buf_w), dtype=np.float32)
                        self.base_x, self.base_y = (buf_w - w) // 2, (buf_h - h) // 2

                    if self.auto_align:
                        # Auto-align: pass self.prev_frame to the helper to handle alignment, composition, and accumulation
                        self.sum_buffer, self.weight_buffer, self.T_cumulative = accumulate_panorama_frame(
                            self.sum_buffer, self.weight_buffer, (self.base_x, self.base_y),
                            self.T_cumulative, self.prev_frame, frame, translation_only=True
                        )
                        
                        # Extract current offsets for UI/logs
                        self.offset_x = float(self.T_cumulative[0, 2])
                        self.offset_y = float(self.T_cumulative[1, 2])
                        self.offset_angle = float(np.arctan2(self.T_cumulative[1, 0], self.T_cumulative[0, 0]) * 180.0 / np.pi)
                        
                        if self.prev_frame is not None and i % 5 == 0:
                            T_step = align_images(self.prev_frame, frame, translation_only=True)
                            event_logger.log(f"Align F{i}: dx={T_step[0, 2]:.1f}, dy={T_step[1, 2]:.1f}, rot={self.offset_angle:.2f}°")
                    else:
                        # Manual drift: manually compose the translation matrix first (for subsequent frames)
                        if self.prev_frame is not None:
                            T_step = np.float32([[1, 0, self.drift_step], [0, 1, 0]])
                            self.T_cumulative = compose_transforms(self.T_cumulative, T_step)
                        
                        # Accumulate without further alignment (pass img_prev=None)
                        self.sum_buffer, self.weight_buffer, self.T_cumulative = accumulate_panorama_frame(
                            self.sum_buffer, self.weight_buffer, (self.base_x, self.base_y),
                            self.T_cumulative, None, frame, translation_only=True
                        )
                        self.offset_x = float(self.T_cumulative[0, 2])
                        self.offset_y = float(self.T_cumulative[1, 2])
                        self.offset_angle = 0.0

                    self.current_frame += 1
                    self.prev_frame = frame
                time.sleep(0.5)
            self._finalize()
        except Exception as e:
            event_logger.log(f"Panorama Error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.is_active = False

    def _finalize(self):
        if self.sum_buffer is None: return
        event_logger.log(f"Finalizing {self.current_frame} frames")
        mask = self.weight_buffer > 0
        res = np.zeros_like(self.sum_buffer, dtype=np.float32)
        res[mask] = self.sum_buffer[mask] / self.weight_buffer[mask][:, np.newaxis]
        final = np.clip(res, 0, 255).astype(np.uint8)
        
        # Crop to bounding box of content
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
            "active": self.is_active, 
            "current": self.current_frame, 
            "total": self.total_frames,
            "progress": (self.current_frame / self.total_frames * 100) if self.total_frames > 0 else 0,
            "offset_x": round(self.offset_x, 1), 
            "offset_y": round(self.offset_y, 1),
            "offset_angle": round(self.offset_angle, 2)
        }

    def stop(self):
        self.is_active = False
        event_logger.log("Panorama cancelled")
