import os
import time
import queue
import threading
from datetime import datetime
from typing import Optional, Dict, Any
import cv2
import numpy as np
from logger import event_logger

class StreamRecorder:
    """
    Asynchronous stream recorder that continuously saves frames into a dedicated
    session directory without blocking the main live view or tracking pipeline.
    """
    def __init__(self, base_dir: Optional[str] = None):
        if base_dir is None:
            self.base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../captures"))
        else:
            self.base_dir = os.path.abspath(base_dir)

        os.makedirs(self.base_dir, exist_ok=True)

        self.is_recording: bool = False
        self.session_name: str = ""
        self.session_path: str = ""
        self.frames_count: int = 0
        self.bytes_stored: int = 0
        self.dropped_frames: int = 0
        self.start_time: float = 0.0
        self.stop_time: float = 0.0

        self.frame_queue: queue.Queue = queue.Queue(maxsize=120)
        self.writer_thread: Optional[threading.Thread] = None
        self.lock = threading.Lock()

    def start_recording(self) -> Dict[str, Any]:
        """
        Starts a new recording session, creating a timestamped folder inside captures/.
        """
        with self.lock:
            if self.is_recording:
                return {
                    "success": False,
                    "error": "Recording is already active",
                    "directory": self.session_name,
                    "frames": self.frames_count
                }

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.session_name = f"session_{timestamp}"
            self.session_path = os.path.join(self.base_dir, self.session_name)
            os.makedirs(self.session_path, exist_ok=True)

            self.frames_count = 0
            self.bytes_stored = 0
            self.dropped_frames = 0
            self.start_time = time.time()
            self.stop_time = 0.0
            self.is_recording = True

            # Clear queue
            while not self.frame_queue.empty():
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    break

            self.writer_thread = threading.Thread(target=self._writer_loop, daemon=True)
            self.writer_thread.start()

            event_logger.log(f"Stream Recording started: saving to {self.session_name}/")
            return {
                "success": True,
                "is_recording": True,
                "directory": self.session_name,
                "session_path": self.session_path
            }

    def add_frame(self, frame: np.ndarray, timestamp: Optional[float] = None) -> None:
        """
        Submits a frame with its capture timestamp to the asynchronous write queue.
        Fast and non-blocking for the acquisition loop.
        """
        if not self.is_recording:
            return

        capture_time = timestamp if timestamp is not None else time.time()
        try:
            self.frame_queue.put_nowait((frame.copy(), capture_time))
        except queue.Full:
            with self.lock:
                self.dropped_frames += 1
                if self.dropped_frames % 30 == 1:
                    event_logger.log(f"Recording warning: write queue full, dropped {self.dropped_frames} frames.")

    def _writer_loop(self):
        """Background disk writer thread."""
        while self.is_recording or not self.frame_queue.empty():
            try:
                item = self.frame_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            try:
                if isinstance(item, tuple):
                    frame, capture_time = item
                else:
                    frame, capture_time = item, time.time()

                with self.lock:
                    curr_idx = self.frames_count + 1

                # Use timestamp for image filename instead of sequential frame number
                dt = datetime.fromtimestamp(capture_time)
                timestamp_str = dt.strftime("%Y%m%d_%H%M%S_%f")
                filename = f"{timestamp_str}.jpg"
                filepath = os.path.join(self.session_path, filename)
                if os.path.exists(filepath):
                    filename = f"{timestamp_str}_{curr_idx:06d}.jpg"
                    filepath = os.path.join(self.session_path, filename)

                # Save frame at high JPEG quality
                cv2.imwrite(filepath, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                size = os.path.getsize(filepath)

                with self.lock:
                    self.frames_count = curr_idx
                    self.bytes_stored += size

                self.frame_queue.task_done()
            except Exception as e:
                event_logger.log(f"Recording write error: {e}")

    def stop_recording(self) -> Dict[str, Any]:
        """
        Stops the recording session and flushes pending frames to disk.
        """
        with self.lock:
            if not self.is_recording:
                return {
                    "success": False,
                    "error": "No recording is currently active",
                    "frames": self.frames_count,
                    "mb_stored": round(self.bytes_stored / (1024 * 1024), 2)
                }
            self.is_recording = False
            self.stop_time = time.time()

        if self.writer_thread and self.writer_thread.is_alive():
            self.writer_thread.join(timeout=3.0)

        mb = round(self.bytes_stored / (1024 * 1024), 2)
        duration = round(self.stop_time - self.start_time, 1)
        event_logger.log(f"Stream Recording stopped: {self.frames_count} frames, {mb} MB in {self.session_name}/")

        return {
            "success": True,
            "is_recording": False,
            "directory": self.session_name,
            "frames": self.frames_count,
            "bytes_stored": self.bytes_stored,
            "mb_stored": mb,
            "duration_sec": duration,
            "dropped_frames": self.dropped_frames
        }

    def get_status(self) -> Dict[str, Any]:
        """Returns live metrics of the current/last recording session."""
        with self.lock:
            now = time.time()
            duration = (now - self.start_time) if self.is_recording else (self.stop_time - self.start_time if self.stop_time > 0 else 0.0)
            mb = round(self.bytes_stored / (1024 * 1024), 2)
            fps = round(self.frames_count / max(0.1, duration), 1) if duration > 0 else 0.0

            return {
                "is_recording": self.is_recording,
                "frames": self.frames_count,
                "bytes_stored": self.bytes_stored,
                "mb_stored": mb,
                "directory": self.session_name,
                "duration_sec": round(duration, 1),
                "fps": fps,
                "dropped_frames": self.dropped_frames
            }
