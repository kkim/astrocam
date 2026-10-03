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
    
    Supports on-the-fly frame accumulation in RAM (e.g. 0ms/raw, 200ms/5FPS, 1000ms/1FPS)
    for high-SNR stacked image recording to microSD / SSD.
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
        self.frames_count: int = 0      # Saved files count on disk
        self.total_raw_frames: int = 0  # Raw frames ingested from sensor
        self.bytes_stored: int = 0
        self.dropped_frames: int = 0
        self.start_time: float = 0.0
        self.stop_time: float = 0.0

        # Accumulation settings
        # 0 = raw (every frame), 200 = 5 FPS target, 1000 = 1 FPS target
        self.accumulation_ms: int = 0
        self.file_format: str = "tif"   # "tif" or "jpg"
        self.acc_buffer: Optional[np.ndarray] = None
        self.acc_count: int = 0
        self.acc_start_time: float = 0.0
        self.acc_first_timestamp: float = 0.0
        self.last_acc_count: int = 1

        self.frame_queue: queue.Queue = queue.Queue(maxsize=120)
        self.writer_thread: Optional[threading.Thread] = None
        self.lock = threading.Lock()

    def start_recording(self, accumulation_ms: Optional[int] = None, file_format: Optional[str] = None) -> Dict[str, Any]:
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

            if accumulation_ms is not None:
                self.accumulation_ms = int(accumulation_ms)
            if file_format is not None:
                fmt = file_format.lower().replace(".", "")
                self.file_format = fmt if fmt in ("tif", "tiff", "jpg", "jpeg") else "tif"

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.session_name = f"session_{timestamp}"
            self.session_path = os.path.join(self.base_dir, self.session_name)
            os.makedirs(self.session_path, exist_ok=True)

            self.frames_count = 0
            self.total_raw_frames = 0
            self.bytes_stored = 0
            self.dropped_frames = 0
            self.start_time = time.time()
            self.stop_time = 0.0
            self.is_recording = True

            # Reset accumulation buffer
            self.acc_buffer = None
            self.acc_count = 0
            self.acc_start_time = 0.0
            self.acc_first_timestamp = 0.0
            self.last_acc_count = 1

            # Clear queue
            while not self.frame_queue.empty():
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    break

            self.writer_thread = threading.Thread(target=self._writer_loop, daemon=True)
            self.writer_thread.start()

            ext_display = self.file_format.upper()
            acc_display = f"{self.accumulation_ms}ms" if self.accumulation_ms > 0 else "RAW"
            event_logger.log(f"Stream Recording started: saving {ext_display} to {self.session_name}/ (accumulation={acc_display})")
            return {
                "success": True,
                "is_recording": True,
                "directory": self.session_name,
                "session_path": self.session_path,
                "accumulation_ms": self.accumulation_ms,
                "format": self.file_format
            }

    def add_frame(self, frame: np.ndarray, timestamp: Optional[float] = None) -> None:
        """
        Submits a frame to the recorder. If accumulation_ms > 0, accumulates frames
        in RAM until the time window has elapsed, then submits the integrated frame.
        Fast and non-blocking for the acquisition loop.
        """
        if not self.is_recording:
            return

        capture_time = timestamp if timestamp is not None else time.time()

        with self.lock:
            self.total_raw_frames += 1

            if self.accumulation_ms <= 0:
                # Raw mode: write every frame immediately
                dt = datetime.fromtimestamp(capture_time)
                timestamp_str = dt.strftime("%Y%m%d_%H%M%S_%f")
                filename = f"{timestamp_str}.{self.file_format}"
                try:
                    self.frame_queue.put_nowait((frame.copy(), filename))
                except queue.Full:
                    self.dropped_frames += 1
                    if self.dropped_frames % 30 == 1:
                        event_logger.log(f"Recording warning: write queue full, dropped {self.dropped_frames} frames.")
            else:
                # Accumulation mode: accumulate over time window in RAM
                if self.acc_buffer is None:
                    self.acc_buffer = frame.astype(np.float32)
                    self.acc_count = 1
                    self.acc_start_time = capture_time
                    self.acc_first_timestamp = capture_time
                else:
                    self.acc_buffer += frame.astype(np.float32)
                    self.acc_count += 1

                elapsed_ms = (capture_time - self.acc_start_time) * 1000.0
                if elapsed_ms >= self.accumulation_ms:
                    self._flush_accumulated_frame_locked()

    def _flush_accumulated_frame_locked(self) -> None:
        """Flushes the current accumulated buffer into the write queue (called with lock)."""
        if self.acc_buffer is None or self.acc_count <= 0:
            return

        avg_frame = (self.acc_buffer / self.acc_count).clip(0, 255).astype(np.uint8)
        dt = datetime.fromtimestamp(self.acc_first_timestamp)
        timestamp_str = dt.strftime("%Y%m%d_%H%M%S_%f")
        # Indicate accumulated frame count in the filename
        filename = f"{timestamp_str}_acc{self.acc_count}.{self.file_format}"
        self.last_acc_count = self.acc_count

        try:
            self.frame_queue.put_nowait((avg_frame, filename))
        except queue.Full:
            self.dropped_frames += 1
            if self.dropped_frames % 30 == 1:
                event_logger.log(f"Recording warning: write queue full, dropped {self.dropped_frames} frames.")

        self.acc_buffer = None
        self.acc_count = 0

    def _writer_loop(self):
        """Background disk writer thread."""
        while self.is_recording or not self.frame_queue.empty():
            try:
                item = self.frame_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            try:
                frame, filename = item

                with self.lock:
                    curr_idx = self.frames_count + 1

                filepath = os.path.join(self.session_path, filename)
                if os.path.exists(filepath):
                    name, ext = os.path.splitext(filename)
                    filepath = os.path.join(self.session_path, f"{name}_{curr_idx:06d}{ext}")

                # Save frame based on format
                if filename.endswith(".tif") or filename.endswith(".tiff"):
                    cv2.imwrite(filepath, frame)
                else:
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

            # Flush any pending partial accumulated frame
            if self.acc_buffer is not None and self.acc_count > 0:
                self._flush_accumulated_frame_locked()

            self.is_recording = False
            self.stop_time = time.time()

        if self.writer_thread and self.writer_thread.is_alive():
            self.writer_thread.join(timeout=3.0)

        mb = round(self.bytes_stored / (1024 * 1024), 2)
        duration = round(self.stop_time - self.start_time, 1)
        acc_info = f" (avg ~{self.last_acc_count} frames/stack)" if self.accumulation_ms > 0 else ""
        event_logger.log(f"Stream Recording stopped: {self.frames_count} files saved{acc_info}, {mb} MB in {self.session_name}/")

        return {
            "success": True,
            "is_recording": False,
            "directory": self.session_name,
            "frames": self.frames_count,
            "raw_frames": self.total_raw_frames,
            "accumulated_frames": self.last_acc_count,
            "accumulation_ms": self.accumulation_ms,
            "format": self.file_format,
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
                "raw_frames": self.total_raw_frames,
                "accumulated_frames": self.last_acc_count,
                "accumulation_ms": self.accumulation_ms,
                "format": self.file_format,
                "bytes_stored": self.bytes_stored,
                "mb_stored": mb,
                "directory": self.session_name,
                "duration_sec": round(duration, 1),
                "fps": fps,
                "dropped_frames": self.dropped_frames
            }
