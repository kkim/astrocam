import os
import sys
import time
import json
import queue
import cv2
import numpy as np
import subprocess
import threading
from typing import Dict, Any, Optional, List, Tuple
from logger import event_logger

MOSAIC_SUBDIR = "mosaic_v1"
ACTIVE_SESSION: Optional[str] = None
QUEUED_SESSIONS: List[Tuple[str, str]] = []
JOB_QUEUE: queue.Queue = queue.Queue()
WORKER_THREAD: Optional[threading.Thread] = None
LOCK = threading.Lock()

def get_session_mosaic_info(s_dir: str, session_id: str) -> Dict[str, Any]:
    """Checks the mosaic status and live progress of a session directory."""
    mosaic_dir = os.path.join(s_dir, MOSAIC_SUBDIR)
    mosaic_hdr_path = os.path.join(mosaic_dir, "mosaic_hdr_asinh_8bit.png")
    real_mosaic_marker = os.path.join(mosaic_dir, "mosaic_linear_16bit.tif")
    proc_marker = os.path.join(mosaic_dir, ".processing")
    progress_file = os.path.join(mosaic_dir, ".progress.json")

    with LOCK:
        is_active = (session_id == ACTIVE_SESSION)
        queue_pos = 0
        for idx, (qid, _) in enumerate(QUEUED_SESSIONS):
            if qid == session_id:
                queue_pos = idx + 1
                break

    # A completed mosaic must have both the HDR render and the linear 16-bit output
    if os.path.exists(mosaic_hdr_path) and os.path.exists(real_mosaic_marker):
        return {
            "has_mosaic": True,
            "mosaic_status": "ready",
            "mosaic_url": f"/captures/{session_id}/{MOSAIC_SUBDIR}/mosaic_hdr_asinh_8bit.png",
            "mosaic_mtime": os.path.getmtime(mosaic_hdr_path),
            "mosaic_progress": {"percent": 100, "message": "Mosaic ready", "stage": "ready"},
            "mosaic_queue_pos": 0,
        }

    # If currently queued waiting for the active job to finish
    if queue_pos > 0:
        mtime = os.path.getmtime(mosaic_hdr_path) if os.path.exists(mosaic_hdr_path) else time.time()
        return {
            "has_mosaic": False,
            "mosaic_status": "queued",
            "mosaic_url": f"/captures/{session_id}/{MOSAIC_SUBDIR}/mosaic_hdr_asinh_8bit.png",
            "mosaic_mtime": mtime,
            "mosaic_progress": {
                "percent": 0,
                "message": f"Queued (#{queue_pos} in line)",
                "stage": "queued"
            },
            "mosaic_queue_pos": queue_pos,
        }

    # Check if the process is orphaned (> 15 minutes old and not in active jobs)
    is_processing = is_active or os.path.exists(proc_marker)
    if is_processing and not is_active and os.path.exists(proc_marker):
        try:
            marker_age = time.time() - os.path.getmtime(proc_marker)
            if marker_age > 900:
                os.remove(proc_marker)
                if os.path.exists(progress_file):
                    os.remove(progress_file)
                is_processing = False
        except Exception:
            pass

    if is_processing:
        mtime = os.path.getmtime(mosaic_hdr_path) if os.path.exists(mosaic_hdr_path) else time.time()
        progress_data = {"percent": 0.0, "message": "Starting alignment...", "stage": "starting"}
        if os.path.exists(progress_file):
            try:
                with open(progress_file, "r") as pf:
                    progress_data = json.load(pf)
            except Exception:
                pass

        return {
            "has_mosaic": False,
            "mosaic_status": "processing",
            "mosaic_url": f"/captures/{session_id}/{MOSAIC_SUBDIR}/mosaic_hdr_asinh_8bit.png",
            "mosaic_mtime": mtime,
            "mosaic_progress": progress_data,
            "mosaic_queue_pos": 0,
        }

    # If only an old placeholder exists without completed mosaic outputs, clean it up
    if os.path.exists(mosaic_hdr_path) and not os.path.exists(real_mosaic_marker):
        try:
            os.remove(mosaic_hdr_path)
            if os.path.exists(progress_file):
                os.remove(progress_file)
        except Exception:
            pass

    return {
        "has_mosaic": False,
        "mosaic_status": "none",
        "mosaic_url": None,
        "mosaic_mtime": 0,
        "mosaic_progress": None,
        "mosaic_queue_pos": 0,
    }

def create_placeholder_mosaic(s_dir: str, output_dir: str, first_frame: str) -> str:
    """Creates a placeholder image based on the first frame with an imprinted HUD status."""
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "mosaic_hdr_asinh_8bit.png")
    in_path = os.path.join(s_dir, first_frame)
    img = None
    if os.path.exists(in_path):
        try:
            img = cv2.imread(in_path, cv2.IMREAD_UNCHANGED)
        except Exception:
            pass

    if img is None:
        img = np.zeros((1080, 1920, 3), dtype=np.uint8)
    else:
        if img.dtype == np.uint16:
            img = np.clip(img / 257.0, 0, 255).astype(np.uint8)
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    h, w = img.shape[:2]
    overlay = img.copy()
    bw = int(w * 0.70)
    bh = int(h * 0.22)
    bx1 = (w - bw) // 2
    by1 = (h - bh) // 2
    bx2 = bx1 + bw
    by2 = by1 + bh

    cv2.rectangle(overlay, (bx1, by1), (bx2, by2), (13, 17, 23), -1)
    cv2.rectangle(overlay, (bx1, by1), (bx2, by2), (88, 166, 255), 3)
    alpha = 0.85
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

    text1 = "Mosaic Being Processed..."
    text2 = "Aligning star centroids & stacking full FOV"

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale1 = max(0.9, w / 1600.0)
    scale2 = scale1 * 0.65

    t1_size = cv2.getTextSize(text1, font, scale1, 2)[0]
    t1_x = (w - t1_size[0]) // 2
    t1_y = by1 + int(bh * 0.45)
    cv2.putText(img, text1, (t1_x, t1_y), font, scale1, (255, 255, 255), 2, cv2.LINE_AA)

    t2_size = cv2.getTextSize(text2, font, scale2, 1)[0]
    t2_x = (w - t2_size[0]) // 2
    t2_y = by1 + int(bh * 0.78)
    cv2.putText(img, text2, (t2_x, t2_y), font, scale2, (180, 210, 255), 1, cv2.LINE_AA)

    cv2.imwrite(out_path, img)
    return out_path

def _queue_worker_loop():
    """Background worker daemon thread that processes mosaic jobs one-by-one safely."""
    global ACTIVE_SESSION
    while True:
        try:
            session_id, s_dir = JOB_QUEUE.get(timeout=2.0)
        except queue.Empty:
            with LOCK:
                ACTIVE_SESSION = None
                if not QUEUED_SESSIONS and JOB_QUEUE.empty():
                    break
            continue

        with LOCK:
            ACTIVE_SESSION = session_id
            QUEUED_SESSIONS[:] = [(qid, qdir) for qid, qdir in QUEUED_SESSIONS if qid != session_id]

        output_dir = os.path.join(s_dir, MOSAIC_SUBDIR)
        proc_marker = os.path.join(output_dir, ".processing")
        progress_file = os.path.join(output_dir, ".progress.json")
        mosaic_hdr_path = os.path.join(output_dir, "mosaic_hdr_asinh_8bit.png")
        real_mosaic_marker = os.path.join(output_dir, "mosaic_linear_16bit.tif")

        try:
            if os.path.exists(real_mosaic_marker):
                event_logger.log(f"Mosaic: {session_id} already completed in {MOSAIC_SUBDIR}, skipping.")
                continue

            with open(proc_marker, "w") as f:
                f.write(str(time.time()))

            files = sorted([
                f for f in os.listdir(s_dir)
                if not f.startswith(".") and f != "thumb.jpg" and f.lower().endswith((".tif", ".tiff", ".jpg", ".jpeg", ".png"))
            ])

            script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../bin/mosaic_stars.py"))
            venv_py = os.path.abspath(os.path.join(os.path.dirname(__file__), "venv/bin/python"))
            py_exec = venv_py if os.path.exists(venv_py) else sys.executable

            event_logger.log(f"Mosaic: Started star alignment on {session_id} ({len(files)} frames)...")
            cmd = [py_exec, script_path, s_dir, "-o", output_dir]
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

            if res.returncode == 0 and os.path.exists(real_mosaic_marker):
                event_logger.log(f"Mosaic: Finished stitching {session_id} successfully!")
            else:
                err_snippet = res.stderr.strip()[-300:] if res.stderr else "Unknown error"
                event_logger.log(f"Mosaic error on {session_id}: {err_snippet}")
                if not os.path.exists(real_mosaic_marker):
                    try:
                        if os.path.exists(mosaic_hdr_path):
                            os.remove(mosaic_hdr_path)
                    except Exception:
                        pass
        except Exception as e:
            event_logger.log(f"Mosaic exception on {session_id}: {e}")
            if not os.path.exists(real_mosaic_marker):
                try:
                    if os.path.exists(mosaic_hdr_path):
                        os.remove(mosaic_hdr_path)
                except Exception:
                    pass
        finally:
            if os.path.exists(proc_marker):
                try:
                    os.remove(proc_marker)
                except Exception:
                    pass
            if os.path.exists(progress_file):
                try:
                    os.remove(progress_file)
                except Exception:
                    pass
            JOB_QUEUE.task_done()
            with LOCK:
                if ACTIVE_SESSION == session_id:
                    ACTIVE_SESSION = None

def _ensure_worker_started_locked():
    global WORKER_THREAD
    if WORKER_THREAD is None or not WORKER_THREAD.is_alive():
        WORKER_THREAD = threading.Thread(target=_queue_worker_loop, daemon=True)
        WORKER_THREAD.start()

def trigger_mosaic_stitching(session_id: str, s_dir: str) -> Dict[str, Any]:
    """
    Submits a mosaic creation job to the single-worker background queue.
    Guarantees only one mosaic runs at a time to protect CPU and RAM.
    """
    output_dir = os.path.join(s_dir, MOSAIC_SUBDIR)
    proc_marker = os.path.join(output_dir, ".processing")
    mosaic_hdr_path = os.path.join(output_dir, "mosaic_hdr_asinh_8bit.png")
    real_mosaic_marker = os.path.join(output_dir, "mosaic_linear_16bit.tif")

    with LOCK:
        # If already genuinely finished
        if os.path.exists(mosaic_hdr_path) and os.path.exists(real_mosaic_marker):
            return {
                "success": True,
                "status": "ready",
                "message": "Mosaic is ready",
                "queue_pos": 0,
                "url": f"/captures/{session_id}/{MOSAIC_SUBDIR}/mosaic_hdr_asinh_8bit.png"
            }

        # If currently actively running
        if session_id == ACTIVE_SESSION or os.path.exists(proc_marker):
            return {
                "success": True,
                "status": "processing",
                "message": "Mosaic is currently processing",
                "queue_pos": 0,
                "url": f"/captures/{session_id}/{MOSAIC_SUBDIR}/mosaic_hdr_asinh_8bit.png"
            }

        # If already queued in line
        for idx, (qid, _) in enumerate(QUEUED_SESSIONS):
            if qid == session_id:
                queue_pos = idx + 1
                return {
                    "success": True,
                    "status": "queued",
                    "message": f"Mosaic is queued (#{queue_pos} in line)",
                    "queue_pos": queue_pos,
                    "url": f"/captures/{session_id}/{MOSAIC_SUBDIR}/mosaic_hdr_asinh_8bit.png"
                }

        files = sorted([
            f for f in os.listdir(s_dir)
            if not f.startswith(".") and f != "thumb.jpg" and f.lower().endswith((".tif", ".tiff", ".jpg", ".jpeg", ".png"))
        ])
        if not files:
            return {"success": False, "error": "No capture frames found in session"}

        # Create placeholder immediately so user can preview right away
        first_frame = files[0]
        create_placeholder_mosaic(s_dir, output_dir, first_frame)

        # Enqueue job into FIFO queue
        is_worker_idle = (ACTIVE_SESSION is None and not QUEUED_SESSIONS)
        QUEUED_SESSIONS.append((session_id, s_dir))
        JOB_QUEUE.put((session_id, s_dir))
        _ensure_worker_started_locked()

        if is_worker_idle:
            status_str = "processing"
            pos = 0
            msg = "Mosaic generation started"
        else:
            status_str = "queued"
            pos = len(QUEUED_SESSIONS)
            msg = f"Mosaic added to queue (#{pos} in line)"

        event_logger.log(f"Mosaic: {session_id} {msg}")

    return {
        "success": True,
        "status": status_str,
        "message": msg,
        "queue_pos": pos,
        "url": f"/captures/{session_id}/{MOSAIC_SUBDIR}/mosaic_hdr_asinh_8bit.png"
    }
