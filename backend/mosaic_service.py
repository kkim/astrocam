import os
import sys
import time
import cv2
import numpy as np
import subprocess
import threading
from typing import Dict, Any, Optional
from logger import event_logger

ACTIVE_JOBS = set()
LOCK = threading.Lock()

def get_session_mosaic_info(s_dir: str, session_id: str) -> Dict[str, Any]:
    """Checks the mosaic status of a session directory."""
    mosaic_dir = os.path.join(s_dir, "mosaic")
    mosaic_hdr_path = os.path.join(mosaic_dir, "mosaic_hdr_asinh_8bit.png")
    proc_marker = os.path.join(mosaic_dir, ".processing")
    
    with LOCK:
        is_in_active_jobs = session_id in ACTIVE_JOBS
    
    is_processing = is_in_active_jobs or os.path.exists(proc_marker)
    
    if is_processing:
        # Check if the process is orphaned (> 15 minutes old and not in active jobs)
        if not is_in_active_jobs and os.path.exists(proc_marker):
            try:
                marker_age = time.time() - os.path.getmtime(proc_marker)
                if marker_age > 900:
                    os.remove(proc_marker)
                    is_processing = False
            except Exception:
                pass

    if is_processing:
        mtime = os.path.getmtime(mosaic_hdr_path) if os.path.exists(mosaic_hdr_path) else time.time()
        return {
            "has_mosaic": False,
            "mosaic_status": "processing",
            "mosaic_url": f"/captures/{session_id}/mosaic/mosaic_hdr_asinh_8bit.png",
            "mosaic_mtime": mtime
        }
    
    if os.path.exists(mosaic_hdr_path):
        return {
            "has_mosaic": True,
            "mosaic_status": "ready",
            "mosaic_url": f"/captures/{session_id}/mosaic/mosaic_hdr_asinh_8bit.png",
            "mosaic_mtime": os.path.getmtime(mosaic_hdr_path)
        }
        
    return {
        "has_mosaic": False,
        "mosaic_status": "none",
        "mosaic_url": None,
        "mosaic_mtime": 0
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

def trigger_mosaic_stitching(session_id: str, s_dir: str) -> Dict[str, Any]:
    """Initiates mosaic creation for a session in a background subprocess."""
    output_dir = os.path.join(s_dir, "mosaic")
    proc_marker = os.path.join(output_dir, ".processing")
    mosaic_hdr_path = os.path.join(output_dir, "mosaic_hdr_asinh_8bit.png")

    with LOCK:
        if session_id in ACTIVE_JOBS or os.path.exists(proc_marker):
            return {
                "success": True,
                "status": "processing",
                "message": "Mosaic is already processing",
                "url": f"/captures/{session_id}/mosaic/mosaic_hdr_asinh_8bit.png"
            }

        # If already finished
        if os.path.exists(mosaic_hdr_path):
            return {
                "success": True,
                "status": "ready",
                "message": "Mosaic is ready",
                "url": f"/captures/{session_id}/mosaic/mosaic_hdr_asinh_8bit.png"
            }

        ACTIVE_JOBS.add(session_id)

    files = sorted([
        f for f in os.listdir(s_dir)
        if not f.startswith(".") and f != "thumb.jpg" and f.lower().endswith((".tif", ".tiff", ".jpg", ".jpeg", ".png"))
    ])
    if not files:
        with LOCK:
            ACTIVE_JOBS.discard(session_id)
        return {"success": False, "error": "No capture frames found in session"}

    first_frame = files[0]
    create_placeholder_mosaic(s_dir, output_dir, first_frame)
    
    with open(proc_marker, "w") as f:
        f.write(str(time.time()))

    script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../bin/mosaic_stars.py"))
    venv_py = os.path.abspath(os.path.join(os.path.dirname(__file__), "venv/bin/python"))
    py_exec = venv_py if os.path.exists(venv_py) else sys.executable

    def _worker():
        try:
            event_logger.log(f"Mosaic: Started star alignment on {session_id} ({len(files)} frames)...")
            cmd = [py_exec, script_path, s_dir, "-o", output_dir]
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if res.returncode == 0:
                event_logger.log(f"Mosaic: Finished stitching {session_id} successfully!")
            else:
                event_logger.log(f"Mosaic error on {session_id}: {res.stderr[:200]}")
        except Exception as e:
            event_logger.log(f"Mosaic exception on {session_id}: {e}")
        finally:
            with LOCK:
                ACTIVE_JOBS.discard(session_id)
            if os.path.exists(proc_marker):
                try:
                    os.remove(proc_marker)
                except Exception:
                    pass

    t = threading.Thread(target=_worker, daemon=True)
    t.start()

    return {
        "success": True,
        "status": "processing",
        "message": "Mosaic generation started",
        "url": f"/captures/{session_id}/mosaic/mosaic_hdr_asinh_8bit.png"
    }
