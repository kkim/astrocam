from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import uvicorn
import cv2
import numpy as np
import os
import asyncio
from datetime import datetime
from astro_rig import get_astro_rig
from logger import event_logger
from panorama import PanoramaManager
from pipeline import AstroPipeline
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional
import glob
import json
from mosaic_service import get_session_mosaic_info, trigger_mosaic_stitching

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure captures directory exists
CAPTURES_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../captures"))
os.makedirs(CAPTURES_DIR, exist_ok=True)

# Mount captures for viewing
app.mount("/captures", StaticFiles(directory=CAPTURES_DIR), name="captures")

@app.get("/gallery")
def list_captures():
    base_dir = CAPTURES_DIR
    if not os.path.exists(base_dir):
        return []

    items = []
    # 1. Sequence recording sessions
    session_dirs = glob.glob(os.path.join(base_dir, "session_*"))
    for s_dir in session_dirs:
        if not os.path.isdir(s_dir):
            continue
        s_name = os.path.basename(s_dir)
        files = sorted([
            f for f in os.listdir(s_dir)
            if not f.startswith(".") and f != "thumb.jpg" and f.lower().endswith((".tif", ".tiff", ".jpg", ".jpeg", ".png"))
        ])
        if not files:
            continue

        first_frame = files[0]
        fmt = first_frame.rsplit(".", 1)[-1].lower()
        thumb_dir = os.path.join(base_dir, "thumbnails")
        os.makedirs(thumb_dir, exist_ok=True)
        thumb_path = os.path.join(thumb_dir, f"{s_name}.jpg")

        # Auto-generate Full HD preview/thumbnail if missing
        if not os.path.exists(thumb_path):
            try:
                img = cv2.imread(os.path.join(s_dir, first_frame), cv2.IMREAD_UNCHANGED)
                if img is not None:
                    if img.dtype == np.uint16:
                        img = np.clip(img / 257.0, 0, 255).astype(np.uint8)
                    cv2.imwrite(thumb_path, img, [cv2.IMWRITE_JPEG_QUALITY, 99])
            except Exception as e:
                event_logger.log(f"Error creating thumbnail for {s_name}: {e}")

        # Parse date and time
        parts = s_name.replace("session_", "").split("_")
        if len(parts) >= 2 and len(parts[0]) == 8 and len(parts[1]) >= 6:
            ymd, hms = parts[0], parts[1]
            date_str = f"{ymd[:4]}-{ymd[4:6]}-{ymd[6:8]}"
            time_str = f"{hms[:2]}:{hms[2:4]}:{hms[4:6]}"
            dt_str = f"{date_str} {time_str}"
        else:
            dt = datetime.fromtimestamp(os.path.getmtime(s_dir))
            date_str = dt.strftime("%Y-%m-%d")
            time_str = dt.strftime("%H:%M:%S")
            dt_str = f"{date_str} {time_str}"

        # Mosaic availability info
        mosaic_info = get_session_mosaic_info(s_dir, s_name)

        thumb_exists = os.path.exists(thumb_path)
        thumb_url = f"/captures/thumbnails/{s_name}.jpg" if thumb_exists else f"/captures/{s_name}/{first_frame}"
        items.append({
            "id": s_name,
            "type": "sequence",
            "name": s_name,
            "time": time_str,
            "date": date_str,
            "datetime": dt_str,
            "frames": len(files),
            "first_frame": first_frame,
            "thumbnail": thumb_url,
            "url": mosaic_info["mosaic_url"] if mosaic_info["mosaic_url"] else (thumb_url if fmt in ("tif", "tiff") else f"/captures/{s_name}/{first_frame}"),
            "first_frame_url": f"/captures/{s_name}/{first_frame}",
            "format": fmt,
            "has_mosaic": mosaic_info["has_mosaic"],
            "mosaic_status": mosaic_info["mosaic_status"],
            "mosaic_url": mosaic_info["mosaic_url"],
            "mosaic_mtime": mosaic_info["mosaic_mtime"],
            "mtime": os.path.getmtime(s_dir)
        })

    # 2. Standalone captures (e.g. panorama_*.jpg, seq_*.jpg, etc.)
    for filepath in glob.glob(os.path.join(base_dir, "*.*")):
        filename = os.path.basename(filepath)
        if filename.startswith(".") or filename.endswith(".tmp"):
            continue
        if not filename.lower().endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff")):
            continue
        mtime = os.path.getmtime(filepath)
        dt = datetime.fromtimestamp(mtime)
        fmt = filename.rsplit(".", 1)[-1].lower()
        items.append({
            "id": filename,
            "type": "image",
            "name": filename,
            "time": dt.strftime("%H:%M:%S"),
            "date": dt.strftime("%Y-%m-%d"),
            "datetime": dt.strftime("%Y-%m-%d %H:%M:%S"),
            "frames": 1,
            "first_frame": filename,
            "thumbnail": f"/captures/{filename}",
            "url": f"/captures/{filename}",
            "first_frame_url": f"/captures/{filename}",
            "format": fmt,
            "has_mosaic": False,
            "mosaic_status": "none",
            "mosaic_url": None,
            "mosaic_mtime": 0,
            "mtime": mtime
        })

    # Sort newest first and return latest 30 captures
    items.sort(key=lambda x: x["mtime"], reverse=True)
    return items[:30]

@app.post("/mosaic/generate/{session_id}")
def generate_mosaic(session_id: str):
    s_dir = os.path.join(CAPTURES_DIR, session_id)
    if not os.path.isdir(s_dir):
        return {"success": False, "error": f"Session {session_id} not found"}
    return trigger_mosaic_stitching(session_id, s_dir)

@app.get("/mosaic/status/{session_id}")
def get_mosaic_status(session_id: str):
    s_dir = os.path.join(CAPTURES_DIR, session_id)
    if not os.path.isdir(s_dir):
        return {"status": "none"}
    return get_session_mosaic_info(s_dir, session_id)

# Configuration persistence
CONFIG_PATH = "config.json"

def load_config():
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r") as f:
                data = json.load(f)
                if "camera_mode" not in data:
                    data["camera_mode"] = data.get("rig_mode", "mock")
                if "mount_mode" not in data:
                    data["mount_mode"] = data.get("rig_mode", "mock")
                return data
        except Exception as e:
            event_logger.log(f"Error loading config: {e}")
    return {"camera_mode": "mock", "mount_mode": "mock", "rig_mode": "mock"}

def save_config(config_data):
    try:
        current = load_config()
        current.update(config_data)
        with open(CONFIG_PATH, "w") as f:
            json.dump(current, f)
    except Exception as e:
        event_logger.log(f"Error saving config: {e}")

config = load_config()

# Unified AstroRig & Processing Pipeline with decoupled camera and mount
camera_mode = config.get("camera_mode", "mock")
mount_mode = config.get("mount_mode", "mock")
rig = get_astro_rig(camera_mode=camera_mode, mount_mode=mount_mode, motor_pin=18)
pipeline = AstroPipeline(rig)
panorama = PanoramaManager(rig)

@app.on_event("shutdown")
def shutdown_event():
    event_logger.log("System shutting down...")
    if rig: rig.close()
    if pipeline: pipeline.close()

class ControlUpdate(BaseModel):
    property: str
    value: float

class MotorSpeedUpdate(BaseModel):
    speed: float

class RigUpdate(BaseModel):
    mode: Optional[str] = None
    camera_mode: Optional[str] = None
    mount_mode: Optional[str] = None

@app.get("/rig")
def get_rig_mode():
    return {
        "mode": f"{camera_mode}_{mount_mode}",
        "camera_mode": camera_mode,
        "mount_mode": mount_mode,
        "rig_mode": camera_mode if camera_mode == mount_mode else "custom"
    }

@app.post("/rig")
def set_rig_mode(update: RigUpdate):
    global rig, camera_mode, mount_mode, pipeline, panorama
    new_camera_mode = update.camera_mode or (update.mode if update.mode else camera_mode)
    new_mount_mode = update.mount_mode or (update.mode if update.mode else mount_mode)

    if new_camera_mode == camera_mode and new_mount_mode == mount_mode:
        return {"success": True, "camera_mode": camera_mode, "mount_mode": mount_mode}

    event_logger.log(f"Switching rig mode: Camera={new_camera_mode.upper()}, Mount={new_mount_mode.upper()}")
    if rig:
        rig.close()

    camera_mode = new_camera_mode
    mount_mode = new_mount_mode
    save_config({
        "camera_mode": camera_mode,
        "mount_mode": mount_mode,
        "rig_mode": camera_mode if camera_mode == mount_mode else "custom"
    })

    rig = get_astro_rig(camera_mode=camera_mode, mount_mode=mount_mode, motor_pin=18)
    pipeline.rig = rig
    panorama.rig = rig
    return {"success": True, "camera_mode": camera_mode, "mount_mode": mount_mode}

@app.post("/camera/reconnect")
def reconnect_camera():
    global rig
    if not rig:
        return {"success": False, "error": "Rig not initialized"}
    success = False
    if hasattr(rig, "reconnect_camera"):
        success = rig.reconnect_camera()
    elif hasattr(rig, "reconnect"):
        success = rig.reconnect()
    status = rig.get_camera_status()
    return {"success": success, "camera_status": status}

@app.get("/logs")
def get_logs():
    return event_logger.get_logs()

@app.get("/motor/status")
def get_motor_status():
    return rig.get_motor_status()

@app.post("/motor/speed")
def set_motor_speed(update: MotorSpeedUpdate):
    if pipeline and pipeline.auto_tracking:
        pipeline.set_auto_tracking(False)
    success = rig.set_motor_speed(update.speed)
    if success:
        event_logger.log(f"Motor speed: {update.speed}%")
    return {"success": success, "speed": update.speed}

class CameraAngleUpdate(BaseModel):
    angle: float

@app.post("/mock/camera_angle")
def set_mock_camera_angle(update: CameraAngleUpdate):
    if not rig: return {"success": False, "error": "Rig not initialized"}
    if hasattr(rig, "set_camera_angle"):
        success = rig.set_camera_angle(update.angle)
        return {"success": success, "angle": update.angle}
    return {"success": False, "error": "Rig does not support setting camera angle"}

class SimDriftUpdate(BaseModel):
    speed: float
    angle: float

@app.post("/mock/sim_drift")
def set_mock_sim_drift(update: SimDriftUpdate):
    if not rig: return {"success": False, "error": "Rig not initialized"}
    if hasattr(rig, "set_sim_drift"):
        success = rig.set_sim_drift(update.speed, update.angle)
        return {"success": success, "speed": update.speed, "angle": update.angle}
    return {"success": False, "error": "Rig does not support setting sim drift"}

@app.get("/stream")
async def stream():
    if not pipeline:
        return Response(content="Pipeline not available", status_code=503)
        
    async def frame_generator():
        while pipeline and pipeline.is_running:
            try:
                frame = pipeline.get_frame()
                if frame:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                await asyncio.sleep(0.03) # ~33fps
            except Exception as e:
                print(f"Stream error: {e}")
                break

    return StreamingResponse(frame_generator(), media_type="multipart/x-mixed-replace; boundary=frame")

class ResolutionUpdate(BaseModel):
    width: int
    height: int

class SequenceUpdate(BaseModel):
    count: int
    interval: float

@app.post("/sequence")
def start_sequence(req: SequenceUpdate):
    if not pipeline: return {"success": False}
    success = pipeline.start_sequence(req.count, req.interval)
    return {"success": success}

@app.get("/sequence/status")
def get_sequence_status():
    if not pipeline: return {"active": False}
    return pipeline.get_sequence_status()

@app.post("/resolution")
def set_resolution(res: ResolutionUpdate):
    return {"success": False, "error": "Not implemented"}

@app.get("/status")
def get_status():
    if not rig or not pipeline:
        return {"connected": False, "mean_brightness": 0, "error": "Rig or pipeline not initialized"}
    status = rig.get_camera_status()
    status["fps"] = pipeline.fps
    status["mean_brightness"] = pipeline.mean_brightness
    motor_status = rig.get_motor_status()
    status["mock_mode"] = motor_status.get("mock_mode", True)
    status["camera_mode"] = camera_mode
    status["mount_mode"] = mount_mode
    status["recording"] = pipeline.get_recording_status()
    return status

@app.get("/controls")
def get_controls():
    if not rig or not pipeline: return {}
    return {**rig.get_camera_params(), "average": pipeline.n_avg}

@app.post("/controls")
def set_control(update: ControlUpdate):
    if not rig or not pipeline: return {"success": False}
    if update.property == "average":
        success = pipeline.set_n_avg(update.value)
    else:
        success = rig.set_camera_param(update.property, update.value)
    if success:
        event_logger.log(f"Control: {update.property} -> {update.value}")
    return {"success": success, "property": update.property, "value": update.value}

class RecordingStartRequest(BaseModel):
    accumulation_ms: Optional[int] = None
    format: Optional[str] = None

@app.post("/recording/start")
def start_recording(req: Optional[RecordingStartRequest] = None):
    if not pipeline: return {"success": False, "error": "Pipeline not initialized"}
    acc_ms = req.accumulation_ms if (req and req.accumulation_ms is not None) else None
    fmt = req.format if (req and req.format is not None) else None
    return pipeline.start_recording(accumulation_ms=acc_ms, file_format=fmt)

@app.post("/recording/stop")
def stop_recording():
    if not pipeline: return {"success": False, "error": "Pipeline not initialized"}
    return pipeline.stop_recording()

@app.get("/recording/status")
def get_recording_status():
    if not pipeline: return {"is_recording": False, "frames": 0, "mb_stored": 0.0}
    return pipeline.get_recording_status()

@app.post("/capture")
def capture():
    if not pipeline: return {"success": False, "error": "Pipeline not initialized"}
    return pipeline.capture_frame()

class TrackingUpdate(BaseModel):
    enable: bool

@app.post("/tracking/toggle")
def set_tracking(req: TrackingUpdate):
    if not pipeline: return {"success": False, "error": "Pipeline not initialized"}
    pipeline.set_auto_tracking(req.enable)
    return {"success": True, "active": req.enable}

@app.get("/tracking/status")
def get_tracking_status():
    if not pipeline: return {"active": False, "status": "inactive"}
    return pipeline.get_tracking_status()

class PanoramaStartRequest(BaseModel):
    frames: int
    drift_step: float = 15.0
    auto_align: bool = True

@app.post("/panorama/start")
def start_panorama(req: PanoramaStartRequest):
    if not panorama:
        return {"success": False, "error": "Panorama manager not initialized"}
    success = panorama.start(frames=req.frames, drift_step=req.drift_step, auto_align=req.auto_align)
    return {"success": success}

@app.post("/panorama/stop")
def stop_panorama():
    if not panorama:
        return {"success": False, "error": "Panorama manager not initialized"}
    panorama.stop()
    return {"success": True}

@app.get("/panorama/status")
def get_panorama_status():
    if not panorama:
        return {"active": False}
    return panorama.get_status()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
