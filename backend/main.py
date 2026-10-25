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
from typing import Optional, List
import glob
import json
import subprocess
import threading
import time
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

        # Nav info (target / coordinates)
        target_name = ""
        camera_settings = {}
        session_nav_file = os.path.join(s_dir, "nav.json")
        if os.path.exists(session_nav_file):
            try:
                with open(session_nav_file, "r") as f:
                    s_nav = json.load(f)
                    target_name = s_nav.get("target", "")
                    camera_settings = s_nav.get("camera_settings", {})
            except Exception:
                pass

        thumb_exists = os.path.exists(thumb_path)
        thumb_url = f"/captures/thumbnails/{s_name}.jpg" if thumb_exists else f"/captures/{s_name}/{first_frame}"
        items.append({
            "id": s_name,
            "type": "sequence",
            "name": s_name,
            "target": target_name,
            "camera_settings": camera_settings,
            "time": time_str,
            "date": date_str,
            "datetime": dt_str,
            "frames": len(files),
            "first_frame": first_frame,
            "thumbnail": thumb_url,
            "url": mosaic_info["mosaic_url"] if mosaic_info["has_mosaic"] else (thumb_url if fmt in ("tif", "tiff") else f"/captures/{s_name}/{first_frame}"),
            "first_frame_url": f"/captures/{s_name}/{first_frame}",
            "format": fmt,
            "has_mosaic": mosaic_info["has_mosaic"],
            "mosaic_status": mosaic_info["mosaic_status"],
            "mosaic_url": mosaic_info["mosaic_url"] if mosaic_info["has_mosaic"] else None,
            "mosaic_mtime": mosaic_info["mosaic_mtime"],
            "mosaic_progress": mosaic_info.get("mosaic_progress"),
            "mosaic_queue_pos": mosaic_info.get("mosaic_queue_pos", 0),
            "mtime": os.path.getmtime(s_dir)
        })

    # 2. Standalone captures (e.g. panorama_*.jpg, seq_*.jpg, etc.)
    for filepath in glob.glob(os.path.join(base_dir, "*.*")):
        filename = os.path.basename(filepath)
        if filename.startswith(".") or filename.endswith(".tmp") or filename.endswith(".json"):
            continue
        if not filename.lower().endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff")):
            continue
        mtime = os.path.getmtime(filepath)
        dt = datetime.fromtimestamp(mtime)
        fmt = filename.rsplit(".", 1)[-1].lower()

        target_name = ""
        camera_settings = {}
        sidecar_path = filepath + ".nav.json"
        if os.path.exists(sidecar_path):
            try:
                with open(sidecar_path, "r") as f:
                    c_nav = json.load(f)
                    target_name = c_nav.get("target", "")
                    camera_settings = c_nav.get("camera_settings", {})
            except Exception:
                pass

        items.append({
            "id": filename,
            "type": "image",
            "name": filename,
            "target": target_name,
            "camera_settings": camera_settings,
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
            "mosaic_progress": None,
            "mosaic_queue_pos": 0,
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
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.json")

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
    return {"camera_mode": "mock", "mount_mode": "mock", "rig_mode": "mock", "camera_format": "YUYV"}

def save_config(config_data):
    try:
        current = load_config()
        current.update(config_data)
        with open(CONFIG_PATH, "w") as f:
            json.dump(current, f, indent=2)
    except Exception as e:
        event_logger.log(f"Error saving config: {e}")

def save_camera_settings():
    try:
        settings = {}
        if rig and hasattr(rig, "get_camera_params"):
            settings.update(rig.get_camera_params())
        if pipeline:
            settings["accumulation_count"] = pipeline.accumulation_count
            settings["average"] = pipeline.accumulation_count
        
        proc_config = pipeline.get_processing_config() if pipeline and hasattr(pipeline, "get_processing_config") else {}
        save_config({
            "camera_settings": settings,
            "pipeline_settings": proc_config
        })
        return True
    except Exception as e:
        event_logger.log(f"Error saving camera settings: {e}")
        return False

def restore_camera_settings():
    try:
        cfg = load_config()
        cam_settings = cfg.get("camera_settings", {})
        if cam_settings and rig:
            # 1. Apply auto_exposure first so V4L2 locks to manual mode before setting manual exposure
            if "auto_exposure" in cam_settings:
                try:
                    rig.set_camera_param("auto_exposure", cam_settings["auto_exposure"])
                except Exception as e:
                    event_logger.log(f"Error restoring auto_exposure: {e}")

            # 2. Apply exposure second (only effective when auto_exposure=0 in V4L2)
            if "exposure" in cam_settings:
                try:
                    rig.set_camera_param("exposure", cam_settings["exposure"])
                except Exception as e:
                    event_logger.log(f"Error restoring exposure: {e}")

            # 3. Apply remaining camera parameters
            for prop, val in cam_settings.items():
                if prop in ("auto_exposure", "exposure", "format", "camera_format", "average", "accumulation_count"):
                    continue
                try:
                    rig.set_camera_param(prop, val)
                except Exception as e:
                    event_logger.log(f"Error restoring camera param {prop}={val}: {e}")

            # 4. Restore accumulation count
            acc_val = cam_settings.get("accumulation_count", cam_settings.get("average"))
            if acc_val is not None and pipeline:
                pipeline.set_accumulation_count(int(acc_val))

            event_logger.log(f"Restored camera settings: {list(cam_settings.keys())}")
        
        proc_settings = cfg.get("pipeline_settings", {})
        if proc_settings and pipeline and hasattr(pipeline, "set_processing_config"):
            try:
                pipeline.set_processing_config(proc_settings)
                event_logger.log(f"Restored pipeline processing settings: {list(proc_settings.keys())}")
            except Exception as e:
                event_logger.log(f"Error restoring pipeline settings: {e}")
    except Exception as e:
        event_logger.log(f"Error restoring settings from config.json: {e}")

config = load_config()

# Unified AstroRig & Processing Pipeline with decoupled camera and mount
camera_mode = config.get("camera_mode", "mock")
mount_mode = config.get("mount_mode", "mock")
camera_format = config.get("camera_format", "YUYV")
rig = get_astro_rig(camera_mode=camera_mode, mount_mode=mount_mode, motor_pin=18, camera_format=camera_format)
pipeline = AstroPipeline(rig)
panorama = PanoramaManager(rig)
restore_camera_settings()

def _on_nav_change_reset_acc(changed_keys):
    if changed_keys & {"ra", "dec", "target"}:
        if pipeline:
            pipeline.reset_accumulation()

from sky_nav import get_sky_nav
get_sky_nav().subscribe(_on_nav_change_reset_acc)

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

    rig = get_astro_rig(camera_mode=camera_mode, mount_mode=mount_mode, motor_pin=18, camera_format=camera_format)
    pipeline.rig = rig
    panorama.rig = rig
    return {"success": True, "camera_mode": camera_mode, "mount_mode": mount_mode, "camera_format": camera_format}

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

class CameraFormatUpdate(BaseModel):
    format: str

@app.post("/camera/format")
def set_camera_format(req: CameraFormatUpdate):
    global camera_format
    if not rig:
        return {"success": False, "error": "Rig not initialized"}
    fmt = req.format.upper()
    if fmt not in ("YUYV", "MJPG"):
        return {"success": False, "error": f"Invalid format: {fmt}. Must be YUYV or MJPG."}
    success = False
    if hasattr(rig, "set_camera_format"):
        success = rig.set_camera_format(fmt)
    if success:
        camera_format = fmt
        save_config({"camera_format": fmt})
        if pipeline:
            pipeline.reset_accumulation()
    current_fmt = rig.get_camera_format() if hasattr(rig, "get_camera_format") else camera_format
    return {"success": success, "camera_format": current_fmt}

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

class SkyUpdate(BaseModel):
    backend: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    ra: Optional[float] = None
    dec: Optional[float] = None
    focal_length_mm: Optional[float] = None
    pixel_size_um: Optional[float] = None
    target: Optional[str] = None

@app.get("/mock/sky")
def get_mock_sky():
    from sky_nav import get_sky_nav
    return {"success": True, **get_sky_nav().get_config()}

@app.post("/mock/sky")
def set_mock_sky(update: SkyUpdate):
    from sky_nav import get_sky_nav
    nav = get_sky_nav()
    success = nav.set(**update.model_dump(exclude_unset=True))
    return {"success": success, **nav.get_config()}

class SkySearch(BaseModel):
    query: str

@app.post("/mock/sky/search")
def search_mock_sky(search: SkySearch):
    from sky_nav import get_sky_nav
    nav = get_sky_nav()
    result = nav.search_target(search.query)
    return {**result, "sky": nav.get_config()}

@app.get("/mock/sky/targets")
def get_mock_sky_targets():
    from sky_nav import get_sky_nav
    from star_sources import get_all_planets_coords
    planets = []
    try:
        nav = get_sky_nav()
        cfg = nav.get_config()
        lat = cfg.get("latitude", 37.77)
        lon = cfg.get("longitude", -122.42)
        planets = get_all_planets_coords(lat_deg=lat, lon_deg=lon)
    except Exception as e:
        event_logger.log(f"Error computing planet coords: {e}")

    targets = []
    targets_file = os.path.join(os.path.dirname(__file__), "targets.json")
    if os.path.exists(targets_file):
        try:
            with open(targets_file, "r") as f:
                targets = json.load(f)
        except Exception as e:
            targets = []
    return planets + targets

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
    status["accumulation_count"] = pipeline.accumulation_count
    status["acc_count"] = pipeline.acc_count
    motor_status = rig.get_motor_status()
    status["mock_mode"] = motor_status.get("mock_mode", True)
    status["camera_mode"] = camera_mode
    status["mount_mode"] = mount_mode
    status["camera_format"] = rig.get_camera_format() if hasattr(rig, "get_camera_format") else camera_format
    status["recording"] = pipeline.get_recording_status()
    status["corrupted_frames_dropped"] = getattr(pipeline, "corrupted_frames_dropped", 0)
    status["histogram"] = pipeline.get_histogram() if hasattr(pipeline, "get_histogram") else []
    status["processing"] = pipeline.get_processing_config() if hasattr(pipeline, "get_processing_config") else {}
    return status

@app.get("/histogram")
def get_histogram():
    if not pipeline or not hasattr(pipeline, "get_histogram"):
        return {"histogram": []}
    return {"histogram": pipeline.get_histogram()}

class ProcessingUpdate(BaseModel):
    histogram: Optional[bool] = None
    corrupted_filter: Optional[bool] = None
    luminance_check: Optional[bool] = None
    roi: Optional[List[int]] = None
    stride: Optional[int] = None

@app.get("/pipeline/processing")
def get_pipeline_processing():
    if not pipeline or not hasattr(pipeline, "get_processing_config"):
        return {"histogram": True, "corrupted_filter": True, "luminance_check": True, "roi": [0, 0, 1920, 1080], "stride": 4}
    return pipeline.get_processing_config()

@app.post("/pipeline/processing")
def set_pipeline_processing(req: ProcessingUpdate):
    if not pipeline or not hasattr(pipeline, "set_processing_config"):
        return {"success": False}
    updates = {k: v for k, v in req.dict().items() if v is not None}
    pipeline.set_processing_config(updates)
    save_camera_settings()
    return {"success": True, **pipeline.get_processing_config()}

@app.get("/controls")
def get_controls():
    if not rig or not pipeline: return {}
    return {**rig.get_camera_params(), "average": pipeline.accumulation_count, "accumulation_count": pipeline.accumulation_count}

@app.post("/controls")
def set_control(update: ControlUpdate):
    if not rig or not pipeline: return {"success": False}
    if update.property in ("average", "accumulation_count"):
        success = pipeline.set_accumulation_count(int(update.value))
    else:
        success = rig.set_camera_param(update.property, update.value)
        pipeline.reset_accumulation()
    if success:
        event_logger.log(f"Control: {update.property} -> {update.value}")
        save_camera_settings()
    return {"success": success, "property": update.property, "value": update.value}

class AccumulationUpdate(BaseModel):
    count: int

@app.post("/pipeline/accumulation")
def set_pipeline_accumulation(req: AccumulationUpdate):
    if not pipeline: return {"success": False}
    success = pipeline.set_accumulation_count(req.count)
    if success:
        save_camera_settings()
    return {"success": success, "accumulation_count": pipeline.accumulation_count}

@app.get("/camera/settings")
def get_saved_camera_settings():
    cfg = load_config()
    return {
        "camera_settings": cfg.get("camera_settings", {}),
        "pipeline_settings": cfg.get("pipeline_settings", {})
    }

@app.post("/camera/settings/save")
def manually_save_camera_settings():
    ok = save_camera_settings()
    cfg = load_config()
    return {
        "success": ok,
        "camera_settings": cfg.get("camera_settings", {}),
        "pipeline_settings": cfg.get("pipeline_settings", {})
    }

@app.post("/camera/settings/restore")
def manually_restore_camera_settings():
    restore_camera_settings()
    return {
        "success": True,
        "controls": get_controls(),
        "processing": pipeline.get_processing_config() if pipeline and hasattr(pipeline, "get_processing_config") else {}
    }

class BlackCalibrationRequest(BaseModel):
    frames: int = 15

@app.post("/calibration/black")
def start_black_calibration(req: Optional[BlackCalibrationRequest] = None):
    if not pipeline:
        return {"success": False, "error": "Pipeline not initialized"}
    frames = req.frames if req else 15
    return pipeline.start_black_calibration(target_frames=frames)

@app.get("/calibration/status")
def get_calibration_status():
    if not pipeline:
        return {"calibrated": False, "is_calibrating": False}
    return pipeline.get_calibration_status()

@app.delete("/calibration/black")
def clear_black_calibration():
    if not pipeline:
        return {"success": False, "error": "Pipeline not initialized"}
    return pipeline.clear_black_calibration()

class BaselineCaptureRequest(BaseModel):
    key: str
    frames: int = 10

class BaselineConfigRequest(BaseModel):
    subtract: Optional[bool] = None
    pedestal: Optional[float] = None
    key: Optional[str] = None

@app.get("/calibration/baseline/status")
def get_baseline_status(key: str = ""):
    if not pipeline:
        return {"exists": False, "subtract": False, "pedestal": 10.0}
    return pipeline.check_baseline_status(key)

@app.post("/calibration/baseline/capture")
def capture_baseline(req: BaselineCaptureRequest):
    if not pipeline:
        return {"success": False, "error": "Pipeline not initialized"}
    return pipeline.start_baseline_capture(key=req.key, frames=req.frames)

@app.post("/calibration/baseline/config")
def update_baseline_config(req: BaselineConfigRequest):
    if not pipeline:
        return {"success": False, "error": "Pipeline not initialized"}
    return pipeline.set_baseline_config(subtract=req.subtract, pedestal=req.pedestal, key=req.key)

class RecordingStartRequest(BaseModel):
    accumulation_count: Optional[int] = None
    accumulation_ms: Optional[int] = None
    format: Optional[str] = None

@app.post("/recording/start")
def start_recording(req: Optional[RecordingStartRequest] = None):
    if not pipeline: return {"success": False, "error": "Pipeline not initialized"}
    acc_count = req.accumulation_count if (req and req.accumulation_count is not None) else None
    acc_ms = req.accumulation_ms if (req and req.accumulation_ms is not None) else None
    fmt = req.format if (req and req.format is not None) else None
    return pipeline.start_recording(accumulation_count=acc_count, accumulation_ms=acc_ms, file_format=fmt)

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

@app.post("/system/restart")
def restart_system():
    event_logger.log("System restart requested from Web UI.")

    def trigger_restart():
        time.sleep(1.0)
        try:
            res = subprocess.run(["systemctl", "is-active", "--quiet", "astrocam"])
            if res.returncode == 0:
                subprocess.Popen(["sudo", "systemd-run", "systemctl", "restart", "astrocam"])
                return
        except Exception as e:
            event_logger.log(f"Error checking systemd status: {e}")

        try:
            subprocess.Popen(["/bin/bash", "/home/kio/projects/astrocam/start.sh"])
        except Exception as e:
            event_logger.log(f"Error executing start.sh: {e}")
        os._exit(0)

    threading.Thread(target=trigger_restart, daemon=True).start()
    return {"success": True, "message": "Restarting AstroCam service..."}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
