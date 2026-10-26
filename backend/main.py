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
import struct
from mosaic_service import get_session_mosaic_info, trigger_mosaic_stitching, MOSAIC_SUBDIR

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

# Cache static session metadata keyed by s_name -> (dir_mtime, meta_dict)
_SESSION_META_CACHE = {}

@app.get("/gallery")
def list_captures():
    base_dir = CAPTURES_DIR
    if not os.path.exists(base_dir):
        return []

    thumb_dir = os.path.join(base_dir, "thumbnails")
    os.makedirs(thumb_dir, exist_ok=True)

    items = []
    # 1. Sequence recording sessions
    session_dirs = glob.glob(os.path.join(base_dir, "session_*"))
    for s_dir in session_dirs:
        if not os.path.isdir(s_dir):
            continue
        s_name = os.path.basename(s_dir)
        dir_mtime = os.path.getmtime(s_dir)

        cached = _SESSION_META_CACHE.get(s_name)
        if cached and cached[0] == dir_mtime:
            s_meta = cached[1]
        else:
            files = sorted([
                f for f in os.listdir(s_dir)
                if not f.startswith(".") and f != "thumb.jpg" and f.lower().endswith((".tif", ".tiff", ".jpg", ".jpeg", ".png"))
            ])
            if not files:
                continue

            first_frame = files[0]
            fmt = first_frame.rsplit(".", 1)[-1].lower()

            # Parse date and time
            parts = s_name.replace("session_", "").split("_")
            if len(parts) >= 2 and len(parts[0]) == 8 and len(parts[1]) >= 6:
                ymd, hms = parts[0], parts[1]
                date_str = f"{ymd[:4]}-{ymd[4:6]}-{ymd[6:8]}"
                time_str = f"{hms[:2]}:{hms[2:4]}:{hms[4:6]}"
                dt_str = f"{date_str} {time_str}"
            else:
                dt = datetime.fromtimestamp(dir_mtime)
                date_str = dt.strftime("%Y-%m-%d")
                time_str = dt.strftime("%H:%M:%S")
                dt_str = f"{date_str} {time_str}"

            # Nav info (target / coordinates / notes)
            target_name = ""
            camera_settings = {}
            s_nav = {}
            session_nav_file = os.path.join(s_dir, "nav.json")
            if os.path.exists(session_nav_file):
                try:
                    with open(session_nav_file, "r") as f:
                        s_nav = json.load(f)
                        target_name = s_nav.get("target", "")
                        camera_settings = s_nav.get("camera_settings", {})
                except Exception:
                    pass

            s_meta = {
                "frames": len(files),
                "first_frame": first_frame,
                "fmt": fmt,
                "date_str": date_str,
                "time_str": time_str,
                "dt_str": dt_str,
                "target_name": target_name,
                "camera_settings": camera_settings,
                "nav": s_nav,
            }
            _SESSION_META_CACHE[s_name] = (dir_mtime, s_meta)

        first_frame = s_meta["first_frame"]
        fmt = s_meta["fmt"]
        thumb_path = os.path.join(thumb_dir, f"{s_name}.jpg")
        thumb_sm_path = os.path.join(thumb_dir, f"{s_name}_sm.jpg")

        # Auto-generate Full HD preview and lightweight 360x202 grid thumbnail if missing
        if not os.path.exists(thumb_path) or not os.path.exists(thumb_sm_path):
            try:
                src_img = None
                if os.path.exists(thumb_path):
                    src_img = cv2.imread(thumb_path, cv2.IMREAD_COLOR)
                if src_img is None:
                    raw_img = cv2.imread(os.path.join(s_dir, first_frame), cv2.IMREAD_UNCHANGED)
                    if raw_img is not None:
                        if raw_img.dtype == np.uint16:
                            raw_img = np.clip(raw_img / 257.0, 0, 255).astype(np.uint8)
                        src_img = raw_img
                        if not os.path.exists(thumb_path):
                            cv2.imwrite(thumb_path, src_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
                if src_img is not None and not os.path.exists(thumb_sm_path):
                    h_img, w_img = src_img.shape[:2]
                    sm_w = 360
                    sm_h = max(1, int(round(h_img * (sm_w / float(w_img)))))
                    sm_img = cv2.resize(src_img, (sm_w, sm_h), interpolation=cv2.INTER_AREA)
                    cv2.imwrite(thumb_sm_path, sm_img, [cv2.IMWRITE_JPEG_QUALITY, 82])
            except Exception as e:
                event_logger.log(f"Error creating thumbnail for {s_name}: {e}")

        # Mosaic availability info
        mosaic_info = get_session_mosaic_info(s_dir, s_name)
        nav_info = mosaic_info.get("nav") or s_meta.get("nav") or {}

        preview_exists = os.path.exists(thumb_path)
        sm_exists = os.path.exists(thumb_sm_path)
        preview_url = f"/captures/thumbnails/{s_name}.jpg" if preview_exists else f"/captures/{s_name}/{first_frame}"
        thumb_url = f"/captures/thumbnails/{s_name}_sm.jpg" if sm_exists else preview_url
        items.append({
            "id": s_name,
            "type": "sequence",
            "name": s_name,
            "target": nav_info.get("target") or s_meta["target_name"],
            "stars": nav_info.get("stars", ""),
            "ra": nav_info.get("ra"),
            "dec": nav_info.get("dec"),
            "orientation": nav_info.get("orientation", ""),
            "note": nav_info.get("note", ""),
            "nav": nav_info,
            "camera_settings": s_meta["camera_settings"],
            "time": s_meta["time_str"],
            "date": s_meta["date_str"],
            "datetime": s_meta["dt_str"],
            "frames": s_meta["frames"],
            "first_frame": first_frame,
            "thumbnail": thumb_url,
            "url": mosaic_info["mosaic_url"] if mosaic_info["has_mosaic"] else (preview_url if fmt in ("tif", "tiff") else f"/captures/{s_name}/{first_frame}"),
            "first_frame_url": f"/captures/{s_name}/{first_frame}",
            "format": fmt,
            "has_mosaic": mosaic_info["has_mosaic"],
            "mosaic_status": mosaic_info["mosaic_status"],
            "mosaic_url": mosaic_info["mosaic_url"] if mosaic_info["has_mosaic"] else None,
            "mosaic_mtime": mosaic_info["mosaic_mtime"],
            "mosaic_progress": mosaic_info.get("mosaic_progress"),
            "mosaic_queue_pos": mosaic_info.get("mosaic_queue_pos", 0),
            "mosaic_meta": mosaic_info.get("mosaic_meta"),
            "mtime": dir_mtime
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

    # Sort by capture datetime (latest shown first) and return latest 30 captures
    items.sort(key=lambda x: (x.get("datetime", ""), x.get("mtime", 0)), reverse=True)
    return items[:30]


class MosaicGenerateRequest(BaseModel):
    force: bool = False
    align_mode: Optional[str] = None
    min_snr: Optional[float] = None
    max_step_px: Optional[float] = None
    static_filter: Optional[bool] = None
    defect_radius: Optional[int] = None
    defect_soft_px: Optional[int] = None
    use_defect_mask: Optional[bool] = None
    feather_px: Optional[int] = None
    stretch_method: Optional[str] = None
    stretch_factor: Optional[float] = None
    fov_mode: Optional[str] = None


@app.post("/mosaic/generate/{session_id}")
def generate_mosaic(session_id: str, req: Optional[MosaicGenerateRequest] = None):
    s_dir = os.path.join(CAPTURES_DIR, session_id)
    if not os.path.isdir(s_dir):
        return {"success": False, "error": f"Session {session_id} not found"}
    opts = req.model_dump(exclude_none=True) if req else {}
    return trigger_mosaic_stitching(session_id, s_dir, options=opts)


class StackNoteUpdate(BaseModel):
    target: Optional[str] = None
    stars: Optional[str] = None
    ra: Optional[float] = None
    dec: Optional[float] = None
    orientation: Optional[str] = None
    note: Optional[str] = None


@app.post("/mosaic/note/{session_id}")
def update_stack_note(session_id: str, req: StackNoteUpdate):
    s_dir = os.path.join(CAPTURES_DIR, session_id)
    if not os.path.isdir(s_dir):
        return {"success": False, "error": f"Session {session_id} not found"}
    nav_path = os.path.join(s_dir, "nav.json")
    nav_data = {}
    if os.path.exists(nav_path):
        try:
            with open(nav_path, "r") as f:
                nav_data = json.load(f)
        except Exception:
            nav_data = {}
    updates = req.model_dump(exclude_unset=True)
    for k, v in updates.items():
        nav_data[k] = v
    try:
        tmp_path = nav_path + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump(nav_data, f, indent=2)
        os.replace(tmp_path, nav_path)
        _SESSION_META_CACHE.pop(session_id, None)
        return {"success": True, "nav": nav_data}
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.get("/mosaic/status/{session_id}")
def get_mosaic_status(session_id: str):
    s_dir = os.path.join(CAPTURES_DIR, session_id)
    if not os.path.isdir(s_dir):
        return {"status": "none"}
    return get_session_mosaic_info(s_dir, session_id)

@app.get("/mosaic/linear16/{session_id}")
def get_mosaic_linear16(session_id: str):
    s_dir = os.path.join(CAPTURES_DIR, session_id)
    tif_path = os.path.join(s_dir, MOSAIC_SUBDIR, "mosaic_linear_16bit.tif")
    if not os.path.exists(tif_path):
        fallback_path = os.path.join(s_dir, "mosaic", "mosaic_linear_16bit.tif")
        if os.path.exists(fallback_path):
            tif_path = fallback_path
        else:
            return Response(content="16-bit linear mosaic not found", status_code=404)

    img = cv2.imread(tif_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return Response(content="Failed to decode 16-bit mosaic", status_code=500)

    if img.dtype == np.uint8:
        img = img.astype(np.uint16) * 257
    elif img.dtype != np.uint16:
        img = np.clip(img, 0, 65535).astype(np.uint16)

    if img.ndim == 3:
        if img.shape[2] == 4:
            img = img[:, :, :3]
        b_ch = img[:, :, 0].astype(np.int32)
        g_ch = img[:, :, 1].astype(np.int32)
        r_ch = img[:, :, 2].astype(np.int32)
        if np.max(np.abs(b_ch - g_ch)) <= 3 and np.max(np.abs(b_ch - r_ch)) <= 3:
            img = img[:, :, 1]
            channels = 1
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            channels = 3
    else:
        channels = 1

    h, w = img.shape[:2]
    valid = img[img > 0]
    if valid.size > 0:
        sample = valid[::4] if valid.size > 500000 else valid
        min_nz = int(np.min(valid))
        max_val = int(np.max(valid))
        med = float(np.median(sample))
        mad = float(np.median(np.abs(sample - med)))
        sigma = max(1.0, 1.4826 * mad)
        p05 = float(np.percentile(sample, 0.5))
        p999 = float(np.percentile(sample, 99.95))
        auto_min = int(max(min_nz, min(p05, med - 1.5 * sigma)))
        auto_max = int(max(auto_min + 16, p999))
    else:
        min_nz, max_val, auto_min, auto_max = 0, 65535, 0, 65535

    header = struct.pack("<8I", 0x36314D41, w, h, channels, auto_min, auto_max, min_nz, max_val)
    arr_le = np.ascontiguousarray(img, dtype="<u2")
    return Response(
        content=header + arr_le.tobytes(),
        media_type="application/octet-stream",
        headers={"Cache-Control": "no-cache"}
    )

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
