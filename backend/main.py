from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import uvicorn
import cv2
import os
import asyncio
from datetime import datetime
from astro_rig import get_astro_rig
from logger import event_logger
from panorama import PanoramaManager
from pydantic import BaseModel

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Unified AstroRig
rig_mode = "real" # Try real by default
rig = get_astro_rig(rig_mode, 0, 18)
panorama = PanoramaManager(rig)

@app.on_event("shutdown")
def shutdown_event():
    event_logger.log("System shutting down...")
    if rig: rig.close()

class ControlUpdate(BaseModel):
    property: str
    value: float

class MotorSpeedUpdate(BaseModel):
    speed: float

class RigUpdate(BaseModel):
    mode: str

@app.get("/rig")
def get_rig_mode():
    return {"mode": rig_mode}

@app.post("/rig")
def set_rig_mode(update: RigUpdate):
    global rig, rig_mode
    if update.mode == rig_mode:
        return {"success": True, "mode": rig_mode}
    
    event_logger.log(f"Switching rig mode to: {update.mode}")
    if rig:
        rig.close()
    
    rig_mode = update.mode
    rig = get_astro_rig(rig_mode, 0, 18)
    return {"success": True, "mode": rig_mode}

@app.get("/logs")
def get_logs():
    return event_logger.get_logs()

@app.get("/motor/status")
def get_motor_status():
    return rig.get_motor_status()

@app.post("/motor/speed")
def set_motor_speed(update: MotorSpeedUpdate):
    success = rig.set_motor_speed(update.speed)
    if success:
        event_logger.log(f"Motor speed: {update.speed}%")
    return {"success": success, "speed": update.speed}

@app.get("/stream")
async def stream():
    if not rig:
        return Response(content="Rig not available", status_code=503)
        
    async def frame_generator():
        while True:
            try:
                frame = rig.get_frame()
                if frame:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                await asyncio.sleep(0.04) # ~25fps
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
    if not rig: return {"success": False}
    success = rig.start_sequence(req.count, req.interval)
    return {"success": success}

@app.get("/sequence/status")
def get_sequence_status():
    if not rig: return {"active": False}
    return rig.get_sequence_status()

@app.post("/resolution")
def set_resolution(res: ResolutionUpdate):
    # Resolution switching needs implementation in rigs if needed
    return {"success": False, "error": "Not implemented in unified rig yet"}

@app.get("/status")
def get_status():
    if not rig:
        return {"connected": False, "mean_brightness": 0, "error": "Rig not initialized"}
    # Merge camera status with motor mock mode info for UI
    status = rig.get_camera_status()
    motor_status = rig.get_motor_status()
    status["mock_mode"] = motor_status.get("mock_mode", True)
    # Mock brightness for UI if needed
    if "mean_brightness" not in status:
        status["mean_brightness"] = 12.75
    return status

@app.get("/controls")
def get_controls():
    if not rig: return {}
    return rig.get_camera_params()

@app.post("/controls")
def set_control(update: ControlUpdate):
    if not rig: return {"success": False}
    success = rig.set_camera_param(update.property, update.value)
    if success:
        event_logger.log(f"Control: {update.property} -> {update.value}")
    return {"success": success, "property": update.property, "value": update.value}

@app.post("/capture")
def capture():
    if not rig: return {"success": False, "error": "Rig not initialized"}
    return rig.capture_frame()

class PanoramaUpdate(BaseModel):
    frames: int
    drift_step: float

@app.post("/panorama/start")
def start_panorama(req: PanoramaUpdate):
    success = panorama.start(req.frames, req.drift_step)
    return {"success": success}

@app.get("/panorama/status")
def get_panorama_status():
    return panorama.get_status()

@app.post("/panorama/stop")
def stop_panorama():
    panorama.stop()
    return {"success": True}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
