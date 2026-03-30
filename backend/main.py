from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import uvicorn
import cv2
import os
import asyncio
from datetime import datetime
from camera import SV205Camera
from motor_control import MotorController
from pydantic import BaseModel

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize camera and motor
cam = SV205Camera(0)
motor = MotorController(18)

class ControlUpdate(BaseModel):
    property: str
    value: float

class MotorSpeedUpdate(BaseModel):
    speed: float

@app.get("/motor/status")
def get_motor_status():
    return motor.get_status()

@app.post("/motor/speed")
def set_motor_speed(update: MotorSpeedUpdate):
    success = motor.set_speed(update.speed)
    return {"success": success, "speed": update.speed}

@app.get("/stream")
async def stream():
    if not cam:
        return Response(content="Camera not available", status_code=503)
        
    async def frame_generator():
        while True:
            try:
                frame = cam.get_frame()
                if frame:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                await asyncio.sleep(0.03) # Limit to ~30fps
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
    if not cam: return {"success": False}
    success = cam.start_sequence(req.count, req.interval)
    return {"success": success}

@app.get("/sequence/status")
def get_sequence_status():
    if not cam: return {"active": False}
    return cam.sequence_info

@app.post("/resolution")
def set_resolution(res: ResolutionUpdate):
    if not cam: return {"success": False}
    success = cam.set_resolution(res.width, res.height)
    return {"success": success, "width": res.width, "height": res.height}

@app.get("/status")
def get_status():
    if not cam:
        return {"connected": False, "mean_brightness": 0, "error": "Camera not initialized"}
    return cam.get_status()

@app.get("/controls")
def get_controls():
    if not cam: return {}
    return cam.get_params()

@app.post("/controls")
def set_control(update: ControlUpdate):
    if not cam: return {"success": False}
    success = cam.set_param(update.property, update.value)
    return {"success": success, "property": update.property, "value": update.value}

@app.post("/capture")
def capture():
    if not cam: return {"success": False}
    frame_data = cam.get_frame()
    if frame_data:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"capture_{timestamp}.jpg"
        filepath = os.path.join("/home/kio/projects/astrocam", filename)
        with open(filepath, "wb") as f:
            f.write(frame_data)
        return {"success": True, "filename": filename}
    return {"success": False, "error": "Could not capture frame"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
