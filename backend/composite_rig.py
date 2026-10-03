from typing import Any, Dict, Optional
import numpy as np
from base_rig import BaseAstroRig

class CompositeAstroRig(BaseAstroRig):
    """
    Composes any camera implementation (RealCamera or MockCamera) with any
    mount implementation (RealMount or MockMount).
    """
    def __init__(self, camera: Any, mount: Any):
        self.camera = camera
        self.mount = mount

        # If camera is simulated, connect it to the mount's duty cycle
        if hasattr(self.camera, "get_duty_callback"):
            self.camera.get_duty_callback = lambda: self.mount.current_duty

    @property
    def current_duty(self) -> float:
        return self.mount.current_duty

    @current_duty.setter
    def current_duty(self, val: float):
        self.mount.current_duty = float(val)

    @property
    def target_duty(self) -> float:
        return self.mount.target_duty

    # Pass-through simulation properties for UI / pipeline if camera supports them
    @property
    def sim_drift_speed(self) -> float:
        return getattr(self.camera, "sim_drift_speed", 0.0)

    @property
    def sim_drift_angle(self) -> float:
        return getattr(self.camera, "sim_drift_angle", 0.0)

    @property
    def camera_angle(self) -> float:
        return getattr(self.camera, "camera_angle", 0.0)

    def set_camera_angle(self, angle: float) -> bool:
        if hasattr(self.camera, "set_camera_angle"):
            return self.camera.set_camera_angle(angle)
        return False

    def set_sim_drift(self, speed: float, angle: float) -> bool:
        if hasattr(self.camera, "set_sim_drift"):
            return self.camera.set_sim_drift(speed, angle)
        return False

    # Camera delegations
    def get_raw_frame(self) -> Optional[np.ndarray]:
        return self.camera.get_raw_frame()

    def set_camera_param(self, prop: str, value: float) -> bool:
        return self.camera.set_camera_param(prop, value)

    def get_camera_params(self) -> Dict[str, Any]:
        return self.camera.get_camera_params()

    def get_camera_status(self) -> Dict[str, Any]:
        return self.camera.get_camera_status()

    def reconnect_camera(self) -> bool:
        if hasattr(self.camera, "reconnect"):
            return self.camera.reconnect()
        return False

    def reconnect(self) -> bool:
        return self.reconnect_camera()

    # Mount delegations
    def set_motor_speed(self, speed: float, ramp_time: float = 0.5) -> bool:
        return self.mount.set_motor_speed(speed, ramp_time=ramp_time)

    def get_motor_status(self) -> Dict[str, Any]:
        return self.mount.get_motor_status()

    def close(self):
        if self.camera:
            try:
                self.camera.close()
            except Exception:
                pass
        if self.mount:
            try:
                self.mount.close()
            except Exception:
                pass
