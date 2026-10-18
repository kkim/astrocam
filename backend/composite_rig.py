from typing import Any, Dict, Optional, Tuple
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

    def get_sky_config(self) -> Optional[Dict[str, Any]]:
        from sky_nav import get_sky_nav
        return get_sky_nav().get_config()

    def set_sky_config(self, **changes) -> bool:
        from sky_nav import get_sky_nav
        return get_sky_nav().set(**changes)

    def search_target(self, query: str) -> Dict[str, Any]:
        from sky_nav import get_sky_nav
        return get_sky_nav().search_target(query)

    # Camera delegations
    def get_raw_frame(self) -> Optional[np.ndarray]:
        return self.camera.get_raw_frame()

    def get_fresh_frame(self, last_frame_id: int = -1) -> Tuple[Optional[np.ndarray], int]:
        if hasattr(self.camera, "get_fresh_frame"):
            return self.camera.get_fresh_frame(last_frame_id)
        frame = self.camera.get_raw_frame()
        return frame, last_frame_id + 1

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
