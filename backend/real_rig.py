from typing import Optional
from real_camera import RealCamera
from real_mount import RealMount, HAS_GPIO
from composite_rig import CompositeAstroRig

class RealAstroRig(CompositeAstroRig):
    """
    RealAstroRig bundles physical USB camera capture and physical GPIO motor control.
    Maintained for direct instantiation and backwards compatibility.
    """
    def __init__(self, camera_id: Optional[int] = None, motor_pin: int = 18):
        camera = RealCamera(camera_id=camera_id)
        mount = RealMount(motor_pin=motor_pin)
        super().__init__(camera, mount)

    @property
    def cap(self):
        return getattr(self.camera, "cap", None)

    @property
    def pin(self):
        return getattr(self.mount, "pin", None)

    @property
    def width(self):
        return getattr(self.camera, "width", 1920)

    @property
    def height(self):
        return getattr(self.camera, "height", 1080)
