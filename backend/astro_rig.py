from typing import Optional
from real_camera import RealCamera
from real_mount import RealMount, HAS_GPIO
from mock_camera import MockCamera
from mock_mount import MockMount
from composite_rig import CompositeAstroRig
from real_rig import RealAstroRig
from mock_rig import MockAstroRig
from logger import event_logger

def get_astro_rig(
    camera_mode: str = "mock",
    mount_mode: Optional[str] = None,
    camera_id: Optional[int] = None,
    motor_pin: int = 18,
    mode: Optional[str] = None
) -> CompositeAstroRig:
    """
    Provisions an AstroRig with decoupled, independently selectable camera and mount engines.
    
    Backwards compatibility:
    - If `mode` is passed, or if only a single mode string is given (e.g. `get_astro_rig('real')`),
      both camera and mount will switch to that mode.
    - If `mount_mode` is explicitly provided, camera and mount are configured independently.
    """
    if mode is not None:
        camera_mode = mode
        mount_mode = mode
    elif mount_mode is None:
        # Legacy single-argument invocation e.g. get_astro_rig("real")
        mount_mode = camera_mode

    event_logger.log(f"Initializing AstroRig (Camera: {camera_mode.upper()}, Mount: {mount_mode.upper()})")

    # Instantiate Camera
    if camera_mode == "real":
        camera = RealCamera(camera_id=camera_id)
    else:
        camera = MockCamera()

    # Instantiate Mount
    if mount_mode == "real":
        mount = RealMount(motor_pin=motor_pin)
    else:
        mount = MockMount()

    return CompositeAstroRig(camera, mount)
