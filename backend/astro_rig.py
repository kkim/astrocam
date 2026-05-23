import cv2
from real_rig import RealAstroRig, HAS_GPIO
from mock_rig import MockAstroRig
from logger import event_logger

def get_astro_rig(mode="mock", camera_id=0, motor_pin=18):
    if mode == "real":
        # Try to open camera hardware to verify availability
        cap = cv2.VideoCapture(camera_id, cv2.CAP_V4L2)
        camera_available = cap.isOpened()
        if camera_available:
            cap.release()
            event_logger.log("Hardware detected: Initializing RealAstroRig")
            return RealAstroRig(camera_id, motor_pin)
        else:
            event_logger.log("Hardware NOT detected: Falling back to MockAstroRig")
            return MockAstroRig()
    else:
        event_logger.log("Initializing MockAstroRig (requested)")
        return MockAstroRig()
