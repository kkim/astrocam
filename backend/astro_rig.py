import cv2
from real_rig import RealAstroRig, HAS_GPIO
from mock_rig import MockAstroRig
from logger import event_logger

def get_astro_rig(mode="mock", camera_id=0, motor_pin=18):
    if mode == "real":
        event_logger.log("Initializing RealAstroRig")
        return RealAstroRig(camera_id, motor_pin)
    else:
        event_logger.log("Initializing MockAstroRig")
        return MockAstroRig()
