from abc import ABC, abstractmethod

class BaseAstroRig(ABC):
    @abstractmethod
    def get_frame(self):
        pass

    @abstractmethod
    def get_raw_frame(self):
        pass

    @abstractmethod
    def set_camera_param(self, prop, value):
        pass

    @abstractmethod
    def get_camera_params(self):
        pass

    @abstractmethod
    def get_camera_status(self):
        pass

    @abstractmethod
    def start_sequence(self, count, interval):
        pass

    @abstractmethod
    def get_sequence_status(self):
        pass

    @abstractmethod
    def set_motor_speed(self, speed, ramp_time=0.5):
        pass

    @abstractmethod
    def get_motor_status(self):
        pass

    @abstractmethod
    def capture_frame(self):
        pass

    @abstractmethod
    def set_auto_tracking(self, enable: bool):
        pass

    @abstractmethod
    def get_tracking_status(self):
        pass

    @abstractmethod
    def close(self):
        pass

    def set_camera_angle(self, angle_deg):
        return False

    def set_sim_drift(self, speed, angle_deg):
        return False

