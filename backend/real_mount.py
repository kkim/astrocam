import threading
import time
from typing import Dict, Any, Optional
from logger import event_logger

try:
    from gpiozero import PWMOutputDevice
    HAS_GPIO = True
except (ImportError, RuntimeError):
    HAS_GPIO = False

class RealMount:
    """
    Controls physical DC motor on the equatorial mount using Raspberry Pi GPIO PWM.
    """
    def __init__(self, motor_pin: int = 18):
        self.motor_pin_number = motor_pin
        self.pin: Optional[Any] = None
        self.current_duty: float = 0.0
        self.target_duty: float = 0.0
        self.ramp_thread: Optional[threading.Thread] = None
        self.stop_ramping = threading.Event()
        
        self._init_motor()

    def _init_motor(self):
        if HAS_GPIO:
            try:
                self.pin = PWMOutputDevice(self.motor_pin_number)
                self.pin.value = 0
                event_logger.log(f"Motor initialized on GPIO {self.motor_pin_number} (PWM)")
            except Exception as e:
                event_logger.log(f"Error: Motor init failed on GPIO {self.motor_pin_number}: {e}")
        else:
            event_logger.log(f"Notice: GPIO not available. Mount running in simulated hardware mode.")

    def set_motor_speed(self, speed: float, ramp_time: float = 0.5) -> bool:
        self.target_duty = float(speed)
        if self.ramp_thread and self.ramp_thread.is_alive():
            self.stop_ramping.set()
            self.ramp_thread.join()
        self.stop_ramping.clear()
        self.ramp_thread = threading.Thread(target=self._ramp_logic, args=(ramp_time,), daemon=True)
        self.ramp_thread.start()
        return True

    def _ramp_logic(self, ramp_time: float):
        start_duty = self.current_duty
        target_duty = self.target_duty
        if start_duty == target_duty:
            return
        
        steps = int(ramp_time / 0.05)
        if steps <= 0:
            self._apply_motor_duty(target_duty)
            self.current_duty = target_duty
            return
            
        duty_step = (target_duty - start_duty) / steps
        for _ in range(steps):
            if self.stop_ramping.is_set():
                break
            start_duty += duty_step
            self._apply_motor_duty(start_duty)
            self.current_duty = start_duty
            time.sleep(0.05)
            
        if not self.stop_ramping.is_set():
            self._apply_motor_duty(target_duty)
            self.current_duty = target_duty

    def _apply_motor_duty(self, duty: float):
        if self.pin:
            try:
                # gpiozero expects value between 0.0 and 1.0
                self.pin.value = max(0.0, min(1.0, duty / 100.0))
            except Exception as e:
                event_logger.log(f"Error applying motor duty: {e}")

    def get_motor_status(self) -> Dict[str, Any]:
        return {
            "duty_cycle": round(self.current_duty, 2),
            "current_duty": round(self.current_duty, 2),
            "target_duty": round(self.target_duty, 2),
            "voltage": round(3.3 * (self.current_duty / 100.0), 2),
            "is_ramping": self.ramp_thread.is_alive() if self.ramp_thread else False,
            "mock_mode": not HAS_GPIO or self.pin is None,
            "mount_mode": "real"
        }

    def close(self):
        if self.ramp_thread and self.ramp_thread.is_alive():
            self.stop_ramping.set()
            self.ramp_thread.join(timeout=0.2)
        if self.pin:
            try:
                self.pin.value = 0
                self.pin.close()
            except Exception:
                pass
            self.pin = None
