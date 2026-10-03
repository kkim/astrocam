import threading
import time
from typing import Dict, Any, Optional

class MockMount:
    """
    Simulates mount motor control with speed ramping and duty cycle state.
    """
    def __init__(self, initial_duty: float = 80.0):
        self.current_duty: float = float(initial_duty)
        self.target_duty: float = float(initial_duty)
        self.ramp_thread: Optional[threading.Thread] = None
        self.stop_ramping = threading.Event()

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
            self.current_duty = target_duty
            return

        duty_step = (target_duty - start_duty) / steps
        for _ in range(steps):
            if self.stop_ramping.is_set():
                break
            start_duty += duty_step
            self.current_duty = start_duty
            time.sleep(0.05)

        if not self.stop_ramping.is_set():
            self.current_duty = target_duty

    def get_motor_status(self) -> Dict[str, Any]:
        return {
            "duty_cycle": round(self.current_duty, 2),
            "current_duty": round(self.current_duty, 2),
            "target_duty": round(self.target_duty, 2),
            "voltage": round(3.3 * (self.current_duty / 100.0), 2),
            "is_ramping": self.ramp_thread.is_alive() if self.ramp_thread else False,
            "mock_mode": True,
            "mount_mode": "mock"
        }

    def close(self):
        if self.ramp_thread and self.ramp_thread.is_alive():
            self.stop_ramping.set()
            self.ramp_thread.join(timeout=0.2)
