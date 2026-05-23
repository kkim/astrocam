import threading
import time

try:
    from gpiozero import PWMOutputDevice
    HAS_GPIO = True
except (ImportError, RuntimeError):
    HAS_GPIO = False
    # Using event_logger here would create a circular dependency with motor_control importing logger
    # and logger being imported by main. Let's keep this as print for now as it's an init error.
    print("GPIO not available, motor control will run in MOCK MODE")

class MotorController:
    def __init__(self, pin=18):
        self.pin_number = pin
        self.pin = None
        self.current_duty = 0.0
        self.target_duty = 0.0
        self.mock_mode = not HAS_GPIO
        self.ramp_thread = None
        self.stop_ramping = threading.Event()
        self._init_motor()

    def _init_motor(self):
        if not self.mock_mode:
            try:
                self.pin = PWMOutputDevice(self.pin_number)
                self.pin.value = 0
            except Exception as e:
                # This needs to be a print, not event_logger.log, to avoid circular dependency
                # as motor_control is instantiated before event_logger is fully ready in some contexts.
                print(f"Failed to initialize GPIO: {e}") 
                self.mock_mode = True

    def set_speed(self, duty_pct, ramp_time=0.5):
        """Sets the motor speed with optional ramping."""
        self.target_duty = max(0.0, min(100.0, duty_pct))
        
        # Stop any existing ramping
        if self.ramp_thread and self.ramp_thread.is_alive():
            self.stop_ramping.set()
            self.ramp_thread.join()
        
        self.stop_ramping.clear()
        self.ramp_thread = threading.Thread(target=self._ramp_logic, args=(ramp_time,), daemon=True)
        self.ramp_thread.start()
        return True

    def _ramp_logic(self, ramp_time):
        start_duty = self.current_duty
        end_duty = self.target_duty
        steps = 20
        delay = ramp_time / steps
        
        for i in range(1, steps + 1):
            if self.stop_ramping.is_set():
                break
            
            # Linear interpolation
            self.current_duty = start_duty + (end_duty - start_duty) * (i / steps)
            value = self.current_duty / 100.0
            
            if not self.mock_mode and self.pin:
                self.pin.value = value
            
            time.sleep(delay)
        
        self.current_duty = end_duty
        # Final log for the target
        # print(f"Motor reached {self.current_duty}%")

    def get_status(self):
        return {
            "duty_cycle": round(self.current_duty, 2),
            "target_duty": self.target_duty,
            "voltage": round(3.3 * (self.current_duty / 100.0), 2),
            "mock_mode": self.mock_mode,
            "pin": self.pin_number
        }

    def close(self):
        self.stop_ramping.set()
        if self.ramp_thread:
            self.ramp_thread.join()
        if self.pin:
            self.pin.value = 0
            self.pin.close()
