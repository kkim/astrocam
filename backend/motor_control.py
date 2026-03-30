try:
    from gpiozero import PWMOutputDevice
    HAS_GPIO = True
except (ImportError, RuntimeError):
    HAS_GPIO = False
    print("GPIO not available, motor control will run in MOCK MODE")

class MotorController:
    def __init__(self, pin=18):
        self.pin_number = pin
        self.pin = None
        self.duty_cycle = 0.0
        self.mock_mode = not HAS_GPIO
        self._init_motor()

    def _init_motor(self):
        if not self.mock_mode:
            try:
                self.pin = PWMOutputDevice(self.pin_number)
                self.pin.value = 0
            except Exception as e:
                print(f"Failed to initialize GPIO: {e}")
                self.mock_mode = True

    def set_speed(self, duty_pct):
        """Sets the motor speed as a percentage (0-100)"""
        self.duty_cycle = max(0.0, min(100.0, duty_pct))
        value = self.duty_cycle / 100.0
        
        if not self.mock_mode and self.pin:
            self.pin.value = value
        
        print(f"Motor speed set to {self.duty_cycle}% ({3.3 * value:.2f}V)")
        return True

    def get_status(self):
        return {
            "duty_cycle": self.duty_cycle,
            "voltage": round(3.3 * (self.duty_cycle / 100.0), 2),
            "mock_mode": self.mock_mode,
            "pin": self.pin_number
        }

    def close(self):
        if self.pin:
            self.pin.value = 0
            self.pin.close()
