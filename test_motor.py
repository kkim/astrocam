from gpiozero import PWMOutputDevice
from time import sleep

# Testing GPIO 18 (PWM Pin)
pin = PWMOutputDevice(18)

# Sequence: 85-90-95-94-93-92-90-91-92-95
sequence = [85, 90, 95, 94, 93, 92, 90, 91, 92, 95]

try:
    print(f"Running custom sequence: {sequence}")
    print("Each step will last 2 seconds.")
    
    for duty_pct in sequence:
        duty = duty_pct / 100.0
        print(f"Duty Cycle: {duty_pct}% ({3.3 * duty:.2f}V)")
        pin.value = duty
        sleep(2.0)
            
    print("Sequence complete.")
            
except KeyboardInterrupt:
    print("\nStopped.")
finally:
    pin.value = 0
    pin.close()
