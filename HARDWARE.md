# Hardware Configuration: Equatorial Mount Control
This document tracks the physical wiring and specifications for the telescope motor control system.

## Control Module
- **Module:** AOD4184 (D4184) N-Channel MOSFET.
- **Why:** Logic-level compatible (fully turns on at 3.3V), high efficiency (~11mΩ), and remains cool under load.
- **Control Signal:** Hardware PWM (Pulse Width Modulation).
- **Target Frequency:** 100Hz - 1000Hz (To be tuned).

## Raspberry Pi Wiring Map (Standard Non-Inverting)
| Component Pin | Connection | RPi Physical Pin | Notes |
| :--- | :--- | :--- | :--- |
| **Gate (G)** | GPIO 18 (PWM) | Pin 12 | Use 220Ω resistor in series for protection. |
| **Source (S)** | GND | Pin 6 | Must share common ground with Pi. |
| **Drain (D)** | Motor (-) | N/A | Connects to the negative terminal of the motor. |
| **Pull-down** | 10K Resistor | N/A | Connect between **Gate** and **Ground** to keep motor OFF at boot. |

## Power & Motor Wiring (Load Side)
- **Motor (+):** Connect directly to **9V External Power (+)**.
- **Motor (-):** Connect to MOSFET **Drain**.
- **9V Power (-):** Connect to MOSFET **Source** (and RPi GND).
- **Flyback Diode:** Place a 1N4007 diode across motor terminals (Cathode to 9V+) to prevent voltage spikes.

## Tested Performance Findings (March 2026)
- **Start-up Threshold:** The motor begins rotation at approximately **70% duty cycle (~2.3V)**. Duty cycles below this may hum but lack the torque to overcome static friction.
- **Speed Control:** Verified stable variable speed control using PWM on GPIO 18.
- **Incremental Precision:** Responds clearly to fine increments (as small as 1%) in the 80% to 100% range.
- **Circuit Stability:** The AOD4184 MOSFET remains cool and provides consistent switching logic (Active HIGH).

## Software Requirements
- **Primary Library:** `gpiozero` (Used for initial validation and PWM control).
- **Alternative Library:** `pigpio` (Recommended for high-precision hardware PWM if jitter becomes an issue).
- **Logic:** 
  - **3.3V (1.0):** Motor ON (Full speed).
  - **0.0V (0.0):** Motor OFF.
