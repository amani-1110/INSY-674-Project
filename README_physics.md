# physics.py — Vehicle physics (Person 2)

Moves an F1 car in an empty 2D plane. No track needed.

```python
from f1_simulator.physics import update_car
new_state = update_car(
    {"x": 120.0, "y": 80.0, "heading": 0.0, "speed": 20.0},   # m, m, rad, m/s
    {"steering": 0.0, "throttle": 1.0, "brake": 0.0},         # rad (+ = left), 0..1, 0..1
    0.1)                                                      # seconds
# -> {'x': 122.05, 'y': 80.0, 'heading': 0.0, 'speed': 21.07}
```

## Model (kinematic bicycle + simple F1 forces)
1. **Speed:** `a = (drive - brake - drag - rolling) / mass`
   - drive = throttle × min(power / v, rear-tyre grip). Power-limited at speed, grip-limited at launch.
   - brake = brake × tyre grip. Grip = μ·(weight + downforce), so braking is stronger at high speed (as in F1).
   - drag = ½·ρ·CdA·v². This sets the top speed.
2. **Turning:** yaw rate = v·tan(δ)/L (bicycle model, wheelbase L = 3.6 m).
   If the steering would need more sideways grip than the tyres have, δ is reduced, so the car runs wide.
3. **Position:** integrated exactly along the arc for the step.

## Assumptions (in `CarParams`)
| Parameter | Value | Effect |
|---|---|---|
| Mass | 800 kg | car + driver |
| Power | 750 kW | ~1000 hp |
| μ (tyre) | 1.7 | warm slicks |
| CdA / ClA | 1.08648 / 4.0 m² | drag / downforce |
| Max steer | 0.40 rad | steering lock |
| Maximum speed | 372.6 km/h (103.5 m/s) | hard speed limit |

Resulting behaviour (vs real F1): 0–100 km/h 2.7 s (≈2.6), top speed 372.6 km/h,
200→0 km/h in ~60 m, ~4 g cornering at 250 km/h.

## Limitations
- No tyre slip, sliding or spinning: the car never loses control, it only runs wide.
- Longitudinal and lateral grip aren't combined (no friction circle), so braking while turning is too generous.
- No gears, no weight transfer, no fuel or tyre wear, no reverse.
- Parameters are public estimates, not calibrated to a specific car.

## Tests
`pytest -q` runs 17 tests: contract keys, reproducibility, straight-line accel/top speed/braking,
constant-steering circle radius = L/tan(δ), left/right sign convention, grip limit, heading wrapping.
