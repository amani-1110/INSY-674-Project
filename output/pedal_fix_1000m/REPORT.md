# Pedal arbitration benchmark

Saved 1000 m model; model weights, steering, track, physics and timestep unchanged.

| Controller | Lap time (s) | Completed | Off-track events | Valid lap |
|---|---:|---|---:|---|
| Original raw pedals | timeout | False | 0 | False |
| Exclusive pedals + launch | 94.7 | True | 11 | False |

This fixes conflicting pedals, not steering or speed planning. An off-track completion remains invalid.
Below 5 m/s, near the centerline on gentle road, throttle is at least 0.4 and brake is zero.
Otherwise brake predictions below 0.5 are ignored; predictions at/above 0.5 select braking with zero throttle.
The brake value is a regression of binary labels, not calibrated brake pressure. Threshold 0.5 is a controller rule.
controls.csv records model outputs and executed actions at every timestep. No model retraining was needed.
