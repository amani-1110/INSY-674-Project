# Curvature lookahead comparison

Same Monza data, training laps 5/11, held-out lap 14, steering, physics, start and timestep.

| Horizon | Time (s) | Off-track events | Valid lap | Throttle MAE | Brake MAE |
|---|---:|---:|---|---:|---:|
| 20 m | 86.2 | 14 | False | 0.0666 | 0.0800 |
| 250 m | 91.6 | 9 | False | 0.0489 | 0.0599 |
| 500 m | 120.0 | 9 | False | 0.0484 | 0.0565 |
| 1000 m | timeout | 0 | False | 0.0475 | 0.0544 |

A completed circuit with off-track events is invalid. Lap times from invalid laps do not establish a performance gain.
This is an exploratory comparison on one circuit/session, not evidence of unseen-circuit performance.
Near-field curvature samples are retained and additional far-field samples are added; feature count changes with horizon.
The steering controller is unchanged, so this tests extended throttle/brake-model features rather than a new driving planner.
