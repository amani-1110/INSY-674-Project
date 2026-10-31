# Monza pipeline results

Training laps: [5, 11]. Held-out lap: 14.
Track length: 5790.694 m. Timestep: 0.1 s. Timeout: 600.0 s.

| Controller | Completed | Valid (no off-track events) | Lap time (s) | Off-track events |
|---|---|---|---|---|
| Baseline | True | True | 228.09999999999133 | 0 |
| Imitation | True | False | 86.19999999999938 | 14 |

Held-out throttle MAE: 0.0666.
Held-out brake MAE: 0.0800.

Both drivers use the same steering function, physics, start and timestep.
Any off-track event makes the lap invalid for performance claims; failed laps
have no lap time. A timeout is reported explicitly. No artificial recovery or
teleport is used. The saved model was trained only on the training laps.

Limitations: these laps share a session and circuit. This does not test unseen
circuits or establish real-world driving performance. Source positions are
approximate; some aligned samples fall outside the estimated road. Public brake
labels are binary, not measured brake pressure. Regression outputs are continuous
control commands. The simulator starts at rest, outside qualifying-speed training
conditions. The physics has no combined friction circle, tyre wear or calibrated
car-specific dynamics. A 20 m curvature horizon can be inadequate at high speed.

See alignment.json, validation.json, benchmark.json, run_manifest.json and the
per-lap CSVs/overlays for the underlying results. Load joblib models only from
trusted sources.
