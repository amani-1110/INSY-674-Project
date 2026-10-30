# Monza-trained model transfer to Silverstone

The same saved 500 m model, planner parameters, steering and physics were used. No Silverstone training data or parameter tuning was used.

| Circuit | Role | Planner-only (s) | Hybrid (s) | Hybrid gain | Departures |
|---|---|---:|---:|---:|---|
| Monza | Training circuit | 208.7 | 205.4 | 3.3 s (1.6%) | 0 / 0 |
| Silverstone | Unseen circuit | 258.3 | 256.5 | 1.8 s (0.7%) | 0 / 0 |

The learned pedal proposals retained a small lap-time benefit on this one unseen circuit while both drivers completed clean laps.
This is limited evidence of transfer within the hybrid controller, not proof that the model learned cornering speeds independently.
The planner had access to Silverstone geometry and set the speed limits; steering remained rule-based.
No held-out real-driver prediction accuracy was measured on Silverstone. Both tests are deterministic single-condition simulator runs, not statistical estimates.
Broader claims require additional circuits and conditions, kept separate from training and controller tuning.

Run: python benchmark_planner_ablation.py --track silverstone --horizon 500
Watch: python -m f1_simulator.demo --track silverstone --controller hybrid --model output/pipeline_500m/imitation_model.joblib
