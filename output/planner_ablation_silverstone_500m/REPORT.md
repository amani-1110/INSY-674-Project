# Does the learned pedal model help?

Matched Silverstone ablation: same track, start at rest, physics, steering, curvature observations, planner and speed-tracking rules.
Model trained on Monza only. Test circuit unseen to model: True. No retraining or planner tuning.
Timestep: 0.1 s. Model horizon: 500 m. Planner horizon: 500 m.

| Controller | Time (s) | Off-track events | Valid lap |
|---|---:|---:|---|
| planner_only | 258.3 | 0 | True |
| hybrid | 256.5 | 0 | True |

Planner-only never calls a trained predictor: neutral pedal proposals leave all acceleration and braking to the shared rule-based target-speed tracker.
The hybrid adds the saved model's pedal suggestions before those same overrides. No model was retrained.
The hybrid was 1.8 s faster on this matched run. This is a limited simulator result, not proof of generalization.

These are deterministic single-condition tests, not statistical confidence estimates or broad evidence of generalization.
The planner can query the evaluation circuit's geometry; unseen means excluded from model training, not hidden from the planner.
No real telemetry from the evaluation circuit is used. This is a closed-loop simulator test, not held-out real-driver prediction accuracy.
The clean laps demonstrate the full controller's behavior, not independently learned corner speeds.
Next learning experiment: imitate successful simulator target speeds/accelerations, with separate rollout/circuit evaluation. Teacher imitation is not independent discovery.
