# Does the learned pedal model help?

Matched Monza ablation: same track, start at rest, physics, steering, curvature observations, planner and speed-tracking rules.
Timestep: 0.1 s. Model horizon: 500 m. Planner horizon: 500 m.

| Controller | Time (s) | Off-track events | Valid lap |
|---|---:|---:|---|
| planner_only | 208.7 | 0 | True |
| hybrid | 205.4 | 0 | True |

Planner-only never calls a trained predictor: neutral pedal proposals leave all acceleration and braking to the shared rule-based target-speed tracker.
The hybrid adds the saved model's pedal suggestions before those same overrides. No model was retrained.
The hybrid was 3.3 s faster on this matched run. This is a limited simulator result, not proof of generalization.

These are deterministic single-condition tests on Monza, not statistical confidence estimates or unseen-circuit validation.
The clean laps demonstrate the full controller's behavior, not independently learned corner speeds.
Next learning experiment: imitate successful simulator target speeds/accelerations, with separate rollout/circuit evaluation. Teacher imitation is not independent discovery.
