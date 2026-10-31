# Driving stages comparison

## Three stages of driving: assistance versus learned control

These are three system configurations, not a controlled ablation of assistance
alone. Stages 1/2 share the original pedal model; stage 3 uses a new full-action
policy and different training labels. Times are measured at 0.1 s simulation steps.

### 1. Without speed assistance

The original 500 m model proposes throttle and brake without the speed planner.
Rule-based steering, exclusive pedals and a launch rule remain active. This is
not fully unassisted AI driving.

![Without speed assistance](media/monza_first_departure_1x.gif)

**Track: Monza**  
**Completion time: 91.8 s - invalid (13 departures).**  
**Track: Silverstone - not tested in this configuration.**

### 2. With assistance

The original learned pedals are combined with hand-written steering, launch
handling and a curvature/braking speed planner. The planner can override pedals.

![With assistance](media/monza_same_turn_with_planner_1x.gif)

**Track: Monza**  
**Completion time: 205.4 s - valid (0 departures).**  
**Track: Silverstone**  
**Completion time: 256.5 s - valid (0 departures).**

### 3. AI-learned steering and pedals - no runtime driving assistance

**Implemented and tested.** A neural network predicts steering, and a learned
ExtraTrees regressor predicts a signed pedal command (positive throttle,
negative brake). The runtime has no corner-speed planner, steering formula,
launch override, target-speed tracker or teacher fallback. Only actuator-range
clipping and conversion of the signed command to throttle/brake remain.

![Learned full-action policy](media/monza_full_action_turn_1x.gif)

**Track: Monza**  
**Completion time: 209.1 s - valid (0 departures).**  
**Track: Silverstone**  
**Completion time: 266.3 s - valid (0 departures).**

Training changed: 40,000 perturbed Monza simulator states were labelled by the
planner-only expert, including low-speed and recovery situations absent from
qualifying telemetry. Steering labels now exist because the teacher is simulated.
This is full-action expert imitation, **not reinforcement learning** or independent
discovery of corner-speed limits. The policy learns the teacher's behavior; the
teacher does not drive the car during inference. No Silverstone states were used
for training. Dataset-aggregation code is available if rollouts fail; this model
completed cleanly on its first fit, so no aggregation refit was performed.

The evaluator replaces planner, rule-steering and pedal-cleanup entry points
with functions that raise errors. Both learned-policy benchmarks completed with
these guards active and the saved model hash unchanged.

All clips are independent 1x recorded-control replays near the same Monza turn,
not synchronized races. A shorter invalid completion is not a valid improvement.
This is a limited deterministic simulator test, not proof of real-car driving.

Run the third stage:

```powershell
python -m f1_simulator.demo --track monza --controller learned
python -m f1_simulator.demo --track silverstone --controller learned
```

