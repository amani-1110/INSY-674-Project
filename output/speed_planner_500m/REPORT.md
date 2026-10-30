# Hybrid speed planner benchmark

## Three stages of driving: assistance versus learned control

These are three system configurations, not a controlled ablation of assistance
alone. Stages 1/2 share the original pedal model; stage 3 uses a new full-action
policy and different training labels. Times are measured at 0.1 s simulation steps.

### 1. Without speed assistance

The original 500 m model proposes throttle and brake without the speed planner.
Rule-based steering, exclusive pedals and a launch rule remain active. This is
not fully unassisted AI driving.

![Without speed assistance](../analysis_pack/media/monza_first_departure_1x.gif)

**Track: Monza**  
**Completion time: 91.8 s - invalid (13 departures).**  
**Track: Silverstone - not tested in this configuration.**

### 2. With assistance

The original learned pedals are combined with hand-written steering, launch
handling and a curvature/braking speed planner. The planner can override pedals.

![With assistance](../analysis_pack/media/monza_same_turn_with_planner_1x.gif)

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

![Learned full-action policy](../analysis_pack/media/monza_full_action_turn_1x.gif)

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


## Original assisted-pedal benchmark

Same model weights, exclusive pedal policy, steering, track, physics, start and timestep.

| Controller | Lap time (s) | Off-track events | Valid lap |
|---|---:|---:|---|
| corrected_imitation | 91.8 | 13 | False |
| hybrid | 205.4 | 0 | True |

The hybrid uses dense 5 m curvature samples up to 500 m, a lateral-acceleration speed limit and backward braking constraints.
Planner limits override learned pedals; acceleration toward the target can override learned braking. Deviations trigger a 12 m/s cap.
This is a hybrid controller, not a retrained model. Both runs retain the original steering.
An off-track completion is invalid. These are exploratory simulator results on Monza, not unseen-circuit validation.

## Smaller timestep check

At 0.05 s steps the same hybrid completed in 204.6 s with 0 off-track events (valid: True).


## What the AI learned, and what the planner supplied

The Random Forest learned to predict throttle and brake labels from recorded
qualifying telemetry. It did not learn the hybrid controller's target corner
speeds. Those targets are calculated by a hand-written curvature/braking planner
with a conservative lateral-acceleration limit of 4 m/s^2 and a 45 m/s top-speed
cap. For sufficiently tight corners, that rule can set a target near 30 km/h.
A separate rule-based controller supplies steering. Pedal arbitration and the
low-speed launch rule are also hand-written.

The planner can override either learned pedal prediction. Therefore, the valid
205.4 s lap demonstrates the performance of the combined hybrid controller in
this simulator; it is not evidence that the Random Forest independently learned
safe cornering or that the learned component caused the improvement. The
corrected learned-pedal controller with the same steering recorded 13 departures.
The smaller-timestep check supports consistency between these two tested
settings, not a general guarantee of reliability.

A matched planner-only ablation is reported separately in
../planner_ablation_500m/REPORT.md. In the initial matched Monza run, both
controllers completed valid laps: planner-only took 208.7 s and the hybrid
205.4 s. The learned pedal proposals provided a modest 3.3 s (about 1.6%) lap-time
benefit in that condition. This does not establish independently learned corner
speeds or unseen-circuit generalization; the rule-based planner supplied the
same speed limits to both drivers.

## Could a model learn cornering speeds?

The original pedal model did not independently learn safe cornering. A follow-up full-action policy now learns steering and pedals from simulated expert labels and completes both tracks without runtime driving helpers; see ../learned_driver/REPORT.md. This is imitation of an expert, not independent discovery of optimal cornering. One
next experiment is to train a model to predict target speed or acceleration
from curvature, speed and vehicle state using successful simulator driving as
teacher data. Learning from this planner would be imitation of a rule-based
teacher, not independent discovery of its speed limits. Closed-loop rollout
and recovery-state examples would help address the mismatch between recorded
qualifying states and the states visited by the simulator.

A later reinforcement-learning experiment could optimize simulator actions
using forward progress, lap completion and track-departure penalties. It would
require careful evaluation against shortcuts and other reward exploits; success
is not assured. More real sessions and circuits are also needed to distinguish
track memorization from transferable driving behavior. Separate circuits must
be reserved for evaluation, and valid lap rate and speed should both be reported.
