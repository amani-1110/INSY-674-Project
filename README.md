# F1 simulator and spline track geometry

## Install and run

From the repository root (Python 3.10 or newer):

```powershell
python -m pip install -r requirements.txt
python -m pytest -q
python visualize_track.py monza
python -m f1_simulator.demo --track monza
```

The plot is saved to `output/monza_track.png`. The interactive pygame simulator
opens a separate window. Select a car with 1-4; use arrows/WASD to drive, R to
reset, C to choose again, and Escape to quit. `--track demo` uses a synthetic
ellipse. The simulator now calls the shared `physics/physics.py` implementation
through `f1_simulator/core/physics.py`; geometry itself has no physics dependency.

## Shared Track API

```python
from track import Track
# Existing package imports also work:
# from f1_simulator.core.track import Track

track = Track("monza")
print(track.length)                  # about 5790.693805 metres for this dataset
x, y = track.position(500.0)
heading = track.heading(500.0)        # radians, CCW from positive x
curvature = track.curvature(500.0)    # 1/metres, positive for left turns
left, right = track.boundaries(500.0)
left_width = track.left_width(500.0)
right_width = track.right_width(500.0)
s, offset = track.project(x, y)       # metres; offset positive to the left
```

All queries wrap `s % track.length`, including negative distances. Coordinates
use Cartesian x right, y up; dimensions are metres, angles radians. Boundaries
are always returned **(left, right)** relative to the input point order, which
defines travel direction. `Track()` retains the synthetic demo for compatibility;
use `Track("monza")` explicitly for real geometry. Unknown circuits raise an error.
No source file is downloaded at runtime.

## Spline and numerical methods

The track uses SciPy's periodic cubic interpolant for x and y, initially indexed
by cumulative input chord distance. That parameter is **not** treated as arc
length. Each segment's speed `sqrt(x'(u)^2 + y'(u)^2)` is integrated with 16-point
Gauss-Legendre quadrature. Cumulative integrals define lap length; Brent's method
inverts the integral within the appropriate segment for each `position(s)` query.
Tests compare integration with independent adaptive quadrature and verify
unit-speed sampling. These are numerical approximations, not symbolic exactness.

Heading comes from `atan2(y', x')`; signed curvature is
`(x' y'' - y' x'') / (x'^2 + y'^2)^(3/2)`. Widths use positive, periodic piecewise
linear interpolation through the same source stations; linear interpolation
avoids cubic width overshoot. A repeated closing row is optional. Invalid inputs,
nonpositive widths, consecutive duplicate points and degenerate tangents are
rejected. Custom data must describe a sensible simple circuit; the constructor
does not certify arbitrary road geometry.

Projection uses a KD-tree nearest input point as an upper bound and conservative
Bezier segment bounding boxes to discard only segments that cannot improve it.
For every remaining segment it evaluates endpoints and all real roots of the
quintic squared-distance derivative. The result is on the **continuous spline**,
not a nearest sampled coordinate. At crossings or equally close branches the
geometrical nearest point can be ambiguous; temporal continuity belongs in the
alignment/session module.

## Add another circuit

Copy a TUM `tracks/<Circuit>.csv` file to `data/<lowercase_name>.csv`, retain its
licence and provenance, then use `Track("lowercase_name")`. Do not use a file
from TUM's `racelines` folder as a centerline.

Alternatively, load any path:

```python
track = Track.from_csv("path/to/circuit.csv")
reversed_track = Track.from_csv("path/to/circuit.csv", reverse=True)
```

Two CSV schemas are accepted (optional UTF-8 BOM, optional TUM `#` header):

```text
# x_m,y_m,w_tr_right_m,w_tr_left_m
```

or

```text
x,y,left_width,right_width
```

TUM puts **right before left**. The loader remaps them explicitly. Rows are
preserved in source order. `reverse=True` reverses travel while preserving the
first point and swaps left/right widths so physical road boundaries stay put.
The loader does not guess a circuit's orientation or rescale coordinates.
For in-memory data, use
`Track(name="custom", track_points=[(x, y, left_width, right_width), ...])`.

```powershell
python visualize_track.py --csv path/to/circuit.csv --output output/custom.png
python visualize_track.py monza --reverse --output output/monza_reverse.png
python -m f1_simulator.demo --track-csv path/to/circuit.csv
```

## Monza source, attribution and limitations

`data/monza.csv` is the **unmodified** TUMFTM racetrack-database
`tracks/Monza.csv`, pinned at commit
`e59595d1f3573b30d1ded6a08984935b957688e0`:
https://github.com/TUMFTM/racetrack-database/tree/e59595d1f3573b30d1ded6a08984935b957688e0

The upstream README specifies metre coordinates and metre right/left widths.
These are local planar coordinates, not latitude/longitude. The supplied files
do not specify a geodetic CRS, global origin or verified official timing-line
station. We preserve the axes and row order. `s=0` is the dataset's first point
and the simulator's finish location, not a claim about an official timing line.
No north-reference conversion is applied; this API's heading is always measured
from positive x. TUM's helper normal convention confirms widths are defined
relative to the tangent/travel direction.

TUM reports that centerlines originate from OpenStreetMap GPS points and are
smoothed, while widths are estimated from satellite imagery. Smoothing can move
the reference line away from the exact middle of the road; accuracy varies.
The measured spline length is about 5790.69 m and is not forced to a nominal
circuit length. Geometry is not a surveyed map or a guarantee of the 2025 layout.
Do not assume its coordinate frame matches FastF1: telemetry needs alignment.
Source attribution includes TUMFTM, Alexander Heilmeier, Andressa de Paula Suiti
(width extraction), and OpenStreetMap contributors.

Upstream licence: LGPL-3.0, preserved in `data/TUM_LICENSE` with the incorporated GPL in `data/GPL-3.0.txt`; the upstream README
is `data/TUM_README.md`. `data/monza.source.json` records the source commit,
original filename, unchanged status and SHA-256. Upstream data remains under
its own licence. OpenStreetMap attribution: https://www.openstreetmap.org/copyright

## Validation and project integration

`test_track.py` covers closure, seam continuity, wrapping, finite outputs,
quadrature accuracy, heading, straight/circle curvature, width mapping,
reversal, boundary sides, signed projection and global nearest-point comparisons.
It also checks centerline and each boundary for proper self-intersections in a
1600-point sampling. This catches obvious crossings, not every possible
sub-sample defect or arbitrary custom-track error. The visualization shows
centerline curvature, both boundaries, direction arrows, dataset start and
signed projection examples with equal aspect ratio.

The full suite also checks the actual simulator with shared physics: baseline
lap completion on the demo track and a stationary controller timeout. These
checks do not establish trained AI performance or a valid autonomous Monza lap.
Evaluation imports use the real simulator; tests no longer inject a fake global
`simulator` module. Imitation exposes the agreed `train_model` and
`predict_action` names as aliases for existing `train` and `predict` functions.
Fresh telemetry extraction requires FastF1; tests use offline fixtures and never
request Formula 1 credentials or download sessions.


## Run the complete offline pipeline

```powershell
python run_pipeline.py
python -m f1_simulator.demo --track monza --controller baseline
python -m f1_simulator.demo --track monza --controller imitation --model output/pipeline_250m/imitation_model.joblib
```

The single pipeline command verifies the saved telemetry hashes, loads TUM Monza,
fits a transform on lap 5 and reuses it across the same session, then exports
aligned and track-relative records and overlay plots. Laps 5 and 11 train the
model, with grouped cross-validation between those training laps. Lap 14 remains
held out and never trains the saved model. Lap identity is attached before
combining records. No sign-in, live download or Formula 1 account is required.

The saved model is reloaded and connected to the baseline steering controller;
only throttle/brake differ between the drivers. Both start from rest under the
same physics, track, timestep and timeout. Results distinguish completion from a
valid lap (zero off-track events); failed laps have no lap time and only two
valid laps can produce a valid time-improvement claim.

Artifacts in `output/pipeline_250m/`:

- `REPORT.md`: readable results and limitations.
- `run_manifest.json`: sources, split, hashes and experiment identity.
- `lap*_aligned.csv`, `lap*_track_relative.csv`, `lap*_overlay.png`: handoffs.
- `alignment.json`: fitted transform and checks per lap.
- `imitation_model.joblib`: model trained only on training laps.
- `validation.json`: grouped validation and held-out metrics.
- `benchmark.json` and `benchmark.csv`: simulator results and comparison.

Change the output directory, timestep, attempt timeout or held-out lap with
`--output`, `--dt`, `--max-time` and `--held-out-lap`. Defaults are 0.1 s steps,
600 simulated seconds per attempt and lap 14 held out. Reruns replace artifacts
in the selected output directory. Use different output directories to retain
experiments. The visual demo uses real-time steps and is for inspection; the
fixed-timestep benchmark is the comparable measurement.

This is a single-session, held-out-lap experiment. It does **not** satisfy the
unseen-circuit generalization goal: more circuit geometry and telemetry are
still needed. Accurate pedal predictions do not guarantee stable driving,
especially when starting from rest outside the qualifying-speed training data.
A failed/off-track AI run remains a reported experimental result.


## Camera and live demo performance

The demo defaults to a follow camera at 5 pixels/metre, so a typical 12 m road
is about 60 pixels wide and the car has a visible position inside its boundaries.
A minimap shows the full circuit and car location. Press **V** to switch between
follow and full-circuit overview, and **+/-** to adjust follow zoom. Use
`--view overview` to start with the whole circuit. The HUD identifies manual or
AI control; keyboard driving controls apply only in manual mode.

The renderer caches spline centerline/boundary samples once per track. Moving
the follow camera transforms those cached points; it does not reintegrate the
track geometry every frame. The camera changes the display only, preserving
physical track widths, model inputs and simulator measurements. Ground grid
markers and the moving minimap marker make progress visible on long straights.
Restart a running demo after updating the source to load these fixes.


## 250 m imitation model

The pipeline now defaults to a 250 m pedal-model horizon, sampled at
`[0, 10, 20, 40, 60, 80, 100, 150, 200, 250]` metres. The simulator's default
three-sample observation contract remains available for existing teammates.
The AI controller requests the offsets recorded inside its model; both the
benchmark and visual demo use those exact offsets, including wraparound.
Saved models carry their feature names, target names and offsets, which are
validated when loaded. Original 20 m models remain loadable.

```powershell
python run_pipeline.py --lookahead 250 --output output/pipeline_250m
python -m f1_simulator.demo --controller imitation --model output/pipeline_250m/imitation_model.joblib
python -m f1_simulator.demo --controller imitation --model output/pipeline/imitation_model.joblib
```

The last command loads the preserved original 20 m model. To train a fresh
20 m run, use `python run_pipeline.py --lookahead 20 --output output/pipeline_20m`.
The existing `output/pipeline/` experiment is retained for comparison.
Default pipeline output and demo model now point to `output/pipeline_250m/`.

The steering rule, physics, starting state and timestep are unchanged. Extending
curvature features retrains the Random Forest's throttle/brake predictions; it
does not add a braking-distance planner or change steering. Report valid laps
and off-track counts alongside lap times. A longer horizon alone is not a
promise of the previously reported 88.7 s/zero-off-track result.


## Compare 500 m and 1,000 m models

```powershell
python run_pipeline.py --lookahead 500
python run_pipeline.py --lookahead 1000
python compare_horizons.py
python -m f1_simulator.demo --controller imitation --model output/pipeline_500m/imitation_model.joblib
python -m f1_simulator.demo --controller imitation --model output/pipeline_1000m/imitation_model.joblib
```

Without `--output`, the pipeline chooses `output/pipeline_<horizon>m/`, keeping
experiments separate. The default horizon remains 250 m. The 500 m model retains
all 250 m samples and adds 300, 350, 400, 450 and 500 m. The 1,000 m model also
adds 600, 700, 800, 900 and 1,000 m. Near-field detail, model hyperparameters,
training/test split, steering and simulation settings are preserved.

`compare_horizons.py` checks matching saved source/split/timestep conditions and
writes `output/lookahead_comparison.md` and `.json`. The original 20 m run is
read from `output/pipeline/`. This is an exploratory Monza comparison; choosing
among these models does not establish generalization to a new circuit.


## Exclusive pedals and launch behavior

The AI controller now defaults to `exclusive_v1` pedal arbitration. Below
5 m/s, within 3 m of the centerline, heading error within 0.5 rad, and nearby
curvature no greater than 0.05 1/m, it supplies at least 40% throttle and no
brake. This handles launching from rest without pretending the model learned
standing starts. Outside that case, brake predictions below 0.5 are discarded;
at/above 0.5 the controller brakes with zero throttle. Brake regression values
come from binary telemetry labels and are not calibrated pressure estimates.
These rules are controller changes; the model weights and steering are unchanged.

Restart the demo to load the policy. The HUD now shows executed throttle/brake:

```powershell
python -m f1_simulator.demo --controller imitation --model output/pipeline_1000m/imitation_model.joblib
python benchmark_pedal_policy.py --horizon 1000
```

The benchmark uses the saved model and matched original timestep/timeout,
checks its hash against the original experiment and saves results separately in
`output/pedal_fix_1000m/`, including a per-step `controls.csv`. Historical
horizon results remain raw-pedal experiments; `compare_horizons.py` rejects
mixed pedal policies. The policy used by new pipeline benchmarks is recorded
in `benchmark.json`. To reproduce the old behavior programmatically, use
`make_ai_controller(model, predict_action, pedal_policy="raw")`.

Removing pedal overlap fixes acceleration, but is not a guarantee of valid laps
or a replacement for a speed-aware braking planner. The same limited Monza
training data and approximate physics still apply.


## Hybrid curvature speed planner

```powershell
python benchmark_speed_planner.py --horizon 500
python -m f1_simulator.demo --controller hybrid --model output/pipeline_500m/imitation_model.joblib
```

The hybrid retains the trained model and original steering. A dense 500 m
curvature preview, sampled every 5 m, estimates corner speed from
`v = sqrt(a_lateral / abs(curvature))`. Backward constraints use
`v_now^2 <= v_corner^2 + 2*a_brake*distance` and a 15 m buffer. Default limits
are 4 m/s^2 lateral acceleration, 5 m/s^2 braking deceleration and 45 m/s maximum
speed. These are conservative controller assumptions, not calibrated F1 limits.
If lateral offset exceeds 2 m or heading error exceeds 0.2 rad, the target is
capped at 12 m/s to help the unchanged steering recover.

Above target speed, the planner brakes and suppresses throttle. When more than
1 m/s below target it can override learned braking and supply acceleration;
otherwise learned pedal suggestions remain bounded. Pedals stay exclusive.
This is a **hybrid learned/rule-based driver**, not a retrained neural network
or proof that the learned model alone improved. The model is a Random Forest.

Track curvature for the planner is cached once. A controller-specific observer
interpolates that cache while supplying the model its original curvature inputs;
the default shared simulator observation API remains unchanged. The cache wraps
at the lap boundary. The benchmark and visual demo use the same observer.

Results and per-step speed/target/pedal traces are stored separately in
`output/speed_planner_500m/`. The comparison uses the same saved 500 m model,
exclusive pedal policy, steering, physics, start and timestep for both runs.
Historical raw-pedal horizon comparisons are not substituted for that control.
Off-track laps remain invalid. Monza performance is an exploratory simulator
result and does not establish unseen-circuit generalization.


## Does the learned pedal model add value?

```powershell
python benchmark_planner_ablation.py --horizon 500
python -m f1_simulator.demo --controller planner
```

The benchmark compares the existing hybrid with a planner-only driver, writing
`output/planner_ablation_500m/benchmark.json`, `REPORT.md` and pedal/speed traces.
Both use the same curvature planner, steering, exclusive pedal/launch rules,
track, start, physics, timestep and speed-tracking overrides. Planner-only uses
neutral throttle/brake proposals and never queries a learned model. The hybrid
uses the saved Random Forest proposals before those shared rules. Matching
curvature observations are used for the benchmark. Model weights are unchanged.

This ablation asks whether learned pedal proposals improve the complete driver;
it does not train a model to infer corner speeds. Clean laps alone cannot answer
that question. A difference measured on one deterministic Monza condition is
not a statistical estimate or evidence of unseen-circuit generalization.


## Transfer test on an unseen circuit

```powershell
python benchmark_planner_ablation.py --track silverstone --horizon 500
python visualize_track.py silverstone --output output/silverstone_track.png
python -m f1_simulator.demo --track silverstone --controller hybrid --model output/pipeline_500m/imitation_model.joblib
python -m f1_simulator.demo --track silverstone --controller planner
```

`data/silverstone.csv` is the unmodified TUM `tracks/Silverstone.csv`, pinned to
the same commit as Monza (`e59595d1f3573b30d1ded6a08984935b957688e0`). Its
`silverstone.source.json` preserves attribution, licence and SHA-256. Existing
TUM LGPL/GPL files apply. Width columns and coordinate conventions use the same
loader. The measured spline length is about 5887.37 m; no nominal-length scaling
or assertion of current surveyed geometry is applied.

The benchmark freezes the Monza-trained 500 m model, original steering, pedal
policy and planner parameters. No Silverstone telemetry is used for training,
alignment or validation. Only the planner queries the new track geometry.
Both controllers receive matched geometry observations on Silverstone. This
measures transfer of the learned pedal contribution inside the hybrid, not
model-only driving or blind discovery of a track.

Results, source/code hashes, unchanged-model checks and controls are saved in
`output/planner_ablation_silverstone_500m/`. The model hash must match its
original training manifest and is checked again after evaluation. An optional
`--output` keeps additional experiments separate. No parameters should be tuned
on this test circuit when reporting it as a held-out evaluation.


The initial frozen-model Silverstone test completed cleanly for both drivers:
planner-only 258.3 s, hybrid 256.5 s (1.8 s / about 0.7% benefit). This is a
small transfer benefit in one deterministic simulator condition, not proof of
independently learned corner speeds. See `output/transfer_comparison.md` for
comparison with Monza and `output/planner_ablation_silverstone_500m/REPORT.md`
for test provenance and limitations.


## Rubric-aligned analysis and media pack

```powershell
python screen_model_families.py
python export_analysis_pack.py
```

The exploratory screen compares a mean baseline, Ridge, Random Forest, RBF SVR
and a small neural network, using two-fold CV grouped by the two training laps.
Scalers are fitted inside folds. It does not overwrite the driving models.

`output/analysis_pack/ANALYSIS.md` maps the evidence and remaining work to the
course rubric. `GALLERY.html`/`GALLERY.md` and `CAPTIONS.csv` organize native
simulator screenshots, 8 GIFs, 11 charts, data profiles and model-CV results.
`PRESENTATION_STORYBOARD.md` suggests a 10-minute storyline. Corner station
labels are dataset distances, not official corner names.

Media use saved-control replays through the actual physics/simulator. The
exporter verifies speed agreement, duration, lap completion and departure counts
against the saved benchmark. Full-lap GIFs are nominal 10x playback; close-up
turn and departure clips are 1x. Curvature-selected turns and the first departure
have explicit selection rules, recorded in `manifest.json`. Pillow is needed
for GIF export. The ZIP is `output/analysis_pack.zip`.

Pending rubric work includes more independent sessions/laps, training-only
hyperparameter tuning, an explicit ensemble comparison and a fresh final test.
Repeatedly inspected lap 14 is exploratory validation, not a pristine final test.


## Working full-action learned policy (stage 3)

```powershell
python train_full_driver.py
python benchmark_learned_driver.py --track monza
python benchmark_learned_driver.py --track silverstone
python -m f1_simulator.demo --controller learned
python -m f1_simulator.demo --track silverstone --controller learned
```

This is a separately trained policy, not the earlier real-telemetry pedal model.
A tanh MLP (32/32 hidden units) predicts steering from lateral/heading error and
current curvature. A 100-tree ExtraTrees model predicts a signed pedal command
from speed, absolute pose errors and 101 curvature samples through 500 m. The
actor has no runtime speed planner, steering formula, launch rule, brake-threshold
cleanup or expert fallback. Actuator clipping and positive-throttle/negative-brake
action encoding are the only output transformations. Geometry remains an observed
input, and the same simulator physics/track-limit rules apply.

Training uses 40,000 perturbed Monza simulator states with action labels from the
planner-only expert, including launch and recovery examples. It is **behavior
cloning / expert imitation, not reinforcement learning**. A dataset-aggregation
loop collects corrective labels if a student rollout fails; this trained policy
completed cleanly on its first fit, so zero aggregation refits were used. Learning
an expert's conservative behavior is not independent discovery of corner speeds.
No Silverstone states or telemetry were used for this training.

Frozen-policy benchmarks with forbidden-helper guards recorded:

- Monza: 209.1 s, zero departures, valid (0.1 s timestep).
- Silverstone: 266.3 s, zero departures, valid (same frozen model).
- Monza at 0.05 s timestep: 208.9 s, zero departures, valid.

The learned policy is slightly slower than the assisted hybrid, but its actions
are all learned. This is a comparison of systems with different training data,
not a controlled assistance-only ablation. Detailed provenance and results are
in `output/learned_driver/REPORT.md`, `training.json`, and each track's benchmark.
Do not infer real-car reliability or broad generalization from these runs.

Media/report refresh after the benchmarks:

```powershell
python export_learned_driver.py
python update_driving_stages.py
```

The three-stage report and gallery now include real full-action learned-policy
GIFs rather than a future-work placeholder. `--policy` selects a different full
policy artifact; it is separate from `--model`, which selects older pedal models.


## Project documentation

See [What We Built and Learned](DOCUMENTATION.md) for the plain-language project narrative, three driving stages, rubric sections, measured results, limitations and reproduction commands. The [Overleaf package](output/F1_Project_Overleaf.zip) includes the LaTeX source, print-friendly comparison cards and numbered frames for optional PDF animation.

Read the [visual project story](output/analysis_pack/PROJECT_STORY.html) for the minimal card-based narrative. The [presentation plan](output/analysis_pack/PRESENTATION_STORYBOARD.md) provides section guidance and speaker cues. [Detailed evidence](PROJECT_EVIDENCE.md) is available separately. Rebuild this version with `python build_concise_documentation.py`.


## Complete driving benchmark

All eight saved driving configurations have now been run on Monza, Silverstone and Spa under matched start, physics, timestep and timeout settings. See [all 24 attempts](output/all_configurations/REPORT.md). Steering, pedal model and speed assistance are explicit; lap outcomes are separate from offline pedal-prediction error. These are expanded development tests on already inspected circuits.

## Submission readiness

See [rubric checklist and final presentation actions](RUBRIC_CHECKLIST.md) for the evidence, evaluation caveats and remaining submission checks.
