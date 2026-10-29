# F1 AI Simulator

A 2D F1 driving simulator with a player car, a telemetry-trained AI car, a
rule-based baseline car, and a replay car following a recorded 2025 Monza lap.

## Install

From the repository root:

```powershell
python -m pip install -r requirements.txt
```

The telemetry handoff is an independent optional component. To run its
FastF1 extraction tests or download new telemetry, install its dependencies
from `Task4_F1_Telemetry_Handoff/requirements.txt` as well. Loading and
validating the bundled CSVs does not require FastF1.

## Run the three-car Monza demo

```powershell
python -m f1_simulator
```

The app trains its throttle/brake model from the three bundled VER qualifying
laps at startup. The pygame window then opens with car selection:

- **You** drive with the arrow keys or WASD.
- **AI** uses the learned throttle/brake model and the baseline steering
  controller.
- **BASELINE** is controlled entirely by the rule-based driver.
- **VER telemetry** replays the fastest bundled lap as a translucent ghost.

Press `R` to reset, `C` to choose another car sprite, and `Esc` to quit.
`python -m f1_simulator.demo` is an equivalent entry point.
The camera follows your car and is zoomed in; each visible car has a name tag.

The Monza route is built from recorded telemetry coordinates, using the
driver's racing line as a reference path and estimated 10 m road widths. It is
an approximation, not an official circuit map or surveyed track boundary. The
AI is trained only from three laps of this same circuit; this demo does not
claim generalization to unseen circuits.

Run the standalone detailed-physics plot demo with:

```powershell
python -m examples.physics_demo
```

It writes `physics_demo.png` to the current working directory.

## Tests

Run the simulator, controller, physics, and evaluation tests from the
repository root:

```powershell
python -m pytest -q
```

Run the separate telemetry test suite from its own folder:

```powershell
Set-Location Task4_F1_Telemetry_Handoff
python -m unittest -v test_telemetry
```

## Package structure

```text
f1_simulator/
  core/          Track geometry, telemetry track, and race sessions
  physics/       Detailed and prototype vehicle models
  controllers/   Baseline, AI adapter, and imitation-learning utilities
  evaluation/    Controller run loop, metrics, and comparison
  data/          Telemetry-to-track alignment
  ui/            Pygame rendering and car selection
  utils/         Shared asset helpers
  assets/        Bundled car sprites
  demo.py        Interactive application
  __main__.py    `python -m f1_simulator` entry point
examples/        Standalone physics demonstration
tests/
  core/          Track and simulator tests
  physics/       Physics contract and behavior tests
  controllers/   Controller tests
  evaluation/    Metrics tests
  integration/   Cross-module tests
Task4_F1_Telemetry_Handoff/
                 Independent telemetry extractor, tests, docs, and sample data
```

## Track and AI limitations

The simulator currently derives its Monza route from the bundled Verstappen
telemetry and estimates equal widths on both sides. The replay shows the
recorded telemetry; the human and AI cars use the derived route for steering
and off-track checks. The lap coordinates overlap at their boundary, so the
loader trims the duplicate lead-in before closing the track path.

The AI learns longitudinal controls (throttle and brake) from the three
available laps. Steering labels are not present in the telemetry, so steering
comes from the baseline controller. There are not yet multiple circuits,
official track geometry, or held-out-circuit validation.

## Imports and physics models

Use package imports in new code:

```python
from f1_simulator.controllers.baseline import baseline_action
from f1_simulator.core.track import Track
from f1_simulator.evaluation import evaluate_controller
from f1_simulator.physics import update_car
```

The race sessions use `f1_simulator.physics.update_car`, the detailed
bicycle-model implementation. `f1_simulator.physics.prototype.update_car`
remains available as the separate compact model.
Existing root-level imports (`physics`, `baseline`, `ai_controller`, and
`evaluation`) and the former `f1_simulator.core` physics/imitation paths remain
as compatibility shims.

The detailed vehicle model is not calibrated F1 dynamics. The Monza route and
road widths are estimated from telemetry and are not an official circuit map.
