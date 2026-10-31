# F1 Driving Simulator

A simulator for comparing rule-based driving with models that learn from examples. The main goal is a completed lap with zero road departures, followed by a faster time.

## Run

Python 3.10 or newer. From this folder:

```powershell
python -m pip install -r requirements.txt
python -m f1_simulator --track monza
python -m pytest -q
```

Use arrows/WASD to drive, R to reset, V to change camera and Escape to quit. The demo offers four car sprites. Monza, Silverstone and Spa geometry is bundled; `--track demo` uses a synthetic oval.

```powershell
# Original learned pedals, with rule-based steering
python -m f1_simulator --controller imitation
# Same pedals with a rule-based speed planner
python -m f1_simulator --controller hybrid
# Learned steering and pedals, without the runtime speed planner
python -m f1_simulator --controller learned
# Evaluate the learned driver on another circuit
python -m f1_simulator --track silverstone --controller learned
```

## Structure

```text
f1_simulator/              Application code
  core/                   Track geometry, simulation state and learned policies
  physics/                Shared vehicle dynamics
  controllers/            Rule drivers, speed planning and evaluation
  telemetry/              Recorded-lap extraction and validation
  alignment/              Match recorded coordinates to track geometry
  ui/                     Rendering, cameras and car selection
  assets/                 Four car sprites required by the renderer
  utils/                  Asset loading
scripts/                  Training, model selection and benchmark commands
tests/                   All automated tests
data/                    Track geometry, real telemetry and source licenses
models/                  Six saved models and their hash manifest
```

`output/` is created by experiments and ignored by Git. Pictures, videos, PDFs, ZIPs and presentation-authoring tools are outside the code repository. Keep generated deliverables separate from source code.

## Train and compare

Run these as modules from the project root:

```powershell
# Prepare recorded laps, fit the original pedal model and evaluate it
python -m scripts.run_pipeline --lookahead 500
# Compare initial model families, then tune candidates and try a blend
python -m scripts.screen_model_families
python -m scripts.complete_model_study
# Retrain the separate full-control policy from simulator teacher examples
python -m scripts.train_full_driver
# Benchmark all eight saved configurations across all three circuits
python -m scripts.benchmark_all_configurations --prepare
python -m scripts.benchmark_all_configurations --track monza
python -m scripts.benchmark_all_configurations --track silverstone
python -m scripts.benchmark_all_configurations --track spa
```

The original pipeline and tuning study write new models to their experiment directories under `output/`. Demos and the all-configuration benchmark use the frozen models in `models/`; `--model` and `--policy` can select other files for the interactive demo. Training the full-control driver explicitly replaces `models/full_control.joblib`. The manifest records the bundled versions, before any local retraining.

Other commands include `scripts.benchmark_learned_driver`, `scripts.benchmark_speed_planner`, `scripts.benchmark_planner_ablation` and `scripts.visualize_track`. Use `--help` for their arguments. Inspect the saved telemetry offline with `python -m f1_simulator.telemetry.demo_telemetry`.

## Method and findings

**Frame the problem.** This is offline supervised imitation learning for a student team comparing driving approaches under the same simulated physics. It is not reinforcement learning. The original model learns accelerator and brake labels from real recordings; the full-control policy copies a rule-based simulator teacher.

**Explore and prepare the data.** The bundled recordings contain 1,774 moments from Verstappen’s 2025 Monza qualifying laps 5, 11 and 14. Each row has time (seconds), x/y position (metres), speed (m/s), accelerator (0–1) and brake (binary). Saved exports have no missing fields, but interpolated gaps up to 1.3 seconds limit measurement detail. Lap identity and source hashes accompany the data. Alignment converts positions into lateral/heading errors, and road curvature supplies the view ahead. Laps 5 and 11 supply training examples; lap 14 was inspected during development.

**Measure and compare.** Initial candidates include a mean predictor, Ridge, Random Forest, SVR and a neural network. `scripts/screen_model_families.py` holds out one whole training lap at a time and reports means and sample standard deviations. Scaling stays within each fold. Random Forest, SVR and the neural network form the tuning shortlist to compare tree, kernel and layered nonlinear predictors. Driving evaluation separately measures completion, departure count and time.

**Fine-tune.** `scripts/complete_model_study.py` tries six settings per shortlisted family and an equal Random Forest/SVR blend. Average accelerator/brake prediction error on the two reused selection folds was:

| Candidate | Mean absolute error ± sample SD |
|---|---:|
| Random Forest | 0.0538 ± 0.0011 |
| Equal RF/SVR blend | 0.0545 ± 0.0003 |
| SVR | 0.0600 ± 0.0021 |
| Neural network | 0.0975 ± 0.0009 |

Random Forest was selected before the original Spa evaluation. The separate full-control policy uses a neural network for steering and Extra Trees for a signed pedal command: positive accelerates, negative brakes. Its 40,000 training situations come from a simulator teacher, including starts and recovery states.

The expanded benchmark tested eight complete configurations on three circuits: 21 clean completions, one invalid completion and two timeouts. It compares complete systems with different assistance and training sources, rather than isolating algorithm quality. Reproduce the evidence with the commands above; generated reports and control traces remain under ignored `output/`.

## Assumptions and limits

- Identical starts and physics support consistent system comparisons. One deterministic start per setup does not establish reliability; vary starting conditions next.
- Three laps from one driver/session provide few independent examples. Collect fresh sessions and drivers before claiming generalization.
- Alignment uses lap 5 before the two-fold comparison. Its positions influence calibration even when that lap is held out. Fit calibration within each training fold, or use independent calibration data, for a stricter estimate.
- Tuning reuses the selection folds and lap 14 was repeatedly inspected. These are preliminary selection/development results. The expanded three-circuit benchmark is not a new untouched test.
- Physics and track widths are approximate. A clean lap checks a reference point on the car, rather than official F1 tyre/kerb rules, and does not establish real-car performance. Calibrate physics and validate boundaries before making those claims.

Track data comes from the TUM racetrack database. Retain `data/TUM_LICENSE`, `data/GPL-3.0.txt` and the `*.source.json` files. Telemetry metadata documents FastF1 provenance and interpolation. Model hashes are recorded in `models/manifest.json`.
