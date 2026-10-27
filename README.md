# F1 AI — Simulator and Visualization

Standalone prototype of Part 3, using a temporary elliptical track and a simplified bicycle-model physics module. Replace `track.py` and `physics.py` with the team's modules once ready; preserve their documented APIs.

## Run on your Windows computer

Open the repository root in VS Code and run in the terminal:

```powershell
python -m pip install -r requirements.txt
python -m f1_simulator.demo
```

Use arrows/WASD to drive, R to reset, Escape to quit. The pygame application opens in a separate window; it is not designed to render inside Jupyter.

## Tests

```powershell
python -m pytest -q
```

Run this command from the repository root, rather than running the test file
directly. `pytest -q` also works. In VS Code, use the Testing panel to discover
and run tests; select a Python interpreter with the requirements installed.

The `core.simulator` module provides `reset_simulation`, `simulate_step`, and `get_observation`; `ui.renderer` provides `render`. Race state is maintained between calls. Units are metres, seconds, m/s, and radians. The demo track is synthetic, not Monza. The physics model is a temporary approximation, not calibrated F1 vehicle dynamics.

## Car selection
Run `python -m f1_simulator.demo` from the repository root. Choose one of four cars by clicking or pressing 1–4. Press C while driving to choose again. Car images are in `f1_simulator/assets/` and rotate with the vehicle heading.

## Project structure

```text
INSY-674-Project/
├── README.md
├── requirements.txt
├── pytest.ini
├── f1_simulator/
│   ├── __init__.py
│   ├── demo.py                # Application entry point and controls
│   ├── core/
│   │   ├── __init__.py
│   │   ├── physics.py         # Vehicle movement
│   │   ├── track.py           # Track geometry
│   │   └── simulator.py       # Simulation steps and race state
│   ├── ui/
│   │   ├── __init__.py
│   │   ├── renderer.py        # Track, car, and HUD drawing
│   │   └── car_selection.py   # Car selection screen
│   ├── utils/
│   │   ├── __init__.py
│   │   └── assets.py          # Shared sprite loading
│   └── assets/               # Car images
└── tests/
    └── test_simulator.py
```

Run all commands above from the repository root. Import modules using the
`f1_simulator` package, for example `from f1_simulator.core.track import Track`.
