# F1 AI — Simulator and Visualization

Standalone prototype of Part 3, using a temporary elliptical track and a simplified bicycle-model physics module. Replace `track.py` and `physics.py` with the team's modules once ready; preserve their documented APIs.

## Run on your Windows computer

Open the extracted folder in VS Code and run in the terminal:

```powershell
python -m pip install -r requirements.txt
python demo.py
```

Use arrows/WASD to drive, R to reset, Escape to quit. The pygame application opens in a separate window; it is not designed to render inside Jupyter.

## Tests

```powershell
python -m pytest -q
```

The simulator exports `reset_simulation`, `simulate_step`, `get_observation`, and `render`. Race state is maintained between calls. Units are metres, seconds, m/s, and radians. The demo track is synthetic, not Monza. The physics model is a temporary approximation, not calibrated F1 vehicle dynamics.

## Car selection
Run `python demo.py`. Choose one of four cars by clicking or pressing 1–4. Press C while driving to choose again. Car images are in `assets/` and rotate with the vehicle heading.
