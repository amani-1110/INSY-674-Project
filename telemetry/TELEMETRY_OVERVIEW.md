# telemetry.py Person 4 F1 telemetry

Retrieves a qualifying lap and produces clean, synchronized driver records for Person 5's alignment module. The handoff also includes three saved Max Verstappen laps from 2025 Monza qualifying, with metadata and plots. The saved-data demo runs offline without a Formula 1 account.

## Required interface

```python
def extract_lap(year: int, circuit: str,
                session_type: str, driver: str) -> list[dict]:
    ...
```

Returns ordered dictionaries with exactly six fields. Each call returns one accepted lap, selected fastest-first under the extraction quality policy.

| Field | Unit or range | Meaning |
| --- | --- | --- |
| time | seconds | Elapsed time since lap start |
| x and y | metres | Position in the telemetry source frame |
| speed | m/s | Vehicle speed |
| throttle | 0 to 1 | Normalized throttle input |
| brake | 0.0 or 1.0 | Brakes off or on |

## Usage with saved data

Run from the extracted handoff folder. This route uses only Python's standard library.

```python
from demo_telemetry import load_lap_csv

lap_data = load_lap_csv(
    "data/monza_three_new_laps/2025_Monza_Q_VER_lap14.csv"
)
print(lap_data[0])

# Person 5's next steps, after importing their functions:
# aligned = align_coordinates(lap_data, track)
# track_data = convert_to_track_coordinates(aligned, track)
```

## Quick check

```bash
python3 demo_telemetry.py
```

Reopens all three CSVs and checks fields, finite values, ranges, increasing time, full lap-time coverage, metadata consistency and manifest hashes. No installation or sign-in is needed for this check.

## Method

1. Select the driver and qualifying session with FastF1. Keep accurate, non-deleted laps without pit in/out and with green track status; try the fastest eligible candidate first.
2. Check native car and position channels for missing values, boundary coverage and large timestamp gaps. Merge the streams using FastF1's get_telemetry(frequency="original"), which interpolates channels and lap boundaries.
3. Convert km/h to m/s, throttle percent to a 0-1 fraction, position tenths of a metre to metres, and session time to lap-relative seconds. Keep brake binary.
4. Remove missing/nonfinite rows and duplicate times, order samples, validate ranges, and export a six-field CSV with metadata. Save speed/control and XY plots; plotted distance is estimated by integrating speed over time.

<!-- pagebreak -->

## Included laps and verified checks

| Lap | Lap time seconds | Samples | Largest reported native gap seconds |
| --- | --- | --- | --- |
| 5 | 79.455 | 590 | 1.240 |
| 11 | 79.422 | 591 | 1.039 |
| 14 | 78.923 | 593 | 1.260 |

The saved-data demo passes for all three CSVs, covering 1,774 samples. All eight offline extraction tests also pass. They use synthetic fixtures and a mocked FastF1 session to check unit conversion, lap-relative time, cleaning, invalid controls, gap rejection, CSV export and the required function interface.

```bash
python3 -m pip install -r requirements.txt
python3 -m unittest -v test_telemetry.py
```

These checks verify software behavior and the supplied files' consistency. Original upstream streams are not included; native-stream statistics are retained from the metadata. Full integration with Person 5's module has not yet been tested.

## Repository placement and fresh extraction

Suggested main-module location is f1_simulator/data/telemetry.py. Add an empty __init__.py in data/ if missing. Retain the demo, CSV checker, metadata, plots and saved data; adjust their imports and file paths if the assembler moves them into packages. Merge the dependencies with the existing simulator requirements.

```python
# After placing telemetry.py in the shared package:
from f1_simulator.data.telemetry import extract_lap

lap_data = extract_lap(2025, "Monza", "Q", "VER")
```

Fresh extraction needs the listed dependencies and access to FastF1 data; Formula 1 sign-in may be required. This call does not load a bundled CSV or write export files. The standalone command-line script performs exports and plotting.

## Assumptions and limits

- The saved CSVs used a documented 1.3-second gap limit; telemetry.py retains its original 1.0-second limit. Therefore fresh extraction does not reproduce these three lap selections under the unchanged policy. Keep the per-lap quality metadata and agree on the policy before further collection.
- Positions remain in the telemetry frame. Person 5 must align a full trajectory with the corresponding circuit's Track geometry. A temporary ellipse is not a valid Monza map. Track-relative offset, heading error and centreline distance are Person 5's outputs.
- No measured steering angle or brake pressure is supplied. Brake is an on/off indicator, and public positions are approximate. Interpolation does not establish precise positions or control timing across gaps.
- Keep laps and their identities separate; each lap restarts its clock. Only Monza is included so far. Additional circuits and a held-out-circuit evaluation are needed for the project's generalization goal.
- Tests do not establish simulator driving performance. Person 6 builds training inputs and labels from Person 5's aligned records; Person 7 evaluates the resulting controller.
