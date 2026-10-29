# Task 4 F1 telemetry handoff

For a two-page summary, read Task4_Telemetry_Overview.pdf. TELEMETRY_OVERVIEW.md contains the editable version.

This component provides telemetry extraction and three saved Max Verstappen lap exports from 2025 Monza qualifying. Person 5 can load the CSVs immediately for coordinate alignment. A Formula 1 account is not required to read the saved files or run the offline demo.

The package includes code, tests, saved data, metadata and plots. It covers one circuit and is a starting dataset for integration; the project's multiple-circuit training and held-out-circuit evaluation dataset is still to be assembled.

## Files

| File | Purpose |
| --- | --- |
| telemetry.py | Required extract_lap interface, multi-lap helper, cleaning, exports and plotting |
| demo_telemetry.py | Loads and validates the three saved exports offline; includes load_lap_csv helper |
| check_csv.py | Checks the six-field schema and numeric ranges of an individual CSV |
| test_telemetry.py | Eight offline tests of extraction logic with synthetic fixtures |
| requirements.txt | Dependency versions for live extraction, plotting and tests |
| data/monza_three_new_laps/ | Three CSVs, matching metadata and plots, manifest and data-quality notes |
| START_HERE.md | Short entry point and fresh-download sign-in guidance |

## Run the saved-data demo

Extract the package and open a terminal in its folder. Use your Python 3 executable:

```bash
python demo_telemetry.py
```

Use `python3` if your system names its Python executable that way. No package installation, network request or account sign-in is needed for this demo. It reopens the CSVs and checks exact fields, numeric ranges, strictly increasing time, lap duration coverage, metadata consistency, distinct lap identities and SHA-256 hashes against the supplied manifest. Native-stream gap statistics come from the supplied metadata; original upstream streams are not bundled and cannot be independently rechecked by this demo.

## Included data

| Lap | Time in seconds | Samples | Largest reported native gap in seconds |
| --- | ---: | ---: | ---: |
| 5 | 79.455 | 590 | 1.240 |
| 11 | 79.422 | 591 | 1.039 |
| 14 | 78.923 | 593 | 1.260 |

Every CSV has exactly these columns:

```text
time,x,y,speed,throttle,brake
```

Time is seconds since lap start. X and Y are metres in the F1 telemetry source frame. Speed is m/s. Throttle is a fraction in [0,1], and brake is a binary indicator in {0,1}. X/Y need alignment with the team's Track coordinate system before projection. Steering angle and brake pressure are not supplied.

Keep each lap separate and retain its metadata. A stable lap ID is the filename stem, such as `2025_Monza_Q_VER_lap14`. All three laps belong to the same circuit and session. Keep this identity when constructing datasets; random row splits do not evaluate unseen-circuit generalization.

## Load records for Person 5

The saved-data route uses Python's standard library only:

```python
from pathlib import Path
from demo_telemetry import load_lap_csv

root = Path(__file__).resolve().parent
lap_data = load_lap_csv(
    root / "data" / "monza_three_new_laps" / "2025_Monza_Q_VER_lap14.csv"
)
# Person 5's functions, once available:
# aligned = align_coordinates(lap_data, track)
# track_records = convert_to_track_coordinates(aligned, track)
```

Run that example from a script in this handoff folder. If the assembler changes package layout, adjust import paths and data paths to match. Each record is a dictionary with the six CSV fields converted to floats. The helper validates the schema and returns list[dict].

The exact team download interface is also available:

```python
from telemetry import extract_lap

lap_data = extract_lap(2025, "Monza", "Q", "VER")
```

That call downloads or reads cached FastF1 data and returns one accepted lap as list[dict]. It does not load a particular bundled CSV and does not save files. It requires FastF1 and access to its data. Use the saved-data helper above when assembling the existing exports offline.

## Quality policy difference

The saved exports were created using a maximum gap limit of 1.3 seconds, as recorded in every metadata JSON and the data README. Their longest reported native gaps are up to 1.26 seconds. FastF1 merges and interpolates car and position streams and creates interpolated lap-boundary samples. Public position observations are approximate, not precise racing-line ground truth.

The supplied telemetry.py retains the original MAX_GAP = 1.0 policy. Therefore an ordinary call to extract_lap or extract_laps does not reproduce the three bundled lap selections: these laps exceed that original native-gap limit. Keep this difference visible during integration. Loading a saved CSV checks it as a saved record set; it does not make it pass the stricter extraction policy. The included tests cover the original 1.0-second policy.

The maximum gaps after merging are approximately 1.006 seconds for lap 5, 0.892 seconds for lap 11 and 1.030 seconds for lap 14. Preserve this information when aligning or training, and make an explicit team decision about acceptable gaps. The original per-lap metadata and manifest have been copied unchanged.

## Tests and dependencies

To use the downloader, regenerate plots or run the extraction tests, install the dependencies in your project's environment:

```bash
python -m pip install -r requirements.txt
python -m unittest -v test_telemetry.py
```

All eight extraction tests passed when this handoff was prepared. They use synthetic fixtures and a mocked session. They verify conversion and software behavior, not the upstream accuracy of the saved real exports. The saved-data demo separately checks the three existing files.

If adding this folder to the simulator repository, merge the dependency requirements with the team's environment. The simulator's root requirements.txt also contains pygame and pytest; keep those dependencies. Retain the handoff's requirements.txt inside its own folder instead of replacing the simulator's root file.

## Fresh downloads

Reading the included files requires no account. A new FastF1 request can require your own Formula 1 sign-in:

```bash
python -m fastf1 auth f1tv --authenticate
python telemetry.py --year 2025 --circuit Monza --session Q --driver VER
```

The unchanged script uses its original 1.0-second limit for new exports. This package does not include account credentials, tokens, a Python environment or a FastF1 cache. No new remote download was performed when preparing this handoff.

## Integration responsibilities

Person 4 provides these source-frame lap records. Person 5 estimates the coordinate transform from a full trajectory and produces track-relative position, offset and heading error. Person 6 constructs model features and labels from aligned records and track geometry. The team still needs laps from additional circuits for the stated cross-circuit objective.
