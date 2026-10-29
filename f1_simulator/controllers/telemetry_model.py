"""Train the demo controller from the bundled telemetry laps."""

from pathlib import Path

from ..data.alignment import convert_to_track_coordinates
from ..data.lap_records import load_closed_lap_csv
from .imitation import prepare_training_data, train


_LAP_DATA = (
    Path(__file__).resolve().parents[2]
    / "Task4_F1_Telemetry_Handoff"
    / "data"
    / "monza_three_new_laps"
)


def load_telemetry_laps(folder=_LAP_DATA):
    """Load the three bundled real-lap CSVs with stable lap identities."""
    laps = []
    for lap_number in (5, 11, 14):
        path = Path(folder) / f"2025_Monza_Q_VER_lap{lap_number:02d}.csv"
        rows = load_closed_lap_csv(path)
        laps.append((lap_number, rows))
    return laps


def train_telemetry_model(track, folder=_LAP_DATA):
    """Train throttle/brake prediction on bundled same-circuit telemetry.

    Track-relative features are calculated against the telemetry-derived
    reference line. This trains for the available Monza laps only; it is not
    evidence of performance on a previously unseen circuit.
    """
    records = []
    for lap_number, lap in load_telemetry_laps(folder):
        aligned = convert_to_track_coordinates(lap, track)
        for record in aligned:
            record["lap_id"] = lap_number
        records.extend(aligned)

    features, targets = prepare_training_data(records, track)
    if not features:
        raise ValueError("No training samples could be prepared from telemetry.")
    return train(features, targets, n_estimators=100, random_state=0)
