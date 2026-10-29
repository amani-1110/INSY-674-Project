"""Helpers for reading and normalizing bundled telemetry lap records."""

import csv
from pathlib import Path

import numpy as np


def load_closed_lap_csv(path):
    """Load a lap and trim an overlapping leading sample near its finish point."""
    with Path(path).open(newline="", encoding="utf-8") as file:
        rows = [
            {key: float(value) for key, value in row.items()}
            for row in csv.DictReader(file)
        ]
    if len(rows) < 20:
        raise ValueError(f"Telemetry lap has too few samples: {path}")

    finish = np.array([rows[-1]["x"], rows[-1]["y"]])
    overlap_count = max(1, min(len(rows) // 10, 30))
    start_candidates = np.array(
        [[row["x"], row["y"]] for row in rows[:overlap_count]]
    )
    distances = np.linalg.norm(start_candidates - finish, axis=1)
    overlap_index = int(np.argmin(distances))
    if overlap_index > 0 and distances[overlap_index] <= 5.0:
        rows = rows[overlap_index:]

    start_time = rows[0]["time"]
    for row in rows:
        row["time"] -= start_time
    return rows
