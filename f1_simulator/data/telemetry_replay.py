"""Time-synchronized playback of a recorded telemetry lap."""

from pathlib import Path

import numpy as np

from .lap_records import load_closed_lap_csv

_DEFAULT_LAP = (
    Path(__file__).resolve().parents[2]
    / "Task4_F1_Telemetry_Handoff"
    / "data"
    / "monza_three_new_laps"
    / "2025_Monza_Q_VER_lap14.csv"
)


class TelemetryReplay:
    """Interpolate a recorded lap into renderable car states."""

    def __init__(self, records):
        if len(records) < 2:
            raise ValueError("A telemetry replay needs at least two samples.")
        self.times = np.asarray([row["time"] for row in records], dtype=float)
        self.x = np.asarray([row["x"] for row in records], dtype=float)
        self.y = np.asarray([row["y"] for row in records], dtype=float)
        self.speed = np.asarray([row["speed"] for row in records], dtype=float)
        if not (
            np.isfinite(self.times).all()
            and np.isfinite(self.x).all()
            and np.isfinite(self.y).all()
            and np.isfinite(self.speed).all()
        ):
            raise ValueError("Telemetry replay samples must be finite.")
        if np.any(np.diff(self.times) <= 0):
            raise ValueError("Telemetry replay times must increase strictly.")

        dx = np.gradient(self.x)
        dy = np.gradient(self.y)
        self.heading = np.unwrap(np.arctan2(dy, dx))
        self.duration = float(self.times[-1])

    @classmethod
    def from_csv(cls, path=_DEFAULT_LAP):
        return cls(load_closed_lap_csv(path))

    def state_at(self, elapsed_time):
        elapsed = float(np.clip(elapsed_time, self.times[0], self.times[-1]))
        finished = elapsed >= self.duration
        return {
            "x": float(np.interp(elapsed, self.times, self.x)),
            "y": float(np.interp(elapsed, self.times, self.y)),
            "heading": float(np.interp(elapsed, self.times, self.heading)),
            "speed": float(np.interp(elapsed, self.times, self.speed)),
            "finished": finished,
        }
