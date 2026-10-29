"""Approximate circuit geometry derived from a recorded telemetry lap."""

import math
from pathlib import Path

import numpy as np

from ..data.lap_records import load_closed_lap_csv

_DEFAULT_LAP = (
    Path(__file__).resolve().parents[2]
    / "Task4_F1_Telemetry_Handoff"
    / "data"
    / "monza_three_new_laps"
    / "2025_Monza_Q_VER_lap14.csv"
)


class TelemetryTrack:
    """Closed track corridor inferred from an ordered GPS racing-line CSV.

    The recorded racing line is used as the reference path, not as an official
    circuit centreline. Road widths are approximate.
    """

    def __init__(self, points, name="Telemetry-derived Monza", width=10.0,
                 smoothing_m=2.0):
        points = np.asarray(points, dtype=float)
        if points.ndim != 2 or points.shape[1] != 2 or len(points) < 20:
            raise ValueError("Track points must contain at least 20 x/y samples.")
        if not np.isfinite(points).all():
            raise ValueError("Track points must contain only finite coordinates.")
        if width <= 0 or smoothing_m < 0:
            raise ValueError("Track width must be positive and smoothing nonnegative.")

        keep = np.r_[True, np.linalg.norm(np.diff(points, axis=0), axis=1) > 1e-6]
        points = points[keep]
        if len(points) < 20:
            raise ValueError("Track needs at least 20 distinct x/y samples.")

        closed_points = np.vstack((points, points[0]))
        source_distance = np.r_[
            0.0,
            np.cumsum(np.linalg.norm(np.diff(closed_points, axis=0), axis=1)),
        ]
        raw_length = float(source_distance[-1])
        segment_count = max(200, int(math.ceil(raw_length / 2.0)))
        self.length = raw_length
        self.spacing = self.length / segment_count
        sample_s = np.arange(segment_count) * self.spacing
        sampled = np.column_stack(
            (
                np.interp(sample_s, source_distance, closed_points[:, 0]),
                np.interp(sample_s, source_distance, closed_points[:, 1]),
            )
        )

        if smoothing_m:
            radius = max(1, int(math.ceil(3 * smoothing_m / self.spacing)))
            offsets = np.arange(-radius, radius + 1)
            weights = np.exp(-0.5 * (offsets * self.spacing / smoothing_m) ** 2)
            weights /= weights.sum()
            sampled = sum(
                weight * np.roll(sampled, int(offset), axis=0)
                for offset, weight in zip(offsets, weights)
            )

        self.name = name
        self.width = float(width)
        self.points = np.vstack((sampled, sampled[0]))
        self.seg = np.diff(self.points, axis=0)
        self.s = np.arange(segment_count + 1) * self.spacing

    @classmethod
    def from_csv(cls, path=_DEFAULT_LAP, **kwargs):
        """Create an approximate circuit from an ordered telemetry CSV."""
        rows = load_closed_lap_csv(path)
        points = [(row["x"], row["y"]) for row in rows]
        return cls(points, **kwargs)

    def position(self, s):
        distance = float(s) % self.length
        x = float(np.interp(distance, self.s, self.points[:, 0]))
        y = float(np.interp(distance, self.s, self.points[:, 1]))
        return x, y

    def heading(self, s):
        before = np.asarray(self.position(s - self.spacing))
        after = np.asarray(self.position(s + self.spacing))
        delta = after - before
        return math.atan2(float(delta[1]), float(delta[0]))

    def curvature(self, s):
        before = self.heading(s - 5.0)
        after = self.heading(s + 5.0)
        delta = (after - before + math.pi) % (2 * math.pi) - math.pi
        return float(delta / 10.0)

    def left_width(self, s):
        return self.width

    def right_width(self, s):
        return self.width

    def boundaries(self, s):
        x, y = self.position(s)
        heading = self.heading(s)
        nx, ny = -math.sin(heading), math.cos(heading)
        return (
            (x + self.width * nx, y + self.width * ny),
            (x - self.width * nx, y - self.width * ny),
        )

    def project(self, x, y):
        point = np.array([x, y], dtype=float)
        lengths_squared = np.einsum("ij,ij->i", self.seg, self.seg)
        fractions = np.clip(
            np.einsum("ij,ij->i", point - self.points[:-1], self.seg)
            / lengths_squared,
            0.0,
            1.0,
        )
        projections = self.points[:-1] + fractions[:, None] * self.seg
        distances_squared = np.einsum(
            "ij,ij->i", projections - point, projections - point
        )
        index = int(np.argmin(distances_squared))
        tangent = self.seg[index] / math.sqrt(lengths_squared[index])
        displacement = point - projections[index]
        offset = tangent[0] * displacement[1] - tangent[1] * displacement[0]
        distance = self.s[index] + fractions[index] * self.spacing
        return float(distance % self.length), float(offset)
