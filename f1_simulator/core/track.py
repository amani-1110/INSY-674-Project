"""Closed spline track geometry. Distances and widths are in metres."""
import csv
from pathlib import Path
import math

import numpy as np
from scipy.spatial import cKDTree
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq


class Track:
    """Periodic cubic centerline with a numerically integrated arc-length parameterization.

    Supply ordered (x, y, left_width, right_width) control points. A repeated
    closing point is optional. The default remains a synthetic demo ellipse;
    circuit names alone never stand in for real circuit data.
    """

    def __init__(self, name="demo", track_points=None, reverse=False):
        if track_points is None:
            if name != "demo":
                if not name or Path(name).name != name or name in (".", ".."):
                    raise ValueError("Use a circuit name or Track.from_csv(path)")
                source = Path(__file__).resolve().parents[2] / "data" / (name.lower() + ".csv")
                if not source.is_file():
                    raise ValueError(f"No bundled circuit {name!r}; use Track.from_csv(path)")
                track_points = self._read_csv(source)
            theta = np.linspace(0, 2 * np.pi, 48, endpoint=False)
            if track_points is None:
                track_points = np.column_stack((155 * np.cos(theta),
                                               105 * np.sin(theta),
                                               np.full(48, 10.), np.full(48, 10.)))
        data = np.asarray(track_points, dtype=float)
        if data.ndim != 2 or data.shape[1] != 4 or not np.isfinite(data).all():
            raise ValueError("Expected finite (x, y, left_width, right_width) rows")
        if len(data) > 1 and np.array_equal(data[0, :2], data[-1, :2]):
            if not np.array_equal(data[0, 2:], data[-1, 2:]):
                raise ValueError("Closing point widths must match the first point")
            data = data[:-1]
        if len(data) < 4 or np.any(data[:, 2:] <= 0):
            raise ValueError("Need at least four points with positive widths")
        if reverse:
            data = data[np.r_[0, np.arange(len(data) - 1, 0, -1)]].copy()
            data[:, [2, 3]] = data[:, [3, 2]]
        closed = np.vstack((data, data[0]))
        chords = np.linalg.norm(np.diff(closed[:, :2], axis=0), axis=1)
        if np.any(chords <= 1e-9):
            raise ValueError("Consecutive track points must be distinct")
        self.name = name
        self._knots = np.r_[0., np.cumsum(chords)]
        self._curve = CubicSpline(self._knots, closed[:, :2], bc_type="periodic")
        self._widths = closed[:, 2:]
        self._gauss_x, self._gauss_w = np.polynomial.legendre.leggauss(16)
        self._arc = np.r_[0., np.cumsum([
            self._integrate_speed(a, b) for a, b in zip(self._knots[:-1], self._knots[1:])])]
        self.length = float(self._arc[-1])
        # Bezier control hulls enclose every cubic segment, giving rigorous
        # bounding boxes for pruning the global closest-point search.
        h = np.diff(self._knots)
        coeff = self._curve.c * h[None, :, None] ** np.array([3, 2, 1, 0])[:, None, None]
        self._coeff = coeff[::-1].transpose(1, 0, 2)  # ascending powers of t in [0,1]
        c0, c1, c2, c3 = self._coeff.transpose(1, 0, 2)
        controls = np.stack((c0, c0+c1/3, c0+2*c1/3+c2/3, c0+c1+c2+c3))
        self._bbox_min, self._bbox_max = controls.min(axis=0), controls.max(axis=0)
        # Reject a cusp even if it lies between the input samples: find all
        # minima of squared tangent magnitude on each cubic.
        for c in self._coeff:
            derivative = np.arange(1, 4)[:, None] * c[1:]
            squared = np.zeros(5)
            for dim in range(2):
                poly = np.polynomial.polynomial.polymul(derivative[:, dim], derivative[:, dim])
                squared[:len(poly)] += poly
            critical = np.polynomial.polynomial.polyroots(
                np.arange(1, len(squared)) * squared[1:])
            samples = [0., 1.] + [r.real for r in critical
                                  if abs(r.imag) < 1e-9 and 0 < r.real < 1]
            if min(np.linalg.norm(np.polynomial.polynomial.polyval(t, derivative))
                   for t in samples) < 1e-8:
                raise ValueError("Track has a degenerate tangent; revise control points")
        self._tree = cKDTree(data[:, :2])
        # Dense polyline retained for rendering and compatibility with existing callers.
        self.s = np.linspace(0, self.length, 2401)
        self.points = np.array([self.position(s) for s in self.s])
        self.seg = np.diff(self.points, axis=0)

    @staticmethod
    def _read_csv(path):
        """Accept TUM's commented right/left header or our named left/right header."""
        with open(path, newline="", encoding="utf-8-sig") as stream:
            lines = [line.strip() for line in stream if line.strip()]
        if not lines:
            raise ValueError("Track CSV is empty")
        header = lines[0].lstrip("# ")
        rows = csv.DictReader([header] + [line for line in lines[1:] if not line.startswith("#")])
        names = set(rows.fieldnames or [])
        if {"x_m", "y_m", "w_tr_right_m", "w_tr_left_m"} <= names:
            fields = ("x_m", "y_m", "w_tr_left_m", "w_tr_right_m")
        elif {"x", "y", "left_width", "right_width"} <= names:
            fields = ("x", "y", "left_width", "right_width")
        else:
            raise ValueError("Unrecognized track CSV header; expected TUM or x,y,left_width,right_width")
        return [[float(row[key]) for key in fields] for row in rows]

    @classmethod
    def from_csv(cls, path, name=None, reverse=False):
        """Load Cartesian metre coordinates; preserve order unless reverse=True."""
        return cls(name=name or Path(path).stem, track_points=cls._read_csv(path), reverse=reverse)

    def _integrate_speed(self, a, b):
        mid, half = (a + b) / 2, (b - a) / 2
        velocity = self._curve(mid + half * self._gauss_x, 1)
        return float(half * np.dot(self._gauss_w, np.linalg.norm(velocity, axis=1)))

    def _parameter(self, s):
        s = float(s)
        if not math.isfinite(s):
            raise ValueError("Track distance must be finite")
        distance = s % self.length
        i = min(int(np.searchsorted(self._arc, distance, side="right") - 1), len(self._knots)-2)
        a, b = self._knots[i:i+2]
        target = distance - self._arc[i]
        return float(brentq(lambda u: self._integrate_speed(a, u) - target,
                            a, b, xtol=1e-10))

    def position(self, s):
        return tuple(map(float, self._curve(self._parameter(s))))

    def heading(self, s):
        dx, dy = self._curve(self._parameter(s), 1)
        return math.atan2(dy, dx)

    def curvature(self, s):
        u = self._parameter(s)
        dx, dy = self._curve(u, 1)
        ddx, ddy = self._curve(u, 2)
        return float((dx * ddy - dy * ddx) / (dx * dx + dy * dy) ** 1.5)

    def left_width(self, s):
        return float(np.interp(self._parameter(s), self._knots, self._widths[:, 0]))

    def right_width(self, s):
        return float(np.interp(self._parameter(s), self._knots, self._widths[:, 1]))

    def boundaries(self, s):
        x, y = self.position(s)
        h = self.heading(s)
        nx, ny = -math.sin(h), math.cos(h)
        left, right = self.left_width(s), self.right_width(s)
        return ((x + left * nx, y + left * ny), (x - right * nx, y - right * ny))

    def project(self, x, y):
        """Globally nearest point on continuous cubics; signed offset is + left.

        Prune only segments whose Bezier bounding box cannot beat a known
        point, then solve every remaining cubic's distance stationary points.
        At exact crossings multiple nearest positions are geometrically ambiguous.
        """
        p = np.array([x, y], dtype=float)
        if not np.isfinite(p).all():
            raise ValueError("Position must be finite")
        distance, _ = self._tree.query(p)
        delta = np.maximum(np.maximum(self._bbox_min - p, p - self._bbox_max), 0.)
        lower = np.sum(delta * delta, axis=1)
        candidates = np.flatnonzero(lower <= distance * distance + 1e-8)
        best = None
        for i in candidates:
            c = self._coeff[i].copy()
            c[0] -= p
            stationary = np.zeros(6)
            for dim in range(2):
                poly = np.polynomial.polynomial.polymul(c[:, dim],
                                                        np.arange(1, 4) * c[1:, dim])
                stationary[:len(poly)] += poly
            roots = np.polynomial.polynomial.polyroots(stationary)
            ts = [0., 1.] + [float(r.real) for r in roots
                            if abs(r.imag) < 1e-8 and 0 < r.real < 1]
            for t in ts:
                residual = np.polynomial.polynomial.polyval(t, c)
                d2 = float(residual @ residual)
                u = float(self._knots[i] + t * (self._knots[i+1] - self._knots[i]))
                if best is None or (d2, u) < best[:2]:
                    best = (d2, u, int(i))
        _, u, i = best
        s = (self._arc[i] + self._integrate_speed(self._knots[i], u)) % self.length
        cx, cy = self._curve(u)
        dx, dy = self._curve(u, 1)
        norm = math.hypot(dx, dy)
        offset = (-dy * (x-cx) + dx * (y-cy)) / norm
        return float(s), float(offset)
