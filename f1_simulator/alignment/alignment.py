"""
Person 5 — Data alignment: match driver coordinates to the track.

Input  : lap_data from telemetry.py (Person 4), one full lap, ordered by time:
         [{"time", "x", "y", "speed", "throttle", "brake"}, ...]
         x/y are in the telemetry's own frame (OpenF1 or FastF1, ~0.1 m units) —
         the source does not matter because scale/rotation/translation are estimated.
Track  : any object with the shared Track API (Tony's track.py, or mock_track.MockTrack).

Public functions (team contract):
    align_coordinates(lap_data, track)              -> list[dict]  (x, y now in track world metres)
    convert_to_track_coordinates(aligned, track)    -> list[dict]  (time, s, lateral_offset,
                                                                    heading_error, speed, throttle, brake)
Extras for checking the work:
    estimate_transform(lap_data, track)             -> dict (scale, rotation, translation, reflected, rms_m)
    alignment_report(training_data, track)          -> dict of sanity checks
    plot_alignment(aligned, track, path)            -> overlay figure
    load_lap_csv(path)                              -> lap_data from Person 4's CSV

Only these Track members are used: length, position, heading, left_width, right_width, project.

Standalone self-test (no repo needed):   python alignment.py
  -> hides a synthetic lap in another coordinate frame with a KNOWN transform,
     recovers it, prints the checks and saves alignment_overlay.png.
     Uses the team's Track if f1_simulator is importable, otherwise a built-in demo track.

Requires: numpy, scipy, matplotlib (pandas only for load_lap_csv).
"""
import numpy as np

_ROWS = ("time", "x", "y", "speed", "throttle", "brake")


# ----------------------------------------------------------------------------
# Small geometry helpers
# ----------------------------------------------------------------------------
def _wrap_angle(a):
    """Wrap to [-pi, pi)."""
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


def _resample_by_arclength(xy, n, closed=False):
    pts = np.vstack([xy, xy[:1]]) if closed else xy
    d = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
    t = np.linspace(0, d[-1], n, endpoint=not closed)
    return np.column_stack([np.interp(t, d, pts[:, 0]), np.interp(t, d, pts[:, 1])]), d[-1]


def _umeyama(src, dst, allow_scale=True, weights=None):
    """Least-squares similarity transform dst ≈ c·R·src + t (Umeyama 1991), no reflection."""
    w = np.ones(len(src)) if weights is None else weights
    w = w / w.sum()
    mu_s, mu_d = w @ src, w @ dst
    S, D = src - mu_s, dst - mu_d
    cov = (D * w[:, None]).T @ S
    U, sig, Vt = np.linalg.svd(cov)
    E = np.diag([1.0, np.sign(np.linalg.det(U @ Vt))])
    R = U @ E @ Vt
    var_s = w @ np.sum(S**2, axis=1)
    c = float(np.trace(np.diag(sig) @ E) / var_s) if allow_scale else 1.0
    t = mu_d - c * R @ mu_s
    return c, R, t


def _apply(c, R, t, xy, reflect):
    p = xy * np.array([1.0, -1.0]) if reflect else xy
    return (c * (R @ p.T)).T + t


def _lap_arrays(lap_data):
    recs = sorted((r for r in lap_data if np.isfinite(r["x"]) and np.isfinite(r["y"])),
                  key=lambda r: r["time"])
    if len(recs) < 20:
        raise ValueError("Need a substantial trajectory (≥20 valid samples); "
                         "a few points cannot determine the coordinate transform.")
    xy = np.array([[r["x"], r["y"]] for r in recs], dtype=float)
    keep = np.r_[True, np.linalg.norm(np.diff(xy, axis=0), axis=1) > 1e-9]  # drop repeats
    return [r for r, k in zip(recs, keep) if k], xy[keep]


# ----------------------------------------------------------------------------
# 1) Coordinate transform: telemetry frame -> track world frame
# ----------------------------------------------------------------------------
def estimate_transform(lap_data, track, n=360, icp_iters=30, allow_reflection=False,
                       enforce_forward=True):
    """
    Estimate the similarity transform that maps telemetry XY onto the track centreline.

    Step A (global, no initial guess needed): resample both the lap and the centreline by
    arc length, then try every start offset × both directions (× mirror) and solve a
    closed-form Procrustes fit for each. The lowest residual wins.
    Step B (refine): ICP — project transformed points onto the centreline with
    track.project, refit with robust (Huber) weights, repeat. This removes the bias from
    assuming the racing line has the same length as the centreline.
    Assumes lap_data covers (close to) one full lap.

    enforce_forward: the driver travels in the direction of increasing track s (the
        simulator's "forward progress"). Keeps symmetric tracks (e.g. an ellipse) from
        being matched backwards.
    allow_reflection: also try a mirrored telemetry frame. FastF1/OpenF1 frames are not
        mirrored, so this is off by default; turn it on only if the overlay looks flipped.
    """
    _, xy = _lap_arrays(lap_data)
    P, _ = _resample_by_arclength(xy, n, closed=False)
    Q = np.array([track.position(s) for s in np.linspace(0, track.length, n, endpoint=False)])

    best = None
    for reflect in ((False, True) if allow_reflection else (False,)):
        Pr = P * np.array([1.0, -1.0]) if reflect else P
        for direction in ((1,) if enforce_forward else (1, -1)):
            Qd = Q if direction == 1 else Q[::-1]
            for k in range(0, n, 2):
                Qk = np.roll(Qd, -k, axis=0)
                c, R, t = _umeyama(Pr, Qk)
                rms = np.sqrt(np.mean(np.sum((_apply(c, R, t, Pr, False) - Qk) ** 2, axis=1)))
                if best is None or rms < best[0]:
                    best = (rms, c, R, t, reflect)
    _, c, R, t, reflect = best

    # Step B: ICP refinement against the true centreline geometry
    src = xy * np.array([1.0, -1.0]) if reflect else xy
    for _ in range(icp_iters):
        cur = _apply(c, R, t, src, False)
        proj = [track.project(px, py) for px, py in cur]
        tgt = np.array([track.position(s) for s, _ in proj])
        resid = np.linalg.norm(cur - tgt, axis=1)
        delta = max(np.median(resid) * 2.0, 1.0)            # Huber threshold (metres)
        w = np.where(resid <= delta, 1.0, delta / np.maximum(resid, 1e-9))
        c_new, R_new, t_new = _umeyama(src, tgt, weights=w)
        moved = abs(c_new - c) / c + abs(np.arctan2(R_new[1, 0], R_new[0, 0]) - np.arctan2(R[1, 0], R[0, 0]))
        c, R, t = c_new, R_new, t_new
        if moved < 1e-7:
            break

    cur = _apply(c, R, t, src, False)
    proj = [track.project(px, py) for px, py in cur]
    tgt = np.array([track.position(s) for s, _ in proj])
    rms = float(np.sqrt(np.mean(np.sum((cur - tgt) ** 2, axis=1))))
    return {"scale": float(c), "rotation": float(np.arctan2(R[1, 0], R[0, 0])),
            "translation": (float(t[0]), float(t[1])), "reflected": bool(reflect),
            "rms_to_centreline_m": rms, "_R": R}


def align_coordinates(lap_data, track, transform=None):
    """Transform an entire driver's X/Y trajectory into track world coordinates.

    Returns records with the same fields as lap_data; only x and y change (now metres in
    Tony's frame). Pass a precomputed `transform` to reuse one fit across several laps of
    the same session (they share a coordinate frame).
    """
    recs, xy = _lap_arrays(lap_data)
    T = transform or estimate_transform(lap_data, track)
    out_xy = _apply(T["scale"], T["_R"], np.asarray(T["translation"]), xy, T["reflected"])
    aligned = []
    for r, (x, y) in zip(recs, out_xy):
        row = {k: r.get(k) for k in _ROWS}
        row["x"], row["y"] = float(x), float(y)
        aligned.append(row)
    return aligned


# ----------------------------------------------------------------------------
# 2) Track-relative records: s, lateral offset, heading error
# ----------------------------------------------------------------------------
def _local_project(track, x, y, s_guess, window, step=0.5):
    """Closest centreline point searched only near s_guess (avoids jumping to a nearby
    but different part of the circuit, e.g. hairpins or parallel straights)."""
    L = track.length
    cand = np.mod(s_guess + np.arange(-window, window + step, step), L)
    pts = np.array([track.position(s) for s in cand])
    i = int(np.argmin(np.sum((pts - [x, y]) ** 2, axis=1)))
    s = float(cand[i])
    cx, cy = pts[i]
    h = track.heading(s)
    offset = -np.sin(h) * (x - cx) + np.cos(h) * (y - cy)   # + = left of travel direction
    return s, float(offset)


def _driver_heading(xy, smooth_m=8.0):
    """Heading from several consecutive positions: smooth along arc length, then differentiate."""
    from scipy.ndimage import gaussian_filter1d
    step = np.median(np.linalg.norm(np.diff(xy, axis=0), axis=1))
    sigma = max(smooth_m / max(step, 1e-6), 1.0)
    xs = gaussian_filter1d(xy[:, 0], sigma, mode="nearest")
    ys = gaussian_filter1d(xy[:, 1], sigma, mode="nearest")
    return np.arctan2(np.gradient(ys), np.gradient(xs))


def convert_to_track_coordinates(aligned_data, track, max_jump_m=25.0):
    """Project positions to s / lateral offset; estimate driver heading error.

    s is distance along Tony's centreline in [0, track.length). Projection is done
    sequentially: the global track.project result is accepted only if it is consistent
    with the previous sample's s plus distance travelled; otherwise a local search is used.
    """
    xy = np.array([[r["x"], r["y"]] for r in aligned_data], dtype=float)
    L = track.length
    heading = _driver_heading(xy)
    out, s_prev = [], None
    for i, (r, (x, y)) in enumerate(zip(aligned_data, xy)):
        s, off = track.project(x, y)
        if s_prev is not None:
            travelled = np.linalg.norm(xy[i] - xy[i - 1])
            expected = s_prev + travelled
            gap = (s - expected + L / 2) % L - L / 2          # circular difference
            if abs(gap) > max_jump_m + travelled:
                s, off = _local_project(track, x, y, expected, window=max_jump_m + 2 * travelled)
        s_prev = s
        out.append({
            "time": r["time"],
            "s": float(s),
            "lateral_offset": float(off),
            "heading_error": float(_wrap_angle(heading[i] - track.heading(s))),
            "speed": r["speed"], "throttle": r["throttle"], "brake": r["brake"],
        })
    return out


# ----------------------------------------------------------------------------
# 3) Checks the team can read ("Done when" criteria)
# ----------------------------------------------------------------------------
def alignment_report(training_data, track):
    L = track.length
    s = np.array([r["s"] for r in training_data])
    off = np.array([r["lateral_offset"] for r in training_data])
    he = np.array([r["heading_error"] for r in training_data])
    ds = (np.diff(s) + L / 2) % L - L / 2
    lw = np.array([track.left_width(v) for v in s])
    rw = np.array([track.right_width(v) for v in s])
    inside = (off <= lw) & (off >= -rw)
    return {
        "samples": len(s),
        "progress_m": float(ds.sum()),
        "progress_fraction_of_lap": float(ds.sum() / L),
        "backward_steps": int((ds < -0.5).sum()),             # progress should follow the lap
        "lateral_offset_mean_m": float(off.mean()),
        "lateral_offset_abs_p95_m": float(np.percentile(np.abs(off), 95)),
        "share_within_track_widths": float(inside.mean()),   # plausible offsets
        "heading_error_abs_p95_rad": float(np.percentile(np.abs(he), 95)),
    }


def plot_alignment(aligned_data, track, path="alignment_overlay.png", title=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ss = np.linspace(0, track.length, 2000, endpoint=False)
    b = np.array([track.boundaries(s) for s in ss])           # (n, 2 sides, 2 coords)
    c = np.array([track.position(s) for s in ss])
    d = np.array([[r["x"], r["y"]] for r in aligned_data])
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.plot(b[:, 0, 0], b[:, 0, 1], color="0.6", lw=0.8, label="Track boundaries")
    ax.plot(b[:, 1, 0], b[:, 1, 1], color="0.6", lw=0.8)
    ax.plot(c[:, 0], c[:, 1], color="0.6", lw=0.6, ls="--", label="Centreline")
    ax.plot(d[:, 0], d[:, 1], color="#d1495b", lw=1.4, label="Aligned driver trajectory")
    ax.scatter(*d[0], color="#d1495b", zorder=5, s=30, label="Lap start")
    ax.set_aspect("equal")
    ax.set(xlabel="x (m)", ylabel="y (m)", title=title or "Driver trajectory aligned to track")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def load_lap_csv(path):
    """Read Person 4's clean lap CSV (time, x, y, speed, throttle, brake)."""
    import pandas as pd
    df = pd.read_csv(path)
    missing = set(_ROWS) - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing columns: {sorted(missing)}")
    return df[list(_ROWS)].astype(float).to_dict("records")


# ============================================================================
# Self-test: built-in demo track + synthetic lap with a known transform
# ============================================================================
from scipy.interpolate import splprep, splev   # noqa: E402
from scipy.spatial import cKDTree              # noqa: E402


class DemoTrack:
    def __init__(self, track_points, n_dense=20000, smoothing=0.0):
        """track_points: list of (x, y, left_width, right_width) in metres, closed loop."""
        pts = np.asarray(track_points, dtype=float)
        xy, lw, rw = pts[:, :2], pts[:, 2], pts[:, 3]
        if np.linalg.norm(xy[0] - xy[-1]) < 1e-6:          # drop duplicated closing point
            xy, lw, rw = xy[:-1], lw[:-1], rw[:-1]
        closed = np.vstack([xy, xy[:1]])
        tck, u = splprep([closed[:, 0], closed[:, 1]], s=smoothing, per=1, k=3)

        u_dense = np.linspace(0, 1, n_dense, endpoint=False)
        x, y = splev(u_dense, tck)
        dx, dy = splev(u_dense, tck, der=1)
        ddx, ddy = splev(u_dense, tck, der=2)
        seg = np.hypot(np.diff(np.r_[x, x[0]]), np.diff(np.r_[y, y[0]]))
        self._s = np.r_[0.0, np.cumsum(seg)[:-1]]
        self.length = float(seg.sum())
        self._xy = np.column_stack([x, y])
        self._heading = np.unwrap(np.arctan2(dy, dx))
        self._kappa = (dx * ddy - dy * ddx) / np.power(dx**2 + dy**2, 1.5)  # + = left turn
        # widths: interpolate the control-point widths along spline parameter
        u_ctrl = u[:-1]
        self._lw = np.interp(u_dense, u_ctrl, lw, period=1.0)
        self._rw = np.interp(u_dense, u_ctrl, rw, period=1.0)
        self._tree = cKDTree(self._xy)

    # ---- helpers --------------------------------------------------------
    def _wrap(self, s):
        return float(np.mod(s, self.length))

    def _interp(self, arr, s, angle=False):
        s = self._wrap(s)
        s_ext = np.r_[self._s, self.length]
        if angle:
            # heading table is unwrapped; close the loop with the total rotation (±2π)
            total = arr[0] + 2 * np.pi * np.round((arr[-1] - arr[0]) / (2 * np.pi))
            a = np.r_[arr, total]
            v = np.interp(s, s_ext, a)
            return float((v + np.pi) % (2 * np.pi) - np.pi)
        return float(np.interp(s, s_ext, np.r_[arr, arr[0]]))

    # ---- shared API -----------------------------------------------------
    def position(self, s):
        return (self._interp(self._xy[:, 0], s), self._interp(self._xy[:, 1], s))

    def heading(self, s):
        return self._interp(self._heading, s, angle=True)

    def curvature(self, s):
        return self._interp(self._kappa, s)

    def left_width(self, s):
        return self._interp(self._lw, s)

    def right_width(self, s):
        return self._interp(self._rw, s)

    def boundaries(self, s):
        x, y = self.position(s)
        h = self.heading(s)
        nx, ny = -np.sin(h), np.cos(h)                     # left normal
        lw, rw = self.left_width(s), self.right_width(s)
        return ((x - rw * nx, y - rw * ny), (x + lw * nx, y + lw * ny))  # (right, left)

    def project(self, x, y):
        """Closest centreline distance s and signed lateral offset (+ = left)."""
        _, i = self._tree.query([x, y])
        best = None
        for j in (i - 1, i):                               # refine on the two adjacent segments
            a, b = self._xy[j % len(self._xy)], self._xy[(j + 1) % len(self._xy)]
            ab = b - a
            t = np.clip(np.dot([x, y] - a, ab) / max(np.dot(ab, ab), 1e-12), 0, 1)
            c = a + t * ab
            d = np.hypot(x - c[0], y - c[1])
            if best is None or d < best[0]:
                s = self._s[j % len(self._s)] + t * np.linalg.norm(ab)
                best = (d, s, c, ab)
        d, s, c, ab = best
        side = np.sign(ab[0] * (y - c[1]) - ab[1] * (x - c[0]))
        return (self._wrap(s), float(side * d))


def demo_track():
    """A closed synthetic circuit (~8 km) with straights, S-bends and tight curves."""
    t = np.linspace(0, 2 * np.pi, 120, endpoint=False)
    r = 900 + 180 * np.cos(3 * t) + 90 * np.sin(5 * t + 0.4)
    x = 1.6 * r * np.cos(t)
    y = 0.75 * r * np.sin(t)
    w = 6.0 + 1.5 * np.sin(4 * t) ** 2
    return DemoTrack([(xi, yi, wi, wi) for xi, yi, wi in zip(x, y, w)], smoothing=0.0)


def _synthetic_lap(track, scale=10.0, rot=0.7, shift=(-3500.0, 1200.0), start_s=None,
                   noise_units=3.0, v=60.0, dt=0.25, seed=0):
    """Driver weaves ±3 m inside the track, then the lap is put into a different
    frame (tele = scale · Rᵀ (world − shift)) — like real FastF1 X/Y (~0.1 m units)."""
    rng = np.random.default_rng(seed)
    L = track.length
    start_s = 0.15 * L if start_s is None else start_s
    n = int(L / (v * dt))
    s_true = np.mod(start_s + np.arange(n) * v * dt, L)
    off_true = 3.0 * np.sin(2 * np.pi * 7 * np.arange(n) / n)
    world = []
    for s, o in zip(s_true, off_true):
        x, y = track.position(s)
        h = track.heading(s)
        world.append((x - o * np.sin(h), y + o * np.cos(h)))
    R = np.array([[np.cos(rot), -np.sin(rot)], [np.sin(rot), np.cos(rot)]])
    tele = scale * (R.T @ (np.array(world) - np.asarray(shift)).T).T
    tele += rng.normal(0, noise_units, tele.shape)
    lap = [{"time": i * dt, "x": float(x), "y": float(y), "speed": v,
            "throttle": 1.0, "brake": 0.0} for i, (x, y) in enumerate(tele)]
    return lap, s_true, off_true


def _self_test():
    tracks = [("built-in demo track", demo_track(), 60.0)]
    try:
        from f1_simulator.core.track import Track
        tracks.append(("team Track (f1_simulator.core.track)", Track(), 10.0))
    except ImportError:
        print("(f1_simulator not found — testing on the built-in demo track only)")
    ok_all = True
    for name, tr, v in tracks:
        lap, s_true, off_true = _synthetic_lap(tr, v=v, noise_units=1.0 if v < 20 else 3.0)
        T = estimate_transform(lap, tr)
        aligned = align_coordinates(lap, tr, transform=T)
        training = convert_to_track_coordinates(aligned, tr)
        rep = alignment_report(training, tr)
        ok = (abs(T["scale"] - 0.1) < 0.001 and rep["backward_steps"] == 0
              and rep["share_within_track_widths"] > 0.97
              and abs(rep["progress_fraction_of_lap"] - 1) < 0.03)
        ok_all &= ok
        print(f"\n== {name} ({tr.length:.0f} m) ==")
        print(f"  recovered scale {T['scale']:.5f} (true 0.10000) | rotation {T['rotation']:.3f} rad "
              f"(true 0.700; ±π on symmetric tracks) | RMS to centreline {T['rms_to_centreline_m']:.2f} m")
        for k, val in rep.items():
            print(f"  {k}: {val:.3f}" if isinstance(val, float) else f"  {k}: {val}")
        print("  first record:", {k: round(val, 3) for k, val in training[0].items()})
        print("  RESULT:", "PASS" if ok else "FAIL")
        if name.startswith("built-in"):
            print("  saved", plot_alignment(aligned, tr, "alignment_overlay.png",
                                          "Synthetic lap: known transform recovered and overlaid"))
    print("\nALL CHECKS PASSED" if ok_all else "\nSOME CHECKS FAILED")
    return ok_all


if __name__ == "__main__":
    import sys
    sys.exit(0 if _self_test() else 1)
