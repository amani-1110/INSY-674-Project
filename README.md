# F1 simulator and spline track geometry

## Install and run

From the repository root (Python 3.10 or newer):

```powershell
python -m pip install -r requirements.txt
python -m pytest -q
python visualize_track.py monza
python -m f1_simulator.demo --track monza
```

The plot is saved to `output/monza_track.png`. The interactive pygame simulator
opens a separate window. Select a car with 1-4; use arrows/WASD to drive, R to
reset, C to choose again, and Escape to quit. `--track demo` uses a synthetic
ellipse. The simulator now calls the shared `physics/physics.py` implementation
through `f1_simulator/core/physics.py`; geometry itself has no physics dependency.

## Shared Track API

```python
from track import Track
# Existing package imports also work:
# from f1_simulator.core.track import Track

track = Track("monza")
print(track.length)                  # about 5790.693805 metres for this dataset
x, y = track.position(500.0)
heading = track.heading(500.0)        # radians, CCW from positive x
curvature = track.curvature(500.0)    # 1/metres, positive for left turns
left, right = track.boundaries(500.0)
left_width = track.left_width(500.0)
right_width = track.right_width(500.0)
s, offset = track.project(x, y)       # metres; offset positive to the left
```

All queries wrap `s % track.length`, including negative distances. Coordinates
use Cartesian x right, y up; dimensions are metres, angles radians. Boundaries
are always returned **(left, right)** relative to the input point order, which
defines travel direction. `Track()` retains the synthetic demo for compatibility;
use `Track("monza")` explicitly for real geometry. Unknown circuits raise an error.
No source file is downloaded at runtime.

## Spline and numerical methods

The track uses SciPy's periodic cubic interpolant for x and y, initially indexed
by cumulative input chord distance. That parameter is **not** treated as arc
length. Each segment's speed `sqrt(x'(u)^2 + y'(u)^2)` is integrated with 16-point
Gauss-Legendre quadrature. Cumulative integrals define lap length; Brent's method
inverts the integral within the appropriate segment for each `position(s)` query.
Tests compare integration with independent adaptive quadrature and verify
unit-speed sampling. These are numerical approximations, not symbolic exactness.

Heading comes from `atan2(y', x')`; signed curvature is
`(x' y'' - y' x'') / (x'^2 + y'^2)^(3/2)`. Widths use positive, periodic piecewise
linear interpolation through the same source stations; linear interpolation
avoids cubic width overshoot. A repeated closing row is optional. Invalid inputs,
nonpositive widths, consecutive duplicate points and degenerate tangents are
rejected. Custom data must describe a sensible simple circuit; the constructor
does not certify arbitrary road geometry.

Projection uses a KD-tree nearest input point as an upper bound and conservative
Bezier segment bounding boxes to discard only segments that cannot improve it.
For every remaining segment it evaluates endpoints and all real roots of the
quintic squared-distance derivative. The result is on the **continuous spline**,
not a nearest sampled coordinate. At crossings or equally close branches the
geometrical nearest point can be ambiguous; temporal continuity belongs in the
alignment/session module.

## Add another circuit

Copy a TUM `tracks/<Circuit>.csv` file to `data/<lowercase_name>.csv`, retain its
licence and provenance, then use `Track("lowercase_name")`. Do not use a file
from TUM's `racelines` folder as a centerline.

Alternatively, load any path:

```python
track = Track.from_csv("path/to/circuit.csv")
reversed_track = Track.from_csv("path/to/circuit.csv", reverse=True)
```

Two CSV schemas are accepted (optional UTF-8 BOM, optional TUM `#` header):

```text
# x_m,y_m,w_tr_right_m,w_tr_left_m
```

or

```text
x,y,left_width,right_width
```

TUM puts **right before left**. The loader remaps them explicitly. Rows are
preserved in source order. `reverse=True` reverses travel while preserving the
first point and swaps left/right widths so physical road boundaries stay put.
The loader does not guess a circuit's orientation or rescale coordinates.
For in-memory data, use
`Track(name="custom", track_points=[(x, y, left_width, right_width), ...])`.

```powershell
python visualize_track.py --csv path/to/circuit.csv --output output/custom.png
python visualize_track.py monza --reverse --output output/monza_reverse.png
python -m f1_simulator.demo --track-csv path/to/circuit.csv
```

## Monza source, attribution and limitations

`data/monza.csv` is the **unmodified** TUMFTM racetrack-database
`tracks/Monza.csv`, pinned at commit
`e59595d1f3573b30d1ded6a08984935b957688e0`:
https://github.com/TUMFTM/racetrack-database/tree/e59595d1f3573b30d1ded6a08984935b957688e0

The upstream README specifies metre coordinates and metre right/left widths.
These are local planar coordinates, not latitude/longitude. The supplied files
do not specify a geodetic CRS, global origin or verified official timing-line
station. We preserve the axes and row order. `s=0` is the dataset's first point
and the simulator's finish location, not a claim about an official timing line.
No north-reference conversion is applied; this API's heading is always measured
from positive x. TUM's helper normal convention confirms widths are defined
relative to the tangent/travel direction.

TUM reports that centerlines originate from OpenStreetMap GPS points and are
smoothed, while widths are estimated from satellite imagery. Smoothing can move
the reference line away from the exact middle of the road; accuracy varies.
The measured spline length is about 5790.69 m and is not forced to a nominal
circuit length. Geometry is not a surveyed map or a guarantee of the 2025 layout.
Do not assume its coordinate frame matches FastF1: telemetry needs alignment.
Source attribution includes TUMFTM, Alexander Heilmeier, Andressa de Paula Suiti
(width extraction), and OpenStreetMap contributors.

Upstream licence: LGPL-3.0, preserved in `data/TUM_LICENSE` with the incorporated GPL in `data/GPL-3.0.txt`; the upstream README
is `data/TUM_README.md`. `data/monza.source.json` records the source commit,
original filename, unchanged status and SHA-256. Upstream data remains under
its own licence. OpenStreetMap attribution: https://www.openstreetmap.org/copyright

## Validation and project integration

`test_track.py` covers closure, seam continuity, wrapping, finite outputs,
quadrature accuracy, heading, straight/circle curvature, width mapping,
reversal, boundary sides, signed projection and global nearest-point comparisons.
It also checks centerline and each boundary for proper self-intersections in a
1600-point sampling. This catches obvious crossings, not every possible
sub-sample defect or arbitrary custom-track error. The visualization shows
centerline curvature, both boundaries, direction arrows, dataset start and
signed projection examples with equal aspect ratio.

The full suite also checks the actual simulator with shared physics: baseline
lap completion on the demo track and a stationary controller timeout. These
checks do not establish trained AI performance or a valid autonomous Monza lap.
Evaluation imports use the real simulator; tests no longer inject a fake global
`simulator` module. Imitation exposes the agreed `train_model` and
`predict_action` names as aliases for existing `train` and `predict` functions.
Fresh telemetry extraction requires FastF1; tests use offline fixtures and never
request Formula 1 credentials or download sessions.
