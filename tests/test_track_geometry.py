"""Geometry validation using real Monza and independent analytical fixtures."""
import math

import numpy as np
import pytest
from scipy.integrate import quad

from f1_simulator.core.track import Track


@pytest.fixture(scope="module")
def monza():
    return Track("monza")


def circular_track(reverse=False):
    theta = np.linspace(0, 2*np.pi, 64, endpoint=False)
    return Track(track_points=np.column_stack((100*np.cos(theta), 100*np.sin(theta),
                                               np.full(64, 4.), np.full(64, 7.))), reverse=reverse)


def circular_error(a, b, length):
    return (a-b+length/2) % length-length/2


def test_arc_length_against_independent_quadrature(monza):
    reference = sum(quad(lambda u: np.linalg.norm(monza._curve(u, 1)), a, b,
                         epsabs=1e-10)[0]
                    for a, b in zip(monza._knots[:-1], monza._knots[1:]))
    assert monza.length == pytest.approx(reference, abs=1e-6)
    # This tests physical distance, rather than simply testing wrapping.
    for s in np.linspace(0, monza.length, 21, endpoint=False):
        u = monza._parameter(s)
        i = np.searchsorted(monza._knots, u, side="right")-1
        actual = monza._arc[i] + quad(lambda v: np.linalg.norm(monza._curve(v, 1)),
                                      monza._knots[i], u, epsabs=1e-10)[0]
        assert actual == pytest.approx(s, abs=1e-6)
        tangent = (np.array(monza.position(s+0.001))-monza.position(s-0.001))/0.002
        assert np.linalg.norm(tangent) == pytest.approx(1, abs=1e-5)
        assert math.atan2(tangent[1], tangent[0]) == pytest.approx(monza.heading(s), abs=1e-5)


def test_closed_periodic_and_finite(monza):
    assert monza.position(0) == pytest.approx(monza.position(monza.length))
    assert monza.position(-3) == pytest.approx(monza.position(monza.length-3))
    assert monza.position(17) == pytest.approx(monza.position(17+3*monza.length))
    assert np.linalg.norm(np.subtract(monza.position(-1e-5), monza.position(1e-5))) < 3e-5
    assert abs(circular_error(monza.heading(-1e-5), monza.heading(1e-5), 2*np.pi)) < 1e-4
    assert monza.curvature(-1e-5) == pytest.approx(monza.curvature(1e-5), abs=1e-6)
    for s in np.linspace(0, monza.length, 501):
        values = [*monza.position(s), monza.heading(s), monza.curvature(s),
                  monza.left_width(s), monza.right_width(s), *np.ravel(monza.boundaries(s))]
        assert np.isfinite(values).all()
        assert monza.left_width(s) > 0 and monza.right_width(s) > 0


def test_source_width_order(monza):
    assert monza.left_width(0) == pytest.approx(5.932)
    assert monza.right_width(0) == pytest.approx(5.739)


@pytest.mark.parametrize("fraction", [0, 0.001, 0.1, 0.25, 0.51, 0.73, 0.99999])
@pytest.mark.parametrize("offset", [-2., 0., 2.])
def test_continuous_projection_and_sides(monza, fraction, offset):
    s = fraction*monza.length
    c = np.array(monza.position(s))
    h = monza.heading(s)
    normal = np.array([-np.sin(h), np.cos(h)])
    projected, lateral = monza.project(*(c+offset*normal))
    assert abs(circular_error(projected, s, monza.length)) < 1e-5
    assert lateral == pytest.approx(offset, abs=1e-6)
    left, right = monza.boundaries(s)
    assert np.dot(np.subtract(left, c), normal) == pytest.approx(monza.left_width(s))
    assert np.dot(np.subtract(right, c), normal) == pytest.approx(-monza.right_width(s))


def test_circle_curvature_and_reversed_widths():
    forward, backward = circular_track(), circular_track(True)
    assert forward.length == pytest.approx(2*np.pi*100, abs=0.001)
    for s in np.linspace(0, forward.length, 30, endpoint=False):
        assert forward.curvature(s) == pytest.approx(0.01, abs=1e-5)
        assert backward.curvature(s) == pytest.approx(-0.01, abs=1e-5)
        assert backward.position(s) == pytest.approx(forward.position(-s), abs=1e-7)
        assert backward.left_width(s) == 7
        assert backward.right_width(s) == 4
        # Reversing preserves the same physical road.
        assert backward.boundaries(s)[0] == pytest.approx(forward.boundaries(-s)[1], abs=1e-7)


def test_straight_curvature(monza):
    values = [abs(monza.curvature(s)) for s in np.linspace(50, 450, 20)]
    assert max(values) < 1e-4


def test_projection_matches_dense_global_search(monza):
    rng = np.random.default_rng(7)
    dense = np.array([monza.position(s) for s in np.linspace(0, monza.length, 12000, endpoint=False)])
    for point in rng.uniform(dense.min(axis=0)-50, dense.max(axis=0)+50, (20, 2)):
        s, _ = monza.project(*point)
        closest_distance = np.linalg.norm(np.subtract(monza.position(s), point))
        assert closest_distance <= np.linalg.norm(dense-point, axis=1).min()+1e-6


def assert_simple_loop(points):
    # Check proper crossings of nonadjacent sampled edges; not a proof for
    # unsampled spline geometry. Closure neighbors are deliberately excluded.
    p = np.asarray(points)
    n = len(p)
    a, b = p, np.roll(p, -1, axis=0)
    def cross(u, v):
        return u[..., 0]*v[..., 1]-u[..., 1]*v[..., 0]
    for i in range(n-2):
        j = np.arange(i+2, n)
        if i == 0:
            j = j[j != n-1]
        d1 = cross(b[i]-a[i], a[j]-a[i])
        d2 = cross(b[i]-a[i], b[j]-a[i])
        d3 = cross(b[j]-a[j], a[i]-a[j])
        d4 = cross(b[j]-a[j], b[i]-a[j])
        assert not np.any((d1*d2 < -1e-10) & (d3*d4 < -1e-10)), f"Crossing near edge {i}"


def test_no_obvious_boundary_intersections(monza):
    ss = np.linspace(0, monza.length, 1600, endpoint=False)
    center = [monza.position(s) for s in ss]
    edges = np.array([monza.boundaries(s) for s in ss])
    assert_simple_loop(center)
    assert_simple_loop(edges[:, 0])
    assert_simple_loop(edges[:, 1])
    # Boundaries must not cross one another either.
    left, right = edges[:, 0], edges[:, 1]
    right_end = np.roll(right, -1, axis=0)
    def cross(a, b):
        return a[..., 0]*b[..., 1]-a[..., 1]*b[..., 0]
    for a, b in zip(left, np.roll(left, -1, axis=0)):
        d1, d2 = cross(b-a, right-a), cross(b-a, right_end-a)
        d3, d4 = cross(right_end-right, a-right), cross(right_end-right, b-right)
        assert not np.any((d1*d2 < -1e-10) & (d3*d4 < -1e-10))



def test_csv_formats_and_reversal(tmp_path):
    path = tmp_path/'square.csv'
    path.write_text('x,y,left_width,right_width\n0,0,2,3\n100,0,2,3\n100,100,2,3\n0,100,2,3\n')
    track = Track.from_csv(path, reverse=True)
    assert track.left_width(0) == 3
    assert track.right_width(0) == 2
    with pytest.raises(ValueError):
        Track('nonexistent')


@pytest.mark.parametrize('points', [[], [[0,0,-1,2]]*4,
    [[0,0,2,2],[0,0,2,2],[100,100,2,2],[0,100,2,2]]])
def test_invalid_control_points(points):
    with pytest.raises(ValueError):
        Track(track_points=points)


def test_source_provenance_hash():
    import hashlib
    import json
    from pathlib import Path
    data = Path(__file__).resolve().parents[1] / "data"
    provenance = json.loads((data / "monza.source.json").read_text())
    assert hashlib.sha256((data / "monza.csv").read_bytes()).hexdigest() == provenance["sha256"]


def test_invalid_queries(monza):
    for value in [float("nan"), float("inf")]:
        with pytest.raises(ValueError):
            monza.position(value)
        with pytest.raises(ValueError):
            monza.project(value, 0)


def test_reversed_real_track_keeps_physical_widths(monza):
    backward = Track("monza", reverse=True)
    for s in [0, 100, 900, 3100]:
        assert backward.position(s) == pytest.approx(monza.position(-s), abs=1e-6)
        assert backward.left_width(s) == pytest.approx(monza.right_width(-s), abs=1e-6)
        assert backward.boundaries(s)[0] == pytest.approx(monza.boundaries(-s)[1], abs=1e-6)


@pytest.fixture(scope="module")
def silverstone():
    return Track("silverstone")


def test_silverstone_source_geometry_and_periodicity(silverstone):
    import hashlib,json
    from pathlib import Path
    folder=Path(__file__).resolve().parents[1]/"data"
    source=json.loads((folder/"silverstone.source.json").read_text())
    assert hashlib.sha256((folder/"silverstone.csv").read_bytes()).hexdigest() == source["sha256"]
    assert source["source_path"] == "tracks/Silverstone.csv"
    assert silverstone.length > 5000
    assert silverstone.position(0) == pytest.approx(silverstone.position(silverstone.length))
    for s in np.linspace(0,silverstone.length,100,endpoint=False):
        assert np.isfinite([*silverstone.position(s),silverstone.curvature(s)]).all()
        assert silverstone.left_width(s)>0 and silverstone.right_width(s)>0


@pytest.mark.parametrize("fraction",[0.,.25,.5,.9])
def test_silverstone_projection_recovers_signed_offsets(silverstone,fraction):
    s=fraction*silverstone.length
    c=np.array(silverstone.position(s));h=silverstone.heading(s)
    normal=np.array([-np.sin(h),np.cos(h)])
    for offset in [-1.,0.,1.]:
        recovered,lateral=silverstone.project(*(c+offset*normal))
        assert abs(circular_error(recovered,s,silverstone.length))<1e-5
        assert lateral == pytest.approx(offset,abs=1e-6)
