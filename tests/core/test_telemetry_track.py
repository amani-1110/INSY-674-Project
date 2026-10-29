import numpy as np
import pytest

from f1_simulator.core.telemetry_track import TelemetryTrack


def test_bundled_lap_builds_a_closed_approximate_monza_track():
    track = TelemetryTrack.from_csv()

    assert 5000 < track.length < 6000
    assert track.name == "Telemetry-derived Monza"
    assert track.points[0] == pytest.approx(track.points[-1])

    distance, offset = track.project(*track.position(0))

    assert min(distance, track.length - distance) < 5
    assert offset == pytest.approx(0.0, abs=1e-6)


def test_telemetry_track_projection_preserves_signed_offsets():
    track = TelemetryTrack.from_csv()
    distance_on_track = 400.0
    x, y = track.position(distance_on_track)
    heading = track.heading(distance_on_track)
    left_normal = np.array([-np.sin(heading), np.cos(heading)])

    distance, offset = track.project(
        *(np.array([x, y]) + 4.0 * left_normal)
    )

    assert distance == pytest.approx(distance_on_track, abs=0.1)
    assert offset == pytest.approx(4.0, abs=0.1)
    assert np.isfinite(track.curvature(distance_on_track))
