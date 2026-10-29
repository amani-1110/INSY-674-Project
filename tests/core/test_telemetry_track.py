import numpy as np
import pytest

from f1_simulator.core.telemetry_track import TelemetryTrack
from f1_simulator.ui.renderer import fit_track_view


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


def test_overview_camera_fits_the_full_track_inside_the_screen():
    track = TelemetryTrack.from_csv()

    center, scale = fit_track_view(track, screen_size=(1100, 750))
    boundary_points = np.asarray(
        [
            point
            for distance in np.linspace(0, track.length, 1000, endpoint=False)
            for point in track.boundaries(distance)
        ]
    )
    screen_positions = np.column_stack(
        (
            550 + (boundary_points[:, 0] - center[0]) * scale,
            375 - (boundary_points[:, 1] - center[1]) * scale,
        )
    )

    assert screen_positions[:, 0].min() >= 69
    assert screen_positions[:, 0].max() <= 1031
    assert screen_positions[:, 1].min() >= 69
    assert screen_positions[:, 1].max() <= 681
    assert scale < 4.5
