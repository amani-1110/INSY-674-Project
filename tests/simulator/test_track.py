import numpy as np
import pytest

from f1_simulator.core.track import Track


@pytest.mark.parametrize("segment", [0, 600, 1200, 1800])
@pytest.mark.parametrize("offset", [-12.0, 0.0, 12.0])
def test_project_preserves_signed_lateral_offset(segment, offset):
    track = Track()
    distance_on_track = (track.s[segment] + track.s[segment + 1]) / 2
    h = track.heading(distance_on_track)
    tangent = np.array([np.cos(h), np.sin(h)])
    left_normal = np.array([-tangent[1], tangent[0]])
    midpoint = np.array(track.position(distance_on_track))
    point = midpoint + offset * left_normal

    distance, actual_offset = track.project(*point)

    assert actual_offset == pytest.approx(offset, abs=2e-5)
    assert distance == pytest.approx(
        (track.s[segment] + track.s[segment + 1]) / 2, abs=2e-5
    )
