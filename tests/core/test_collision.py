import math

import pytest

from f1_simulator.core.collision import resolve_car_collisions


def test_head_on_collision_separates_cars_and_stops_closing_motion():
    first = {
        "state": {"x": 0.0, "y": 0.0, "heading": 0.0, "speed": 10.0}
    }
    second = {
        "state": {"x": 3.0, "y": 0.0, "heading": math.pi, "speed": 10.0}
    }

    resolve_car_collisions([first, second])

    distance = math.hypot(
        second["state"]["x"] - first["state"]["x"],
        second["state"]["y"] - first["state"]["y"],
    )
    assert distance >= 4.4
    assert first["state"]["speed"] == pytest.approx(0.0)
    assert second["state"]["speed"] == pytest.approx(0.0)


def test_telemetry_replay_is_an_immovable_collision_barrier():
    ghost = {
        "state": {"x": 0.0, "y": 0.0, "heading": 0.0, "speed": 10.0},
        "ghost": True,
    }
    car = {
        "state": {"x": 3.0, "y": 0.0, "heading": math.pi, "speed": 10.0}
    }
    ghost_position = (ghost["state"]["x"], ghost["state"]["y"])

    resolve_car_collisions([ghost, car])

    assert (ghost["state"]["x"], ghost["state"]["y"]) == ghost_position
    assert math.hypot(
        car["state"]["x"] - ghost["state"]["x"],
        car["state"]["y"] - ghost["state"]["y"],
    ) >= 4.4


def test_collision_resolution_updates_race_session_state():
    class Session:
        state = {"x": 0.0, "y": 0.0, "heading": 0.0, "speed": 0.0}

    session = Session()
    car = {
        "state": {"x": 0.0, "y": 0.0, "heading": 0.0, "speed": 0.0},
        "session": session,
    }

    resolve_car_collisions([car])

    assert session.state == car["state"]
