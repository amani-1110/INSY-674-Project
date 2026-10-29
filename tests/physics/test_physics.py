"""Tests for physics.py — run with:  pytest -q"""

import copy
import math

import pytest

from f1_simulator.physics import (
    DEFAULT_PARAMS,
    effective_steering,
    grip_force,
    update_car,
    wrap_angle,
)

DT = 0.01


def start(speed=0.0, heading=0.0):
    return {"x": 0.0, "y": 0.0, "heading": heading, "speed": speed}


def run(state, action, seconds, dt=DT):
    for _ in range(int(round(seconds / dt))):
        state = update_car(state, action, dt)
    return state


# ----------------------------------------------------------------- contract
def test_output_has_exactly_the_contract_keys():
    out = update_car(start(20), {"steering": 0, "throttle": 1, "brake": 0}, 0.1)
    assert set(out) == {"x", "y", "heading", "speed"}
    assert all(isinstance(v, float) for v in out.values())


def test_team_doc_example_moves_forward_and_speeds_up():
    out = update_car({"x": 120.0, "y": 80.0, "heading": 0.0, "speed": 20.0},
                     {"steering": 0.0, "throttle": 1.0, "brake": 0.0}, 0.1)
    assert out["x"] == pytest.approx(122.0, abs=0.2)
    assert out["y"] == pytest.approx(80.0)
    assert out["heading"] == pytest.approx(0.0)
    assert out["speed"] > 20.0


def test_reproducible_and_inputs_not_mutated():
    s, a = start(35, 0.3), {"steering": 0.05, "throttle": 0.7, "brake": 0.0}
    s0, a0 = copy.deepcopy(s), copy.deepcopy(a)
    assert update_car(s, a, 0.1) == update_car(s, a, 0.1)
    assert s == s0 and a == a0


def test_dt_zero_returns_same_state_and_negative_dt_rejected():
    s = start(30, 1.0)
    assert update_car(s, {"throttle": 1}, 0.0) == pytest.approx(s)
    with pytest.raises(ValueError):
        update_car(s, {}, -0.1)


def test_missing_state_key_raises():
    with pytest.raises(KeyError):
        update_car({"x": 0, "y": 0, "speed": 1}, {}, 0.1)


def test_out_of_range_pedals_are_clipped():
    s = start(30)
    assert update_car(s, {"throttle": 5.0}, 0.1) == update_car(s, {"throttle": 1.0}, 0.1)
    assert update_car(s, {"brake": -2.0}, 0.1) == update_car(s, {"brake": 0.0}, 0.1)


# --------------------------------------------------------- straight line
def test_straight_line_acceleration_is_f1_like():
    s, t = start(0), 0.0
    while s["speed"] < 100 / 3.6:
        s = update_car(s, {"throttle": 1.0}, DT)
        t += DT
    assert 2.0 < t < 3.5                  # real F1: ~2.6 s for 0-100 km/h
    assert s["y"] == pytest.approx(0.0)   # no drift sideways
    assert s["heading"] == pytest.approx(0.0)


def test_top_speed_plateaus_at_realistic_value():
    s = run(start(0), {"throttle": 1.0}, 60)
    top_kmh = s["speed"] * 3.6
    assert 320 < top_kmh < 380
    s2 = update_car(s, {"throttle": 1.0}, 1.0)
    assert abs(s2["speed"] - s["speed"]) < 0.05  # terminal speed reached


def test_braking_stops_car_in_realistic_distance_and_never_reverses():
    s, t = start(200 / 3.6), 0.0
    while s["speed"] > 0:
        s = update_car(s, {"brake": 1.0}, DT)
        t += DT
        assert s["speed"] >= 0.0
    assert 40 < s["x"] < 90               # real F1: roughly 55-70 m from 200 km/h
    s = update_car(s, {"brake": 1.0}, 1.0)
    assert s["speed"] == 0.0 and s["x"] > 0   # stays stopped, does not roll back


def test_coasting_slows_down():
    s = run(start(80), {}, 2.0)
    assert s["speed"] < 80


def test_more_brake_stops_sooner():
    def stop_dist(b):
        s = start(60)
        while s["speed"] > 0:
            s = update_car(s, {"brake": b}, DT)
        return s["x"]
    assert stop_dist(1.0) < stop_dist(0.5)


# --------------------------------------------------------------- steering
def _circle_fit_radius(points):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    cx, cy = (max(xs) + min(xs)) / 2, (max(ys) + min(ys)) / 2
    radii = [math.hypot(x - cx, y - cy) for x, y in points]
    return sum(radii) / len(radii), max(radii) - min(radii)


def test_constant_steering_low_speed_drives_geometric_circle():
    """At low speed grip is not limiting: radius = L / tan(delta)."""
    delta, speed = 0.2, 10.0
    expected_r = DEFAULT_PARAMS.wheelbase / math.tan(delta)
    s, pts = start(speed), []
    # hold speed constant by resetting it (pure steering test)
    for _ in range(int(2 * math.pi * expected_r / speed / DT) + 10):
        s = update_car(s, {"steering": delta}, DT)
        s["speed"] = speed
        pts.append((s["x"], s["y"]))
    r, spread = _circle_fit_radius(pts)
    assert r == pytest.approx(expected_r, rel=0.02)
    assert spread < 0.05 * expected_r
    assert math.hypot(s["x"], s["y"]) < 1.0   # back near the start after one lap


def test_positive_steering_turns_left_negative_turns_right():
    left = run(start(20), {"steering": 0.1}, 0.5)
    right = run(start(20), {"steering": -0.1}, 0.5)
    assert left["heading"] > 0 and left["y"] > 0
    assert right["heading"] < 0 and right["y"] < 0
    assert left["y"] == pytest.approx(-right["y"])


def test_grip_limit_widens_turn_at_high_speed():
    """Same steering at high speed cannot exceed tyre grip -> wider circle."""
    delta = 0.1
    v = 70.0
    eff = effective_steering(v, delta)
    assert eff < delta
    lat_acc = v * v * math.tan(eff) / DEFAULT_PARAMS.wheelbase
    assert lat_acc == pytest.approx(grip_force(v) / DEFAULT_PARAMS.mass, rel=1e-6)
    assert 3.0 * 9.81 < lat_acc < 6.0 * 9.81  # F1-like 3-6 g at high speed


def test_steering_lock_is_respected():
    assert effective_steering(1.0, 2.0) == pytest.approx(DEFAULT_PARAMS.max_steer)


def test_heading_stays_wrapped():
    s = run(start(15), {"steering": 0.3, "throttle": 0.2}, 30)
    assert -math.pi <= s["heading"] < math.pi
    assert wrap_angle(3 * math.pi) == pytest.approx(-math.pi)


def test_zero_speed_with_steering_does_not_move():
    s = update_car(start(0), {"steering": 0.3}, 0.1)
    assert (s["x"], s["y"], s["heading"]) == pytest.approx((0.0, 0.0, 0.0))
