from f1_simulator.controllers.baseline import baseline_action
from f1_simulator.physics import update_car


def test_baseline_action_is_accepted_by_vehicle_physics():
    state = {"x": 0.0, "y": 0.0, "heading": 0.0, "speed": 10.0}
    observation = {
        "speed": state["speed"],
        "lateral_offset": 0.0,
        "heading_error": 0.0,
        "curvature_ahead": [0.0, 0.0, 0.0],
    }

    action = baseline_action(observation)
    new_state = update_car(state, action, dt=0.1)

    assert set(new_state) == {"x", "y", "heading", "speed"}


def test_baseline_action_drives_forward_over_multiple_steps():
    state = {"x": 0.0, "y": 0.0, "heading": 0.0, "speed": 0.0}
    for _ in range(50):
        observation = {
            "speed": state["speed"],
            "lateral_offset": 0.0,
            "heading_error": 0.0,
            "curvature_ahead": [0.0, 0.0, 0.0],
        }
        state = update_car(state, baseline_action(observation), dt=0.1)

    assert state["x"] > 0
    assert state["speed"] > 0


def test_baseline_corner_action_turns_the_physics_model():
    state = {"x": 0.0, "y": 0.0, "heading": 0.0, "speed": 10.0}
    observation = {
        "speed": state["speed"],
        "lateral_offset": 0.0,
        "heading_error": 0.0,
        "curvature_ahead": [0.05, 0.04, 0.03],
    }

    action = baseline_action(observation)
    new_state = update_car(state, action, dt=0.1)

    assert action["steering"] > 0
    assert new_state["heading"] > 0