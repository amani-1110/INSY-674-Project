import sys
import types

# Create a temporary simulator for testing
mock_simulator = types.ModuleType("simulator")


def reset_simulation(track):
    state = {"x": 0.0, "y": 0.0}
    race_info = {"off_track": False, "completed_lap": False}
    return state, race_info


def get_observation(state, track):
    return {
        "speed": 10.0,
        "lateral_offset": 0.0,
        "heading_error": 0.0,
        "curvature_ahead": [0.0, 0.0, 0.0],
    }


def simulate_step(state, action, dt, track):
    new_state = {
        "x": state["x"] + 10 * dt,
        "y": state["y"],
    }

    race_info = {
       "off_track": 3.0 <= new_state["x"] < 5.0,
        "lap_complete": False,
    }

    return new_state, race_info


mock_simulator.reset_simulation = reset_simulation
mock_simulator.get_observation = get_observation
mock_simulator.simulate_step = simulate_step

sys.modules["simulator"] = mock_simulator

from evaluation import evaluate_controller
from baseline import baseline_action

results = evaluate_controller(baseline_action, track={})

print("Evaluation results:", results)

assert results["completed_lap"] is False
assert results["lap_time"] is None
assert abs(results["distance_travelled"] - 3000.0) < 1e-6
assert results["off_track_events"] == 1

print("All controller integration tests passed!")
