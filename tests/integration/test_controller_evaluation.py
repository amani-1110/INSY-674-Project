from f1_simulator.controllers.baseline import baseline_action
from f1_simulator.evaluation import evaluate_controller
from f1_simulator.evaluation import runner


def test_evaluation_uses_packaged_simulator_and_counts_off_track_entries(monkeypatch):
    def reset_simulation(track):
        return {"x": 0.0, "y": 0.0}, {"off_track": False}

    def get_observation(state, track):
        return {
            "speed": 10.0,
            "lateral_offset": 0.0,
            "heading_error": 0.0,
            "curvature_ahead": [0.0, 0.0, 0.0],
        }

    def simulate_step(state, action, dt, track):
        new_state = {"x": state["x"] + 10 * dt, "y": state["y"]}
        return new_state, {
            "off_track": 3.0 <= new_state["x"] < 5.0,
            "lap_complete": False,
        }

    monkeypatch.setattr(runner.simulator, "reset_simulation", reset_simulation)
    monkeypatch.setattr(runner.simulator, "get_observation", get_observation)
    monkeypatch.setattr(runner.simulator, "simulate_step", simulate_step)

    results = evaluate_controller(baseline_action, track={})

    assert results["completed_lap"] is False
    assert results["lap_time"] is None
    assert results["distance_travelled"] == 3000.0
    assert results["off_track_events"] == 1
