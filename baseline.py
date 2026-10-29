"""Compatibility import for the baseline driving controller."""

from f1_simulator.controllers.baseline import baseline_action

__all__ = ["baseline_action"]


if __name__ == "__main__":
    observation = {
        "speed": 25.0,
        "lateral_offset": 0.0,
        "heading_error": 0.0,
        "curvature_ahead": [0.05, 0.04, 0.03],
    }
    print("Driving AI action:", baseline_action(observation))
