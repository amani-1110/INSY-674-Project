"""Compatibility imports for the detailed vehicle physics model."""

from f1_simulator.physics.detailed import (
    DEFAULT_PARAMS,
    MAX_SPEED_KMH,
    MAX_SPEED_MPS,
    CarParams,
    G,
    effective_steering,
    grip_force,
    longitudinal_accel,
    update_car,
    wrap_angle,
)

__all__ = [
    "DEFAULT_PARAMS",
    "MAX_SPEED_KMH",
    "MAX_SPEED_MPS",
    "CarParams",
    "G",
    "effective_steering",
    "grip_force",
    "longitudinal_accel",
    "update_car",
    "wrap_angle",
]


if __name__ == "__main__":
    state = {"x": 120.0, "y": 80.0, "heading": 0.0, "speed": 20.0}
    action = {"steering": 0.0, "throttle": 1.0, "brake": 0.0}
    print(update_car(state, action, 0.1))
