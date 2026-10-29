"""Simple circular car-to-car collision resolution."""

import math

from ..physics.detailed import MAX_SPEED_MPS


def resolve_car_collisions(cars, radius=2.2, iterations=3):
    """Prevent car overlap and apply a low-speed, inelastic contact response.

    Each item contains a mutable ``state`` dictionary. ``ghost=True`` makes
    the car an immovable but moving barrier, suitable for telemetry replays.
    """
    if radius <= 0 or iterations < 1:
        raise ValueError("Collision radius and iteration count must be positive.")

    for _ in range(iterations):
        for first_index in range(len(cars)):
            for second_index in range(first_index + 1, len(cars)):
                first = cars[first_index]
                second = cars[second_index]
                first_state = first["state"]
                second_state = second["state"]

                dx = second_state["x"] - first_state["x"]
                dy = second_state["y"] - first_state["y"]
                distance = math.hypot(dx, dy)
                min_distance = 2 * radius
                if distance >= min_distance:
                    continue

                if distance <= 1e-9:
                    normal = (1.0, 0.0)
                    distance = 0.0
                else:
                    normal = (dx / distance, dy / distance)

                first_inverse_mass = 0.0 if first.get("ghost") else 1.0
                second_inverse_mass = 0.0 if second.get("ghost") else 1.0
                inverse_mass_sum = first_inverse_mass + second_inverse_mass
                if inverse_mass_sum == 0:
                    continue

                penetration = min_distance - distance + 1e-3
                first_state["x"] -= (
                    normal[0] * penetration * first_inverse_mass / inverse_mass_sum
                )
                first_state["y"] -= (
                    normal[1] * penetration * first_inverse_mass / inverse_mass_sum
                )
                second_state["x"] += (
                    normal[0] * penetration * second_inverse_mass / inverse_mass_sum
                )
                second_state["y"] += (
                    normal[1] * penetration * second_inverse_mass / inverse_mass_sum
                )

                first_velocity = _velocity(first_state)
                second_velocity = _velocity(second_state)
                relative_normal_speed = (
                    (second_velocity[0] - first_velocity[0]) * normal[0]
                    + (second_velocity[1] - first_velocity[1]) * normal[1]
                )
                if relative_normal_speed >= 0:
                    continue

                impulse = -relative_normal_speed / inverse_mass_sum
                if first_inverse_mass:
                    first_velocity[0] -= impulse * normal[0]
                    first_velocity[1] -= impulse * normal[1]
                    _set_velocity(first_state, first_velocity)
                if second_inverse_mass:
                    second_velocity[0] += impulse * normal[0]
                    second_velocity[1] += impulse * normal[1]
                    _set_velocity(second_state, second_velocity)

    for car in cars:
        session = car.get("session")
        if session is not None:
            session.state = car["state"].copy()


def _velocity(state):
    return [
        state["speed"] * math.cos(state["heading"]),
        state["speed"] * math.sin(state["heading"]),
    ]


def _set_velocity(state, velocity):
    speed = math.hypot(*velocity)
    if speed > 1e-9:
        state["heading"] = math.atan2(velocity[1], velocity[0])
    state["speed"] = min(speed, MAX_SPEED_MPS)
