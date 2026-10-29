"""
physics.py — Vehicle physics for the F1 AI project (Person 2).

Moves a car in an empty 2D plane. No track dependency.

Public contract (do not change without telling the team):

    update_car(state: dict, action: dict, dt: float) -> dict

    state  = {"x": m, "y": m, "heading": rad, "speed": m/s}
    action = {"steering": rad (front-wheel angle, + = left),
              "throttle": 0..1, "brake": 0..1}
    dt     = seconds (> 0)

Conventions (shared team conventions):
    x right, y up, angles counter-clockwise, heading wrapped to [-pi, pi).
    Positive steering turns the car left (counter-clockwise).

Model: kinematic bicycle model (rear-axle reference point) with
    * power-limited drive force, capped by tyre traction,
    * aerodynamic drag and rolling resistance,
    * braking force that grows with speed (downforce),
    * a lateral-grip limit: if the requested steering would need more
      lateral acceleration than the tyres can give, the effective steering
      is reduced (the car "understeers" and runs wide instead of turning
      impossibly tight).

All parameters are approximate public-domain figures for a modern F1 car.
They are assumptions, not team-calibrated data. Change them in CarParams.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

G = 9.81  # m/s²
MAX_SPEED_KMH = 372.6
MAX_SPEED_MPS = MAX_SPEED_KMH / 3.6


@dataclass(frozen=True)
class CarParams:
    wheelbase: float = 3.6          # m
    mass: float = 800.0             # kg, car + driver (2024 minimum ~798 kg)
    max_power: float = 750_000.0    # W (~1000 hp)
    mu: float = 1.7                 # tyre friction coefficient (slick, warm)
    air_density: float = 1.225      # kg/m³
    drag_area: float = 1.08648      # Cd·A in m² -> terminal speed ≈ 103.5 m/s
    downforce_area: float = 4.0     # Cl·A in m²  -> ~3.5 g grip at 70 m/s
    rolling_coeff: float = 0.015    # rolling-resistance coefficient
    drive_traction_share: float = 0.6  # share of total grip usable for drive (rear-wheel drive)
    max_steer: float = 0.40         # rad, maximum front-wheel angle
    min_drive_speed: float = 1.0    # m/s, avoids power/0 at standstill
    max_speed: float = MAX_SPEED_MPS


DEFAULT_PARAMS = CarParams()

_STATE_KEYS = ("x", "y", "heading", "speed")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def wrap_angle(a: float) -> float:
    """Wrap an angle to [-pi, pi)."""
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def _clip(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


def grip_force(speed: float, p: CarParams = DEFAULT_PARAMS) -> float:
    """Total tyre force available (N): mu * (weight + downforce)."""
    downforce = 0.5 * p.air_density * p.downforce_area * speed * speed
    return p.mu * (p.mass * G + downforce)


def longitudinal_accel(speed: float, throttle: float, brake: float,
                       p: CarParams = DEFAULT_PARAMS) -> float:
    """Net forward acceleration (m/s²) for the given speed and pedal inputs."""
    grip = grip_force(speed, p)
    drive_limit = p.drive_traction_share * grip
    drive = throttle * min(p.max_power / max(speed, p.min_drive_speed), drive_limit)
    braking = brake * grip
    drag = 0.5 * p.air_density * p.drag_area * speed * speed
    rolling = p.rolling_coeff * p.mass * G if speed > 0 else 0.0
    return (drive - braking - drag - rolling) / p.mass


def effective_steering(speed: float, steering: float,
                       p: CarParams = DEFAULT_PARAMS) -> float:
    """
    Steering actually achieved after the mechanical lock and the grip limit.

    In a kinematic bicycle, lateral acceleration = v² · tan(δ) / L.
    Tyres can supply at most grip_force / mass, so tan(δ) is capped.
    """
    delta = _clip(steering, -p.max_steer, p.max_steer)
    if speed <= 0.0:
        return delta
    a_lat_max = grip_force(speed, p) / p.mass
    tan_max = a_lat_max * p.wheelbase / (speed * speed)
    tan_delta = math.tan(delta)
    if abs(tan_delta) > tan_max:
        tan_delta = math.copysign(tan_max, tan_delta)
    return math.atan(tan_delta)


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def update_car(state: dict, action: dict, dt: float,
               params: CarParams = DEFAULT_PARAMS) -> dict:
    """Return car state after dt simulated seconds.

    The input dicts are not modified. Missing action keys default to 0.
    Throttle/brake are clipped to [0, 1]; steering to ±max_steer.
    Speed never goes negative (no reverse gear).
    """
    if dt < 0:
        raise ValueError(f"dt must be >= 0, got {dt}")
    missing = [k for k in _STATE_KEYS if k not in state]
    if missing:
        raise KeyError(f"state is missing keys: {missing}")

    x = float(state["x"])
    y = float(state["y"])
    heading = float(state["heading"])
    v0 = _clip(float(state["speed"]), 0.0, params.max_speed)

    steering = float(action.get("steering", 0.0))
    throttle = _clip(float(action.get("throttle", 0.0)), 0.0, 1.0)
    brake = _clip(float(action.get("brake", 0.0)), 0.0, 1.0)

    if dt == 0:
        return {"x": x, "y": y, "heading": wrap_angle(heading), "speed": v0}

    # 1) Speed update. If the car would stop inside this step, stop it there
    #    and only integrate position over the time it was actually moving.
    a = longitudinal_accel(v0, throttle, brake, params)
    v1 = v0 + a * dt
    move_dt = dt
    if v1 < 0.0:
        move_dt = v0 / -a if a < 0 else 0.0
        v1 = 0.0
    v1 = min(v1, params.max_speed)
    v_avg = 0.5 * (v0 + v1)

    # 2) Heading update (kinematic bicycle): yaw rate = v · tan(δ) / L.
    delta = effective_steering(v_avg, steering, params)
    yaw_rate = v_avg * math.tan(delta) / params.wheelbase
    dpsi = yaw_rate * move_dt

    # 3) Position update. Exact arc integration for constant v and yaw rate;
    #    falls back to a straight line when the car is not turning.
    dist = v_avg * move_dt
    if abs(dpsi) < 1e-9:
        x += dist * math.cos(heading)
        y += dist * math.sin(heading)
    else:
        radius = dist / dpsi
        x += radius * (math.sin(heading + dpsi) - math.sin(heading))
        y -= radius * (math.cos(heading + dpsi) - math.cos(heading))
    heading = wrap_angle(heading + dpsi)

    return {"x": x, "y": y, "heading": heading, "speed": v1}


if __name__ == "__main__":
    # The team-doc example.
    s = {"x": 120.0, "y": 80.0, "heading": 0.0, "speed": 20.0}
    a = {"steering": 0.0, "throttle": 1.0, "brake": 0.0}
    print(update_car(s, a, 0.1))
