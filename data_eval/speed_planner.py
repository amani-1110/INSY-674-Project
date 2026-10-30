"""Conservative curvature speed envelope around learned pedal predictions."""
from dataclasses import asdict, dataclass
import math
import numpy as np

from data_eval.ai_controller import make_ai_controller, resolve_pedals, PEDAL_POLICY
from data_eval.baseline import baseline_action


@dataclass(frozen=True)
class PlannerConfig:
    horizon_m: float = 500.
    sample_spacing_m: float = 5.
    max_speed_mps: float = 45.
    lateral_accel_mps2: float = 4.
    braking_decel_mps2: float = 5.
    braking_buffer_m: float = 15.

    def __post_init__(self):
        for name, value in asdict(self).items():
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")


def target_speed(curvatures, offsets, config=PlannerConfig()):
    """v_corner=sqrt(a_lat/abs(k)); v_now^2<=v_corner^2+2*a_brake*d.

    Each future point constrains the current speed. A distance buffer starts
    braking early to account for discrete sampling and controller delay.
    """
    curvature, distances = np.asarray(curvatures, dtype=float), np.asarray(offsets, dtype=float)
    if (curvature.ndim != 1 or curvature.shape != distances.shape or len(distances) == 0
            or distances[0] != 0 or np.any(np.diff(distances) <= 0)
            or not np.isfinite(curvature).all() or not np.isfinite(distances).all()):
        raise ValueError("Need finite curvature and strictly increasing offsets starting at zero")
    corner_squared = np.minimum(config.max_speed_mps**2,
                                config.lateral_accel_mps2 / np.maximum(np.abs(curvature), 1e-9))
    available = np.maximum(distances-config.braking_buffer_m, 0.)
    return float(np.sqrt(np.min(corner_squared + 2*config.braking_decel_mps2*available)))


def make_hybrid_controller(model, predict_action, config=PlannerConfig(), track=None):
    return _make_planned_controller(make_ai_controller(model, predict_action), config, track)


def make_planner_controller(config=PlannerConfig(), track=None, lookahead_offsets=(0.,10.,20.)):
    """No learned model: shared planner/steering with rule-based speed tracking.

    A neutral pedal proposal leaves acceleration/braking entirely to the same
    speed-tracking overrides used by the hybrid. This isolates learned pedals.
    """
    def rules(observation):
        pedals = resolve_pedals(dict(throttle=0., brake=0.), observation)
        return dict(steering=baseline_action(observation)["steering"], **pedals)
    rules.lookahead_offsets = tuple(lookahead_offsets)
    rules.pedal_policy = PEDAL_POLICY.copy()
    return _make_planned_controller(rules, config, track)


def _make_planned_controller(learned, config, track):
    model_offsets = tuple(learned.lookahead_offsets)
    planner_offsets = tuple(np.r_[np.arange(0., config.horizon_m, config.sample_spacing_m), config.horizon_m])
    offsets = model_offsets if track is not None else tuple(sorted(set(model_offsets) | set(planner_offsets)))
    if track is not None:
        sample_s = np.linspace(0, track.length, math.ceil(track.length/config.sample_spacing_m)+1)
        sample_k = np.array([track.curvature(s) for s in sample_s])
    model_indices = [offsets.index(d) for d in model_offsets]
    planner_indices = None if track is not None else [offsets.index(d) for d in planner_offsets]

    def controller(observation):
        curvature = observation["curvature_ahead"]
        if len(curvature) != len(offsets):
            raise ValueError("Hybrid controller needs its configured dense curvature observations")
        model_observation = dict(observation, curvature_ahead=[curvature[i] for i in model_indices])
        action = learned(model_observation)
        planning_curvature = (observation["planner_curvature_ahead"] if track is not None else
                              [curvature[i] for i in planner_indices])
        target = target_speed(planning_curvature, planner_offsets, config)
        # Slow down if the unchanged steering controller is already deviating.
        if abs(observation["lateral_offset"]) > 2. or abs(observation["heading_error"]) > .2:
            target = min(target, 12.)
        error = target - observation["speed"]
        if error < -.3:
            action.update(throttle=0., brake=min(1., .08-.12*error))
        elif error > 1.:
            action.update(throttle=max(action["throttle"], min(.5, .12*error)), brake=0.)
        elif action["throttle"] > 0:
            action["throttle"] = min(action["throttle"], .15)
        controller.last_target_speed = target
        return action

    if track is not None:
        def observe(state, active_track):
            if active_track is not track:
                raise ValueError("Hybrid track cache belongs to another track")
            from f1_simulator.core.simulator import get_observation
            obs = get_observation(state, track, lookahead_offsets=model_offsets)
            s, _ = track.project(state["x"], state["y"])
            distances = np.mod(s + np.asarray(planner_offsets), track.length)
            obs["planner_curvature_ahead"] = np.interp(distances, sample_s, sample_k).tolist()
            return obs
        controller.observe = observe
    controller.lookahead_offsets = offsets
    controller.planner_config = asdict(config)
    controller.last_target_speed = None
    controller.pedal_policy = learned.pedal_policy
    return controller
