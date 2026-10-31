"""Join learned pedals to shared steering, with explicit pedal arbitration."""
import math

from f1_simulator.controllers.baseline import baseline_action

PEDAL_POLICY = {"name": "exclusive_v1", "brake_threshold": .5,
                "launch_speed_mps": 5., "launch_throttle": .4}


def resolve_pedals(prediction, observation):
    """Convert averaged labels into mutually exclusive driving commands.

    Binary FastF1 brake labels are averaged by the regressor: use 0.5 to
    choose braking rather than applying small averaged brake values as pressure.
    At walking speeds near the centerline, provide a modest launch input.
    These are controller rules, not additional learned behavior.
    """
    throttle, brake = float(prediction["throttle"]), float(prediction["brake"])
    if not math.isfinite(throttle) or not math.isfinite(brake):
        raise ValueError("Model pedal predictions must be finite")
    throttle, brake = max(0., min(1., throttle)), max(0., min(1., brake))
    speed = float(observation["speed"])
    centered = (abs(observation["lateral_offset"]) <= 3. and
                abs(observation["heading_error"]) <= .5)
    near_curvature = observation["curvature_ahead"][:3]
    gentle = max((abs(k) for k in near_curvature), default=0.) <= .05
    if speed < PEDAL_POLICY["launch_speed_mps"] and centered and gentle:
        return dict(throttle=max(throttle, PEDAL_POLICY["launch_throttle"]), brake=0.)
    if brake >= PEDAL_POLICY["brake_threshold"]:
        return dict(throttle=0., brake=brake)
    return dict(throttle=throttle, brake=0.)


def make_ai_controller(model, predict_action, pedal_policy="exclusive"):
    """Use the same steering as baseline; arbitrate learned pedals by default.

    pedal_policy="raw" reproduces the original simultaneous-pedal experiments.
    """
    if pedal_policy not in ("exclusive", "raw"):
        raise ValueError("pedal_policy must be exclusive or raw")

    def ai_controller(observation):
        baseline = baseline_action(observation)
        prediction = predict_action(model, observation)
        pedals = resolve_pedals(prediction, observation) if pedal_policy == "exclusive" else prediction
        return dict(steering=baseline["steering"], **pedals)

    ai_controller.lookahead_offsets = getattr(model, "lookahead_offsets_", (0., 10., 20.))
    ai_controller.pedal_policy = PEDAL_POLICY.copy() if pedal_policy == "exclusive" else {"name": "raw"}
    return ai_controller
