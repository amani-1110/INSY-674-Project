from types import SimpleNamespace
import pytest
from f1_simulator.controllers.ai_controller import resolve_pedals, make_ai_controller
from f1_simulator.controllers.baseline import baseline_action


def observation(speed=20., offset=0., heading=0., curvature=0.):
    return dict(speed=speed, lateral_offset=offset, heading_error=heading,
                curvature_ahead=[curvature, 0., 0.])


@pytest.mark.parametrize("throttle,brake", [(.3,.13), (.8,.5), (.9,.9), (0.,0.), (2.,-1.)])
def test_mutually_exclusive_commands(throttle, brake):
    result = resolve_pedals(dict(throttle=throttle, brake=brake), observation())
    assert result["throttle"]*result["brake"] == 0
    assert 0 <= result["throttle"] <= 1
    assert 0 <= result["brake"] <= 1


def test_launch_and_normal_braking():
    prediction = dict(throttle=.1, brake=.9)
    assert resolve_pedals(prediction, observation(speed=0)) == dict(throttle=.4, brake=0.)
    assert resolve_pedals(prediction, observation(speed=10)) == dict(throttle=0., brake=.9)


@pytest.mark.parametrize("obs", [observation(speed=0, offset=4), observation(speed=0, heading=1),
                                  observation(speed=0, curvature=.1)])
def test_launch_does_not_override_braking_when_misaligned_or_in_tight_corner(obs):
    assert resolve_pedals(dict(throttle=.1, brake=.9), obs) == dict(throttle=0., brake=.9)


def test_raw_reproduction_and_unchanged_steering():
    prediction = dict(throttle=.3, brake=.13)
    model = SimpleNamespace(lookahead_offsets_=(0.,10.,20.))
    obs = observation()
    predict = lambda model, observation: prediction
    raw = make_ai_controller(model,predict,pedal_policy="raw")
    corrected = make_ai_controller(model,predict)
    assert raw(obs)["brake"] == .13
    assert corrected(obs)["brake"] == 0
    assert raw(obs)["steering"] == corrected(obs)["steering"] == baseline_action(obs)["steering"]
