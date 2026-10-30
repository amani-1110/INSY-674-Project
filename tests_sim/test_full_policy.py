import numpy as np
import pytest
from sklearn.dummy import DummyRegressor
from f1_simulator.core.full_policy import OFFSETS,LearnedDriver,model_features
from track import Track
from f1_simulator.core.simulator import reset_simulation


def policy_bundle(steer=.25,pedal=.6):
    return dict(schema="full_action_v1",offsets=OFFSETS,
                steering_model=DummyRegressor(strategy="constant",constant=steer).fit(np.zeros((2,3)),np.zeros(2)),
                pedal_model=DummyRegressor(strategy="constant",constant=pedal).fit(np.zeros((2,104)),np.zeros(2)))


def test_runtime_uses_learned_actions_without_teacher_or_planner(monkeypatch):
    import data_eval.speed_planner as planner
    import data_eval.baseline as baseline
    import data_eval.ai_controller as pedals
    def forbidden(*args,**kwargs):
        raise AssertionError("Driving assistance must not run inside the learned policy")
    monkeypatch.setattr(planner,"target_speed",forbidden)
    monkeypatch.setattr(baseline,"baseline_action",forbidden)
    monkeypatch.setattr(pedals,"resolve_pedals",forbidden)
    track=Track();driver=LearnedDriver(policy_bundle(),track)
    state,_=reset_simulation(track)
    result=driver(driver.observe(state,track))
    assert result == pytest.approx(dict(steering=.1,throttle=.6,brake=0.))
    assert not hasattr(driver,"last_target_speed")


@pytest.mark.parametrize("pedal,throttle,brake",[(.7,.7,0.),(-.8,0.,.8),(0.,0.,0.),(2.,1.,0.)])
def test_signed_action_encoding_and_no_launch_override(pedal,throttle,brake):
    track=Track();driver=LearnedDriver(policy_bundle(pedal=pedal),track)
    state,_=reset_simulation(track)
    action=driver(driver.observe(state,track))
    assert action["throttle"] == throttle
    assert action["brake"] == brake
    assert action["throttle"]*action["brake"] == 0


def test_input_schema_and_nan_rejected():
    with pytest.raises(ValueError):
        model_features(dict(speed=0,lateral_offset=0,heading_error=0,curvature_ahead=[0]*3))
    with pytest.raises(ValueError):
        model_features(dict(speed=float("nan"),lateral_offset=0,heading_error=0,curvature_ahead=[0]*101))
