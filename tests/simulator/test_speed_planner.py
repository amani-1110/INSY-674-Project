import math
from types import SimpleNamespace
import pytest
from f1_simulator.controllers.speed_planner import PlannerConfig,target_speed,make_hybrid_controller
from f1_simulator.controllers.ai_controller import make_ai_controller


def test_straight_and_corner_limits():
    config=PlannerConfig()
    assert target_speed([0,0,0],[0,100,500],config) == config.max_speed_mps
    assert target_speed([.04,0],[0,100],config) == pytest.approx(10.)


def test_future_corner_starts_braking_in_advance():
    near=target_speed([0,.04],[0,40])
    far=target_speed([0,.04],[0,100])
    assert 10 < near < far < 45
    assert near == pytest.approx(math.sqrt(100+2*5*(40-15)))


def test_hybrid_preserves_steering_and_suppresses_unsafe_throttle():
    model=SimpleNamespace(lookahead_offsets_=(0.,10.,20.))
    predict=lambda model,obs:dict(throttle=1.,brake=0.)
    hybrid=make_hybrid_controller(model,predict)
    offsets=hybrid.lookahead_offsets
    obs=dict(speed=30.,lateral_offset=0.,heading_error=0.,
             curvature_ahead=[.04 for _ in offsets])
    action=hybrid(obs)
    plain=make_ai_controller(model,predict)
    assert action["steering"] == plain(dict(obs,curvature_ahead=[.04]*3))["steering"]
    assert action["throttle"] == 0
    assert action["brake"] > 0


def test_hybrid_does_not_stall_from_model_braking_on_straight():
    model=SimpleNamespace(lookahead_offsets_=(0.,10.,20.))
    hybrid=make_hybrid_controller(model,lambda model,obs:dict(throttle=0.,brake=1.))
    obs=dict(speed=10.,lateral_offset=0.,heading_error=0.,curvature_ahead=[0.]*len(hybrid.lookahead_offsets))
    action=hybrid(obs)
    assert action["throttle"] > 0 and action["brake"] == 0


@pytest.mark.parametrize("kwargs", [dict(max_speed_mps=0),dict(braking_decel_mps2=-1)])
def test_invalid_config(kwargs):
    with pytest.raises(ValueError):
        PlannerConfig(**kwargs)


def test_cached_planner_samples_track_once_and_wraps(monkeypatch):
    from f1_simulator.core.track import Track
    from f1_simulator.core.simulator import reset_simulation
    track=Track()
    model=SimpleNamespace(lookahead_offsets_=(0.,10.,20.))
    hybrid=make_hybrid_controller(model,lambda model,obs:dict(throttle=.5,brake=0.),track=track)
    state,_=reset_simulation(track)
    x,y=track.position(track.length-2)
    state.update(x=x,y=y,heading=track.heading(track.length-2))
    original=track.curvature
    calls=[]
    def curvature(s):
        calls.append(s)
        return original(s)
    monkeypatch.setattr(track,"curvature",curvature)
    obs=hybrid.observe(state,track)
    assert len(calls) == 3  # model features only; planner uses cached interpolation
    assert len(obs["planner_curvature_ahead"]) == 101
    assert obs["planner_curvature_ahead"][1] == pytest.approx(original(3),abs=1e-5)
    action=hybrid(obs)
    assert action["throttle"]*action["brake"] == 0
    assert 0 < hybrid.last_target_speed <= 45


@pytest.mark.parametrize("speed,offset,heading,curvature", [
    (0.,0.,0.,0.), (10.,0.,0.,0.), (30.,0.,0.,.04),
    (20.,3.,0.,0.), (20.,0.,.3,0.), (9.5,0.,0.,.04)])
def test_planner_only_is_exact_neutral_model_ablation(speed,offset,heading,curvature):
    from f1_simulator.controllers.speed_planner import make_planner_controller
    model=SimpleNamespace(lookahead_offsets_=(0.,10.,20.))
    calls=[]
    def neutral_predict(model,obs):
        calls.append(1)
        return dict(throttle=0.,brake=0.)
    hybrid=make_hybrid_controller(model,neutral_predict)
    rules=make_planner_controller()
    assert rules.lookahead_offsets == hybrid.lookahead_offsets
    obs=dict(speed=speed,lateral_offset=offset,heading_error=heading,
             curvature_ahead=[curvature]*len(rules.lookahead_offsets))
    assert rules(obs) == hybrid(obs)
    assert len(calls) == 1
    assert rules.last_target_speed == hybrid.last_target_speed
