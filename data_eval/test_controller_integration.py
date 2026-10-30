import pytest
from data_eval.baseline import baseline_action
from data_eval.evaluation import evaluate_controller
from f1_simulator.core.track import Track


def test_baseline_completes_real_simulated_lap():
    result = evaluate_controller(baseline_action, Track(), dt=0.05)
    assert result['completed_lap']
    assert result['off_track_events'] == 0
    assert 0 < result['lap_time'] < 300


def test_stopped_controller_times_out():
    result = evaluate_controller(
        lambda obs: dict(steering=0., throttle=0., brake=1.), Track(), dt=1.)
    assert not result['completed_lap']
    assert result['lap_time'] is None
    assert result['distance_travelled'] == 0


@pytest.mark.parametrize('dt', [0, -0.1, float('nan'), 2])
def test_invalid_timestep(dt):
    with pytest.raises(ValueError):
        evaluate_controller(baseline_action, Track(), dt=dt)
