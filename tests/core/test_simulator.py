from f1_simulator.core.track import Track
from f1_simulator.core.simulator import (
    get_observation,
    get_race_status,
    reset_simulation,
    simulate_step,
)

def test_contracts():
    track = Track()
    state, info = reset_simulation(track)
    assert set(info) == {'distance','elapsed_time','off_track','lap_complete'}
    state, info = simulate_step(state, {'steering':0.,'throttle':1.,'brake':0.}, .1, track)
    assert info['elapsed_time'] == .1
    obs = get_observation(state, track)
    assert set(obs) == {'speed','lateral_offset','heading_error','curvature_ahead'}
    assert len(obs['curvature_ahead']) == 3

def test_race_status_exposes_counters_without_private_session_access():
    track = Track()
    state, _ = reset_simulation(track)
    state, _ = simulate_step(
        state,
        {'steering': 0., 'throttle': 1., 'brake': 0.},
        .1,
        track,
    )

    status = get_race_status(state, track)

    assert status['elapsed_time'] == .1
    assert status['laps'] == 0
    assert status['off_track_events'] == 0
