from track import Track
from simulator import reset_simulation, simulate_step, get_observation

def test_contracts():
    track = Track()
    state, info = reset_simulation(track)
    assert set(info) == {'distance','elapsed_time','off_track','lap_complete'}
    state, info = simulate_step(state, {'steering':0.,'throttle':1.,'brake':0.}, .1, track)
    assert info['elapsed_time'] == .1
    obs = get_observation(state, track)
    assert set(obs) == {'speed','lateral_offset','heading_error','curvature_ahead'}
    assert len(obs['curvature_ahead']) == 3
