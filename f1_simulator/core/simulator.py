import math
import numpy as np

_session = {}

def reset_simulation(track):
    global _session
    x, y = track.position(0)
    _session = dict(time=0., prev_s=0., progress=0., laps=0, off_events=0, was_off=False)
    state = dict(x=x, y=y, heading=track.heading(0), speed=0.)
    return state, _race_info(state, track, False)

def _race_info(state, track, lap_complete):
    s, offset = track.project(state['x'], state['y'])
    return dict(distance=float(s), elapsed_time=_session['time'], off_track=bool(offset > track.left_width(s) or offset < -track.right_width(s)), lap_complete=lap_complete)

def simulate_step(state, action, dt, track):
    from .physics import update_car
    if not 0 < dt <= 1:
        raise ValueError('dt must be between 0 and 1 second')
    new = update_car(state, action, dt)
    s, _ = track.project(new['x'], new['y'])
    delta = (s - _session['prev_s'] + track.length / 2) % track.length - track.length / 2
    if abs(delta) > max(5, (state['speed'] + 10) * dt * 2):
        delta = 0
    _session['progress'] = max(0, _session['progress'] + delta)
    _session['prev_s'] = s
    _session['time'] += dt
    laps = int(_session['progress'] // track.length)
    complete = laps > _session['laps']
    _session['laps'] = laps
    info = _race_info(new, track, complete)
    if info['off_track'] and not _session['was_off']:
        _session['off_events'] += 1
    _session['was_off'] = info['off_track']
    return new, info

def get_observation(state, track, lookahead_offsets=None):
    if lookahead_offsets is None:
        lookahead_offsets = (0., 10., 20.)
    s, offset = track.project(state['x'], state['y'])
    error = (state['heading'] - track.heading(s) + math.pi) % (2 * math.pi) - math.pi
    return dict(speed=float(state['speed']), lateral_offset=float(offset), heading_error=float(error), curvature_ahead=[float(track.curvature(s + d)) for d in lookahead_offsets])

