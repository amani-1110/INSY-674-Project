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
    from physics import update_car
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

def get_observation(state, track):
    s, offset = track.project(state['x'], state['y'])
    error = (state['heading'] - track.heading(s) + math.pi) % (2 * math.pi) - math.pi
    return dict(speed=float(state['speed']), lateral_offset=float(offset), heading_error=float(error), curvature_ahead=[float(track.curvature(s + d)) for d in (0, 10, 20)])

_renderer = None
_selected_car = 0
_car_images = None

def set_car(index):
    """Select one of the four car sprites (0 through 3)."""
    global _selected_car
    if index not in range(4):
        raise ValueError("car index must be 0, 1, 2, or 3")
    _selected_car = index

def render(track, state):
    global _renderer, _car_images
    import pygame
    if _renderer is None:
        pygame.init()
        screen = pygame.display.set_mode((1100, 750))
        pygame.display.set_caption('F1 AI Simulator')
        pts = np.array([track.position(s) for s in np.linspace(0, track.length, 600)])
        center = pts.mean(axis=0)
        scale = min(900 / max(np.ptp(pts[:, 0]), 1), 550 / max(np.ptp(pts[:, 1]), 1))
        from pathlib import Path
        folder = Path(__file__).resolve().parent / "assets"
        _car_images = [pygame.image.load(str(folder / f"car_{i}.png")).convert_alpha() for i in range(1, 5)]
        _renderer = (screen, center, scale, pygame.font.SysFont('consolas', 20))
    screen, center, scale, font = _renderer
    def pixel(p):
        return (int(550 + (p[0] - center[0]) * scale), int(415 - (p[1] - center[1]) * scale))
    screen.fill((30, 85, 55))
    ss = np.linspace(0, track.length, 600)
    left = [pixel(track.boundaries(s)[0]) for s in ss]
    right = [pixel(track.boundaries(s)[1]) for s in ss]
    pygame.draw.polygon(screen, (95, 98, 103), left + right[::-1])
    pygame.draw.lines(screen, (235, 235, 235), True, left, 3)
    pygame.draw.lines(screen, (235, 235, 235), True, right, 3)
    a, b = track.boundaries(0)
    pygame.draw.line(screen, (255, 255, 255), pixel(a), pixel(b), 5)
    h = state['heading']
    f = np.array([math.cos(h), math.sin(h)])
    n = np.array([-math.sin(h), math.cos(h)])
    p = np.array([state['x'], state['y']])
    car = [pixel(p + 4 * f), pixel(p - 3 * f + 2 * n), pixel(p - 3 * f - 2 * n)]
    # scale the top-down sprite to the track, then rotate its nose to match heading
    sprite = _car_images[_selected_car]
    car_length = max(24, min(75, int(8 * scale)))
    car_width = max(12, int(car_length * sprite.get_width() / sprite.get_height()))
    sprite = pygame.transform.smoothscale(sprite, (car_width, car_length))
    sprite = pygame.transform.rotate(sprite, math.degrees(h) - 90)
    screen.blit(sprite, sprite.get_rect(center=pixel(p)))
    info = _race_info(state, track, False)
    labels = [f"SPEED: {state['speed'] * 3.6:.1f} km/h", f"TIME: {_session['time']:.1f} s", f"LAPS: {_session['laps']}", f"DISTANCE: {info['distance']:.1f} / {track.length:.1f} m", f"OFF TRACK: {info['off_track']}", f"OFF-TRACK EVENTS: {_session['off_events']}"]
    for i, line in enumerate(labels):
        screen.blit(font.render(line, True, (255, 255, 255)), (20, 16 + 27 * i))
    screen.blit(font.render('Arrows/WASD: drive | R: reset | C: choose car | Esc: quit', True, (255,255,255)), (20, 710))
    pygame.display.flip()

def close_renderer():
    global _renderer
    if _renderer:
        import pygame
        pygame.quit()
        _renderer = None
