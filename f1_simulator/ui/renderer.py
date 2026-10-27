import math
import numpy as np

from ..core import simulator as simulation
from ..utils.assets import load_car_images

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
        _car_images = load_car_images()
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
    info = simulation._race_info(state, track, False)
    labels = [f"SPEED: {state['speed'] * 3.6:.1f} km/h", f"TIME: {simulation._session['time']:.1f} s", f"LAPS: {simulation._session['laps']}", f"DISTANCE: {info['distance']:.1f} / {track.length:.1f} m", f"OFF TRACK: {info['off_track']}", f"OFF-TRACK EVENTS: {simulation._session['off_events']}"]
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
