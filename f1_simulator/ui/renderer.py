"""Cached circuit rendering with follow camera and overview minimap."""
import math
import numpy as np

from ..core import simulator as simulation
from ..utils.assets import load_car_images

_renderer = None
_selected_car = 0


def set_car(index):
    """Select one of the four car sprites (0 through 3)."""
    global _selected_car
    if index not in range(4):
        raise ValueError("car index must be 0, 1, 2, or 3")
    _selected_car = index


def _pixels(points, center, scale):
    points = np.asarray(points)
    return np.rint((points - center) * [scale, -scale] + [550, 415]).astype(int)


def _initialize(track):
    import pygame
    pygame.init()
    screen = pygame.display.set_mode((1100, 750))
    pygame.display.set_caption("F1 AI Simulator")
    ss = np.linspace(0, track.length, 2401)
    # All expensive spline geometry is computed only once per track.
    boundary = np.array([track.boundaries(s) for s in ss])
    centerline = np.array([track.position(s) for s in ss])
    center = (centerline.min(axis=0) + centerline.max(axis=0)) / 2
    scale = min(900 / max(np.ptp(centerline[:, 0]), 1),
                550 / max(np.ptp(centerline[:, 1]), 1))
    background = pygame.Surface(screen.get_size()).convert()
    background.fill((30, 85, 55))
    _draw_road(background, boundary, centerline, center, scale, False)
    return dict(track=track, screen=screen, boundary=boundary, centerline=centerline,
                center=center, scale=scale, background=background,
                minimap=pygame.transform.smoothscale(background, (264, 180)),
                cars=load_car_images(), font=pygame.font.SysFont("consolas", 20))


def _draw_road(surface, boundary, centerline, center, scale, detailed):
    import pygame
    left = _pixels(boundary[:, 0], center, scale)
    right = _pixels(boundary[:, 1], center, scale)
    if detailed:
        # Filling a whole kilometre-scale ring at high zoom is expensive even
        # when clipped. Draw only road quads that overlap the viewport.
        quads = np.stack((left[:-1], left[1:], right[1:], right[:-1]), axis=1)
        lo, hi = quads.min(axis=1), quads.max(axis=1)
        visible = (hi[:, 0] >= 0) & (lo[:, 0] < surface.get_width()) & (hi[:, 1] >= 0) & (lo[:, 1] < surface.get_height())
        for quad in quads[visible]:
            pygame.draw.polygon(surface, (88, 92, 99), quad)
    else:
        pygame.draw.polygon(surface, (88, 92, 99), np.vstack((left, right[::-1])))
    pygame.draw.lines(surface, (240, 240, 235), True, left, 2 if detailed else 1)
    pygame.draw.lines(surface, (240, 240, 235), True, right, 2 if detailed else 1)
    if detailed:
        middle = _pixels(centerline, center, scale)
        for i in range(0, len(middle)-2, 8):
            pygame.draw.lines(surface, (150, 151, 149), False, middle[i:i+4], 1)
    pygame.draw.line(surface, (255, 255, 255), left[0], right[0], 4 if detailed else 2)


def render(track, state, controller_name="manual", camera_mode="follow", zoom=5., action=None):
    """Follow at zoom pixels/metre, or show the whole circuit in overview mode."""
    global _renderer
    import pygame
    if _renderer is None or _renderer["track"] is not track:
        _renderer = _initialize(track)
    context = _renderer
    screen, font = context["screen"], context["font"]
    car_position = np.array([state["x"], state["y"]])
    if camera_mode == "follow":
        center, scale = car_position, float(zoom)
        screen.fill((30, 85, 55))
        # Ground markers make movement visible even on a long straight.
        for x in np.arange(math.floor((center[0]-550/scale)/25)*25, center[0]+550/scale, 25):
            px = int(_pixels([x, center[1]], center, scale)[0])
            pygame.draw.line(screen, (34, 92, 61), (px, 0), (px, 750))
        for y in np.arange(math.floor((center[1]-415/scale)/25)*25, center[1]+415/scale, 25):
            py = int(_pixels([center[0], y], center, scale)[1])
            pygame.draw.line(screen, (34, 92, 61), (0, py), (1100, py))
        _draw_road(screen, context["boundary"], context["centerline"], center, scale, True)
    else:
        center, scale = context["center"], context["scale"]
        screen.blit(context["background"], (0, 0))
    sprite = context["cars"][_selected_car]
    # Six metres by two metres in follow view; overview uses a visible marker.
    car_length = max(14, round(6 * scale))
    car_width = max(6, round(2 * scale))
    sprite = pygame.transform.smoothscale(sprite, (car_width, car_length))
    sprite = pygame.transform.rotate(sprite, math.degrees(state["heading"]) - 90)
    screen.blit(sprite, sprite.get_rect(center=_pixels(car_position, center, scale)))
    if camera_mode == "follow":
        screen.blit(context["minimap"], (816, 20))
        pygame.draw.rect(screen, (180, 190, 180), (816, 20, 264, 180), 1)
        map_point = _pixels(car_position, context["center"], context["scale"])
        marker = (816 + round(map_point[0] * 264/1100), 20 + round(map_point[1] * 180/750))
        pygame.draw.circle(screen, (255, 210, 55), marker, 4)
    info = simulation._race_info(state, track, False)
    pygame.draw.rect(screen, (20, 47, 34), (10, 10, 390, 248), border_radius=8)
    labels = [f"SPEED: {state['speed'] * 3.6:.1f} km/h",
              f"TIME: {simulation._session['time']:.1f} s",
              f"LAPS: {simulation._session['laps']}",
              f"DISTANCE: {info['distance']:.1f} / {track.length:.1f} m",
              f"OFF TRACK: {info['off_track']}",
              f"OFF-TRACK EVENTS: {simulation._session['off_events']}",
              f"VIEW: {camera_mode.upper()}"]
    if action is not None:
        labels.append(f"THR: {100*action['throttle']:.0f}%  BRK: {100*action['brake']:.0f}%")
    for i, line in enumerate(labels):
        screen.blit(font.render(line, True, (255, 255, 255)), (20, 16 + 27*i))
    pygame.draw.rect(screen, (20, 47, 34), (0, 700, 1100, 50))
    mode = "WASD/arrows: drive" if controller_name == "manual" else f"AI: {controller_name}"
    text = mode + " | V: view | +/-: zoom | R: reset | C: car | Esc: quit"
    screen.blit(font.render(text, True, (255, 255, 255)), (20, 714))
    pygame.display.flip()


def close_renderer():
    global _renderer
    if _renderer is not None:
        import pygame
        pygame.quit()
        _renderer = None
