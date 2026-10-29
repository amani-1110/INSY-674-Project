import math
import numpy as np

from ..core import simulator as simulation
from ..utils.assets import load_car_images

_renderer = None
_selected_car = 0
_car_images = None
_CAMERA_SCALE = 4.5
_CAR_LENGTH_PIXELS = 42
_CAR_LABEL_COLORS = {
    "YOU": (40, 130, 255),
    "AI": (235, 65, 65),
    "VER": (255, 190, 45),
    "BASELINE": (50, 210, 170),
}


def fit_track_view(track, screen_size=(1100, 750), margin=70):
    """Return a center and scale that fit the full track within the viewport."""
    screen_width, screen_height = screen_size
    distances = np.linspace(0, track.length, 1000, endpoint=False)
    points = []
    for distance in distances:
        points.extend(track.boundaries(distance))
    points = np.asarray(points, dtype=float)
    minimum = points.min(axis=0)
    maximum = points.max(axis=0)
    span = np.maximum(maximum - minimum, 1.0)
    available_width = max(screen_width - 2 * margin, 1)
    available_height = max(screen_height - 2 * margin - 50, 1)
    scale = min(available_width / span[0], available_height / span[1])
    center = tuple((minimum + maximum) / 2)
    return center, float(scale)


def set_car(index):
    """Select one of the four car sprites (0 through 3)."""
    global _selected_car
    if index not in range(4):
        raise ValueError("car index must be 0, 1, 2, or 3")
    _selected_car = index


def render(
    track,
    state,
    cars=None,
    hud_lines=None,
    camera_scale=None,
    camera_center=None,
    race_message=None,
    race_lights=0,
    lights_go=False,
):
    global _renderer, _car_images
    import pygame

    if _renderer is None:
        pygame.init()
        screen = pygame.display.set_mode((1100, 750))
        pygame.display.set_caption("F1 AI Simulator")
        _car_images = load_car_images()
        _renderer = (
            screen,
            _CAMERA_SCALE,
            pygame.font.SysFont("consolas", 20),
            pygame.font.SysFont("consolas", 15, bold=True),
        )
    screen, default_scale, font, tag_font = _renderer
    screen_width, screen_height = screen.get_size()
    scale = default_scale if camera_scale is None else camera_scale
    camera = (
        (state["x"], state["y"])
        if camera_center is None
        else camera_center
    )

    def pixel(point):
        return (
            int(screen_width / 2 + (point[0] - camera[0]) * scale),
            int(screen_height / 2 - (point[1] - camera[1]) * scale),
        )

    screen.fill((30, 85, 55))
    sample_count = max(300, min(1600, int(track.length / 3)))
    distances = np.linspace(0, track.length, sample_count, endpoint=False)
    boundaries = [track.boundaries(distance) for distance in distances]
    left = [pixel(pair[0]) for pair in boundaries]
    right = [pixel(pair[1]) for pair in boundaries]
    pygame.draw.polygon(screen, (95, 98, 103), left + right[::-1])
    pygame.draw.lines(screen, (235, 235, 235), True, left, 2)
    pygame.draw.lines(screen, (235, 235, 235), True, right, 2)
    start_left, start_right = track.boundaries(0)
    pygame.draw.line(
        screen,
        (255, 255, 255),
        pixel(start_left),
        pixel(start_right),
        3,
    )
    if cars is None:
        cars = [{
            "state": state,
            "label": "CAR",
            "car_index": _selected_car,
        }]
    for car in cars:
        car_state = car["state"]
        heading = car_state["heading"]
        position = (car_state["x"], car_state["y"])
        screen_position = pixel(position)
        sprite = _car_images[car.get("car_index", _selected_car)]
        car_width = max(10, int(_CAR_LENGTH_PIXELS * sprite.get_width() / sprite.get_height()))
        sprite = pygame.transform.smoothscale(
            sprite, (car_width, _CAR_LENGTH_PIXELS)
        )
        sprite = pygame.transform.rotate(sprite, math.degrees(heading) - 90)
        if car.get("ghost"):
            sprite.set_alpha(150)
        screen.blit(sprite, sprite.get_rect(center=screen_position))

        name = car.get("label")
        if name and (
            -50 < screen_position[0] < screen_width + 50
            and -50 < screen_position[1] < screen_height + 50
        ):
            tag = tag_font.render(name, True, (255, 255, 255))
            tag_background = pygame.Surface(
                (tag.get_width() + 12, tag.get_height() + 6),
                pygame.SRCALPHA,
            )
            color = _CAR_LABEL_COLORS.get(name, (25, 30, 38))
            tag_background.fill((*color, 220))
            tag_position = (
                screen_position[0] - tag.get_width() // 2 - 6,
                screen_position[1] - _CAR_LENGTH_PIXELS // 2 - tag.get_height() - 10,
            )
            screen.blit(tag_background, tag_position)
            screen.blit(
                tag,
                (tag_position[0] + 6, tag_position[1] + 3),
            )

    labels = hud_lines
    if labels is None:
        info = simulation.get_race_status(state, track)
        labels = [
            f"SPEED: {state['speed'] * 3.6:.1f} km/h",
            f"TIME: {info['elapsed_time']:.1f} s",
            f"LAPS: {info['laps']}",
            f"DISTANCE: {info['distance']:.1f} / {track.length:.1f} m",
            f"OFF TRACK: {info['off_track']}",
            f"OFF-TRACK EVENTS: {info['off_track_events']}",
        ]
    for i, line in enumerate(labels):
        label = font.render(line, True, (255, 255, 255))
        background = pygame.Surface((label.get_width() + 16, label.get_height() + 6))
        background.set_alpha(180)
        background.fill((12, 20, 28))
        screen.blit(background, (14, 12 + 25 * i))
        screen.blit(label, (22, 15 + 25 * i))
    instructions = "Arrows/WASD: drive | R: reset | C: choose car | Esc: quit"
    screen.blit(
        font.render(instructions, True, (255, 255, 255)),
        (20, screen_height - 40),
    )
    if race_message is not None:
        panel = pygame.Rect(
            screen_width // 2 - 210,
            screen_height // 2 - 82,
            420,
            164,
        )
        pygame.draw.rect(screen, (12, 18, 26), panel, border_radius=18)
        pygame.draw.rect(
            screen,
            (180, 190, 200),
            panel,
            width=2,
            border_radius=18,
        )
        title = font.render(race_message, True, (255, 255, 255))
        screen.blit(title, title.get_rect(center=(screen_width // 2, panel.y + 38)))
        for index in range(5):
            color = (
                (35, 220, 95)
                if lights_go
                else (235, 45, 50)
                if index < race_lights
                else (65, 65, 70)
            )
            center = (
                screen_width // 2 - 120 + index * 60,
                panel.y + 105,
            )
            pygame.draw.circle(screen, (5, 5, 8), center, 23)
            pygame.draw.circle(screen, color, center, 16)
    pygame.display.flip()


def close_renderer():
    global _renderer
    if _renderer:
        import pygame
        pygame.quit()
        _renderer = None
