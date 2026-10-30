import pygame

from .core.track import Track
from .core.simulator import reset_simulation, simulate_step
from .ui.renderer import render, close_renderer, set_car
from .ui.car_selection import choose_car

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Drive on a spline track")
    parser.add_argument("--track", default="monza", help="Bundled circuit or demo")
    parser.add_argument("--track-csv", help="Custom centerline and widths CSV")
    args = parser.parse_args()
    track = Track.from_csv(args.track_csv) if args.track_csv else Track(args.track)
    pygame.init()
    screen = pygame.display.set_mode((1100, 750))
    pygame.display.set_caption('F1 AI Simulator - Car Selection')
    selection = choose_car(screen)
    if selection is None:
        pygame.quit()
        return
    set_car(selection)
    state, info = reset_simulation(track)
    clock = pygame.time.Clock()
    running = True
    while running:
        dt = min(clock.tick(60) / 1000, .05)
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                state, info = reset_simulation(track)
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_c:
                selection = choose_car(screen)
                if selection is None:
                    running = False
                else:
                    set_car(selection)
                    state, info = reset_simulation(track)
        if not running:
            break
        keys = pygame.key.get_pressed()
        action = dict(
            steering=.28 * (int(keys[pygame.K_LEFT] or keys[pygame.K_a]) - int(keys[pygame.K_RIGHT] or keys[pygame.K_d])),
            throttle=float(keys[pygame.K_UP] or keys[pygame.K_w]),
            brake=float(keys[pygame.K_DOWN] or keys[pygame.K_s]),
        )
        state, info = simulate_step(state, action, dt, track)
        render(track, state)
    close_renderer()


if __name__ == '__main__':
    main()
