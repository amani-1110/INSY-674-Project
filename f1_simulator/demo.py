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
    parser.add_argument("--controller", choices=["manual", "baseline", "imitation", "hybrid", "planner", "learned"], default="manual")
    parser.add_argument("--model", default="output/pipeline_250m/imitation_model.joblib")
    parser.add_argument("--policy", default="output/learned_driver/policy.joblib")
    parser.add_argument("--view", choices=["follow", "overview"], default="follow")
    args = parser.parse_args()
    track = Track.from_csv(args.track_csv) if args.track_csv else Track(args.track)
    controller = None
    if args.controller == "baseline":
        from data_eval.baseline import baseline_action
        controller = baseline_action
    elif args.controller == "learned":
        from .core.full_policy import load_learned_driver
        controller = load_learned_driver(args.policy, track)
    elif args.controller == "planner":
        from data_eval.speed_planner import make_planner_controller
        controller = make_planner_controller(track=track)
    elif args.controller in ("imitation", "hybrid"):
        from f1_simulator.core.imitation import load_model, predict_action
        from data_eval.ai_controller import make_ai_controller
        model = load_model(args.model)
        model.n_jobs = 1
        if args.controller == "hybrid":
            from data_eval.speed_planner import make_hybrid_controller
            controller = make_hybrid_controller(model, predict_action, track=track)
        else:
            controller = make_ai_controller(model, predict_action)
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
    camera_mode, zoom = args.view, 5.
    running = True
    while running:
        dt = min(clock.tick(60) / 1000, .05)
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_v:
                camera_mode = "overview" if camera_mode == "follow" else "follow"
            elif event.type == pygame.KEYDOWN and event.key in (pygame.K_EQUALS, pygame.K_PLUS, pygame.K_KP_PLUS):
                zoom = min(10., zoom + 1.)
            elif event.type == pygame.KEYDOWN and event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                zoom = max(2., zoom - 1.)
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
        if controller is not None:
            from .core.simulator import get_observation
            observer = getattr(controller, "observe", None)
            observation = (observer(state, track) if observer is not None else
                           get_observation(state, track,
                               lookahead_offsets=getattr(controller, "lookahead_offsets", None)))
            action = controller(observation)
        state, info = simulate_step(state, action, max(dt, .001), track)
        render(track, state, controller_name=args.controller, camera_mode=camera_mode, zoom=zoom, action=action)
    close_renderer()


if __name__ == '__main__':
    main()
