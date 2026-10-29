"""Interactive Monza session: player, learned controller, and telemetry replay."""

import math

import pygame

from .controllers.ai import make_ai_controller
from .controllers.baseline import baseline_action
from .controllers.imitation import predict
from .controllers.telemetry_model import train_telemetry_model
from .core.collision import resolve_car_collisions
from .core.simulator import RaceSession
from .core.telemetry_track import TelemetryTrack
from .data.telemetry_replay import TelemetryReplay
from .ui.car_selection import choose_car
from .ui.renderer import close_renderer, fit_track_view, render, set_car


def _new_sessions(track):
    return (
        RaceSession(track, start_s=0.0, lateral_offset=-6.0),
        RaceSession(track, start_s=track.length - 40.0, lateral_offset=3.0),
        RaceSession(track, start_s=track.length - 20.0),
    )


def _status_line(label, state, session):
    status = session.status(state)
    return (
        f"{label}: {state['speed'] * 3.6:5.1f} km/h | "
        f"lap {status['laps']} | {session.progress:.0f}/{session.track.length:.0f} m | "
        f"off track {status['off_track_events']}"
    )


def _race_presentation(
    presentation_time,
    overview_duration=1.5,
    zoom_duration=1.5,
    countdown_duration=3.0,
    race_finished=False,
):
    """Return the race-light message, illuminated lights, and green-light state."""
    if race_finished:
        return "FINISH - ONE LAP COMPLETE", 5, True
    countdown_start = overview_duration + zoom_duration
    if presentation_time < countdown_start:
        return None, 0, False

    countdown_elapsed = presentation_time - countdown_start
    if countdown_elapsed < countdown_duration:
        seconds_left = max(1, math.ceil(countdown_duration - countdown_elapsed))
        lights_on = min(5, 1 + 2 * int(countdown_elapsed))
        return str(seconds_left), lights_on, False
    if countdown_elapsed < countdown_duration + 0.8:
        return "GO!", 5, True
    return None, 0, False


def main():
    pygame.init()
    screen = pygame.display.set_mode((1100, 750))
    pygame.display.set_caption("F1 Simulator - Telemetry-derived Monza")
    selection = choose_car(screen)
    if selection is None:
        pygame.quit()
        return

    track = TelemetryTrack.from_csv()
    model = train_telemetry_model(track)
    ai_controller = make_ai_controller(model, predict)
    telemetry = TelemetryReplay.from_csv()
    set_car(selection)

    player, ai, baseline = _new_sessions(track)
    elapsed_time = 0.0
    presentation_time = 0.0
    overview_duration = 1.5
    zoom_duration = 1.5
    countdown_duration = 3.0
    transition_duration = overview_duration + zoom_duration + countdown_duration
    follow_scale = 4.5
    overview_center, overview_scale = fit_track_view(track, screen.get_size())
    race_finished = False
    clock = pygame.time.Clock()
    running = True
    try:
        while running:
            dt = min(clock.tick(60) / 1000, 0.05)
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (
                    event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
                ):
                    running = False
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                    player, ai, baseline = _new_sessions(track)
                    elapsed_time = 0.0
                    presentation_time = 0.0
                    race_finished = False
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_c:
                    selection = choose_car(screen)
                    if selection is None:
                        running = False
                    else:
                        set_car(selection)

            if not running:
                break

            presentation_time += dt

            race_started = presentation_time >= transition_duration
            player_state = player.state
            ai_state = ai.state
            baseline_state = baseline.state
            telemetry_state = telemetry.state_at(elapsed_time)

            if race_started and not race_finished:
                keys = pygame.key.get_pressed()
                player_action = {
                    "steering": 0.28
                    * (
                        int(keys[pygame.K_LEFT] or keys[pygame.K_a])
                        - int(keys[pygame.K_RIGHT] or keys[pygame.K_d])
                    ),
                    "throttle": float(keys[pygame.K_UP] or keys[pygame.K_w]),
                    "brake": float(keys[pygame.K_DOWN] or keys[pygame.K_s]),
                }

                player_state, player_info = player.step(player_action, dt)
                ai_state, _ = ai.step(ai_controller(ai.observation()), dt)
                baseline_state, _ = baseline.step(
                    baseline_action(baseline.observation()),
                    dt,
                )
                elapsed_time += dt
                telemetry_state = telemetry.state_at(elapsed_time)

                cars_for_collision = [
                    {"state": player_state, "session": player},
                    {"state": ai_state, "session": ai},
                    {"state": baseline_state, "session": baseline},
                    {"state": telemetry_state, "ghost": True},
                ]
                resolve_car_collisions(cars_for_collision)
                player_state = player.state
                ai_state = ai.state
                baseline_state = baseline.state
                race_finished = player_info["lap_complete"] or player.laps >= 1

            cars = [
                {
                    "state": player_state,
                    "label": "YOU",
                    "car_index": selection,
                },
                {
                    "state": ai_state,
                    "label": "AI",
                    "car_index": (selection + 1) % 4,
                },
                {
                    "state": baseline_state,
                    "label": "BASELINE",
                    "car_index": (selection + 3) % 4,
                },
                {
                    "state": telemetry_state,
                    "label": "VER",
                    "car_index": (selection + 2) % 4,
                    "ghost": True,
                },
            ]
            zoom_progress = max(
                0.0,
                min(
                    1.0,
                    (presentation_time - overview_duration)
                    / zoom_duration,
                ),
            )
            zoom_progress = zoom_progress * zoom_progress * (
                3 - 2 * zoom_progress
            )
            camera_scale = overview_scale + (
                follow_scale - overview_scale
            ) * zoom_progress
            camera_center = (
                overview_center[0] * (1 - zoom_progress)
                + player_state["x"] * zoom_progress,
                overview_center[1] * (1 - zoom_progress)
                + player_state["y"] * zoom_progress,
            )
            hud_lines = [
                "MONZA REFERENCE: telemetry-derived approximation; widths estimated",
                _status_line("YOU", player_state, player),
                _status_line("AI", ai_state, ai),
                _status_line("BASELINE", baseline_state, baseline),
                (
                    f"VER TELEMETRY: {telemetry_state['speed'] * 3.6:5.1f} km/h | "
                    f"{min(elapsed_time, telemetry.duration):.1f}/"
                    f"{telemetry.duration:.1f} s"
                    + (" | replay finished" if telemetry_state["finished"] else "")
                ),
            ]
            race_message, race_lights, lights_go = _race_presentation(
                presentation_time,
                overview_duration,
                zoom_duration,
                countdown_duration,
                race_finished,
            )

            render(
                track,
                player_state,
                cars=cars,
                hud_lines=hud_lines,
                camera_scale=camera_scale,
                camera_center=camera_center,
                race_message=race_message,
                race_lights=race_lights,
                lights_go=lights_go,
            )
    finally:
        close_renderer()


if __name__ == "__main__":
    main()
