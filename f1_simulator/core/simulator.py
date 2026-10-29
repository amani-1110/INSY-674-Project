"""Track-aware race sessions and the shared simulator API."""

import math

from ..physics.detailed import update_car


class RaceSession:
    """Independent simulation state for one car on a shared track."""

    def __init__(self, track, start_s=0.0, lateral_offset=0.0):
        self.track = track
        self.time = 0.0
        self.start_s = float(start_s) % track.length
        self.progress = 0.0
        self.laps = 0
        self.off_track_events = 0
        self.was_off_track = False

        x, y = track.position(self.start_s)
        heading = track.heading(self.start_s)
        x -= math.sin(heading) * lateral_offset
        y += math.cos(heading) * lateral_offset
        self.state = {
            "x": float(x),
            "y": float(y),
            "heading": float(heading),
            "speed": 0.0,
        }
        self.previous_s = self.start_s

    def _race_info(self, state, lap_complete=False):
        distance, offset = self.track.project(state["x"], state["y"])
        return {
            "distance": float(distance),
            "elapsed_time": self.time,
            "off_track": bool(
                offset > self.track.left_width(distance)
                or offset < -self.track.right_width(distance)
            ),
            "lap_complete": lap_complete,
        }

    def observation(self, state=None):
        state = self.state if state is None else state
        distance, offset = self.track.project(state["x"], state["y"])
        heading_error = (
            state["heading"] - self.track.heading(distance) + math.pi
        ) % (2 * math.pi) - math.pi
        return {
            "speed": float(state["speed"]),
            "lateral_offset": float(offset),
            "heading_error": float(heading_error),
            "curvature_ahead": [
                float(self.track.curvature(distance + lookahead))
                for lookahead in (0, 10, 20)
            ],
        }

    def step(self, action, dt):
        if not 0 < dt <= 1:
            raise ValueError("dt must be between 0 and 1 second")

        new_state = update_car(self.state, action, dt)
        distance, _ = self.track.project(new_state["x"], new_state["y"])
        delta = (
            distance - self.previous_s + self.track.length / 2
        ) % self.track.length - self.track.length / 2
        if abs(delta) > max(5, (self.state["speed"] + 10) * dt * 2):
            delta = 0.0
        self.progress = max(0.0, self.progress + delta)
        self.previous_s = distance
        self.time += dt

        laps = int(self.progress // self.track.length)
        lap_complete = laps > self.laps
        self.laps = laps
        self.state = new_state

        race_info = self._race_info(new_state, lap_complete)
        if race_info["off_track"] and not self.was_off_track:
            self.off_track_events += 1
        self.was_off_track = race_info["off_track"]
        return new_state, race_info

    def status(self, state=None):
        info = self._race_info(self.state if state is None else state)
        info.update(
            laps=self.laps,
            off_track_events=self.off_track_events,
        )
        return info


_default_session = None


def reset_simulation(track):
    """Reset the module-level session and return its starting state."""
    global _default_session
    _default_session = RaceSession(track)
    return _default_session.state.copy(), _default_session._race_info(
        _default_session.state
    )


def simulate_step(state, action, dt, track):
    """Advance the module-level session by one timestep."""
    if _default_session is None or _default_session.track is not track:
        raise RuntimeError("Call reset_simulation(track) before simulate_step.")
    return _default_session.step(action, dt)


def get_observation(state, track):
    """Return the controller observation for the module-level session."""
    if _default_session is None or _default_session.track is not track:
        raise RuntimeError("Call reset_simulation(track) before get_observation.")
    return _default_session.observation(state)


def get_race_status(state, track):
    """Return race information and counters for renderers and clients."""
    if _default_session is None or _default_session.track is not track:
        raise RuntimeError("Call reset_simulation(track) before requesting status.")
    return _default_session.status(state)
